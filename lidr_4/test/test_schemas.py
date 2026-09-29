"""Tests del contrato HTTP de estimaciones.

Escriben sueltos, sin herencia de lidr_3. El criterio: cada test falla por una
razón distinta y el nombre dice cuál, para que un rojo apunte a una causa y no
a "algo de schemas".

Dos cosas que no se pueden probar acá, y conviene tener presentes:

- La respuesta real del endpoint. `/estimate` no existe hasta WU6, así que
  nada de lo de abajo toca FastAPI, ni HTTP, ni al proveedor.
- La sincronía entre CONTRATO_MIN/MAX_CHARS (app.config) y el Field de
  EstimationRequest. Son constantes duplicadas porque config.py no puede
  importar al schema (el schema importa get_settings: sería un ciclo), y la
  duplicación la vigila `test_contrato_y_constantes_no_se_desincronizan`.
"""

import pytest
from pydantic import ValidationError

from app.config import (
    CONTRATO_MAX_CHARS,
    CONTRATO_MIN_CHARS,
)
from app.schemas.estimation import (
    EstimationRequest,
    EstimationResponse,
)

# Values válidos de los tres enums, para no repetirlos en cada test.
PETICION_VALIDA = {
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}

# "x" * 20: 20 caracteres exactos, el mínimo del contrato.
DESC_MINIMA = "x" * CONTRATO_MIN_CHARS
DESC_MAXIMA = "x" * CONTRATO_MAX_CHARS


def peticion(**cambios) -> dict:
    """Petición válida con los campos de `cambios` sobrescritos."""
    return {**PETICION_VALIDA, **cambios}


def construir(descripcion: str, **cambios) -> EstimationRequest:
    return EstimationRequest(description=descripcion, **peticion(**cambios))


# --- El contrato: lo que el schema declara y Swagger documenta.


def test_contrato_y_constantes_no_se_desincronizan():
    """La duplicación de CONTRATO_* y el Field no puede quedar vieja.

    config.py no puede importar el schema (el schema importa get_settings), así
    que los números 20/2000 viven en dos lugares. Este test es la única red:
    si alguien cambia uno y no el otro, la app valida 50 mientras Swagger dice
    20, que es el bug exacto que tenían los defaults de WU1."""
    description = EstimationRequest.model_json_schema()["properties"]["description"]
    assert description["minLength"] == CONTRATO_MIN_CHARS
    assert description["maxLength"] == CONTRATO_MAX_CHARS


@pytest.mark.parametrize(
    ("longitud", "acepta"),
    [
        (CONTRATO_MIN_CHARS - 1, False),
        (CONTRATO_MIN_CHARS, True),
        (CONTRATO_MAX_CHARS, True),
        (CONTRATO_MAX_CHARS + 1, False),
    ],
)
def test_limites_del_contrato_en_los_bordes(longitud, acepta):
    """Los cuatro bordes exactos. Los intermedios no aportan: la validación de
    longitud no tiene分支, es una comparación."""
    if acepta:
        assert construir("x" * longitud).description == "x" * longitud
    else:
        with pytest.raises(ValidationError):
            construir("x" * longitud)


def test_swagger_emite_los_enums_como_listas_de_valores():
    """El cliente tiene que poder generar sus propios desplegables desde esto.

    Si los enums documentaran nombres内部 en vez de valores, un cliente en
    otro lenguaje no podría mapearlos. `use_enum_values` está apagado a
    propósito, así que el valor en el JSON Schema es el valor del cable."""
    defs = EstimationRequest.model_json_schema()["$defs"]
    assert defs["ProjectType"]["enum"] == [
        "mobile_app",
        "web_saas",
        "internal_tool",
        "data_pipeline",
    ]
    assert defs["DetailLevel"]["enum"] == ["summary", "medium", "detailed"]
    assert defs["OutputFormat"]["enum"] == ["phases_table", "line_items", "narrative"]


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        ("project_type", "mobile_app"),
        ("project_type", "internal_tool"),
        ("project_type", "data_pipeline"),
        ("detail_level", "summary"),
        ("detail_level", "detailed"),
        ("output_format", "line_items"),
        ("output_format", "narrative"),
    ],
)
def test_todos_los_valores_de_los_enums_son_validos(campo, valor):
    """Ningún valor de los tres enums puede quedar sin probar: son 9 en total y
    un typo en la definición de uno no se ve en el código que los usa."""
    assert construir(DESC_MINIMA, **{campo: valor}) is not None


@pytest.mark.parametrize("campo", list(PETICION_VALIDA))
def test_los_tres_campos_del_formulario_son_obligatorios(campo):
    """Sin default, los tres son obligatorios: una request sin tipo de proyecto
    no es estimable. Sin `default`, Pydantic los marca required en el schema y
    el 422 llega antes de gastar un token."""
    incomplete = peticion()
    del incomplete[campo]
    with pytest.raises(ValidationError) as error:
        EstimationRequest(description=DESC_MINIMA, **incomplete)
    assert campo in str(error.value)


def test_la_respuesta_expone_texto_y_version_de_prompt():
    """`prompt_version` viaja en la respuesta y no solo en los headers: es lo
    que permite correlacionar una estimación con la plantilla que la generó,
    que es lo que hace falta para invalidar caché al cambiarla."""
    respuesta = EstimationResponse(text="3 fases, 6 semanas", prompt_version="v1")
    assert respuesta.text == "3 fases, 6 semanas"
    assert respuesta.prompt_version == "v1"


# --- El techo del operador: Settings, y solo puede estrechar.


def test_el_techo_por_defecto_es_el_contrato(settings_predeterminada):
    """El default no es un valor más permisivo a propósito. Con 50/5000 el
    `min_length=20` del schema no se cumpliría nunca y nadie se enteraría."""
    assert settings_predeterminada.descripcion_min_chars == CONTRATO_MIN_CHARS
    assert settings_predeterminada.descripcion_max_chars == CONTRATO_MAX_CHARS


@pytest.mark.parametrize(
    ("min_chars", "max_chars", "motivo"),
    [
        (50, 2000, "estrechar el mínimo"),
        (20, 500, "bajar el techo de coste"),
        (100, 300, "estrechar ambos"),
        (2000, 2000, "dejar un único largo válido"),
    ],
)
def test_aceptar_techos_que_estrechan_el_rango(construir_settings, min_chars, max_chars, motivo):
    """Estrechar es la razón de ser del techo: bajar el máximo sin redeploy.
    Todas estas configs arrancan."""
    settings = construir_settings(descripcion_min_chars=min_chars, descripcion_max_chars=max_chars)
    assert settings.descripcion_min_chars == min_chars, motivo
    assert settings.descripcion_max_chars == max_chars, motivo


@pytest.mark.parametrize(
    ("min_chars", "max_chars", "variable"),
    [
        (CONTRATO_MIN_CHARS - 1, 2000, "DESCRIPCION_MIN_CHARS"),
        (20, CONTRATO_MAX_CHARS + 1, "DESCRIPCION_MAX_CHARS"),
        (10, 9999, "DESCRIPCION_MIN_CHARS"),
    ],
)
def test_rechazar_techos_mas_laxos_que_el_contrato(
    construir_settings, min_chars, max_chars, variable
):
    """Un techo más laxo rompe la documentación: el `Field` sigue cortando en
    20/2000 pero el parche de OpenAPI escribe el número laxo, así que Swagger
    promete un rango que el servicio no acepta.

    Falla al arrancar, no en runtime: si pasara, el síntoma sería un 422
    inexplicable en producción."""
    with pytest.raises(ValidationError) as error:
        construir_settings(descripcion_min_chars=min_chars, descripcion_max_chars=max_chars)
    assert variable in str(error.value)


def test_rechazar_un_rango_vacio(construir_settings):
    """min > max deja el campo sin ningún valor válido. Es un `.env` mal
    puesto, no un caso degenerado aceptable."""
    with pytest.raises(ValidationError, match="ningún texto sería aceptado"):
        construir_settings(descripcion_min_chars=500, descripcion_max_chars=100)


def test_el_techo_estrechado_rechaza_lo_que_el_contrato_aceptaba(monkeypatch):
    """El efecto observable de la C: con el máximo bajado a 500, una descripción
    de 600 caracteres es legal según el contrato y la rechaza el servicio.

    Es la reducción de coste sin redespliegue, que es lo que `lidr_3` buscaba y
    lo que un `Field` fijo no puede dar.

    No hace falta vaciar la caché a mano: `limpiar_cache_settings` es autouse y
    corre antes que el cuerpo del test, así que el setenv de acá es lo primero
    que ve la lectura de Settings."""
    monkeypatch.setenv("DESCRIPCION_MAX_CHARS", "500")

    # 600 caracteres: dentro de 20/2000, fuera del techo de 500.
    with pytest.raises(ValidationError, match="500"):
        construir("x" * 600)

    # Y un texto que el nuevo techo sí permite pasa sin tocar el schema.
    assert construir("x" * 500) is not None


def test_el_mensaje_del_techo_diga_la_variable_que_hay_que_cambiar(monkeypatch):
    """El 422 tiene que ser accionable: un cliente que recibe "500 caracteres"
    sin más no puede saber que hay un `.env` detrás. El número recibido va en el
    mensaje a propósito, para que se vea la diferencia."""
    monkeypatch.setenv("DESCRIPCION_MAX_CHARS", "500")

    with pytest.raises(ValidationError) as error:
        construir("x" * 600)
    assert "500" in str(error.value)
    assert "600" in str(error.value)


@pytest.mark.parametrize("descripcion", [DESC_MINIMA, "y" * 100, DESC_MAXIMA])
def test_un_techo_igual_al_contrato_no_cambia_nada(monkeypatch, descripcion):
    """Default = contrato. En ese punto el validador de Settings no puede
    disparar: es una no-op, que es exactamente lo que se busca."""
    monkeypatch.setenv("DESCRIPCION_MIN_CHARS", str(CONTRATO_MIN_CHARS))
    monkeypatch.setenv("DESCRIPCION_MAX_CHARS", str(CONTRATO_MAX_CHARS))
    assert construir(descripcion) is not None


# --- `log_level`: dominio cerrado sin romper la regla de "vacío = default".
#
# No es parte del contrato HTTP, pero el `Literal` lo introduction una
# regresión real que casi se cuela: `Literal` rechaza `""` antes de que
# `aplicar_defaults` pueda convertirlo, así que un `LOG_LEVEL=` vacío pasaba a
# romper el arranque. El principio 2 del docstring de config.py dice que cadena
# vacía = "no configurado". Estos tests lo fijan.


def test_log_level_vacio_usa_el_default_no_rompe_el_arranque(construir_settings):
    """`LOG_LEVEL=` en un .env a medio completar no puede tumbar el servicio.

    Es el principio 2 de lidr_3, y el `Literal` lo rompía: sin el
    BeforeValidator, `""` moría en la validación de dominio."""
    assert construir_settings(log_level="").log_level == "INFO"


@pytest.mark.parametrize("nivel", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_log_level_acepta_los_cinco_niveles(construir_settings, nivel):
    assert construir_settings(log_level=nivel).log_level == nivel


@pytest.mark.parametrize("nivel", ["info", "trace", "warn", "fatal", "  "])
def test_log_level_invalido_falla_al_arrancar(construir_settings, nivel):
    """Un valor presente pero inválido muere acá, no a mitad de una request.

    `info` en minúscula NO se normaliza a `INFO` a propósito: es un `.env` mal
    puesto, y silenciarlo a un default esconde el error. La normalización solo
    cubre el vacío."""
    with pytest.raises(ValidationError) as error:
        construir_settings(log_level=nivel)
    assert "log_level" in str(error.value)


def test_el_error_de_log_level_lista_los_validos(construir_settings):
    """El valor de `Literal` es mejor que el de la stdlib justamente por esto:
    `Unknown level: 'info'` no le dice a nadie qué escribir en el `.env`."""
    with pytest.raises(ValidationError) as error:
        construir_settings(log_level="info")
    mensaje = str(error.value)
    for nivel in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        assert nivel in mensaje
