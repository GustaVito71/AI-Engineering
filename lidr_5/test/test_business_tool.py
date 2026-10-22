"""Tests de `app/business/` (Paso 12): cliente al backend de negocio y herramienta
`find_similar_projects`. Todos con falsos: `httpx.MockTransport`, sin red ni backend real.

QUÉ SE COMPRUEBA

- Cliente: construye la petición con la clave y la consulta correctas, parsea la respuesta,
  lanza `BusinessBackendUnavailableError` ante un error HTTP o una respuesta inválida, y la
  clave nunca aparece en un mensaje de error.
- Herramienta: formato del texto, tope de consultas por petición, mensaje cuando el backend
  no está disponible, mensaje cuando no hay resultados, y registro condicional por Settings.
"""

from __future__ import annotations

import httpx
import pytest
from structlog.testing import capture_logs

from app.business.client import BusinessBackendClient, BusinessBackendUnavailableError
from app.business.tool import TOOL_NAME, build_find_similar_projects_tool, build_tool_from_settings
from app.config import Settings
from app.observability import start_turn_cost

CLAVE = "negocio-SECRETO-123"
BASE_URL = "http://business-backend:8003"

PROYECTO = {
    "project_type": "migración de API",
    "hours_estimated": 320,
    "hours_actual": 410,
    "deviation_pct": 28.1,
    "risks_materialized": ["dependencias de terceros desactualizadas"],
}


def _cliente(handler, *, api_key: str = CLAVE) -> BusinessBackendClient:
    """Cliente de prueba sobre un `MockTransport` con el handler dado."""
    return BusinessBackendClient(
        base_url=BASE_URL,
        api_key=api_key,
        timeout_seconds=5,
        transport=httpx.MockTransport(handler),
    )


# ----------------------------------------------------------------------------- cliente
async def test_envia_la_clave_y_la_consulta_correctas() -> None:
    """La petición lleva `X-API-Key` y `query` con lo que se pidió."""
    capturada = {}

    def handler(request: httpx.Request) -> httpx.Response:
        capturada["x_api_key"] = request.headers.get("x-api-key")
        capturada["query"] = request.url.params.get("query")
        return httpx.Response(200, json={"results": [PROYECTO]})

    resultado = await _cliente(handler).find_similar("migración de API")
    assert capturada == {"x_api_key": CLAVE, "query": "migración de API"}
    assert resultado[0].project_type == "migración de API"
    assert resultado[0].deviation_pct == 28.1


async def test_401_lanza_sin_filtrar_la_clave() -> None:
    """Un 401 del backend lanza `BusinessBackendUnavailableError` sin la clave en el texto."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "X-API-Key ausente o incorrecta"})

    with pytest.raises(BusinessBackendUnavailableError) as info:
        await _cliente(handler).find_similar("api")
    assert CLAVE not in str(info.value)


async def test_timeout_lanza() -> None:
    """Un timeout del backend se homogeneiza en `BusinessBackendUnavailableError`."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("lento")

    with pytest.raises(BusinessBackendUnavailableError):
        await _cliente(handler).find_similar("api")


async def test_respeta_el_maximo_de_tres_resultados() -> None:
    """Aunque el backend devuelva 5, el cliente se queda con 3."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [PROYECTO] * 5})

    assert len(await _cliente(handler).find_similar("api")) == 3


# --------------------------------------------------------------------------- herramienta
def _herramienta(handler, *, tope: int = 2):
    """La herramienta `find_similar_projects` sobre un backend falso."""
    return build_find_similar_projects_tool(_cliente(handler), max_per_turn=tope)


async def test_el_resultado_es_texto_con_los_datos_clave() -> None:
    """El texto incluye el tipo, las horas, la desviación y los riesgos."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [PROYECTO]})

    texto = await _herramienta(handler).run({"project_type": "migración de API"})
    assert "320" in texto and "410" in texto and "+28.1%" in texto
    assert "dependencias de terceros desactualizadas" in texto


async def test_sin_resultados_lo_dice() -> None:
    """Un backend que no encuentra nada da un texto claro, no una lista vacía muda."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    texto = await _herramienta(handler).run({"project_type": "algo raro"})
    assert "no se encontraron" in texto.lower()


async def test_sin_argumento_no_consulta() -> None:
    """Sin `project_type` no se llama al backend."""
    llamadas = []

    def handler(request: httpx.Request) -> httpx.Response:
        llamadas.append(1)
        return httpx.Response(200, json={"results": []})

    texto = await _herramienta(handler).run({})
    assert "falta" in texto and not llamadas


async def test_el_tope_por_peticion_se_aplica() -> None:
    """La tercera consulta de una petición con tope 2 no llega al backend."""
    start_turn_cost()
    llamadas = []

    def handler(request: httpx.Request) -> httpx.Response:
        llamadas.append(1)
        return httpx.Response(200, json={"results": [PROYECTO]})

    herramienta = _herramienta(handler, tope=2)
    await herramienta.run({"project_type": "uno"})
    await herramienta.run({"project_type": "dos"})
    texto = await herramienta.run({"project_type": "tres"})
    assert "límite de 2" in texto and len(llamadas) == 2


async def test_backend_caido_devuelve_texto_y_no_lanza() -> None:
    """Si el backend no responde, el modelo recibe un aviso, no una excepción."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("caído")

    texto = await _herramienta(handler).run({"project_type": "api"})
    assert "no disponible" in texto


async def test_ningun_evento_ni_error_filtra_la_clave() -> None:
    """Ante un fallo, ni el log ni el texto devuelto contienen la clave."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"caído, clave {CLAVE}")

    with capture_logs() as logs:
        texto = await _herramienta(handler).run({"project_type": "api"})
    assert CLAVE not in texto and CLAVE not in str(logs)


# ------------------------------------------------------------------ Settings y registro
def _settings(**extra) -> Settings:
    """Settings de prueba sin leer el `.env` real."""
    return Settings(_env_file=None, **extra)


def test_la_herramienta_no_se_crea_sin_url_o_sin_clave() -> None:
    """Sin URL, sin clave, o sin ninguna de las dos, no hay herramienta."""
    assert build_tool_from_settings(_settings()) is None
    assert build_tool_from_settings(_settings(business_api_url=BASE_URL)) is None
    assert build_tool_from_settings(_settings(business_api_key=CLAVE)) is None


def test_la_herramienta_se_crea_con_url_y_clave() -> None:
    """Con las dos, la herramienta existe y se llama `find_similar_projects`."""
    herramienta = build_tool_from_settings(
        _settings(business_api_url=BASE_URL, business_api_key=CLAVE)
    )
    assert herramienta is not None and herramienta.name == TOOL_NAME == "find_similar_projects"


@pytest.mark.parametrize("campo", ["business_max_per_turn", "business_timeout_seconds"])
def test_limites_invalidos_fallan_al_arrancar(campo) -> None:
    """Un límite en cero es un error de configuración, igual que en el Paso 11."""
    with pytest.raises(ValueError):
        _settings(**{campo: 0})


def test_valores_por_defecto_aprobados() -> None:
    """Los valores acordados: 2 consultas por petición, 5 segundos de timeout."""
    s = _settings()
    assert (s.business_max_per_turn, s.business_timeout_seconds) == (2, 5.0)


def test_el_registro_incluye_find_similar_projects_con_bucle_y_backend(monkeypatch) -> None:
    """Bucle encendido + URL y clave del backend → `find_similar_projects` registrada."""
    from types import SimpleNamespace as NS

    from app import dependencies

    monkeypatch.setattr(
        dependencies,
        "get_settings",
        lambda: _settings(
            agent_loop_enabled=True, business_api_url=BASE_URL, business_api_key=CLAVE
        ),
    )
    peticion = NS(app=NS(state=NS()))
    registro = dependencies.get_tool_registry(peticion)
    assert registro.names == ["find_similar_projects"]
