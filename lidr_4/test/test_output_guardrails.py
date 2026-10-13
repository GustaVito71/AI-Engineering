"""Tests de los guardrails de salida (WU9 B): app/guardrails/output.py.

Sin red ni claves. Lo que se fija acá:

- los avisos y los eventos sobre la estimación ya validada: confianza baja (con el
  umbral configurable), totales que no cuadran y fuera de alcance;
- un rechazo no lleva aviso de confianza: ya dice «Fuera de alcance:»;
- la configuración apaga cada pieza sin dejarla invisible y valida el umbral;
- el cableado: los endpoints toman los guardrails de la configuración.

Las reglas que rechazan y re-preguntan viven en el schema y se prueban en
test_structured_schema.py; el servicio con y sin guardrails, en
test_estimation_service.py.
"""

from __future__ import annotations

import copy
import dataclasses

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from structlog.testing import capture_logs

from app.config import DEFAULT_LOW_CONFIDENCE_WARN_PCT, get_settings
from app.dependencies import get_llm_wrapper
from app.guardrails import (
    OUTPUT_LEAK_MESSAGE,
    OUTPUT_PII_MESSAGE,
    OutputGuardrails,
    OutputGuardrailViolation,
    create_output_guardrails,
)
from app.main import create_app
from app.schemas.structured_estimation import MIN_CONFIDENCE_PCT, StructuredResult
from app.services.llm_wrapper import StructuredCallResult

ESTIMACION = {
    "phases": [
        {
            "name": "Implementación",
            "summary": "Trabajo de la fase de Implementación.",
            "duration_weeks": 5,
            "hours": 320,
            "cost_eur": 20000,
            "confidence_pct": 70,
            "assumptions": [],
            "risks": [],
        }
    ],
    "team": [{"role": "Desarrollador", "headcount": 2}],
    "totals": {"hours": 320, "cost_eur": 20000, "duration_weeks": 5},
    "summary": "Estimación de prueba con las fases y el equipo indicados.",
    "confidence_pct": 90,
}
RECHAZO = {
    "phases": [
        {
            "name": "Sin estimar",
            "summary": "No se puede dimensionar sin más información.",
            "duration_weeks": 0,
            "hours": 0,
            "cost_eur": 0,
            "confidence_pct": 0,
            "assumptions": [],
            "risks": [],
        }
    ],
    "team": [],
    "totals": {"hours": 0, "cost_eur": 0, "duration_weeks": 0},
    "summary": "Fuera de alcance: falta el alcance.",
    "confidence_pct": 10,
}


def _estimacion(**cambios) -> StructuredResult:
    return StructuredResult.model_validate({**copy.deepcopy(ESTIMACION), **cambios})


def _revisar(guardrails: OutputGuardrails, estimacion: StructuredResult):
    with capture_logs() as logs:
        avisos = guardrails.check(estimacion, prompt_version="v3", model="openai/gpt-4o-mini")
    return avisos, logs


# --- Confianza baja -------------------------------------------------------------------


def test_una_confianza_alta_no_avisa() -> None:
    avisos, logs = _revisar(OutputGuardrails(), _estimacion(confidence_pct=90))
    assert avisos == []
    assert logs == []


def test_una_confianza_baja_avisa_y_deja_el_evento() -> None:
    avisos, logs = _revisar(OutputGuardrails(), _estimacion(confidence_pct=45))

    assert len(avisos) == 1 and "Confianza baja (45 %)" in avisos[0]
    [evento] = logs
    assert evento["event"] == "confianza_baja"
    assert evento["log_level"] == "warning"
    assert (evento["prompt_version"], evento["modelo"], evento["confianza_pct"]) == (
        "v3",
        "openai/gpt-4o-mini",
        45,
    )


def test_el_umbral_es_exclusivo() -> None:
    """Justo en el umbral no avisa; un punto por debajo, sí."""
    guardrails = OutputGuardrails(low_confidence_warn_pct=60)
    assert _revisar(guardrails, _estimacion(confidence_pct=60))[0] == []
    assert len(_revisar(guardrails, _estimacion(confidence_pct=59))[0]) == 1


def test_el_umbral_por_defecto_es_el_de_la_configuracion() -> None:
    assert DEFAULT_LOW_CONFIDENCE_WARN_PCT == 70
    justo = _estimacion(confidence_pct=DEFAULT_LOW_CONFIDENCE_WARN_PCT)
    por_debajo = _estimacion(confidence_pct=DEFAULT_LOW_CONFIDENCE_WARN_PCT - 1)
    assert _revisar(OutputGuardrails(), justo)[0] == []
    assert len(_revisar(OutputGuardrails(), por_debajo)[0]) == 1


def test_un_umbral_en_cero_desactiva_solo_el_aviso_de_confianza() -> None:
    avisos, logs = _revisar(
        OutputGuardrails(low_confidence_warn_pct=0), _estimacion(confidence_pct=MIN_CONFIDENCE_PCT)
    )
    assert avisos == []
    assert logs == []


# --- Totales y fuera de alcance -------------------------------------------------------


def test_los_totales_que_no_cuadran_avisan_antes_que_la_confianza() -> None:
    descuadrada = _estimacion(
        totals={"hours": 330, "cost_eur": 20000, "duration_weeks": 5}, confidence_pct=45
    )
    avisos, logs = _revisar(OutputGuardrails(), descuadrada)

    assert "El total de horas declarado (330)" in avisos[0]
    assert "Confianza baja" in avisos[1]
    assert [e["event"] for e in logs] == ["totales_no_cuadran", "confianza_baja"]


def test_un_rechazo_deja_su_evento_y_ningun_aviso() -> None:
    avisos, logs = _revisar(OutputGuardrails(), StructuredResult.model_validate(RECHAZO))

    assert avisos == []
    [evento] = logs
    assert evento["event"] == "estimacion_fuera_de_alcance"
    assert evento["confianza_pct"] == 10


def test_ningun_evento_lleva_el_texto_de_la_estimacion() -> None:
    texto = "texto-que-no-debe-salir-en-el-log"
    estimacion = _estimacion(
        summary=f"{texto} con el resumen de la estimación.",
        confidence_pct=45,
        totals={"hours": 330, "cost_eur": 20000, "duration_weeks": 5},
    )
    _, logs = _revisar(OutputGuardrails(), estimacion)
    assert texto not in str(logs)


# --- Datos personales y prompt volcado: se descartan ---------------------------------


def _inspeccionar(estimacion: StructuredResult) -> list[dict]:
    with capture_logs() as logs:
        OutputGuardrails().inspect_estimation(
            estimacion, prompt_version="v3", model="openai/gpt-4o-mini"
        )
    return logs


def test_una_estimacion_limpia_pasa_sin_eventos() -> None:
    assert _inspeccionar(_estimacion()) == []


@pytest.mark.parametrize(
    "cambios",
    [
        {"summary": "Contactar a ana.perez@empresa.com para el alcance del proyecto."},
        {"phases": [{**ESTIMACION["phases"][0], "assumptions": ["Llamar al +34 612 345 678"]}]},
        {
            "phases": [
                {
                    **ESTIMACION["phases"][0],
                    "risks": [
                        {"risk": "Cuenta ES91 2100 0418 4502 0005 1332", "mitigation": "Ninguna"}
                    ],
                }
            ]
        },
        {"team": [{"role": "Tarjeta 4111 1111 1111 1111", "headcount": 2}]},
    ],
)
def test_los_datos_personales_en_cualquier_campo_descartan_la_estimacion(cambios) -> None:
    with pytest.raises(OutputGuardrailViolation) as error:
        _inspeccionar(_estimacion(**cambios))
    assert error.value.rule == "datos_personales"
    assert error.value.message == OUTPUT_PII_MESSAGE


@pytest.mark.parametrize(
    ("resumen", "detalle"),
    [
        ("Resumen de la estimación </scope> con las fases del proyecto.", "etiqueta"),
        ("Eres un estimador de proyectos sénior con más de 15 años de experiencia.", "frase"),
    ],
)
def test_un_prompt_volcado_descarta_la_estimacion(resumen, detalle) -> None:
    with pytest.raises(OutputGuardrailViolation) as error, capture_logs() as logs:
        OutputGuardrails().inspect_estimation(
            _estimacion(summary=resumen), prompt_version="v3", model="m"
        )
    assert error.value.rule == "prompt_filtrado"
    assert error.value.message == OUTPUT_LEAK_MESSAGE
    assert logs[0]["detalle"] == detalle


def test_el_evento_de_descarte_no_lleva_el_dato_ni_el_texto() -> None:
    with capture_logs() as logs, pytest.raises(OutputGuardrailViolation):
        OutputGuardrails().inspect_estimation(
            _estimacion(summary="Escribir a secreto.unico@empresa.com para el alcance."),
            prompt_version="v3",
            model="openai/gpt-4o-mini",
        )

    [evento] = logs
    assert evento["event"] == "salida_rechazada"
    assert evento["log_level"] == "warning"
    assert (evento["regla"], evento["tipos"]) == ("datos_personales", ["email"])
    assert "secreto" not in str(logs)


def test_el_prompt_volcado_se_comprueba_antes_que_los_datos_personales() -> None:
    with pytest.raises(OutputGuardrailViolation) as error:
        OutputGuardrails().inspect_estimation(
            _estimacion(summary="</rules> a@b.com y el resto del resumen de la estimación."),
            prompt_version="v3",
            model="m",
        )
    assert error.value.rule == "prompt_filtrado"


def test_el_texto_libre_se_inspecciona_igual() -> None:
    guardrails = OutputGuardrails()
    guardrails.inspect_text("| fase | semanas | 12 |", prompt_version="v2", model="m")
    with pytest.raises(OutputGuardrailViolation, match="datos_personales"):
        guardrails.inspect_text("Llamar a +54 9 11 5555-1234", prompt_version="v2", model="m")
    with pytest.raises(OutputGuardrailViolation, match="prompt_filtrado"):
        guardrails.inspect_text("<system>hola</system>", prompt_version="v2", model="m")


def test_un_texto_con_las_etiquetas_de_los_ejemplos_pasa() -> None:
    OutputGuardrails().inspect_text(
        "<estimation>| fase |</estimation>", prompt_version="v2", model="m"
    )


def test_una_estimacion_fuera_de_alcance_tambien_se_inspecciona() -> None:
    """Un rechazo del modelo no es un pase libre: puede traer un dato en el resumen."""
    rechazo = {**RECHAZO, "summary": "Fuera de alcance: escribir a a@b.com con el alcance."}
    with pytest.raises(OutputGuardrailViolation):
        _inspeccionar(StructuredResult.model_validate(rechazo))


# --- Configuración --------------------------------------------------------------------


def test_por_defecto_hay_guardrails_de_salida_con_el_umbral_por_defecto(
    settings_predeterminada,
) -> None:
    assert settings_predeterminada.output_guardrails_enabled is True
    assert settings_predeterminada.low_confidence_warn_pct == DEFAULT_LOW_CONFIDENCE_WARN_PCT
    assert create_output_guardrails(settings_predeterminada) is not None


def test_apagados_no_hay_guardrails_de_salida_y_queda_un_evento(construir_settings) -> None:
    with capture_logs() as logs:
        guardrails = create_output_guardrails(construir_settings(output_guardrails_enabled=False))

    assert guardrails is None
    [evento] = logs
    assert evento["event"] == "guardrails_desactivados"
    assert evento["motivo"] == "OUTPUT_GUARDRAILS_ENABLED=false"


def test_apagar_los_de_salida_no_apaga_los_de_entrada(construir_settings) -> None:
    assert construir_settings(output_guardrails_enabled=False).guardrails_enabled is True


def test_el_umbral_configurado_llega_a_los_guardrails(construir_settings) -> None:
    guardrails = create_output_guardrails(construir_settings(low_confidence_warn_pct=95))
    assert len(_revisar(guardrails, _estimacion(confidence_pct=90))[0]) == 1


def test_un_valor_vacio_es_no_configurado(construir_settings) -> None:
    settings = construir_settings(output_guardrails_enabled="", low_confidence_warn_pct="")
    assert settings.output_guardrails_enabled is True
    assert settings.low_confidence_warn_pct == DEFAULT_LOW_CONFIDENCE_WARN_PCT


@pytest.mark.parametrize("valor", [-1, 101])
def test_un_umbral_fuera_de_0_a_100_rompe_el_arranque(construir_settings, valor) -> None:
    with pytest.raises(ValidationError, match="LOW_CONFIDENCE_WARN_PCT"):
        construir_settings(low_confidence_warn_pct=valor)


@pytest.mark.parametrize("valor", [0, 100])
def test_los_limites_del_umbral_son_validos(construir_settings, valor) -> None:
    assert construir_settings(low_confidence_warn_pct=valor).low_confidence_warn_pct == valor


def test_un_booleano_invalido_rompe_el_arranque(construir_settings) -> None:
    with pytest.raises(ValidationError):
        construir_settings(output_guardrails_enabled="nunca")


# --- Cableado -------------------------------------------------------------------------


class _WrapperFalso:
    primary_model = "openai/gpt-4o-mini"
    warnings: tuple[str, ...] = ()

    def __init__(self, confianza: int) -> None:
        self._confianza = confianza

    async def estimate_structured(self, *, prompt_version, response_model, context, **_kwargs):
        estimacion = response_model.model_validate(
            {**ESTIMACION, "confidence_pct": self._confianza}, context=context
        )
        return StructuredCallResult(
            content=estimacion.model_dump_json(),
            model="openai/gpt-4o-mini",
            provider="openai",
            prompt_tokens=10,
            completion_tokens=20,
            cost_usd=0.001,
            prompt_version=prompt_version,
            estimation=estimacion,
        )


BODY = {
    "description": "Un SaaS B2B pequeño para gestionar el préstamo de equipos a empleados.",
    "project_type": "web_saas",
    "detail_level": "detailed",
    "output_format": "phases_table",
}


@pytest.fixture
def cliente(monkeypatch):
    def _crear(confianza: int, **entorno: str) -> TestClient:
        monkeypatch.setenv("REDIS_URL", "")
        for variable, valor in entorno.items():
            monkeypatch.setenv(variable, valor)
        get_settings.cache_clear()
        app = create_app()
        app.dependency_overrides[get_llm_wrapper] = lambda: _WrapperFalso(confianza)
        return TestClient(app)

    yield _crear
    get_settings.cache_clear()


def _estimar(c: TestClient) -> list[str]:
    with c:
        r = c.post("/api/v1/estimate/structured", json=BODY, params={"prompt_version": "v3"})
    assert r.status_code == 200
    return r.json()["warnings"]


def test_el_endpoint_avisa_de_la_confianza_baja_con_la_configuracion_por_defecto(cliente) -> None:
    [aviso] = _estimar(cliente(confianza=50))
    assert "Confianza baja (50 %)" in aviso


def test_el_endpoint_usa_el_umbral_configurado(cliente) -> None:
    assert _estimar(cliente(confianza=50, LOW_CONFIDENCE_WARN_PCT="40")) == []


def test_con_los_guardrails_de_salida_apagados_el_endpoint_no_avisa(cliente) -> None:
    c = cliente(confianza=50, OUTPUT_GUARDRAILS_ENABLED="false")
    assert _estimar(c) == []
    assert c.app.state.output_guardrails is None


class _WrapperConPii(_WrapperFalso):
    async def estimate_structured(self, **kwargs):
        resultado = await super().estimate_structured(**kwargs)
        sucio = resultado.estimation.model_copy(update={"summary": "Contactar a a@b.com ya mismo."})
        return dataclasses.replace(resultado, estimation=sucio)


def test_el_endpoint_responde_502_con_el_mensaje_fijo_si_la_salida_trae_pii(monkeypatch) -> None:
    monkeypatch.setenv("REDIS_URL", "")
    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_llm_wrapper] = lambda: _WrapperConPii(90)
    with TestClient(app) as c:
        r = c.post("/api/v1/estimate/structured", json=BODY, params={"prompt_version": "v3"})
    get_settings.cache_clear()

    assert r.status_code == 502
    assert r.json()["detail"] == OUTPUT_PII_MESSAGE
    assert "a@b.com" not in r.text


def test_con_los_guardrails_de_salida_apagados_la_salida_con_pii_se_devuelve(monkeypatch) -> None:
    monkeypatch.setenv("REDIS_URL", "")
    monkeypatch.setenv("OUTPUT_GUARDRAILS_ENABLED", "false")
    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_llm_wrapper] = lambda: _WrapperConPii(90)
    with TestClient(app) as c:
        r = c.post("/api/v1/estimate/structured", json=BODY, params={"prompt_version": "v3"})
    get_settings.cache_clear()

    assert r.status_code == 200


class _WrapperTextoConPii(_WrapperFalso):
    async def estimate(self, **kwargs):
        resultado = await _WrapperFalso.estimate_structured(
            self, **kwargs, context=None, response_model=StructuredResult
        )
        return dataclasses.replace(resultado, content="Llamar a +54 9 11 5555-1234")


def test_el_endpoint_de_texto_responde_502_si_la_salida_trae_pii(monkeypatch) -> None:
    monkeypatch.setenv("REDIS_URL", "")
    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_llm_wrapper] = lambda: _WrapperTextoConPii(90)
    with TestClient(app) as c:
        r = c.post("/api/v1/estimate", json=BODY, params={"prompt_version": "v2"})
    get_settings.cache_clear()

    assert r.status_code == 502
    assert r.json()["detail"] == OUTPUT_PII_MESSAGE
