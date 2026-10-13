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

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from structlog.testing import capture_logs

from app.config import DEFAULT_LOW_CONFIDENCE_WARN_PCT, get_settings
from app.dependencies import get_llm_wrapper
from app.guardrails import OutputGuardrails, create_output_guardrails
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
