"""Tests de `EstimationService`: el pipeline entre el router y el LLMWrapper.

El wrapper se reemplaza por uno falso (sin red ni Redis). Lo que se verifica es
lo que hace el servicio y no el router ni el wrapper:

- arma el prompt de la versión pedida y se lo pasa al wrapper;
- quita las etiquetas del prompt que el modelo copia en el texto;
- junta los avisos del wrapper con los totales que no cuadran, y deja los
  eventos de negocio en el log;
- deja pasar sin tocar los errores del proveedor, para que el router los traduzca;
- un fallo al armar el prompt sale como `PromptRenderError`, que el router no
  convierte en 502.
"""

from __future__ import annotations

import litellm
import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from app.config import get_settings
from app.dependencies import get_llm_wrapper
from app.main import create_app
from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType
from app.schemas.structured_estimation import (
    OUT_OF_SCOPE_PREFIX,
    UNESTIMATED_PHASE,
    StructuredResult,
)
from app.services import estimation as modulo_servicio
from app.services.estimation import (
    EstimationService,
    PromptRenderError,
    StructuredEstimation,
    TextEstimation,
)
from app.services.llm_wrapper import LLMCallResult, StructuredCallResult, StructuredOutputError

DESCRIPCION = "Un SaaS B2B pequeño para gestionar el préstamo de equipos a empleados."
SOLICITUD = EstimationRequest(
    description=DESCRIPCION,
    project_type=ProjectType.WEB_SAAS,
    detail_level=DetailLevel.MEDIUM,
    output_format=OutputFormat.PHASES_TABLE,
)
ESTIMACION = {
    "phases": [
        {
            "name": "Descubrimiento",
            "summary": "Trabajo de la fase de Descubrimiento.",
            "duration_weeks": 1.0,
            "hours": 70,
            "cost_eur": 3950.0,
            "confidence_pct": 85,
            "assumptions": [],
            "risks": [],
        },
        {
            "name": "Implementación",
            "summary": "Trabajo de la fase de Implementación.",
            "duration_weeks": 5.0,
            "hours": 320,
            "cost_eur": 20000.0,
            "confidence_pct": 70,
            "assumptions": [],
            "risks": [],
        },
    ],
    "team": [{"role": "Desarrollador", "headcount": 2}],
    "totals": {"hours": 390, "cost_eur": 23950.0, "duration_weeks": 6.0},
    "summary": "Estimación de prueba con las fases y el equipo indicados.",
    "confidence_pct": 70,
}
FUERA_DE_ALCANCE = {
    "phases": [
        {
            "name": UNESTIMATED_PHASE,
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
    "summary": f"{OUT_OF_SCOPE_PREFIX} falta el alcance.",
    "confidence_pct": 10,
}


class _WrapperFalso:
    """Reemplaza al LLMWrapper: devuelve un resultado fijo o lanza `error`."""

    def __init__(
        self,
        *,
        error: Exception | None = None,
        warnings: tuple[str, ...] = (),
        texto: str = "| fase | semanas |",
        estimacion: dict = ESTIMACION,
    ) -> None:
        self._error = error
        self.warnings = warnings
        self._texto = texto
        self._estimacion = estimacion
        self.llamadas: list[dict] = []

    async def estimate(self, *, prompt_version: str, **kwargs) -> LLMCallResult:
        self.llamadas.append({"prompt_version": prompt_version, **kwargs})
        if self._error is not None:
            raise self._error
        return LLMCallResult(
            content=self._texto,
            model="openai/gpt-4o-mini",
            provider="openai",
            prompt_tokens=10,
            completion_tokens=20,
            cost_usd=0.001,
            prompt_version=prompt_version,
        )

    async def estimate_structured(
        self, *, prompt_version: str, response_model, context, **kwargs
    ) -> StructuredCallResult:
        self.llamadas.append(
            {
                "prompt_version": prompt_version,
                "response_model": response_model,
                "context": context,
                **kwargs,
            }
        )
        if self._error is not None:
            raise self._error
        estimacion = response_model.model_validate(self._estimacion, context=context)
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


# --- Texto libre ----------------------------------------------------------------------


async def test_el_texto_arma_el_prompt_de_la_version_pedida() -> None:
    wrapper = _WrapperFalso()
    resultado = await EstimationService(wrapper).estimate_text(SOLICITUD, "v2")

    [llamada] = wrapper.llamadas
    assert llamada["prompt_version"] == "v2"
    assert llamada["system_prompt"] and DESCRIPCION in llamada["user_prompt"]
    assert resultado == TextEstimation(text="| fase | semanas |", prompt_version="v2", warnings=[])


async def test_el_texto_pierde_las_etiquetas_que_el_modelo_copia_del_prompt() -> None:
    wrapper = _WrapperFalso(texto="<estimation>\n| fase | semanas |\n</estimation>\n")
    resultado = await EstimationService(wrapper).estimate_text(SOLICITUD, "v2")
    assert resultado.text == "| fase | semanas |"


async def test_el_texto_lleva_los_avisos_del_wrapper() -> None:
    wrapper = _WrapperFalso(warnings=("Sin respaldo.",))
    resultado = await EstimationService(wrapper).estimate_text(SOLICITUD, "v2")
    assert resultado.warnings == ["Sin respaldo."]


# --- Salida estructurada --------------------------------------------------------------


async def test_lo_estructurado_pasa_el_schema_y_el_contexto_al_wrapper() -> None:
    wrapper = _WrapperFalso()
    resultado = await EstimationService(wrapper).estimate_structured(
        SOLICITUD, "v3", StructuredResult, context={"output_format": "phases_table"}
    )

    [llamada] = wrapper.llamadas
    assert llamada["prompt_version"] == "v3"
    assert llamada["response_model"] is StructuredResult
    assert llamada["context"] == {"output_format": "phases_table"}
    assert isinstance(resultado, StructuredEstimation)
    assert resultado.call.estimation.totals.hours == 390
    assert resultado.warnings == []


async def test_los_totales_que_no_cuadran_son_un_aviso_y_un_evento() -> None:
    descuadrada = {**ESTIMACION, "totals": {**ESTIMACION["totals"], "hours": 400}}
    wrapper = _WrapperFalso(warnings=("Sin respaldo.",), estimacion=descuadrada)

    with capture_logs() as logs:
        resultado = await EstimationService(wrapper).estimate_structured(
            SOLICITUD, "v3", StructuredResult
        )

    assert resultado.warnings == [
        "Sin respaldo.",
        "El total de horas declarado (400) no coincide con la suma de las fases (390).",
    ]
    [evento] = [e for e in logs if e["event"] == "totales_no_cuadran"]
    assert evento["prompt_version"] == "v3"
    assert evento["modelo"] == "openai/gpt-4o-mini"


async def test_una_estimacion_fuera_de_alcance_deja_un_evento_y_no_es_un_fallo() -> None:
    wrapper = _WrapperFalso(estimacion=FUERA_DE_ALCANCE)

    with capture_logs() as logs:
        resultado = await EstimationService(wrapper).estimate_structured(
            SOLICITUD, "v3", StructuredResult
        )

    assert resultado.call.estimation.out_of_scope
    assert resultado.warnings == []
    [evento] = [e for e in logs if e["event"] == "estimacion_fuera_de_alcance"]
    assert evento["confianza_pct"] == 10


async def test_una_estimacion_normal_no_deja_eventos_de_negocio() -> None:
    with capture_logs() as logs:
        await EstimationService(_WrapperFalso()).estimate_structured(
            SOLICITUD, "v3", StructuredResult
        )
    assert [e["event"] for e in logs] == []


# --- Errores ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        litellm.Timeout(message="lento", model="gpt-4o-mini", llm_provider="openai"),
        StructuredOutputError(attempts=3, cost_usd=0.004, last_error="rendered sin tabla"),
        RuntimeError("el proveedor falló"),
    ],
)
async def test_los_errores_del_proveedor_llegan_tal_cual(error) -> None:
    """El servicio no traduce errores: eso lo hace el router."""
    servicio = EstimationService(_WrapperFalso(error=error))

    with pytest.raises(type(error)) as texto:
        await servicio.estimate_text(SOLICITUD, "v2")
    assert texto.value is error

    with pytest.raises(type(error)) as estructurada:
        await servicio.estimate_structured(SOLICITUD, "v3", StructuredResult)
    assert estructurada.value is error


async def test_un_fallo_al_armar_el_prompt_es_un_error_propio(monkeypatch) -> None:
    def roto(*_args, **_kwargs):
        raise ValueError("plantilla rota")

    monkeypatch.setattr(modulo_servicio, "render_estimation_prompt", roto)
    wrapper = _WrapperFalso()
    servicio = EstimationService(wrapper)

    with pytest.raises(PromptRenderError, match="No se pudo armar el prompt v2: ValueError"):
        await servicio.estimate_text(SOLICITUD, "v2")
    with pytest.raises(PromptRenderError, match="No se pudo armar el prompt v3"):
        await servicio.estimate_structured(SOLICITUD, "v3", StructuredResult)
    assert wrapper.llamadas == []


# --- Con el router ---------------------------------------------------------------------


@pytest.fixture
def cliente(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "")
    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_llm_wrapper] = lambda: _WrapperFalso()
    # Sin raise_server_exceptions, un error interno llega como 500 y no como excepción.
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    get_settings.cache_clear()


@pytest.mark.parametrize(
    ("ruta", "version"),
    [
        ("/api/v1/estimate", "v2"),
        ("/api/v1/estimate/structured", "v3"),
        ("/api/v1/estimate/rendered", "v4"),
    ],
)
def test_un_prompt_roto_es_un_500_y_no_un_502_del_proveedor(
    cliente, monkeypatch, ruta, version
) -> None:
    def roto(*_args, **_kwargs):
        raise ValueError("plantilla rota")

    monkeypatch.setattr(modulo_servicio, "render_estimation_prompt", roto)
    cuerpo = {
        "description": DESCRIPCION,
        "project_type": "web_saas",
        "detail_level": "medium",
        "output_format": "phases_table",
    }

    r = cliente.post(ruta, json=cuerpo, params={"prompt_version": version})

    assert r.status_code == 500
    assert "proveedor" not in r.text
