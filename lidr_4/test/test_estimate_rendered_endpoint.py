"""Tests de POST /api/v1/estimate/rendered y de lo que agrega a /health.

Mismo esquema que test_estimate_structured_endpoint.py: sin red ni Redis, con el
wrapper reemplazado por `dependency_overrides`. El wrapper falso valida con el
schema y el contexto que le pasa el endpoint, así se comprueba que el endpoint
pide `RenderedResult` con el `output_format` de la request. Lo propio de este
endpoint:

- solo acepta versiones de salida renderizada (v4), y entre los tres endpoints
  cada uno dice en cuál pedir una versión de otro tipo;
- un `rendered` que no cumple el formato en ningún intento es un 502;
- la caché, los avisos y los errores del proveedor son los de /estimate/structured.
"""

from __future__ import annotations

import litellm
import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from app.config import get_settings
from app.dependencies import get_llm_wrapper
from app.main import create_app
from app.routers.estimations import INVALID_OUTPUT_MESSAGE, TIMEOUT_MESSAGE
from app.schemas.structured_estimation import RenderedResult
from app.services.llm_wrapper import StructuredCallResult, StructuredOutputError

RUTA = "/api/v1/estimate/rendered"
BODY = {
    "description": "Un SaaS B2B pequeño para gestionar el préstamo de equipos a empleados.",
    "project_type": "web_saas",
    "detail_level": "summary",
    "output_format": "line_items",
}
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
    "rendered": (
        "1. **Descubrimiento** — 1 semana — 70 horas — 3.950 EUR. Entrevistas.\n"
        "2. **Implementación** — 5 semanas — 320 horas — 20.000 EUR. Desarrollo.\n\n"
        "**Total:** 6 semanas, 390 horas y 23.950 EUR."
    ),
}


@pytest.fixture
def cliente(monkeypatch):
    def _crear(**entorno: str) -> TestClient:
        monkeypatch.setenv("REDIS_URL", "")
        for variable, valor in entorno.items():
            monkeypatch.setenv(variable, valor)
        get_settings.cache_clear()
        return TestClient(create_app())

    yield _crear
    get_settings.cache_clear()


class _WrapperFalso:
    """Reemplaza al LLMWrapper: valida `estimacion` con el schema y el contexto que
    recibe, como hace el wrapper real, o lanza `error`."""

    def __init__(
        self,
        error: Exception | None = None,
        estimacion: dict = ESTIMACION,
        warnings: tuple[str, ...] = (),
    ) -> None:
        self._error = error
        self._estimacion = estimacion
        self.warnings = warnings
        self.llamadas: list[dict] = []

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


def _con_wrapper(cliente, wrapper: _WrapperFalso, **entorno: str) -> TestClient:
    c = cliente(**entorno)
    c.app.dependency_overrides[get_llm_wrapper] = lambda: wrapper
    return c


# --- 200 -----------------------------------------------------------------------------


def test_devuelve_la_estimacion_con_su_presentacion(cliente) -> None:
    wrapper = _WrapperFalso()
    with _con_wrapper(cliente, wrapper) as c:
        r = c.post(RUTA, json=BODY)

    assert r.status_code == 200
    assert r.json() == {
        "estimation": ESTIMACION,
        "prompt_version": "v4",
        "cached": False,
        "warnings": [],
    }


def test_pide_rendered_result_con_el_output_format_de_la_request(cliente) -> None:
    wrapper = _WrapperFalso()
    with _con_wrapper(cliente, wrapper) as c:
        c.post(RUTA, json=BODY)

    [llamada] = wrapper.llamadas
    assert llamada["response_model"] is RenderedResult
    assert llamada["context"] == {"output_format": "line_items"}
    # El prompt v4 renderizado pide la lista numerada en rendered.
    assert 'output_format = "line_items". En rendered:' in llamada["system_prompt"]


def test_structured_no_pasa_contexto_ni_pide_rendered(cliente) -> None:
    """/estimate/structured sigue pidiendo StructuredResult, sin contexto."""
    wrapper = _WrapperFalso(estimacion={k: v for k, v in ESTIMACION.items() if k != "rendered"})
    with _con_wrapper(cliente, wrapper) as c:
        r = c.post("/api/v1/estimate/structured", json=BODY)

    assert r.status_code == 200
    assert "rendered" not in r.json()["estimation"]
    [llamada] = wrapper.llamadas
    assert llamada["response_model"].__name__ == "StructuredResult"
    assert llamada["context"] is None


def test_la_version_por_defecto_sale_de_rendered_prompt_version(cliente) -> None:
    """Ni PROMPT_VERSION ni STRUCTURED_PROMPT_VERSION intervienen."""
    wrapper = _WrapperFalso()
    with _con_wrapper(cliente, wrapper, PROMPT_VERSION="v1", STRUCTURED_PROMPT_VERSION="v3") as c:
        r = c.post(RUTA, json=BODY, params={"prompt_version": "v4"})
        r_defecto = c.post(RUTA, json=BODY)
    assert r.json()["prompt_version"] == r_defecto.json()["prompt_version"] == "v4"


def test_un_total_que_no_cuadra_llega_como_aviso(cliente) -> None:
    descuadrada = {
        **ESTIMACION,
        "totals": {**ESTIMACION["totals"], "hours": 400},
        "rendered": ESTIMACION["rendered"].replace("390 horas", "400 horas"),
    }
    with _con_wrapper(cliente, _WrapperFalso(estimacion=descuadrada)) as c:
        r = c.post(RUTA, json=BODY)

    assert r.status_code == 200
    assert r.json()["warnings"] == [
        "El total de horas declarado (400) no coincide con la suma de las fases (390)."
    ]


# --- 422 por versión -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pedida", "tipo", "endpoint"),
    [
        ("v2", "texto libre", "/api/v1/estimate"),
        ("v3", "salida estructurada", "/api/v1/estimate/structured"),
    ],
)
def test_rechaza_una_version_de_otro_tipo_y_dice_donde_pedirla(
    cliente, pedida, tipo, endpoint
) -> None:
    wrapper = _WrapperFalso()
    with _con_wrapper(cliente, wrapper) as c:
        r = c.post(RUTA, json=BODY, params={"prompt_version": pedida})

    assert r.status_code == 422
    assert r.json() == {
        "detail": (
            f"La versión de prompt '{pedida}' es de {tipo}: pedila en POST {endpoint}. "
            "Versiones disponibles en este endpoint: v4."
        )
    }
    assert wrapper.llamadas == []


@pytest.mark.parametrize(
    ("ruta", "disponibles"),
    [("/api/v1/estimate", "v1, v2"), ("/api/v1/estimate/structured", "v3")],
)
def test_los_otros_endpoints_rechazan_v4_y_dicen_donde_pedirla(cliente, ruta, disponibles):
    with _con_wrapper(cliente, _WrapperFalso()) as c:
        r = c.post(ruta, json=BODY, params={"prompt_version": "v4"})

    assert r.status_code == 422
    assert r.json() == {
        "detail": (
            "La versión de prompt 'v4' es de salida renderizada: pedila en POST "
            f"/api/v1/estimate/rendered. Versiones disponibles en este endpoint: {disponibles}."
        )
    }


# --- 503 por configuración -----------------------------------------------------------


@pytest.mark.parametrize("configurada", ["v3", "v99"])
def test_rendered_prompt_version_invalida_es_un_503(cliente, configurada) -> None:
    wrapper = _WrapperFalso()
    with _con_wrapper(cliente, wrapper, RENDERED_PROMPT_VERSION=configurada) as c:
        r = c.post(RUTA, json=BODY)

    assert r.status_code == 503
    assert r.json()["detail"] == (
        f"RENDERED_PROMPT_VERSION='{configurada}' no es una versión de salida renderizada "
        "publicada. Versiones disponibles: v4."
    )
    assert wrapper.llamadas == []


# --- 502 y 504 -------------------------------------------------------------------------


def test_sin_rendered_valido_es_un_502_con_su_mensaje(cliente) -> None:
    error = StructuredOutputError(attempts=3, cost_usd=0.004, last_error="rendered sin tabla")
    with _con_wrapper(cliente, _WrapperFalso(error=error)) as c, capture_logs() as logs:
        r = c.post(RUTA, json=BODY)

    assert r.status_code == 502
    assert r.json() == {"detail": INVALID_OUTPUT_MESSAGE}
    [evento] = [e for e in logs if e["event"] == "estimacion_fallida"]
    assert evento["salida"] == "renderizada"
    assert evento["intentos"] == 3


def test_timeout_del_proveedor_es_un_504(cliente) -> None:
    error = litellm.Timeout(message="lento", model="gpt-4o-mini", llm_provider="openai")
    with _con_wrapper(cliente, _WrapperFalso(error=error)) as c:
        r = c.post(RUTA, json=BODY)
    assert r.status_code == 504
    assert r.json() == {"detail": TIMEOUT_MESSAGE}


# --- /health y OpenAPI ------------------------------------------------------------------


def test_health_lista_las_versiones_renderizadas(cliente) -> None:
    with cliente() as c:
        h = c.get("/health").json()

    assert h["rendered_prompt_version"] == "v4"
    assert h["rendered_prompt_versions"] == ["v4"]
    # Sin cambios para los clientes anteriores.
    assert h["prompt_versions"] == ["v1", "v2"]
    assert h["structured_prompt_versions"] == ["v3"]


def test_openapi_documenta_el_endpoint(cliente) -> None:
    with cliente() as c:
        schema = c.get("/openapi.json").json()

    operacion = schema["paths"][RUTA]["post"]
    respuesta = operacion["responses"]["200"]["content"]["application/json"]["schema"]
    assert respuesta == {"$ref": "#/components/schemas/RenderedEstimationResponse"}
    assert "rendered" in schema["components"]["schemas"]["RenderedResult"]["properties"]
