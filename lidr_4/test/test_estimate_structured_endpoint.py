"""Tests de POST /api/v1/estimate/structured y de lo que agrega a /health.

Mismo esquema que test_estimate_endpoint.py: sin red ni Redis, con el wrapper
reemplazado por `dependency_overrides` en los casos de 200, 502 y 504. Lo propio
de este endpoint:

- solo acepta versiones de salida estructurada, y al pedirle una de texto (o a
  /estimate una estructurada) dice en qué endpoint pedirla;
- un total que no cuadra no es un error: llega en `warnings` y queda en el log;
- que el modelo no cumpla el schema en ningún intento es un 502 con su mensaje.
"""

from __future__ import annotations

import litellm
import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from app.config import get_settings
from app.dependencies import get_llm_wrapper
from app.main import create_app
from app.routers.estimations import (
    INVALID_OUTPUT_MESSAGE,
    PROVIDER_FAILURE_MESSAGE,
    TIMEOUT_MESSAGE,
)
from app.schemas.structured_estimation import StructuredResult
from app.services.llm_wrapper import StructuredCallResult, StructuredOutputError

RUTA = "/api/v1/estimate/structured"
BODY = {
    "description": "Un SaaS B2B pequeño para gestionar el préstamo de equipos a empleados.",
    "project_type": "web_saas",
    "detail_level": "detailed",
    "output_format": "phases_table",
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
            "assumptions": ["Sin dependencias externas"],
            "risks": [{"risk": "Cambios de alcance", "mitigation": "Cerrar el alcance antes"}],
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
DETALLE_INTERNO = "org-ACME-1234: upstream said no (request_id=abc)"


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
    """Reemplaza al LLMWrapper: devuelve `estimacion` validada o lanza `error`."""

    def __init__(
        self,
        error: Exception | None = None,
        warnings: tuple[str, ...] = (),
        estimacion: dict = ESTIMACION,
        cached: bool = False,
    ) -> None:
        self._cached = cached
        self._error = error
        self.warnings = warnings
        self._estimacion = estimacion
        self.llamadas: list[dict] = []

    async def estimate_structured(self, *, prompt_version: str, **kwargs) -> StructuredCallResult:
        self.llamadas.append({"prompt_version": prompt_version, **kwargs})
        if self._error is not None:
            raise self._error
        estimacion = StructuredResult.model_validate(self._estimacion)
        return StructuredCallResult(
            content=estimacion.model_dump_json(),
            model="openai/gpt-4o-mini",
            provider="openai",
            prompt_tokens=10,
            completion_tokens=20,
            cost_usd=0.001,
            prompt_version=prompt_version,
            estimation=estimacion,
            cached=self._cached,
        )

    async def estimate(self, **_kwargs):
        raise AssertionError("/estimate/structured no debe llamar a la estimación de texto")


def _con_wrapper(cliente, wrapper: _WrapperFalso, **entorno: str) -> TestClient:
    c = cliente(**entorno)
    c.app.dependency_overrides[get_llm_wrapper] = lambda: wrapper
    return c


# --- 200 -----------------------------------------------------------------------------


def test_devuelve_la_estimacion_estructurada(cliente) -> None:
    wrapper = _WrapperFalso()
    with _con_wrapper(cliente, wrapper) as c:
        r = c.post(RUTA, json=BODY)

    assert r.status_code == 200
    assert r.json() == {
        "estimation": ESTIMACION,
        "prompt_version": "v3",
        "cached": False,
        "warnings": [],
    }

    # El wrapper recibió el prompt v3 renderizado con el formulario.
    [llamada] = wrapper.llamadas
    assert llamada["prompt_version"] == "v3"
    assert "un único objeto JSON" in llamada["system_prompt"]
    assert BODY["description"] in llamada["user_prompt"]


def test_la_version_por_defecto_sale_de_structured_prompt_version(cliente) -> None:
    """PROMPT_VERSION no interviene: es la de /estimate."""
    wrapper = _WrapperFalso()
    with _con_wrapper(cliente, wrapper, PROMPT_VERSION="v1") as c:
        r = c.post(RUTA, json=BODY)
    assert r.json()["prompt_version"] == "v3"


def test_acepta_v3_en_la_query(cliente) -> None:
    wrapper = _WrapperFalso()
    with _con_wrapper(cliente, wrapper) as c:
        r = c.post(RUTA, json=BODY, params={"prompt_version": "v3"})
    assert r.status_code == 200
    assert wrapper.llamadas[0]["prompt_version"] == "v3"


def test_los_avisos_del_wrapper_llegan_al_cliente(cliente) -> None:
    aviso = "El modelo de respaldo no está disponible."
    with _con_wrapper(cliente, _WrapperFalso(warnings=(aviso,))) as c:
        r = c.post(RUTA, json=BODY)
    assert r.json()["warnings"] == [aviso]


def test_un_total_que_no_cuadra_llega_como_aviso_y_queda_en_el_log(cliente) -> None:
    descuadrada = {**ESTIMACION, "totals": {**ESTIMACION["totals"], "cost_eur": 24050.0}}
    aviso_respaldo = "El modelo de respaldo no está disponible."
    wrapper = _WrapperFalso(estimacion=descuadrada, warnings=(aviso_respaldo,))

    with _con_wrapper(cliente, wrapper) as c, capture_logs() as logs:
        r = c.post(RUTA, json=BODY)

    assert r.status_code == 200
    assert r.json()["estimation"]["totals"]["cost_eur"] == 24050.0  # se devuelve tal cual
    assert r.json()["warnings"] == [
        aviso_respaldo,
        "El coste total declarado (24.050 EUR) no coincide con la suma de las fases (23.950 EUR).",
    ]
    [evento] = [e for e in logs if e["event"] == "totales_no_cuadran"]
    assert evento["log_level"] == "warning"
    assert evento["prompt_version"] == "v3"


# --- 422 por versión ---------------------------------------------------------------------


@pytest.mark.parametrize("pedida", ["v1", "v2"])
def test_rechaza_una_version_de_texto_y_dice_donde_pedirla(cliente, pedida) -> None:
    wrapper = _WrapperFalso()
    with _con_wrapper(cliente, wrapper) as c:
        r = c.post(RUTA, json=BODY, params={"prompt_version": pedida})

    assert r.status_code == 422
    assert r.json() == {
        "detail": (
            f"La versión de prompt '{pedida}' es de texto libre: pedila en POST "
            "/api/v1/estimate. Versiones disponibles en este endpoint: v3."
        )
    }
    assert wrapper.llamadas == []


def test_estimate_rechaza_v3_y_dice_donde_pedirla(cliente) -> None:
    with cliente() as c:
        c.app.dependency_overrides[get_llm_wrapper] = lambda: _WrapperFalso()
        r = c.post("/api/v1/estimate", json=BODY, params={"prompt_version": "v3"})

    assert r.status_code == 422
    assert r.json() == {
        "detail": (
            "La versión de prompt 'v3' es de salida estructurada: pedila en POST "
            "/api/v1/estimate/structured. Versiones disponibles en este endpoint: v1, v2."
        )
    }


@pytest.mark.parametrize("pedida", ["v99", "../v2", ""])
def test_rechaza_una_version_que_no_existe(cliente, pedida) -> None:
    wrapper = _WrapperFalso()
    with _con_wrapper(cliente, wrapper) as c:
        r = c.post(RUTA, json=BODY, params={"prompt_version": pedida})

    assert r.status_code == 422
    assert r.json() == {
        "detail": f"La versión de prompt '{pedida}' no existe. Versiones disponibles: v3."
    }
    assert wrapper.llamadas == []


# --- 503 por configuración -----------------------------------------------------------------


@pytest.mark.parametrize("configurada", ["v2", "v99"])
def test_structured_prompt_version_invalida_es_un_503(cliente, configurada) -> None:
    """Un error de configuración del servicio, no del cliente: nombra la variable."""
    wrapper = _WrapperFalso()
    with _con_wrapper(cliente, wrapper, STRUCTURED_PROMPT_VERSION=configurada) as c:
        r = c.post(RUTA, json=BODY)

    assert r.status_code == 503
    assert f"STRUCTURED_PROMPT_VERSION='{configurada}'" in r.json()["detail"]
    assert "Versiones disponibles: v3." in r.json()["detail"]
    assert wrapper.llamadas == []


def test_sin_la_clave_del_primario_es_un_503(cliente) -> None:
    """Con el wrapper real: la clave se exige al construirlo, como en /estimate."""
    with cliente(ANTHROPIC_API_KEY="sk-anthropic") as c:
        r = c.post(RUTA, json=BODY)
    assert r.status_code == 503
    assert "OPENAI_API_KEY" in r.json()["detail"]


# --- 502 y 504 -------------------------------------------------------------------------------


def test_sin_estimacion_valida_es_un_502_con_su_mensaje(cliente) -> None:
    error = StructuredOutputError(attempts=3, cost_usd=0.004, last_error=DETALLE_INTERNO)
    with _con_wrapper(cliente, _WrapperFalso(error=error)) as c, capture_logs() as logs:
        r = c.post(RUTA, json=BODY)

    assert r.status_code == 502
    assert r.json() == {"detail": INVALID_OUTPUT_MESSAGE}
    assert DETALLE_INTERNO not in r.text
    [evento] = [e for e in logs if e["event"] == "estimacion_fallida"]
    assert evento["codigo_http"] == 502
    assert evento["tipo_error"] == "StructuredOutputError"
    assert evento["intentos"] == 3
    assert evento["coste_usd"] == 0.004
    assert evento["salida"] == "estructurada"
    assert DETALLE_INTERNO in evento["detalle"]


def test_timeout_del_proveedor_es_un_504(cliente) -> None:
    error = litellm.Timeout(message=DETALLE_INTERNO, model="gpt-4o-mini", llm_provider="openai")
    with _con_wrapper(cliente, _WrapperFalso(error=error)) as c:
        r = c.post(RUTA, json=BODY)
    assert r.status_code == 504
    assert r.json() == {"detail": TIMEOUT_MESSAGE}


def test_otro_fallo_del_proveedor_es_un_502(cliente) -> None:
    error = litellm.APIConnectionError(
        message=DETALLE_INTERNO, model="gpt-4o-mini", llm_provider="openai"
    )
    with _con_wrapper(cliente, _WrapperFalso(error=error)) as c, capture_logs() as logs:
        r = c.post(RUTA, json=BODY)

    assert r.status_code == 502
    assert r.json() == {"detail": PROVIDER_FAILURE_MESSAGE}
    assert DETALLE_INTERNO not in r.text
    [evento] = [e for e in logs if e["event"] == "estimacion_fallida"]
    assert evento["salida"] == "estructurada"


# --- /health ----------------------------------------------------------------------------------


def test_health_lista_las_versiones_por_tipo(cliente) -> None:
    with cliente() as c:
        h = c.get("/health").json()

    assert h["prompt_version"] == "v2"
    assert h["prompt_versions"] == ["v1", "v2"]  # sin cambios para los clientes de texto
    assert h["structured_prompt_version"] == "v3"
    assert h["structured_prompt_versions"] == ["v3"]


def test_openapi_documenta_el_endpoint(cliente) -> None:
    with cliente() as c:
        schema = c.get("/openapi.json").json()

    operacion = schema["paths"][RUTA]["post"]
    respuesta = operacion["responses"]["200"]["content"]["application/json"]["schema"]
    assert respuesta == {"$ref": "#/components/schemas/StructuredEstimationResponse"}
    assert "StructuredResult" in schema["components"]["schemas"]


def test_cached_dice_si_la_estimacion_salio_de_la_cache(cliente) -> None:
    with _con_wrapper(cliente, _WrapperFalso(cached=True)) as c:
        r = c.post(RUTA, json=BODY)
    assert r.json()["cached"] is True


def test_un_rechazo_se_devuelve_y_queda_en_el_log(cliente) -> None:
    rechazo = {
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
        "summary": "Fuera de alcance: la descripción no dice qué hay que construir.",
        "confidence_pct": 10,
    }
    with _con_wrapper(cliente, _WrapperFalso(estimacion=rechazo)) as c, capture_logs() as logs:
        r = c.post(RUTA, json=BODY)

    assert r.status_code == 200  # no es un fallo: el modelo explica qué falta
    assert r.json()["estimation"]["summary"].startswith("Fuera de alcance:")
    assert r.json()["warnings"] == []
    [evento] = [e for e in logs if e["event"] == "estimacion_fuera_de_alcance"]
    assert evento["confianza_pct"] == 10
    assert evento["prompt_version"] == "v3"
