"""Tests del endpoint POST /api/v1/estimate: respuesta normal y errores.

Sin red ni Redis: REDIS_URL vacío desactiva la caché. Los casos de 503 usan el
wrapper real (sin la clave que falta ni siquiera llega a construirse); los de
200, 502 y 504 reemplazan el wrapper con `dependency_overrides`.
"""

from __future__ import annotations

import httpx
import litellm
import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from app.config import get_settings
from app.dependencies import get_llm_wrapper
from app.main import create_app
from app.routers.estimations import MENSAJE_FALLO_PROVEEDOR, MENSAJE_TIMEOUT
from app.services.llm_wrapper import LLMCallResult

BODY = {
    "description": "A small B2B SaaS to manage employee equipment loans.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}

# Texto que un proveedor podría devolver y que nunca debe llegar al cliente.
DETALLE_INTERNO = "org-ACME-1234: upstream said no (request_id=abc)"


@pytest.fixture
def cliente(monkeypatch):
    """App real con el entorno que fije cada test; las claves arrancan vacías."""

    def _crear(**entorno: str) -> TestClient:
        for variable in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.setenv(variable, "")
        monkeypatch.setenv("REDIS_URL", "")
        for variable, valor in entorno.items():
            monkeypatch.setenv(variable, valor)
        get_settings.cache_clear()
        return TestClient(create_app())

    yield _crear
    get_settings.cache_clear()


class _WrapperFalso:
    """Reemplaza al LLMWrapper: devuelve un resultado fijo o lanza `error`."""

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    async def estimate(self, *, prompt_version: str, **_kwargs) -> LLMCallResult:
        if self._error is not None:
            raise self._error
        return LLMCallResult(
            content="| phase | duration_weeks | cost_eur | confidence_pct |",
            model="openai/gpt-4o-mini",
            provider="openai",
            prompt_tokens=10,
            completion_tokens=20,
            cost_usd=0.001,
            prompt_version=prompt_version,
        )


def _con_wrapper(cliente, wrapper: _WrapperFalso) -> TestClient:
    c = cliente()
    c.app.dependency_overrides[get_llm_wrapper] = lambda: wrapper
    return c


def _error_de_conexion() -> litellm.APIConnectionError:
    return litellm.APIConnectionError(
        message=DETALLE_INTERNO, llm_provider="openai", model="gpt-4o-mini"
    )


def _timeout() -> litellm.Timeout:
    return litellm.Timeout(message=DETALLE_INTERNO, model="gpt-4o-mini", llm_provider="openai")


def _error_de_estado() -> litellm.RateLimitError:
    return litellm.RateLimitError(
        message=DETALLE_INTERNO,
        llm_provider="openai",
        model="gpt-4o-mini",
        response=httpx.Response(429, request=httpx.Request("POST", "https://x")),
    )


# --- Camino normal --------------------------------------------------------------


def test_returns_200_with_text_and_prompt_version(cliente) -> None:
    with _con_wrapper(cliente, _WrapperFalso()) as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert r.status_code == 200
    assert r.json() == {
        "text": "| phase | duration_weeks | cost_eur | confidence_pct |",
        "prompt_version": "v1",
    }


# --- Falta configuración: 503 ---------------------------------------------------


def test_missing_primary_key_returns_503_naming_the_variable(cliente) -> None:
    with cliente(ANTHROPIC_API_KEY="sk-ant-secreta") as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert r.status_code == 503
    assert "Falta OPENAI_API_KEY" in r.json()["detail"]


def test_missing_fallback_key_returns_503_naming_the_variable(cliente) -> None:
    with cliente(OPENAI_API_KEY="sk-openai-secreta") as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert r.status_code == 503
    assert "Falta ANTHROPIC_API_KEY" in r.json()["detail"]


def test_503_never_leaks_the_configured_key(cliente) -> None:
    with cliente(OPENAI_API_KEY="sk-openai-secreta") as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert "sk-openai-secreta" not in r.text


def test_health_still_answers_without_keys(cliente) -> None:
    """El servicio arranca sin claves y /health lo informa (no hay crashloop)."""
    with cliente() as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["llm_configured"] is False


# --- Falla el proveedor: 502 y 504 con mensaje limpio ----------------------------


@pytest.mark.parametrize(
    ("crear_error", "codigo", "mensaje"),
    [
        (_error_de_conexion, 502, MENSAJE_FALLO_PROVEEDOR),
        (_error_de_estado, 502, MENSAJE_FALLO_PROVEEDOR),
        (lambda: RuntimeError(DETALLE_INTERNO), 502, MENSAJE_FALLO_PROVEEDOR),
        (_timeout, 504, MENSAJE_TIMEOUT),
    ],
    ids=["conexion", "rate-limit", "inesperado", "timeout"],
)
def test_provider_failure_returns_clean_message(cliente, crear_error, codigo, mensaje) -> None:
    with _con_wrapper(cliente, _WrapperFalso(crear_error())) as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert r.status_code == codigo
    assert r.json() == {"detail": mensaje}
    assert DETALLE_INTERNO not in r.text


def test_provider_failure_detail_goes_to_the_log(cliente) -> None:
    """Lo que el cliente no ve tiene que quedar en el log para poder diagnosticar."""
    with _con_wrapper(cliente, _WrapperFalso(_error_de_conexion())) as c, capture_logs() as logs:
        c.post("/api/v1/estimate", json=BODY)
    fallos = [e for e in logs if e["event"] == "estimacion_fallida"]
    assert len(fallos) == 1
    assert fallos[0]["log_level"] == "error"
    assert fallos[0]["codigo_http"] == 502
    assert fallos[0]["tipo_error"] == "APIConnectionError"
    assert DETALLE_INTERNO in fallos[0]["detalle"]
