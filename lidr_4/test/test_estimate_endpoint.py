"""Tests del endpoint POST /api/v1/estimate y de /health: respuesta normal, avisos y errores.

Sin red ni Redis: REDIS_URL vacío desactiva la caché. Los casos de 503 usan el
wrapper real (sin la clave que falta ni siquiera llega a construirse); los de
200, 502 y 504 reemplazan el wrapper con `dependency_overrides`.

El `.env` de quien corre los tests no interviene (ver `entorno_aislado` en
conftest.py), y los tests de claves corren con cada proveedor como primario
(fixture `modelos`).
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
    """App real con el entorno que fije cada test; sin claves ni `.env` de partida."""

    def _crear(**entorno: str) -> TestClient:
        monkeypatch.setenv("REDIS_URL", "")
        for variable, valor in entorno.items():
            monkeypatch.setenv(variable, valor)
        get_settings.cache_clear()
        return TestClient(create_app())

    yield _crear
    get_settings.cache_clear()


class _WrapperFalso:
    """Reemplaza al LLMWrapper: devuelve un resultado fijo o lanza `error`."""

    def __init__(self, error: Exception | None = None, avisos: tuple[str, ...] = ()) -> None:
        self._error = error
        self.avisos = avisos

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
        "avisos": [],
    }


# --- Falta configuración: 503 ---------------------------------------------------


def _entorno(modelos, **claves_por_rol: str) -> dict[str, str]:
    """Modelos fijados más las claves pedidas por rol ("primario"/"respaldo")."""
    (primario, var_primario, _), (respaldo, var_respaldo, _) = modelos
    variables = {"primario": var_primario, "respaldo": var_respaldo}
    entorno = {"PRIMARY_MODEL": primario, "FALLBACK_MODEL": respaldo}
    entorno.update({variables[rol]: valor for rol, valor in claves_por_rol.items()})
    return entorno


def test_missing_primary_key_returns_503_naming_the_variable(cliente, modelos) -> None:
    (_, var_primario, _), _ = modelos
    with cliente(**_entorno(modelos, respaldo="sk-respaldo")) as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert r.status_code == 503
    assert f"Falta {var_primario}" in r.json()["detail"]


def test_503_never_leaks_the_configured_key(cliente, modelos) -> None:
    with cliente(**_entorno(modelos, respaldo="sk-respaldo-secreta")) as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert r.status_code == 503
    assert "sk-respaldo-secreta" not in r.text


def test_missing_fallback_key_still_estimates_and_warns(cliente, monkeypatch, modelos) -> None:
    """Sin la clave del respaldo la estimación sale con el primario, y la
    respuesta trae el aviso para el usuario."""
    _, (_, var_respaldo, _) = modelos

    async def sin_red(self, **kwargs):
        return await litellm.acompletion(
            model=kwargs["model"], messages=kwargs["messages"], mock_response="estimación"
        )

    monkeypatch.setattr(litellm.Router, "acompletion", sin_red)
    with cliente(**_entorno(modelos, primario="sk-primario-secreta")) as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["text"] == "estimación"
    [aviso] = cuerpo["avisos"]
    assert var_respaldo in aviso
    assert "sk-primario-secreta" not in r.text


def test_avisos_reach_the_response(cliente) -> None:
    with _con_wrapper(cliente, _WrapperFalso(avisos=("aviso de prueba",))) as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert r.json()["avisos"] == ["aviso de prueba"]


def test_health_reports_missing_fallback(cliente, modelos) -> None:
    _, (_, var_respaldo, _) = modelos
    with cliente(**_entorno(modelos, primario="sk-primario")) as c:
        h = c.get("/health").json()
    assert h["llm_configured"] is True
    assert h["fallback_configured"] is False
    [aviso] = h["avisos"]
    assert var_respaldo in aviso


def test_health_reports_missing_primary(cliente, modelos) -> None:
    with cliente(**_entorno(modelos, respaldo="sk-respaldo")) as c:
        h = c.get("/health").json()
    assert h["llm_configured"] is False


def test_health_without_notices_when_both_keys_are_set(cliente, modelos) -> None:
    with cliente(**_entorno(modelos, primario="sk-p", respaldo="sk-r")) as c:
        h = c.get("/health").json()
    assert h["llm_configured"] is True
    assert h["fallback_configured"] is True
    assert h["avisos"] == []


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
