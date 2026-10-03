"""Tests del endpoint POST /api/v1/estimate ante configuración incompleta.

Sin red ni Redis: REDIS_URL vacío desactiva la caché, y sin la clave que falta
el wrapper ni siquiera llega a construirse.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app

BODY = {
    "description": "A small B2B SaaS to manage employee equipment loans.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}


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


def test_missing_primary_key_returns_503_naming_the_variable(cliente) -> None:
    with cliente(ANTHROPIC_API_KEY="sk-ant-secreta") as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert r.status_code == 503
    assert "OPENAI_API_KEY" in r.json()["detail"]


def test_missing_fallback_key_returns_503_naming_the_variable(cliente) -> None:
    with cliente(OPENAI_API_KEY="sk-openai-secreta") as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert r.status_code == 503
    assert "ANTHROPIC_API_KEY" in r.json()["detail"]


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
