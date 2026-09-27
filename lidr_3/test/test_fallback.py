"""Tests del wrapper con fallback (Paso 3).

El contrato que prueba este archivo:
- un error transitorio del primario (can_fallback=True) dispara UN salto
  al secundario configurado en LLM_FALLBACK;
- un error permanente (can_fallback=False) falla rápido sin gastar el
  fallback;
- el resultado expone el proveedor REAL que respondió y `used_fallback`;
- si ambos fallan, el cliente ve el mensaje fijo y la traza de los dos
  intentos queda encadenada en el servidor (from exc);
- la key del fallback se exige en el punto de uso: configurar un fallback
  sin su API key arranca y falla recién cuando se intenta usar.
"""

from __future__ import annotations

import pytest
from conftest import make_settings
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.providers import LLMProviderError, LLMResponse
from app.services import llm_service

TRANSCRIPCION = (
    "Reunión de planificación del sprint para el módulo de facturación. "
    "El cliente quiere alta de clientes, emisión de comprobantes y reporte "
    "de cobranzas. Se definieron los límites del MVP para esta iteración."
)


def _respuesta_ok(model: str = "fake") -> LLMResponse:
    return LLMResponse(
        content="## Total\n**80 horas**\n",
        model=model,
        truncated=False,
        usage={"input_tokens": 50, "output_tokens": 20},
    )


async def test_error_transitorio_del_primario_dispara_fallback(monkeypatch):
    """429 del primario -> el wrapper intenta el secundario y responde."""
    llamadas: list[str] = []

    def fake_create(name, api_key, model, *, timeout, max_retries):
        llamadas.append(name)

        class FakeProvider:
            async def chat(self, messages, *, max_tokens=None, temperature=None):
                if name == "openai":
                    raise LLMProviderError(provider="openai", detail="rate limit", status_code=429)
                return _respuesta_ok(model="claude-haiku-4-5")

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)
    settings = make_settings(
        llm_provider="openai",
        openai_api_key="k",
        llm_fallback="anthropic",
        anthropic_api_key="k2",
    )
    resultado = await llm_service.generate_estimation(TRANSCRIPCION, settings)

    assert llamadas == ["openai", "anthropic"]
    assert resultado.used_fallback is True
    assert resultado.provider == "anthropic"


async def test_error_permanente_no_gasta_fallback(monkeypatch):
    """4xx del primario (can_fallback=False) -> falla rápido: el mismo request
    va a fallar igual en el secundario."""
    llamadas: list[str] = []

    def fake_create(name, api_key, model, *, timeout, max_retries):
        llamadas.append(name)

        class FakeProvider:
            async def chat(self, messages, *, max_tokens=None, temperature=None):
                raise LLMProviderError(provider="openai", detail="input inválido", status_code=422)

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)
    settings = make_settings(
        llm_provider="openai",
        openai_api_key="k",
        llm_fallback="anthropic",
        anthropic_api_key="k2",
    )

    with pytest.raises(llm_service.LLMServiceError):
        await llm_service.generate_estimation(TRANSCRIPCION, settings)

    assert llamadas == ["openai"], "el fallback no debería gastarse en un 4xx"


async def test_sin_fallback_configurado_falla_rapido(monkeypatch):
    """Sin LLM_FALLBACK, un error transitorio del primario NO intenta nada."""
    llamadas: list[str] = []

    def fake_create(name, api_key, model, *, timeout, max_retries):
        llamadas.append(name)

        class FakeProvider:
            async def chat(self, messages, *, max_tokens=None, temperature=None):
                raise LLMProviderError(provider="openai", detail="rate limit", status_code=429)

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)
    settings = make_settings(openai_api_key="k")  # llm_fallback por defecto: None

    with pytest.raises(llm_service.LLMServiceError):
        await llm_service.generate_estimation(TRANSCRIPCION, settings)

    assert llamadas == ["openai"]


async def test_primario_ok_no_usa_fallback(monkeypatch):
    def fake_create(name, api_key, model, *, timeout, max_retries):
        class FakeProvider:
            async def chat(self, messages, *, max_tokens=None, temperature=None):
                return _respuesta_ok()

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)
    settings = make_settings(
        llm_provider="openai",
        openai_api_key="k",
        llm_fallback="anthropic",
        anthropic_api_key="k2",
    )
    resultado = await llm_service.generate_estimation(TRANSCRIPCION, settings)

    assert resultado.used_fallback is False
    assert resultado.provider == "openai"


async def test_si_ambos_fallan_se_encadena_y_el_error_nombra_los_dos(monkeypatch):
    def fake_create(name, api_key, model, *, timeout, max_retries):
        class FakeProvider:
            async def chat(self, messages, *, max_tokens=None, temperature=None):
                raise LLMProviderError(provider=name, detail="caído", status_code=500)

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)
    settings = make_settings(
        llm_provider="openai",
        openai_api_key="k",
        llm_fallback="anthropic",
        anthropic_api_key="k2",
    )

    with pytest.raises(llm_service.LLMServiceError) as excinfo:
        await llm_service.generate_estimation(TRANSCRIPCION, settings)

    mensaje = str(excinfo.value)
    assert "openai" in mensaje and "anthropic" in mensaje
    # La causa encadenada conserva el primer fallo para el log del servidor.
    assert isinstance(excinfo.value.__cause__, LLMProviderError)
    assert excinfo.value.__cause__.provider == "anthropic"


async def test_fallback_sin_key_falla_en_el_punto_de_uso(monkeypatch):
    """LLM_FALLBACK=anthropic sin ANTHROPIC_API_KEY: el error aparece recién
    cuando el fallback se intenta, no al arrancar (decisión punto de uso)."""
    llamadas: list[str] = []

    def fake_create(name, api_key, model, *, timeout, max_retries):
        llamadas.append(name)

        class FakeProvider:
            async def chat(self, messages, *, max_tokens=None, temperature=None):
                if name == "openai":
                    raise LLMProviderError(provider="openai", detail="rate limit", status_code=429)
                return _respuesta_ok()

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)
    settings = make_settings(
        llm_provider="openai",
        openai_api_key="k",
        llm_fallback="anthropic",
        anthropic_api_key=None,  # la key del fallback NO está configurada
    )

    with pytest.raises(llm_service.LLMConfigurationError) as excinfo:
        await llm_service.generate_estimation(TRANSCRIPCION, settings)

    assert "ANTHROPIC_API_KEY" in str(excinfo.value)
    # No llegó a construirse el cliente del fallback (fake_create solo vio openai).
    assert llamadas == ["openai"]


async def test_el_fallback_usa_el_modelo_del_fallback_no_el_override_global(monkeypatch):
    """LLM_MODEL es un override del activo; el fallback usa su propio default
    (resolve_model por proveedor), así un LLM_MODEL de openai jamás contamina
    el modelo de anthropic."""
    modelos: list[str] = []

    def fake_create(name, api_key, model, *, timeout, max_retries):
        modelos.append(model)

        class FakeProvider:
            async def chat(self, messages, *, max_tokens=None, temperature=None):
                if name == "openai":
                    raise LLMProviderError(provider="openai", detail="rate limit", status_code=429)
                return _respuesta_ok()

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)
    settings = make_settings(
        llm_provider="openai",
        openai_api_key="k",
        llm_model="gpt-5-mini",  # override del ACTIVO
        openai_model="gpt-4o-mini",
        llm_fallback="anthropic",
        anthropic_api_key="k2",
        anthropic_model="claude-haiku-4-5",
    )

    await llm_service.generate_estimation(TRANSCRIPCION, settings)

    assert modelos == ["gpt-5-mini", "claude-haiku-4-5"], (
        "el override LLM_MODEL no debe contaminar el modelo del fallback"
    )


def test_http_expone_provider_real_y_used_fallback(client, monkeypatch):
    """El contrato visible: la respuesta JSON muestra quién respondió de
    verdad (anthropic) y que hubo fallback, no un provider mentiroso."""

    def fake_create(name, api_key, model, *, timeout, max_retries):
        class FakeProvider:
            async def chat(self, messages, *, max_tokens=None, temperature=None):
                if name == "openai":
                    raise LLMProviderError(provider="openai", detail="rate limit", status_code=429)
                return _respuesta_ok(model="claude-haiku-4-5")

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)
    from app.config import Settings

    settings = Settings(
        _env_file=None,
        openai_api_key="k",
        llm_fallback="anthropic",
        anthropic_api_key="k2",
    )
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as c:
        r = c.post("/api/v1/estimate", json={"transcription": TRANSCRIPCION})
    app.dependency_overrides.clear()

    assert r.status_code == 200
    data = r.json()
    assert data["provider"] == "anthropic"
    assert data["used_fallback"] is True
