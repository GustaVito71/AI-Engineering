"""Tests del cache Redis (Paso 4).

Qué se prueba:
- hit: la estimación se devuelve desde el cache SIN armar el prompt ni
  llamar al proveedor (ni gastar fallback);
- miss -> la respuesta del primario se guarda y el siguiente request es hit;
- fail soft: un Redis caído / entrada corrupta no tumba la estimación, se
  genera igual;
- una respuesta del FALLBACK no se cachea: la clave se arma con el modelo
  del proveedor activo, y cachearla bajo esa clave mentiría sobre el origen
  y el costo de una respuesta futura.

Se usa fakeredis (no un Redis real): misma interfaz async, cero red.
"""

from __future__ import annotations

import fakeredis.aioredis
from redis.exceptions import ConnectionError

from app.cache import build_cache_key, get_cached_estimation, set_cached_estimation
from app.config import Settings
from app.providers import LLMProviderError, LLMResponse
from app.services import llm_service

TRANSCRIPCION = (
    "Reunión de planificación del sprint para el módulo de facturación. "
    "El cliente quiere alta de clientes, emisión de comprobantes y reporte "
    "de cobranzas. Se definieron los límites del MVP para esta iteración."
)


def _respuesta_ok(model: str = "gpt-4o-mini") -> LLMResponse:
    return LLMResponse(
        content="## Total\n**80 horas**\n",
        model=model,
        truncated=False,
        usage={"input_tokens": 50, "output_tokens": 20},
    )


def _make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, openai_api_key="k", **overrides)


def test_build_cache_key_es_determinista_sobre_las_mismas_entradas():
    settings = _make_settings()
    assert build_cache_key(TRANSCRIPCION, settings) == build_cache_key(TRANSCRIPCION, settings)


def test_build_cache_key_cambia_con_transcripcion_o_modelo():
    settings = _make_settings()
    base = build_cache_key(TRANSCRIPCION, settings)
    assert build_cache_key(TRANSCRIPCION + " más deuda técnica", settings) != base
    assert build_cache_key(TRANSCRIPCION, _make_settings(llm_model="gpt-5-mini")) != base


async def test_hit_devuelve_sin_llamar_al_proveedor(monkeypatch):
    llamadas: list[str] = []

    def fake_create(name, api_key, model, *, timeout, max_retries):
        llamadas.append(name)

        class FakeProvider:
            async def chat(self, messages, *, max_tokens=None, temperature=None):
                return _respuesta_ok()

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)
    cache = fakeredis.aioredis.FakeRedis(decode_responses=True)
    settings = _make_settings()

    # Miss: genera y guarda. Hit: ni arma el prompt ni llama al proveedor.
    primero = await llm_service.generate_estimation(TRANSCRIPCION, settings, cache)
    segundo = await llm_service.generate_estimation(TRANSCRIPCION, settings, cache)

    assert llamadas == ["openai"], "el segundo request no debe tocar al proveedor"
    assert primero == segundo
    assert segundo.provider == "openai"
    assert segundo.used_fallback is False


async def test_dos_transcripciones_distintas_no_comparten_cache(monkeypatch):
    llamadas: list[str] = []

    def fake_create(name, api_key, model, *, timeout, max_retries):
        llamadas.append(name)

        class FakeProvider:
            async def chat(self, messages, *, max_tokens=None, temperature=None):
                return _respuesta_ok()

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)
    cache = fakeredis.aioredis.FakeRedis(decode_responses=True)
    settings = _make_settings()

    await llm_service.generate_estimation(TRANSCRIPCION, settings, cache)
    await llm_service.generate_estimation(TRANSCRIPCION + " otra reunión", settings, cache)

    assert llamadas == ["openai", "openai"], "entradas distintas -> miss cada una"


async def test_fallback_no_se_cachea(monkeypatch):
    """Si el primario falla y responde el fallback, la respuesta NO alimenta
    el cache: un request siguiente con el primario sano vuelve a generar."""

    def fake_create(name, api_key, model, *, timeout, max_retries):
        class FakeProvider:
            async def chat(self, messages, *, max_tokens=None, temperature=None):
                if name == "openai":
                    # El primario SIEMPRE cae en esta simulación.
                    raise LLMProviderError(provider="openai", detail="rate limit", status_code=429)
                return _respuesta_ok(model="claude-haiku-4-5")

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)
    cache = fakeredis.aioredis.FakeRedis(decode_responses=True)
    settings = _make_settings(
        llm_fallback="anthropic",
        anthropic_api_key="k2",
    )

    resultado = await llm_service.generate_estimation(TRANSCRIPCION, settings, cache)
    assert resultado.used_fallback is True

    # La respuesta del fallback NO quedó: el cache está vacío para esta clave.
    clave = build_cache_key(TRANSCRIPCION, settings)
    assert await cache.get(clave) is None


async def test_fail_soft_redis_caido_en_lectura(monkeypatch):
    """Un Redis que falla al leer no tumba la estimación: se genera igual."""
    llamadas: list[str] = []

    def fake_create(name, api_key, model, *, timeout, max_retries):
        llamadas.append(name)

        class FakeProvider:
            async def chat(self, messages, *, max_tokens=None, temperature=None):
                return _respuesta_ok()

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)

    class RedisCaido:
        async def get(self, key: str) -> str | None:  # pragma: no cover
            raise ConnectionError("Redis no está corriendo")

        async def set(self, key: str, value: str, *, ex: int | None = None) -> None:
            raise ConnectionError("Redis no está corriendo")

    settings = _make_settings()
    resultado = await llm_service.generate_estimation(TRANSCRIPCION, settings, RedisCaido())

    assert llamadas == ["openai"]
    assert resultado.provider == "openai"


async def test_fail_soft_entrada_corrupta(monkeypatch):
    """Un JSON corrupto en el cache se descarta y se genera igual (miss)."""
    llamadas: list[str] = []

    cache = fakeredis.aioredis.FakeRedis(decode_responses=True)
    settings = _make_settings()
    clave = build_cache_key(TRANSCRIPCION, settings)
    await cache.set(clave, "{esto no es json")

    def fake_create(name, api_key, model, *, timeout, max_retries):
        llamadas.append(name)

        class FakeProvider:
            async def chat(self, messages, *, max_tokens=None, temperature=None):
                return _respuesta_ok()

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)

    resultado = await llm_service.generate_estimation(TRANSCRIPCION, settings, cache)

    assert llamadas == ["openai"], "la entrada corrupta se trató como miss"
    assert resultado.provider == "openai"


async def test_get_cached_estimation_roundtrip():
    """El cache guarda y devuelve el mismo dict serializado (roundtrip)."""
    cache = fakeredis.aioredis.FakeRedis(decode_responses=True)
    clave = "clave-de-prueba"
    original = {
        "estimation": "## Total\n**80 horas**\n",
        "truncated": False,
        "model": "gpt-4o-mini",
        "provider": "openai",
        "used_fallback": False,
        "usage": {"input_tokens": 50, "output_tokens": 20},
        "cost_usd": 0.003,
        "cost_note": "estimado",
    }

    await set_cached_estimation(cache, clave, original, ttl=3600)
    recuperado = await get_cached_estimation(cache, clave)

    assert recuperado == original
