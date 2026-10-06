"""Tests de la caché exact-match: fail soft y un único cliente de Redis.

Sin Redis real: el caso "caído" apunta a un puerto donde no escucha nadie, y
los casos que necesitan un Redis que funcione usan fakeredis.
"""

from __future__ import annotations

import time

import litellm
import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient
from redis.asyncio import Redis

from app.cache import close_cache_client, create_cache_client
from app.config import Settings, get_settings
from app.services.cache import EstimationCache, _NoOpCache, create_estimation_cache
from app.services.llm_wrapper import LLMWrapper

# Puerto 1: nadie escucha, la conexión se rechaza al instante.
REDIS_CAIDO = "redis://127.0.0.1:1/0"


def _cache_fake(ttl: int = 60) -> EstimationCache:
    return EstimationCache(FakeAsyncRedis(decode_responses=True), ttl=ttl)


async def _cliente_caido() -> Redis:
    """Cliente real hacia un Redis que no responde, creado como en el lifespan."""
    return await create_cache_client(Settings(_env_file=None, redis_url=REDIS_CAIDO))


@pytest.fixture(autouse=True)
def _limpiar_settings():
    yield
    get_settings.cache_clear()


# --- Redis funcionando -------------------------------------------------------


async def test_round_trip() -> None:
    cache = _cache_fake()
    key = cache.make_key("system", "user")
    await cache.set(key, {"content": "ok", "cost_usd": 0.01})
    assert await cache.get(key) == {"content": "ok", "cost_usd": 0.01}


async def test_miss_returns_none() -> None:
    cache = _cache_fake()
    assert await cache.get(cache.make_key("nunca", "guardado")) is None


async def test_ttl_is_applied() -> None:
    cache = _cache_fake(ttl=123)
    key = cache.make_key("s", "u")
    await cache.set(key, {"content": "ok"})
    assert 0 < await cache._client.ttl(key) <= 123


def test_key_changes_with_prompt() -> None:
    """El prompt entero entra en la clave: otra plantilla, otra entrada."""
    cache = _cache_fake()
    assert cache.make_key("v1", "u") != cache.make_key("v2", "u")
    assert cache.make_key("v1", "u") == cache.make_key("v1", "u")


async def test_corrupt_entry_is_a_miss() -> None:
    cache = _cache_fake()
    key = cache.make_key("s", "u")
    await cache._client.set(key, "{esto no es json")
    assert await cache.get(key) is None


# --- Redis caído: fail soft ---------------------------------------------------


async def test_get_with_redis_down_is_a_miss() -> None:
    cliente = await _cliente_caido()
    cache = create_estimation_cache(cliente, ttl=60)
    inicio = time.monotonic()
    assert await cache.get("cualquier-clave") is None
    assert time.monotonic() - inicio < 3  # acotado por el timeout, no cuelga
    await close_cache_client(cliente)


async def test_set_with_redis_down_does_not_raise() -> None:
    cliente = await _cliente_caido()
    cache = create_estimation_cache(cliente, ttl=60)
    await cache.set("cualquier-clave", {"content": "ok"})  # no lanza
    await close_cache_client(cliente)


# --- Caché desactivada ----------------------------------------------------------


async def test_no_client_disables_cache() -> None:
    """REDIS_URL vacío: el lifespan no crea cliente y la caché queda desactivada."""
    cliente = await create_cache_client(Settings(_env_file=None, redis_url=""))
    assert cliente is None
    cache = create_estimation_cache(cliente, ttl=60)
    assert isinstance(cache, _NoOpCache)
    await cache.set(cache.make_key("s", "u"), {"content": "ok"})
    assert await cache.get(cache.make_key("s", "u")) is None


# --- De punta a punta -----------------------------------------------------------


async def test_estimate_works_with_redis_down(monkeypatch) -> None:
    """La estimación sale aunque Redis no responda."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
    get_settings.cache_clear()
    cliente = await _cliente_caido()
    wrapper = LLMWrapper(settings=get_settings(), cache=create_estimation_cache(cliente, ttl=60))
    original = wrapper._router.acompletion

    async def sin_red(**kwargs):
        return await original(**kwargs, mock_response="estimación simulada")

    wrapper._router.acompletion = sin_red

    res = await wrapper.estimate(system_prompt="s", user_prompt="u", prompt_version="v1")
    assert res.content == "estimación simulada"
    await close_cache_client(cliente)


def test_app_opens_a_single_redis_client_and_the_wrapper_uses_it(monkeypatch) -> None:
    """Un solo cliente de Redis en toda la app: el del lifespan, que usa el wrapper."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
    monkeypatch.setenv("REDIS_URL", REDIS_CAIDO)
    get_settings.cache_clear()

    creados: list[Redis] = []
    from_url_original = Redis.from_url

    def contar(*args, **kwargs):
        cliente = from_url_original(*args, **kwargs)
        creados.append(cliente)
        return cliente

    monkeypatch.setattr(Redis, "from_url", contar)

    async def acompletion_sin_red(self, **kwargs):
        return await litellm.acompletion(
            model=kwargs["model"], messages=kwargs["messages"], mock_response="ok"
        )

    monkeypatch.setattr(litellm.Router, "acompletion", acompletion_sin_red)

    from app.main import create_app

    app = create_app()
    body = {
        "description": "A small B2B SaaS to manage employee equipment loans.",
        "project_type": "web_saas",
        "detail_level": "medium",
        "output_format": "phases_table",
    }
    with TestClient(app) as c:
        respuesta = c.post("/api/v1/estimate", json=body)
        assert respuesta.status_code == 200
        assert len(creados) == 1
        assert app.state.llm_wrapper._cache._client is app.state.cache_client
