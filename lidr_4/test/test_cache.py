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
from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType
from app.services.cache import EstimationCache, _NoOpCache, create_estimation_cache
from app.services.estimation import EstimationService
from app.services.llm_wrapper import LLMWrapper

# Puerto 1: nadie escucha, la conexión se rechaza al instante.
REDIS_CAIDO = "redis://127.0.0.1:1/0"

_SOLICITUD = EstimationRequest(
    description="Un SaaS B2B pequeño para gestionar el préstamo de equipos a empleados.",
    project_type=ProjectType.WEB_SAAS,
    detail_level=DetailLevel.MEDIUM,
    output_format=OutputFormat.PHASES_TABLE,
)


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


def test_la_clave_cambia_con_cada_parte() -> None:
    """Todas las partes entran en la clave: modelo, versión, parámetros, descripción."""
    cache = _cache_fake()
    base = ("openai/gpt-4o-mini", "v2", "web_saas", "medium", "phases_table", "una descripción")
    assert cache.make_key(*base) == cache.make_key(*base)
    for posicion in range(len(base)):
        otra = [*base]
        otra[posicion] = "otro"
        assert cache.make_key(*otra) != cache.make_key(*base)


def test_las_partes_no_se_pueden_confundir_entre_si() -> None:
    """Unir las partes con un separador haría chocar ("a|b", "c") con ("a", "b|c")."""
    cache = _cache_fake()
    assert cache.make_key("a|b", "c") != cache.make_key("a", "b|c")
    assert cache.make_key("ab", "c") != cache.make_key("a", "bc")


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
    wrapper = LLMWrapper(settings=get_settings())
    servicio = EstimationService(wrapper, cache=create_estimation_cache(cliente, ttl=60))
    original = wrapper._router.acompletion

    async def sin_red(**kwargs):
        return await original(**kwargs, mock_response="estimación simulada")

    wrapper._router.acompletion = sin_red

    res = await servicio.estimate_text(_SOLICITUD, "v2")
    assert res.text == "estimación simulada"
    await close_cache_client(cliente)


def test_app_opens_a_single_redis_client_and_the_service_uses_it(monkeypatch) -> None:
    """Un solo cliente de Redis en toda la app: el del lifespan, que usa el servicio."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
    monkeypatch.setenv("REDIS_URL", REDIS_CAIDO)
    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "false")
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
        assert app.state.estimation_cache._client is app.state.cache_client
