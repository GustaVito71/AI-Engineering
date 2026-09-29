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
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError

from app import main
from app.cache import (
    build_cache_key,
    cerrar_cliente_cache,
    crear_cliente_cache,
    get_cache_client,
    get_cached_estimation,
    set_cached_estimation,
)
from app.config import Settings, get_settings
from app.main import create_app
from app.providers import LLMProviderError, LLMResponse, StreamChunk, StreamDone
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


# ─── El cableado: lifespan publica el cliente, el router lo consume ─────
#
# Los tests de arriba verifican la SEMANTICA de la cache llamando al
# servicio directamente con un cliente pasado a mano. Estos verifican lo que
# faltaba: que el cliente llegue solo. Si `get_cache_client` no estuviera en la
# dependency del router, o el lifespan no publicara nada, TODOS los tests de
# arriba seguirían en verde y el cache no funcionaría en la app real.


def _app_con_cache(cliente) -> TestClient:
    """App con el cache inyectado, saltando el lifespan real (que abriría el
    Redis de la máquina). Se sobreescriben las DOS dependencies: los settings
    para no leer la OPENAI_API_KEY del shell, y el cliente para no tocar
    Redis."""
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: _make_settings()
    app.dependency_overrides[get_cache_client] = lambda: cliente
    return TestClient(app)


def test_el_estimador_consume_el_cliente_inyectado(monkeypatch):
    """Dos POST idénticos: el segundo debe servirse del cache.

    La prueba de que el cableado existe es el CONTEO DE LLAMADAS al proveedor:
    si el router no pasara el cliente, serían dos y el assert falla."""
    llamadas: list[str] = []

    def fake_create(name, api_key, model, *, timeout, max_retries):
        llamadas.append(name)

        class FakeProvider:
            async def chat(self, messages, *, max_tokens=None, temperature=None):
                return _respuesta_ok()

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)
    cache = fakeredis.aioredis.FakeRedis(decode_responses=True)

    with _app_con_cache(cache) as c:
        primero = c.post("/api/v1/estimate", json={"transcription": TRANSCRIPCION})
        segundo = c.post("/api/v1/estimate", json={"transcription": TRANSCRIPCION})

    assert primero.status_code == 200
    assert segundo.status_code == 200
    assert llamadas == ["openai"], f"el cache no evitó la segunda llamada: {llamadas}"
    assert primero.json() == segundo.json()


def test_el_stream_tambien_consume_el_cliente(monkeypatch):
    """El endpoint SSE también recibe el cliente: mismo contrato, mismo cache."""
    llamadas: list[str] = []

    def fake_create(name, api_key, model, *, timeout, max_retries):
        class FakeProvider:
            async def chat(self, messages, *, max_tokens=None, temperature=None):
                return _respuesta_ok()

            async def chat_stream(self, messages, *, max_tokens=None, temperature=None):
                llamadas.append(name)
                yield StreamChunk(delta="## Total\n")
                yield StreamChunk(delta="**80 horas**\n")
                yield StreamDone(
                    model="gpt-4o-mini",
                    truncated=False,
                    usage={"input_tokens": 50, "output_tokens": 20},
                )

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)
    cache = fakeredis.aioredis.FakeRedis(decode_responses=True)

    with _app_con_cache(cache) as c:
        primero = c.post("/api/v1/estimate/stream", json={"transcription": TRANSCRIPCION})
        segundo = c.post("/api/v1/estimate/stream", json={"transcription": TRANSCRIPCION})

    assert primero.status_code == 200
    assert "event: meta" in primero.text, "un miss debe abrir con meta"
    assert "event: estimation" in primero.text
    # Cache hit: el contrato es SOLO `estimation`, sin meta ni delta.
    assert "event: estimation" in segundo.text
    assert "event: meta" not in segundo.text, "un cache hit no debe emitir meta"
    assert "event: delta" not in segundo.text, "un cache hit no debe emitir delta"
    assert llamadas == ["openai"], f"el stream no consultó el cache: {llamadas}"


def test_sin_cliente_de_cache_la_estimacion_sigue_funcionando(monkeypatch):
    """`REDIS_URL` vacío -> sin cache. La estimación NO puede depender de que
    haya Redis: es una optimización, no un requisito."""
    llamadas: list[str] = []

    def fake_create(name, api_key, model, *, timeout, max_retries):
        llamadas.append(name)

        class FakeProvider:
            async def chat(self, messages, *, max_tokens=None, temperature=None):
                return _respuesta_ok()

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)

    with _app_con_cache(None) as c:
        r = c.post("/api/v1/estimate", json={"transcription": TRANSCRIPCION})

    assert r.status_code == 200
    assert r.json()["provider"] == "openai"
    assert llamadas == ["openai"]


def test_redis_caido_no_tumba_la_estimacion_por_http(monkeypatch):
    """El fail-soft del servicio tiene que sobrevivir también a través de HTTP,
    no solo en la llamada directa al servicio."""
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

    with _app_con_cache(RedisCaido()) as c:
        r = c.post("/api/v1/estimate", json={"transcription": TRANSCRIPCION})

    assert r.status_code == 200, "un Redis caído no puede producir un 5xx"
    assert r.json()["provider"] == "openai"


async def test_crear_cliente_cache_devuelve_none_si_redis_url_esta_vacio():
    """El escape hatch por configuración: REDIS_URL vacío = cache desactivado,
    sin tocar código."""
    assert await crear_cliente_cache(_make_settings(redis_url="")) is None


async def test_el_lifespan_publica_el_cliente_y_lo_cierra(monkeypatch):
    """El cliente vive en app.state y se cierra al salir: sin pool de
    conexiones colgando entre tests ni en un shutdown."""
    app = create_app()
    cliente = fakeredis.aioredis.FakeRedis(decode_responses=True)
    cerrados: list[str] = []

    async def fake_crear(settings):
        return cliente

    async def fake_cerrar(c):
        cerrados.append("cerrado")
        await c.aclose()

    monkeypatch.setattr(main, "crear_cliente_cache", fake_crear)
    monkeypatch.setattr(main, "cerrar_cliente_cache", fake_cerrar)
    app.dependency_overrides[get_settings] = lambda: _make_settings()

    with TestClient(app) as c:
        assert c.app.state.cache_client is cliente, "el lifespan no publicó el cliente"
        assert c.get("/health").status_code == 200

    assert cerrados == ["cerrado"], "el lifespan no cerró el cliente al salir"


async def test_cerrar_cliente_cache_es_no_op_sin_cliente():
    """Sin cliente (cache desactivado) no hay nada que cerrar, y eso no es un
    error: es el estado normal de un despliegue con REDIS_URL vacío."""
    await cerrar_cliente_cache(None)


async def test_cerrar_cliente_cache_no_propaga_un_fallo_de_redis():
    """Un `aclose()` que falla no puede tumbar el shutdown de la app."""
    from redis.exceptions import RedisError

    class ClienteQueNoCierra:
        async def aclose(self) -> None:
            raise RedisError("conexión colgada")

    await cerrar_cliente_cache(ClienteQueNoCierra())
