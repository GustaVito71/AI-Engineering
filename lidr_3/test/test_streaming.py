"""Tests del streaming con chat_stream() (Paso 6).

Qué se prueba:
- adaptadores: cada uno cede StreamChunk por fragmento y UN StreamDone con
  usage/truncated normalizados; un error del SDK al abrir el stream se
  traduce a LLMProviderError (mismo contrato que chat()).
- servicio stream_estimation: camino primario (meta -> fragmentos -> final),
  cache hit que cede SOLO el final (sin meta), fallback solo si el primario
  falla ANTES de emitir un fragmento, NO hay fallback a mitad de stream,
  ambos fallan pre-primero -> LLMServiceError, y el primario exitoso
  alimenta el cache.
- endpoint SSE: el pre-arranque traduce a HTTP real el 503 (config) y el 502
  (ambos proveedores), y un fallo a mitad baja como evento `error`.

Los adaptadores se prueban con fakes del SDK (SimpleNamespace): el objetivo
es el contrato del adaptador, no los objetos del SDK. El servicio se prueba
con fakes de provider vía monkeypatch de create_provider, igual que
test_cache.py. El endpoint usa TestClient con `client.stream(...)`.

Nota de wiring: igual que el endpoint no-streaming, el endpoint SSE no recibe
cache (el cache es capa opcional del servicio). El cache hit del stream se
prueba a nivel servicio.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import fakeredis.aioredis
import httpx
import pytest
from anthropic import RateLimitError as AnthropicRateLimitError
from fastapi.testclient import TestClient
from openai import RateLimitError as OpenAIRateLimitError
from structlog.testing import capture_logs

from app.cache import build_cache_key, set_cached_estimation
from app.config import Settings, get_settings
from app.main import create_app
from app.providers import LLMProviderError, Message, StreamChunk, StreamDone
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.openai_provider import OpenAIProvider
from app.services import llm_service
from app.services.llm_service import (
    LLMServiceError,
    StreamFinal,
    StreamMeta,
    stream_estimation,
)

TRANSCRIPCION = (
    "Reunión de planificación del sprint para el módulo de facturación. "
    "El cliente quiere alta de clientes, emisión de comprobantes y reporte "
    "de cobranzas. Se definieron los límites del MVP para esta iteración."
)


def _make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, **{"openai_api_key": "k", **overrides})


# ---------------------------------------------------------------------------
# Fakes de SDK (para los adaptadores)
# ---------------------------------------------------------------------------


class _FakeSdkStream:
    """Iterable async que simula el CM de `responses.stream` de OpenAI."""

    def __init__(self, eventos: list, error: Exception | None = None):
        self._eventos = list(eventos)
        self.error = error

    async def __aenter__(self):
        if self.error is not None:
            raise self.error
        return self

    async def __aexit__(self, *exc_info):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._eventos:
            raise StopAsyncIteration
        return self._eventos.pop(0)


class _FakeAnthropicManager:
    """Simula el manager de `messages.stream` abiding SU contrato real.

    Importante: el doble reproduce la API del SDK 1.8.0, no una versión
    conveniente. `AsyncMessageStream` es asíncrono iterable (se itera con
    `async for chunk in stream`), y `until_done()` es una corrutina anotada
    `-> None` que consume el stream hasta el final y NO devuelve el iterador.

    Una versión anterior de este doble hacía que `until_done()` devolviera un
    generador, que es lo que el adaptador asumía por error. Con ese doble el
    bug pasaba la suite entera y reventaba contra la API real: un test que
    miente sobre la API hace invisible el bug que justamente_probea.
    """

    def __init__(self, eventos: list, error: Exception | None = None):
        self._eventos = list(eventos)
        self.error = error
        self._consumido = False

    async def __aenter__(self):
        if self.error is not None:
            raise self.error
        self._consumido = False
        return self

    async def __aexit__(self, *exc_info):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._eventos:
            raise StopAsyncIteration
        return self._eventos.pop(0)

    async def until_done(self) -> None:
        # Contrato real: consume y devuelve None. Si el adaptador lo usara
        # como iterador, `async for` fallaría con NoneType.
        self._consumido = True
        self._eventos.clear()


def _sdk_error(cls, status: int) -> Exception:
    """Un RateLimitError real del SDK (con response httpx de verdad)."""
    request = httpx.Request("POST", "https://api.example.com/v1/responses")
    return cls(
        "rate limit de prueba",
        response=httpx.Response(status, request=request),
        body=None,
    )


# ---------------------------------------------------------------------------
# Adaptadores
# ---------------------------------------------------------------------------


async def test_openai_stream_cede_fragmentos_y_cierre():
    provider = OpenAIProvider(api_key="k", model="gpt-4o-mini")
    provider.client.responses.stream = lambda **kwargs: _FakeSdkStream(
        [
            SimpleNamespace(type="response.output_text.delta", delta="## Total\n"),
            SimpleNamespace(type="response.output_text.delta", delta="**80 horas**\n"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    model="gpt-4o-mini",
                    status="completed",
                    usage=SimpleNamespace(input_tokens=50, output_tokens=20),
                    incomplete_details=None,
                ),
            ),
        ]
    )

    eventos = [
        e
        async for e in provider.chat_stream([Message(role="user", content="hola")], max_tokens=100)
    ]

    assert eventos[:2] == [StreamChunk("## Total\n"), StreamChunk("**80 horas**\n")]
    done = eventos[-1]
    assert isinstance(done, StreamDone)
    assert done.model == "gpt-4o-mini"
    assert done.truncated is False
    assert done.usage == {"input_tokens": 50, "output_tokens": 20}
    # El contrato: fragmentos + UN cierre, y nada más.
    assert len(eventos) == 3


async def test_openai_stream_truncado_cierra_con_incomplete_y_no_se_queda_mudo():
    """Un stream cortado cierra con `response.incomplete`, NO con `response.completed`.

    Si el provider escucha solo el cierre feliz, nunca emite `StreamDone`, el
    generador termina mudo y `llm_service` revienta con "terminó sin cierre" —
    el error que ocurrió en producción. El contrato: un cierre SIEMPRE, y
    `truncated=True` cuando lo cortó el techo de `max_output_tokens`.
    """
    provider = OpenAIProvider(api_key="k", model="gpt-5.1")
    provider.client.responses.stream = lambda **kwargs: _FakeSdkStream(
        [
            SimpleNamespace(type="response.output_text.delta", delta="## Total\n"),
            SimpleNamespace(
                type="response.incomplete",
                response=SimpleNamespace(
                    model="gpt-5.1",
                    status="incomplete",
                    usage=SimpleNamespace(input_tokens=50, output_tokens=100),
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                ),
            ),
        ]
    )

    eventos = [
        e
        async for e in provider.chat_stream([Message(role="user", content="hola")], max_tokens=100)
    ]

    # Lo que se había roto: el stream igual cierra, no se queda sin cierre.
    assert len(eventos) == 2
    done = eventos[-1]
    assert isinstance(done, StreamDone)
    assert done.truncated is True
    assert done.model == "gpt-5.1"
    assert done.usage == {"input_tokens": 50, "output_tokens": 100}


async def test_openai_stream_error_inicial_se_traduce_a_llm_provider_error():
    provider = OpenAIProvider(api_key="k", model="gpt-4o-mini")
    provider.client.responses.stream = lambda **kwargs: _FakeSdkStream(
        [], error=_sdk_error(OpenAIRateLimitError, 429)
    )

    with pytest.raises(LLMProviderError) as excinfo:
        async for _ in provider.chat_stream([Message(role="user", content="hola")], max_tokens=100):
            pass

    assert excinfo.value.provider == "openai"
    assert excinfo.value.status_code == 429
    assert excinfo.value.can_fallback is True


async def test_anthropic_stream_cede_fragmentos_y_cierre():
    provider = AnthropicProvider(api_key="k", model="claude-haiku-4-5")
    provider.client.messages.stream = lambda **kwargs: _FakeAnthropicManager(
        [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    model="claude-haiku-4-5",
                    usage=SimpleNamespace(input_tokens=50),
                ),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="## Total\n"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="**80 horas**\n"),
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn"),
                usage=SimpleNamespace(output_tokens=20),
            ),
        ]
    )

    eventos = [
        e
        async for e in provider.chat_stream([Message(role="user", content="hola")], max_tokens=100)
    ]

    assert eventos[:2] == [StreamChunk("## Total\n"), StreamChunk("**80 horas**\n")]
    done = eventos[-1]
    assert isinstance(done, StreamDone)
    assert done.model == "claude-haiku-4-5"
    assert done.truncated is False
    assert done.usage == {"input_tokens": 50, "output_tokens": 20}
    assert len(eventos) == 3


async def test_anthropic_stream_marca_truncado_con_stop_reason_max_tokens():
    provider = AnthropicProvider(api_key="k", model="claude-haiku-4-5")
    provider.client.messages.stream = lambda **kwargs: _FakeAnthropicManager(
        [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    model="claude-haiku-4-5",
                    usage=SimpleNamespace(input_tokens=50),
                ),
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="max_tokens"),
                usage=SimpleNamespace(output_tokens=100),
            ),
        ]
    )

    eventos = [
        e
        async for e in provider.chat_stream([Message(role="user", content="hola")], max_tokens=100)
    ]

    assert len(eventos) == 1
    assert eventos[0].truncated is True


async def test_anthropic_stream_error_inicial_se_traduce_a_llm_provider_error():
    provider = AnthropicProvider(api_key="k", model="claude-haiku-4-5")
    provider.client.messages.stream = lambda **kwargs: _FakeAnthropicManager(
        [], error=_sdk_error(AnthropicRateLimitError, 500)
    )

    with pytest.raises(LLMProviderError) as excinfo:
        async for _ in provider.chat_stream([Message(role="user", content="hola")], max_tokens=100):
            pass

    assert excinfo.value.provider == "anthropic"
    assert excinfo.value.status_code == 500
    assert excinfo.value.can_fallback is True


# ---------------------------------------------------------------------------
# Servicio: stream_estimation
# ---------------------------------------------------------------------------


def _fake_provider_factory(
    llamadas: list[str],
    *,
    primario_falla: str | None = None,
    falla_a_mitad: bool = False,
    primer_chunk_delay: float = 0.0,
):
    """Fábrica de providers con chat_stream simulado.

    primario_falla: texto del error del primario antes del primer fragmento
    (pretende simular un 429). falla_a_mitad: el primario cede un fragmento y
    después lanza -> en ese caso nadie hereda el stream.
    primer_chunk_delay: segundos de espera antes del PRIMER fragmento, para
    simular el tiempo de procesamiento real del proveedor antes del primer
    token (el resto de los fragmentos salen sin demora)."""

    def fake_create(name, api_key, model, *, timeout, max_retries):
        llamadas.append(name)

        class FakeProvider:
            async def chat_stream(self, messages, *, max_tokens=None, temperature=None):
                if name == "openai" and primario_falla is not None:
                    raise LLMProviderError(
                        provider="openai", detail=primario_falla, status_code=429
                    )
                if primer_chunk_delay:
                    await asyncio.sleep(primer_chunk_delay)
                if name == "openai" and falla_a_mitad:
                    yield StreamChunk(delta="Empezó a escribir ")
                    raise LLMProviderError(
                        provider="openai", detail="se cayó a mitad", status_code=500
                    )
                yield StreamChunk(delta="## Total\n")
                yield StreamChunk(delta="**80 horas**\n")
                yield StreamDone(
                    model=model,
                    truncated=False,
                    usage={"input_tokens": 50, "output_tokens": 20},
                )

        return FakeProvider()

    return fake_create


async def test_stream_primario_cede_meta_fragmentos_y_final(monkeypatch):
    llamadas: list[str] = []
    monkeypatch.setattr(llm_service, "create_provider", _fake_provider_factory(llamadas))
    settings = _make_settings()

    eventos = [e async for e in stream_estimation(TRANSCRIPCION, settings)]

    assert llamadas == ["openai"]
    assert eventos[0] == StreamMeta(proveedor="openai", camino="primario")
    assert isinstance(eventos[1], StreamChunk)
    assert eventos[1].delta == "## Total\n"
    assert isinstance(eventos[-1], StreamFinal)
    final = eventos[-1].resultado
    assert final.provider == "openai"
    assert final.used_fallback is False
    assert final.estimation == "## Total\n**80 horas**\n"
    assert final.model == "gpt-4o-mini"


async def test_stream_cache_hit_cede_solo_el_final(monkeypatch):
    llamadas: list[str] = []
    monkeypatch.setattr(llm_service, "create_provider", _fake_provider_factory(llamadas))
    settings = _make_settings()
    cache = fakeredis.aioredis.FakeRedis(decode_responses=True)

    # Pre-popular el cache con el shape de EstimationResult.
    clave = build_cache_key(TRANSCRIPCION, settings)
    await set_cached_estimation(
        cache,
        clave,
        {
            "estimation": "## Total\n**80 horas**\n",
            "truncated": False,
            "model": "gpt-4o-mini",
            "provider": "openai",
            "used_fallback": False,
            "usage": {"input_tokens": 50, "output_tokens": 20},
            "cost_usd": 0.003,
            "cost_note": "estimado",
        },
        ttl=3600,
    )

    eventos = [e async for e in stream_estimation(TRANSCRIPCION, settings, cache)]

    # Contrato: cache hit = UN solo StreamFinal, sin meta ni fragmentos.
    assert len(eventos) == 1
    assert isinstance(eventos[0], StreamFinal)
    assert llamadas == [], "el cache hit no toca al proveedor"


async def test_stream_ttft_mide_el_primer_token_real(monkeypatch):
    """`ttft_ms` must cover the real wait for the provider's first token.

    The service pre-starts the stream by consuming its first event, so a
    stopwatch started after the stream opens measures a no-op gap and reports
    ~0.5ms while the provider really took over a second. Injecting a known
    delay before the first chunk pins the semantics: ttft_ms is measured from
    request submission to the first content chunk, so it MUST include it.
    """
    llamadas: list[str] = []
    monkeypatch.setattr(
        llm_service,
        "create_provider",
        _fake_provider_factory(llamadas, primer_chunk_delay=0.25),
    )
    settings = _make_settings()

    with capture_logs() as eventos:
        [e async for e in stream_estimation(TRANSCRIPCION, settings)]

    cierres = [e for e in eventos if e["event"] == "estimacion_completada"]
    assert len(cierres) == 1
    final = cierres[0]
    assert final["ttft_ms"] is not None
    # Tolerant lower bound: 250ms injected, assert >= 200ms so a slow CI box
    # does not flake, but the old ~0.5ms bug still fails loudly.
    assert final["ttft_ms"] >= 200, (
        f"ttft_ms={final['ttft_ms']} did not include the injected 250ms provider delay"
    )


async def test_stream_cache_hit_reporta_ttft_none(monkeypatch):
    """A cache hit never opens a provider stream, so there is no first token
    to measure: `ttft_ms` stays None. Guards the ttft stopwatch move above from
    leaking a bogus 0.0 into the cache-hit path."""
    llamadas: list[str] = []
    monkeypatch.setattr(llm_service, "create_provider", _fake_provider_factory(llamadas))
    settings = _make_settings()
    cache = fakeredis.aioredis.FakeRedis(decode_responses=True)

    # Miss first (outside the capture): generates and populates the cache.
    [e async for e in stream_estimation(TRANSCRIPCION, settings, cache)]

    # Hit: no provider stream at all.
    with capture_logs() as eventos:
        eventos_cache = [e async for e in stream_estimation(TRANSCRIPCION, settings, cache)]

    assert len(eventos_cache) == 1
    assert isinstance(eventos_cache[0], StreamFinal)
    cierres = [e for e in eventos if e["event"] == "estimacion_completada"]
    assert len(cierres) == 1
    final = cierres[0]
    assert final["camino"] == "cache_hit"
    assert final["ttft_ms"] is None
    assert final["latencia_llm_ms"] is None


async def test_stream_fallback_solo_si_el_primario_no_emitio(monkeypatch):
    llamadas: list[str] = []
    monkeypatch.setattr(
        llm_service,
        "create_provider",
        _fake_provider_factory(llamadas, primario_falla="rate limit"),
    )
    settings = _make_settings(llm_fallback="anthropic", anthropic_api_key="k2")

    eventos = [e async for e in stream_estimation(TRANSCRIPCION, settings)]

    assert llamadas == ["openai", "anthropic"]
    assert eventos[0] == StreamMeta(proveedor="anthropic", camino="fallback")
    final = eventos[-1].resultado
    assert final.used_fallback is True
    assert final.provider == "anthropic"
    assert final.estimation == "## Total\n**80 horas**\n"


async def test_stream_no_hay_fallback_a_mitad(monkeypatch):
    """Una vez que el primario emitió texto, el fallback NO salva: cambiar de
    modelo a mitad de respuesta mentiría sobre quién generó cada parte."""
    llamadas: list[str] = []
    monkeypatch.setattr(
        llm_service, "create_provider", _fake_provider_factory(llamadas, falla_a_mitad=True)
    )
    settings = _make_settings(llm_fallback="anthropic", anthropic_api_key="k2")

    with pytest.raises(LLMServiceError):
        [e async for e in stream_estimation(TRANSCRIPCION, settings)]

    assert llamadas == ["openai"], "el fallback no se intenta a mitad de stream"


async def test_stream_ambos_fallan_pre_primer_fragmento_y_son_502_real(monkeypatch):
    """Ambos proveedores fallan antes de emitir un fragmento: la versión
    no-streaming también traduce esto a 502; el router usa el pre-arranque
    para responder el status REAL en lugar de un body roto de un 200."""
    llamadas: list[str] = []
    monkeypatch.setattr(
        llm_service,
        "create_provider",
        _fake_provider_factory(llamadas, primario_falla="rate limit", falla_a_mitad=True),
    )
    # falla_a_mitad en el primario no importa: el primario falla antes de yield.
    settings = _make_settings(llm_fallback="anthropic", anthropic_api_key="k2")

    # Necesitamos que el fallback TAMBIÉN falle pre-primero: extendemos el fake.
    def ambos_fallan(name, api_key, model, *, timeout, max_retries):
        llamadas.append(name)

        class FakeProvider:
            async def chat_stream(self, messages, *, max_tokens=None, temperature=None):
                raise LLMProviderError(provider=name, detail="rate limit", status_code=429)
                yield  # pragma: no cover — convierte el método en generador async

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", ambos_fallan)

    with pytest.raises(LLMServiceError):
        [e async for e in stream_estimation(TRANSCRIPCION, settings)]

    assert llamadas == ["openai", "anthropic"]


async def test_stream_primario_exitoso_alimenta_cache(monkeypatch):
    llamadas: list[str] = []
    monkeypatch.setattr(llm_service, "create_provider", _fake_provider_factory(llamadas))
    settings = _make_settings()
    cache = fakeredis.aioredis.FakeRedis(decode_responses=True)

    eventos = [e async for e in stream_estimation(TRANSCRIPCION, settings, cache)]

    clave = build_cache_key(TRANSCRIPCION, settings)
    assert await cache.get(clave) is not None, "el primario exitoso se cachea"

    # Un generate_estimation posterior debe pegarle al cache (sin proveedor).
    segundo = await llm_service.generate_estimation(TRANSCRIPCION, settings, cache)
    assert llamadas == ["openai"], "el hit posterior no vuelve a llamar"
    assert segundo.estimation == eventos[-1].resultado.estimation


# ---------------------------------------------------------------------------
# Endpoint SSE
# ---------------------------------------------------------------------------


def _cliente_con(settings: Settings):
    """TestClient con settings inyectados (mismo patrón que conftest)."""
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    return app


def _parsear_sse(lineas: list[str]) -> list[tuple[str, dict]]:
    """Convierte las líneas `event:`/`data:` de un body SSE en tuplas."""
    eventos: list[tuple[str, dict]] = []
    actual: dict | None = None
    for linea in lineas:
        if linea.startswith("event: "):
            actual = {"tipo": linea[len("event: ") :]}
        elif linea.startswith("data: ") and actual is not None:
            actual["datos"] = json.loads(linea[len("data: ") :])
            eventos.append((actual["tipo"], actual["datos"]))
            actual = None
    return eventos


def test_endpoint_sse_emite_meta_delta_y_estimation(monkeypatch):
    llamadas: list[str] = []
    monkeypatch.setattr(llm_service, "create_provider", _fake_provider_factory(llamadas))
    app = _cliente_con(_make_settings())

    with (
        TestClient(app) as client,
        client.stream(
            "POST", "/api/v1/estimate/stream", json={"transcription": TRANSCRIPCION}
        ) as respuesta,
    ):
        lineas = [l for l in respuesta.iter_lines() if l]
        assert respuesta.status_code == 200
        assert respuesta.headers["content-type"].startswith("text/event-stream")

    eventos = _parsear_sse(lineas)
    tipos = [tipo for tipo, _ in eventos]
    assert tipos == ["meta", "delta", "delta", "estimation"]
    assert eventos[0][1] == {"proveedor": "openai", "camino": "primario"}
    assert eventos[0][0] == "meta"
    assert "".join(datos["texto"] for tipo, datos in eventos if tipo == "delta") == (
        "## Total\n**80 horas**\n"
    )
    assert eventos[-1][0] == "estimation"
    assert eventos[-1][1]["estimation"] == "## Total\n**80 horas**\n"


def test_endpoint_sse_responde_502_real_si_ambos_fallan_antes_del_primer_byte(monkeypatch):
    llamadas: list[str] = []

    def ambos_fallan(name, api_key, model, *, timeout, max_retries):
        llamadas.append(name)

        class FakeProvider:
            async def chat_stream(self, messages, *, max_tokens=None, temperature=None):
                raise LLMProviderError(provider=name, detail="rate limit", status_code=429)
                yield  # pragma: no cover — convierte el método en generador async

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", ambos_fallan)
    app = _cliente_con(_make_settings(llm_fallback="anthropic", anthropic_api_key="k2"))

    with (
        TestClient(app) as client,
        client.stream(
            "POST", "/api/v1/estimate/stream", json={"transcription": TRANSCRIPCION}
        ) as respuesta,
    ):
        assert respuesta.status_code == 502
        body = json.loads(respuesta.read())
        assert "Inténtalo de nuevo" in body["detail"]

    assert llamadas == ["openai", "anthropic"]


def test_endpoint_sse_responde_503_real_sin_api_key():
    app = _cliente_con(_make_settings(openai_api_key=None))

    with (
        TestClient(app) as client,
        client.stream(
            "POST", "/api/v1/estimate/stream", json={"transcription": TRANSCRIPCION}
        ) as respuesta,
    ):
        assert respuesta.status_code == 503


def test_endpoint_sse_muerte_a_mitad_baja_como_evento_error(monkeypatch):
    llamadas: list[str] = []
    monkeypatch.setattr(
        llm_service, "create_provider", _fake_provider_factory(llamadas, falla_a_mitad=True)
    )
    app = _cliente_con(_make_settings(llm_fallback="anthropic", anthropic_api_key="k2"))

    with (
        TestClient(app) as client,
        client.stream(
            "POST", "/api/v1/estimate/stream", json={"transcription": TRANSCRIPCION}
        ) as respuesta,
    ):
        assert respuesta.status_code == 200
        lineas = [l for l in respuesta.iter_lines() if l]

    eventos = _parsear_sse(lineas)
    tipos = [tipo for tipo, _ in eventos]
    # El primer fragmento salió (texto comprometido) y el error baja como
    # evento SSE honesto; NO hay `estimation` porque la respuesta no cerró.
    assert tipos == ["meta", "delta", "error"]
    assert "No se pudo completar" in eventos[-1][1]["detail"]
    assert llamadas == ["openai"], "no se intenta el fallback a mitad"


async def test_stream_fallback_sin_key_es_fallo_del_salto_y_no_config(monkeypatch):
    """El mismo bug que el de `chat`, en el camino de streaming.

    LLM_FALLBACK declarado sin su key es un fallo del salto, no un rechazo de
    la request. Si `LLMConfigurationError` se escapara del `except` del
    fallback, el router lo serviría como 503 y el cliente leería que su
    configuración está rota, cuando lo que pasó fue un rate limit del
    primario y un fallback sin key: la acción correcta es reintentar.
    """
    llamadas: list[str] = []
    monkeypatch.setattr(
        llm_service,
        "create_provider",
        _fake_provider_factory(llamadas, primario_falla="rate limit"),
    )
    settings = _make_settings(
        llm_provider="openai",
        openai_api_key="k",
        llm_fallback="anthropic",
        anthropic_api_key=None,  # el fallback NO tiene key
    )

    with pytest.raises(llm_service.LLMServiceError) as excinfo:
        [e async for e in stream_estimation(TRANSCRIPCION, settings)]

    assert not isinstance(excinfo.value, llm_service.LLMConfigurationError)
    assert "openai" in str(excinfo.value) and "anthropic" in str(excinfo.value)
    # El fallback ni se construyó: la key se exige antes de abrir el stream.
    assert llamadas == ["openai"]


async def test_anthropic_stream_se_itera_a_si_mismo_no_con_until_done():
    """El adaptador debe iterar el manager, no su `until_done()`.

    Fijado explícitamente porque el doble anterior hacía que `until_done()`
    devolviera un iterador: el test pasaba mientras la llamada real fallaba con
    "'async for' requires an object with __aiter__ method". Si alguien
    reintroduce `until_done()`, este test lo dice en el mensaje.
    """
    provider = AnthropicProvider(api_key="k", model="claude-haiku-4-5")
    eventos = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                model="claude-haiku-4-5-20251001",
                usage=SimpleNamespace(input_tokens=7),
            ),
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="hola"),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=3),
        ),
    ]
    capturado = {}

    def fake_stream(**kwargs):
        manager = _FakeAnthropicManager(eventos)
        capturado["manager"] = manager
        return manager

    provider.client.messages.stream = fake_stream

    recibidos = [
        e
        async for e in provider.chat_stream([Message(role="user", content="hola")], max_tokens=100)
    ]

    assert [type(e).__name__ for e in recibidos] == ["StreamChunk", "StreamDone"]
    # La prueba: `until_done()` NO se usó, así que el flag sigue en False.
    assert capturado["manager"]._consumido is False, (
        "el adaptador está usando until_done() en vez de iterar el stream: "
        "contra la API real eso es un TypeError, aunque el doble lo tolerase"
    )
