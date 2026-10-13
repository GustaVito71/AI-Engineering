"""Tests de trazabilidad LLM (Paso 5): las 3 dimensiones del logging.

Qué se prueba:
- Dimensión 2 (costo) y 3 (camino): el evento `estimacion_completada`
  reporta camino (cache_hit/primario/fallback), intentos_proveedor y
  latencias por fase, más tokens y costo económico;
- Dimensión 1 (contenido): el evento `contenido_intento` con el prompt
  completo y la respuesta literal SIEMPRE se emite marcado DEBUG — con el
  nivel INFO de producción (LOG_LEVEL por defecto) el backend stdlib lo
  descarta antes del render, así el contenido sensible no llega al log;
- cada fallo de la API del proveedor queda registrado sanitizado en
  `provider_error` (tipo + status, nunca el cuerpo crudo).

Se captura con `capture_logs` de structlog (reemplaza los procesadores
globales durante el with): captura fiel de TODOS los eventos y campos,
sin depender del orden de la suite ni de qué backend (print/stdlib) quedó
configurado. La emisión usa `emitir`, que crea un proxy fresco por evento
y por eso se liga a la configuración vigente en ese instante.
"""

from __future__ import annotations

import logging

import fakeredis.aioredis
import httpx
import pytest
from openai import RateLimitError
from structlog.stdlib import add_log_level
from structlog.testing import capture_logs

from app.config import Settings
from app.main import configure_logging
from app.providers import LLMProviderError, LLMResponse, Message
from app.providers.openai_provider import OpenAIProvider
from app.services import llm_service

TRANSCRIPCION = (
    "Reunión de planificación del sprint para el módulo de facturación. "
    "El cliente quiere alta de clientes, emisión de comprobantes y reporte "
    "de cobranzas. Se definieron los límites del MVP para esta iteración."
)

# Third-party namespaces that must never write request/response headers or
# transport traces into the log. Deliberately duplicated from app.main instead
# of imported: this list is the test's own claim about the world, so a drift
# between the two (a renamed transport, a forgotten pin) fails here instead of
# silently passing. Note the `httpcore2` entry: httpx 0.28 ships httpcore2, so
# pinning only "httpcore" left the real emitter untouched.
_HTTP_STACK_ROOTS = frozenset({"httpx", "httpcore", "httpcore2", "openai", "aiohttp", "h11"})


@pytest.fixture
def _restore_logging():
    """Snapshot and restore global logging state.

    `configure_logging` mutates the ROOT logger and pins third-party loggers
    process-wide. Without restoring it, these tests would leak DEBUG root
    level and handler changes into the rest of the suite depending on
    execution order — the same class of bug as a leaking lru_cache.
    """
    root = logging.getLogger()
    saved_root = (root.level, list(root.handlers))
    noisy = ("httpx", "httpcore", "httpcore.connection", "httpcore.proxy", "httpcore.http11")
    saved_noisy = {name: logging.getLogger(name).level for name in noisy}
    yield
    root.setLevel(saved_root[0])
    root.handlers = saved_root[1]
    for name, level in saved_noisy.items():
        logging.getLogger(name).setLevel(level)


def _respuesta_ok(model: str = "gpt-4o-mini") -> LLMResponse:
    return LLMResponse(
        content="## Total\n**80 horas**\n",
        model=model,
        truncated=False,
        usage={"input_tokens": 50, "output_tokens": 20},
    )


def _make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, openai_api_key="k", **overrides)


def _fake_provider_ok(model: str = "gpt-4o-mini"):
    """Factory de provider fake sano: el nombre del proveedor define el modelo."""

    def factory(name, api_key, model, *, timeout, max_retries):
        class FakeProvider:
            async def chat(self, messages, *, max_tokens=None, temperature=None):
                # Respuesta que refleja el proveedor pedido, para poder
                # asertar quién respondió en la traza.
                return _respuesta_ok(model=model)

        return FakeProvider()

    return factory


async def test_estimacion_exitosa_traza_las_tres_dimensiones(monkeypatch):
    monkeypatch.setattr(llm_service, "create_provider", _fake_provider_ok())
    settings = _make_settings()

    with capture_logs(processors=[add_log_level]) as eventos:
        await llm_service.generate_estimation(TRANSCRIPCION, settings)

    # Dimensión 3 (camino): un solo proveedor, sin fallback, sin cache.
    cierres = [e for e in eventos if e["event"] == "estimacion_completada"]
    assert len(cierres) == 1
    final = cierres[0]
    assert final["camino"] == "primario"
    assert final["cache_hit"] is False
    assert final["uso_fallback"] is False
    assert final["intentos_proveedor"] == 1
    assert final["proveedor"] == "openai"
    assert final["latencia_cache_ms"] is None
    assert final["latencia_total_ms"] >= 0
    assert final["latencia_llm_ms"] >= 0

    # Dimensión 2 (costo): tokens y costo económico.
    assert final["input_tokens"] == 50
    assert final["output_tokens"] == 20
    assert final["modelo"] == "gpt-4o-mini"
    assert final["costo_usd"] is not None

    # Dimensión 1 (qué se envió): el resumen del intento está en INFO.
    intentos = [e for e in eventos if e["event"] == "intento_proveedor"]
    assert len(intentos) == 1
    assert intentos[0]["provider"] == "openai"
    assert intentos[0]["max_tokens"] == settings.llm_max_tokens


async def test_cache_hit_traza_camino_cache_y_costo_historico(monkeypatch):
    monkeypatch.setattr(llm_service, "create_provider", _fake_provider_ok())
    cache = fakeredis.aioredis.FakeRedis(decode_responses=True)
    settings = _make_settings()

    # Miss: genera y guarda (fuera de la captura).
    await llm_service.generate_estimation(TRANSCRIPCION, settings, cache)

    # Hit: no toca proveedor, y la traza lo dice.
    with capture_logs(processors=[add_log_level]) as eventos:
        await llm_service.generate_estimation(TRANSCRIPCION, settings, cache)

    cierres = [e for e in eventos if e["event"] == "estimacion_completada"]
    assert len(cierres) == 1
    final = cierres[0]
    assert final["camino"] == "cache_hit"
    assert final["cache_hit"] is True
    assert final["intentos_proveedor"] == 0
    assert final["latencia_llm_ms"] is None
    assert final["latencia_cache_ms"] >= 0
    # Costo histórico de la generación original, no 0: el hit no recalcula.
    assert final["costo_usd"] is not None

    # En un hit no hay intento de proveedor ni contenido.
    assert not [e for e in eventos if e["event"] == "intento_proveedor"]


async def test_fallback_traza_dos_intentos(monkeypatch):
    def factory(name, api_key, model, *, timeout, max_retries):
        class FakeProvider:
            async def chat(self, messages, *, max_tokens=None, temperature=None):
                if name == "openai":
                    raise LLMProviderError(provider="openai", detail="rate limit", status_code=429)
                return _respuesta_ok(model="claude-haiku-4-5")

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", factory)
    settings = _make_settings(llm_fallback="anthropic", anthropic_api_key="k2")

    with capture_logs(processors=[add_log_level]) as eventos:
        resultado = await llm_service.generate_estimation(TRANSCRIPCION, settings)

    assert resultado.used_fallback is True
    cierres = [e for e in eventos if e["event"] == "estimacion_completada"]
    assert len(cierres) == 1
    final = cierres[0]
    assert final["camino"] == "fallback"
    assert final["uso_fallback"] is True
    assert final["intentos_proveedor"] == 2
    assert final["proveedor"] == "anthropic"
    assert final["latencia_llm_ms"] >= 0

    # Solo el intento EXITOSO emite `intento_proveedor` (documenta qué se
    # envió y qué se recibió); el primario que falló por 429 no tiene
    # respuesta que documentar. Su fallo queda en el evento final
    # (camino=fallback, intentos_proveedor=2) y, si viene del adaptador
    # real, en `provider_error` sanitizado.
    intentos = [e for e in eventos if e["event"] == "intento_proveedor"]
    assert len(intentos) == 1
    assert intentos[0]["provider"] == "anthropic"


async def test_contenido_solo_en_debug(monkeypatch):
    """Dimensión 1 (prompt completo + respuesta literal) en DEBUG, no en INFO:
    el contenido es grande y sensato esconderlo del log de producción."""
    monkeypatch.setattr(llm_service, "create_provider", _fake_provider_ok())
    settings = _make_settings()

    with capture_logs(processors=[add_log_level]) as eventos:
        await llm_service.generate_estimation(TRANSCRIPCION, settings)

    contenidos = [e for e in eventos if e["event"] == "contenido_intento"]
    assert len(contenidos) == 1
    detalle = contenidos[0]
    # Marcado DEBUG: con LOG_LEVEL=INFO (producción) el backend stdlib lo
    # descarta antes del render — el contenido no llega al log.
    assert detalle["log_level"] == "debug"
    # Prompt completo: rol system (instrucciones + CAG) y user (transcripción).
    assert [m["role"] for m in detalle["mensajes"]] == ["system", "user"]
    assert detalle["respuesta_modelo"] == "## Total\n**80 horas**\n"

    # Los eventos de resumen (camino/costo) sí son INFO: visibles en el
    # nivel de producción sin exponer el contenido.
    niveles_resumen = {e["event"]: e["log_level"] for e in eventos}
    assert niveles_resumen["intento_proveedor"] == "info"
    assert niveles_resumen["estimacion_completada"] == "info"


async def test_provider_error_traza_sanitizada():
    """El adaptador real registra cada fallo de la API del proveedor con
    datos sanitizados: tipo y status, nunca el cuerpo crudo de la respuesta."""
    provider = OpenAIProvider(api_key="k", model="gpt-4o-mini")

    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    respuesta_http = httpx.Response(429, request=request, json={"error": {"message": "rate limit"}})

    async def explotar(*args, **kwargs):
        raise RateLimitError("rate limit", response=respuesta_http, body=respuesta_http.json())

    provider.client.responses.create = explotar

    with capture_logs(processors=[add_log_level]) as eventos, pytest.raises(LLMProviderError):
        await provider.chat([Message(role="user", content="hola")])

    errores = [e for e in eventos if e["event"] == "provider_error"]
    assert len(errores) == 1
    fallo = errores[0]
    assert fallo["provider"] == "openai"
    assert fallo["tipo_error"] == "RateLimitError"
    assert fallo["status_code"] == 429
    assert fallo["max_retries"] == 2
    # Sanitizado: el mensaje del error no aparece en la traza (solo tipo + status).
    assert not any(
        "rate limit" in str(v).lower() or "rate limit" in k
        for e in eventos
        for k, v in e.items()
        if k != "event"
    )


def test_configure_logging_filtra_los_loggers_http_de_terceros(_restore_logging):
    """Third-party loggers are pinned to WARNING even at LOG_LEVEL=DEBUG.

    HTTP transport libraries log request and response HEADERS at DEBUG, which is
    how account identifiers (e.g. `openai-organization`) and Cloudflare
    `set-cookie` values leaked into the log file. The fix must make DEBUG
    safe — not forbid DEBUG.
    """
    configure_logging("DEBUG")

    # The project's own level IS respected: only library verbosity is pinned,
    # the project's structlog events still get the level they asked for.
    assert logging.getLogger().level == logging.DEBUG

    # Every HTTP transport namespace is pinned by its ROOT, so child loggers
    # created lazily later still inherit WARNING instead of falling back to the
    # DEBUG root. Asserting the root names covers both current and future
    # transports without hardcoding every child.
    for name in ("httpx", "httpcore", "httpcore2", "openai", "aiohttp", "h11"):
        assert logging.getLogger(name).level == logging.WARNING, name

    # Guard the behaviour, not just the level we happened to set: no logger
    # belonging to a known third-party HTTP stack may be effectively noisier
    # than WARNING. This is what catches a transport package being renamed or
    # replaced — the previous test asserted the level of "httpcore" and stayed
    # green while the real emitter ("httpcore2") kept dumping headers.
    for logger_name, logger_obj in list(logging.root.manager.loggerDict.items()):
        if not isinstance(logger_obj, logging.Logger):
            continue
        if logger_name.split(".")[0] in _HTTP_STACK_ROOTS:
            assert logger_obj.getEffectiveLevel() >= logging.WARNING, (
                f"{logger_name} would emit DEBUG/INFO into the log: "
                f"effective level {logging.getLevelName(logger_obj.getEffectiveLevel())}"
            )
