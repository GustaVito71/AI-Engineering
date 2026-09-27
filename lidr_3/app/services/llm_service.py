"""Servicio de estimación (dominio, agnóstico de HTTP).

El servicio no sabe qué es un 502. Dice «esto falló» con sus propias
excepciones; el router traduce. Por eso este módulo es reutilizable desde
un worker, un CLI o una cola sin arrastrar FastAPI detrás.

Flujo de una estimación:
1. Extraer la key en el punto de uso (falta -> LLMConfigurationError).
2. Construir el prompt: system (instrucciones + cache CAG) y user (la
   transcripción dentro de un delimitador impredecible por petición).
3. Llamar al proveedor y traducir sus errores normalizados a LLMServiceError.
4. Señalar si la respuesta se truncó: en CAG un 200 con la respuesta a medio
   escribir es la peor clase de fallo, la que no se ve.

Fallback (wrapper): si el proveedor primario falla con un error que otro
proveedor tiene chance de salvar (`LLMProviderError.can_fallback`, decidido
por HTTP status: 401/429/5xx/red), se hace UN salto al secundario
(`settings.llm_fallback`). Los errores permanentes (4xx) fallan rápido: el
mismo request va a fallar igual en el otro proveedor. El resultado expone el
proveedor que realmente respondió y `used_fallback`.

Cache (opcional): si se pasa un `CacheClient`, la estimación mira primero la
clave determinista (system prompt CAG + transcripción + params). En un hit
no se arma el prompt ni se llama a un proveedor. Solo se cachea la respuesta
del primario exitoso: la clave se arma con el modelo del proveedor activo y
cachear una respuesta del fallback bajo esa clave mentiría sobre el origen.
Fail soft: si el cache falla (Redis caído, entrada corrupta), se loguea y se
genera igual; un cache nunca tumba la estimación.

Trazabilidad (las 3 dimensiones del logging LLM, en structlog):
- Dimensión 1 (qué se envió y qué se recibió): evento `intento_proveedor`
  (INFO, resumen: provider, modelo, max_tokens, latencia) y `contenido_intento`
  (DEBUG, literal: prompt completo system+user y respuesta del modelo). El
  contenido es grande y sensible: se activa con LOG_LEVEL=DEBUG, no llena el
  log de producción.
- Dimensión 2 (cuánto costó): en `estimacion_completada` van input/output
  tokens, modelo y costo económico (costo_usd). Un cache hit reporta el
  costo histórico de la generación original, no 0.
- Dimensión 3 (qué camino siguió): `estimacion_completada` reporta
  camino (cache_hit | primario | fallback), cantidad de proveedores
  llamados (intentos_proveedor), y latencias por fase (cache, LLM, total).
  El evento `provider_error` de cada adaptador deja ver cada fallo de la API
  del proveedor; los reintentos internos del SDK (backoff) no se cuentan uno
  a uno porque el SDK no los expone sin APIs privadas.

El servicio es async de punta a punta: `generate_estimation` es corrutina y
hace `await provider.chat(...)`. Quien la use desde un contexto síncrono
(CLI, worker) debe envolver la llamada con `asyncio.run(...)`.

Streaming (`stream_estimation`): misma disciplina de cache, fallback y traza,
pero cede eventos tipados (StreamMeta/StreamChunk/StreamFinal) en lugar de
una sola respuesta. La diferencia crítica es el pre-arranque: cada proveedor
se abre consumiendo su PRIMER evento, así el camino (primario/fallback) se
decide antes de emitir un solo fragmento y los fallos pre-primer-fragmento se
pueden responder con 502 HTTP real. Un fallo a mitad de stream (ya hubo
deltas) NO hace fallback: el texto que arrancó es comprometido.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Protocol

from ..cache import build_cache_key, get_cached_estimation, set_cached_estimation
from ..config import LLMConfigurationError as _ConfigError
from ..config import Settings
from ..context.examples import build_system_prompt, transcription_delimiter
from ..providers import (
    BaseProvider,
    LLMProviderError,
    LLMResponse,
    Message,
    StreamChunk,
    StreamDone,
    UnknownProviderError,
    create_provider,
)
from ..tracing import emitir
from .pricing import estimate_cost


class CacheClient(Protocol):
    """Interfaz mínima que necesita el servicio de un cache.

    Protocol (duck-typing estructural): el dominio no conoce redis ni
    fakeredis; cualquier objeto con get/set async entra. Lo cumplen
    `redis.asyncio.Redis` y `fakeredis.aioredis.FakeRedis`."""

    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, *, ex: int | None = None) -> None: ...


class LLMServiceError(Exception):
    """Error de dominio: falló la generación de la estimación (-> 502).

    Pensado para ser traducido por el router. Nunca contiene detalles del
    SDK: esos se quedan en el log del servidor."""


class LLMConfigurationError(LLMServiceError):
    """Falta configuración local para llamar al LLM (-> 503).

    Hereda del ServiceError pero merece un status distinto: el cliente no
    debería reintentar, es un problema del que opera el servicio."""


@dataclass(frozen=True)
class EstimationResult:
    estimation: str
    truncated: bool
    model: str
    provider: str
    used_fallback: bool
    usage: dict[str, int | None] | None
    cost_usd: float | None
    cost_note: str | None


@dataclass(frozen=True)
class StreamMeta:
    """Primer evento de un stream real (miss de cache): quién responde.

    Se emite recién cuando el proveedor efectivo ARRANCÓ (primer fragmento o
    cierre recibido), nunca antes: el camino primario/fallback se decide en
    ese momento. En un cache hit NO se emite — el cliente recibe un único
    StreamFinal, igual que el contrato de la respuesta no-streaming."""

    proveedor: str
    camino: str


@dataclass(frozen=True)
class StreamFinal:
    """Cierre de un stream: la EstimationResult completa."""

    resultado: EstimationResult


def _build_messages(transcription: str) -> list[Message]:
    """arma los mensajes con roles y fronteras explícitas.

    system: instrucciones + cache CAG (la transcripción NO va aquí).
    user: la transcripción como DATOS, entre un par de etiquetas cuyo nombre
    solo se conoce en esta llamada. El usuario no puede cerrarlas porque no
    conoce el sufijo aleatorio."""
    apertura, cierre = transcription_delimiter()  # Delimitador aleatorio por llamada
    return [
        Message(role="system", content=build_system_prompt()),  # rol system + cache CAG
        Message(role="user", content=f"{apertura}\n{transcription}\n{cierre}"),  # rol user
    ]


def _preparar_provider(
    transcription: str,
    settings: Settings,
    provider_name: str,
) -> tuple[BaseProvider, list[Message]]:
    """Key, fábrica y mensajes para un proveedor concreto.

    Compartido por `chat()` y `chat_stream()`: el streaming abre el mismo
    provider con los mismos mensajes; lo único distinto es el transporte.
    La key se exige en el punto de uso con el nombre correcto, el modelo se
    resuelve POR proveedor y la fábrica aísla la creación."""
    try:
        # La key se valida en el punto de uso, no al importar.
        key = settings.require_api_key(provider_name)
    except _ConfigError as exc:
        # config.py y este servicio define cada uno su error de config:
        # el de abajo es capa de infraestructura, este es de dominio.
        # Traducción explícita para no atrapar config en el router.
        raise LLMConfigurationError(str(exc)) from exc

    try:
        # El modelo se resuelve POR proveedor: el override LLM_MODEL solo
        # aplica al activo; el fallback usa el default de su proveedor.
        provider = create_provider(
            name=provider_name,
            api_key=key,
            model=settings.resolve_model(provider_name),
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
    except UnknownProviderError as exc:
        raise LLMConfigurationError(str(exc)) from exc

    # Los mensajes (system + user) se construyen recién acá, como argumento
    # de la llamada: después de validar config y crear el provider, antes de
    # que la corrutina toque la red.
    return provider, _build_messages(transcription)


async def _chat_con_provider(
    transcription: str,
    settings: Settings,
    provider_name: str,
) -> tuple[LLMResponse, str, float]:
    """Prepara y llama a un proveedor concreto (primario o fallback).

    El mismo flujo del intento original, parametrizado por nombre. Devuelve
    la respuesta, el nombre real del proveedor y la latencia del chat (ms) —
    para que el resultado no mienta sobre quién respondió y la traza final
    pueda reportar cuánto tardó cada fase sin volver a medir.
    Un LLMProviderError propaga tal cual: generate_estimation decide según
    can_fallback si vale un salto."""
    provider, mensajes = _preparar_provider(transcription, settings, provider_name)
    inicio = time.perf_counter()
    response = await provider.chat(
        mensajes,
        max_tokens=settings.llm_max_tokens,
    )
    latencia_ms = (time.perf_counter() - inicio) * 1000

    # Dimensión 1 (qué se envió y qué se recibió). El resumen va en INFO; el
    # contenido literal (prompt completo + respuesta del modelo) en DEBUG,
    # porque es grande y sensible y no debe llenar el log de producción.
    emitir(
        __name__,
        "intento_proveedor",
        provider=provider_name,
        modelo=response.model,
        max_tokens=settings.llm_max_tokens,
        latencia_ms=round(latencia_ms, 1),
        ok=True,
    )
    emitir(
        __name__,
        "contenido_intento",
        nivel="debug",
        provider=provider_name,
        mensajes=[asdict(m) for m in mensajes],
        respuesta_modelo=response.content,
    )
    return response, provider_name, latencia_ms


async def _abrir_stream_con_provider(
    transcription: str,
    settings: Settings,
    provider_name: str,
) -> tuple[
    AsyncIterator[StreamChunk | StreamDone],
    StreamChunk | StreamDone,
    list[Message],
]:
    """Abre el stream de un proveedor y consume SU PRIMER EVENTO.

    El generador devuelto por `chat_stream()` no ejecuta nada hasta el primer
    `__anext__()`: ese adelanto es lo que permite saber ANTES de emitir un
    solo fragmento si el proveedor arrancó o falló. El primer evento ya
    consumido se devuelve para que el servicio lo procese como el resto.

    Emite `intento_proveedor` solo cuando el intento es EXITOSO y con la
    latencia hasta el primer evento (aproximación del TTFT); el fallo del
    intento previo queda registrado por `provider_error` del adaptador y por
    el evento final (camino=fallback, intentos_proveedor=2), igual que el
    criterio de la traza no-streaming. Un stream que no produce ni un evento
    es un fallo de dominio (vacío, nadie lo hereda): LLMServiceError."""
    provider, mensajes = _preparar_provider(transcription, settings, provider_name)
    inicio = time.perf_counter()
    stream = provider.chat_stream(mensajes, max_tokens=settings.llm_max_tokens)
    try:
        primero = await stream.__anext__()
    except StopAsyncIteration as exc:
        raise LLMServiceError(f"Stream del proveedor '{provider_name}' no produjo eventos") from exc
    # Un LLMProviderError propaga tal cual: stream_estimation decide según
    # can_fallback si vale un salto.
    latencia_ms = (time.perf_counter() - inicio) * 1000
    # Dimensión 1 (qué se envió y qué se recibió), resumen del intento con la
    # latencia al primer evento (TTFT). El contenido literal completo de la
    # respuesta se emite al cierre del stream, cuando la respuesta existe.
    emitir(
        __name__,
        "intento_proveedor",
        provider=provider_name,
        modelo=settings.resolve_model(provider_name),
        max_tokens=settings.llm_max_tokens,
        latencia_ms=round(latencia_ms, 1),
        streaming=True,
        ok=True,
    )
    return stream, primero, mensajes


def _log_estimacion_final(
    *,
    camino: str,
    cache_hit: bool,
    uso_fallback: bool,
    intentos_proveedor: int,
    proveedor: str,
    modelo: str,
    truncated: bool,
    uso: dict[str, int | None] | None,
    costo_usd: float | None,
    costo_nota: str | None,
    latencia_total_ms: float,
    latencia_cache_ms: float | None,
    latencia_llm_ms: float | None,
    ttft_ms: float | None = None,
) -> None:
    """Una sola fuente de verdad para el evento de cierre de la estimación.

    Las tres dimensiones de la traza LLM en un INFO:
    - 2 (cuánto costó): tokens de entrada/salida, modelo, costo económico.
    - 3 (qué camino siguió): cache_hit / primario / fallback, cuántos
      proveedores se llamaron y latencias por fase (cache, LLM, total).
    La dimensión 1 (contenido literal) vive en `contenido_intento` (DEBUG).
    `ttft_ms` (time-to-first-token) es la métrica propia del streaming: se
    mide desde el envío del request hasta el PRIMER fragmento de contenido
    del proveedor, así que incluye red + procesamiento del proveedor. En una
    respuesta no-streaming queda None."""
    emitir(
        __name__,
        "estimacion_completada",
        camino=camino,
        cache_hit=cache_hit,
        uso_fallback=uso_fallback,
        intentos_proveedor=intentos_proveedor,
        proveedor=proveedor,
        modelo=modelo,
        truncated=truncated,
        input_tokens=uso.get("input_tokens") if uso else None,
        output_tokens=uso.get("output_tokens") if uso else None,
        costo_usd=costo_usd,
        costo_nota=costo_nota,
        latencia_total_ms=round(latencia_total_ms, 1),
        latencia_cache_ms=(round(latencia_cache_ms, 1) if latencia_cache_ms is not None else None),
        latencia_llm_ms=(round(latencia_llm_ms, 1) if latencia_llm_ms is not None else None),
        ttft_ms=(round(ttft_ms, 1) if ttft_ms is not None else None),
    )


async def generate_estimation(
    transcription: str,
    settings: Settings,
    cache: CacheClient | None = None,
) -> EstimationResult:
    inicio_total = time.perf_counter()
    # Cache-first: si esta entrada ya tiene una respuesta, no se arma el
    # prompt ni se toca un proveedor. La clave es determinista sobre las
    # ENTRADAS (system prompt CAG + transcripción cruda + params), no sobre
    # el prompt final, porque el delimitador de la transcripción es aleatorio
    # por llamada y rompería el hit.
    cache_key = build_cache_key(transcription, settings) if cache is not None else None
    latencia_cache_ms: float | None = None
    if cache is not None and cache_key is not None:
        inicio_cache = time.perf_counter()
        cached = await get_cached_estimation(cache, cache_key)
        latencia_cache_ms = (time.perf_counter() - inicio_cache) * 1000
        if cached is not None:
            # El dict cacheado tiene exactamente los campos de EstimationResult.
            resultado = EstimationResult(**cached)
            _log_estimacion_final(
                camino="cache_hit",
                cache_hit=True,
                uso_fallback=False,
                intentos_proveedor=0,
                proveedor=resultado.provider,
                modelo=resultado.model,
                truncated=resultado.truncated,
                uso=resultado.usage,
                costo_usd=resultado.cost_usd,
                costo_nota=resultado.cost_note,
                latencia_total_ms=(time.perf_counter() - inicio_total) * 1000,
                latencia_cache_ms=latencia_cache_ms,
                latencia_llm_ms=None,
            )
            return resultado

    # Intento primario. Si falla por un error que el fallback puede salvar
    # (can_fallback: 401/429/5xx/red), se hace UN salto al secundario; un
    # error permanente (4xx) o la ausencia de fallback configurado falla
    # rápido: no se gasta una llamada que va a fallar igual.
    try:
        response, provider_real, latencia_llm_ms = await _chat_con_provider(
            transcription, settings, settings.llm_provider
        )
        used_fallback = False
    except LLMProviderError as exc:
        if not settings.llm_fallback or not exc.can_fallback:
            raise LLMServiceError(f"Fallo del proveedor '{settings.llm_provider}'") from exc
        try:
            response, provider_real, latencia_llm_ms = await _chat_con_provider(
                transcription, settings, settings.llm_fallback
            )
            used_fallback = True
        except LLMProviderError as exc2:
            # El error que ve el cliente es uno fijo; la traza de AMBOS
            # intentos queda encadenada en el log del servidor.
            raise LLMServiceError(
                f"Fallo del proveedor '{settings.llm_provider}' "
                f"y del fallback '{settings.llm_fallback}'"
            ) from exc2
    # NO se captura Exception: los bugs internos deben salir como 500,
    # no disfrazados de fallo del proveedor.

    # Costo del uso real
    cost_usd, cost_note = estimate_cost(response.model, response.usage)
    resultado = EstimationResult(
        estimation=response.content,
        truncated=response.truncated,
        model=response.model,
        provider=provider_real,
        used_fallback=used_fallback,
        usage=response.usage,
        cost_usd=cost_usd,
        cost_note=cost_note,
    )
    # Solo se cachea la respuesta del PRIMARIO. La clave se construyó con el
    # modelo del proveedor activo; si el fallback respondió, cachearla bajo
    # esa clave haría que un hit futuro mienta sobre quién generó la
    # respuesta y sobre su costo. El fallback es la excepción de emergencia:
    # no merece alimentar el cache.
    if cache is not None and cache_key is not None and not used_fallback:
        await set_cached_estimation(cache, cache_key, asdict(resultado), settings.cache_ttl)
    _log_estimacion_final(
        camino="fallback" if used_fallback else "primario",
        cache_hit=False,
        uso_fallback=used_fallback,
        intentos_proveedor=2 if used_fallback else 1,
        proveedor=provider_real,
        modelo=response.model,
        truncated=response.truncated,
        uso=response.usage,
        costo_usd=cost_usd,
        costo_nota=cost_note,
        latencia_total_ms=(time.perf_counter() - inicio_total) * 1000,
        latencia_cache_ms=latencia_cache_ms,
        latencia_llm_ms=latencia_llm_ms,
    )
    return resultado


async def stream_estimation(
    transcription: str,
    settings: Settings,
    cache: CacheClient | None = None,
) -> AsyncIterator[StreamMeta | StreamChunk | StreamFinal]:
    """Versión streaming de `generate_estimation`, con los mismos contratos.

    Eventos que cede (dominio; el router los traduce a SSE):
    - StreamMeta: primer evento de un miss real — quién responde (proveedor
      y camino primario/fallback). NO se emite en cache hit.
    - StreamChunk: cada fragmento de texto del proveedor, sin tocar.
    - StreamFinal: cierre con la EstimationResult completa.

    Diferencias con generate_estimation:
    - Pre-arranque: antes de emitir el meta, el servicio consume el primer
      evento del stream interno. Si el primario falla ahí (sin un solo
      fragmento emitido), salta al fallback; si ambos fallan pre-primer
      fragmento, lanza LLMServiceError antes de rendir nada — el router puede
      responder 502 HTTP real porque todavía no arrancó el cuerpo.
    - Fallo a MITAD (ya se emitieron deltas): NO se hace fallback y se lanza
      LLMServiceError. Texto que arrancó = comprometido: cambiar de modelo a
      mitad de una respuesta sería mentir sobre quién generó cada parte. El
      router lo traduce a un evento `error` SSE (el status HTTP ya es 200).
    - Cache: hit → un único StreamFinal sin meta (el cache no re-streams;
      eso falsificaría la latencia). Miss → al cierre exitoso del PRIMARIO se
      cachea; un fallback NUNCA alimenta el cache (misma regla).
    - Traza: intento_proveedor con la latencia al primer evento, y
      estimacion_completada con `ttft_ms` (time-to-first-token) además de las
      latencias por fase. `ttft_ms` se mide desde el envío del request (antes
      de abrir el stream), no desde el primer fragmento: el pre-arranque ya se
      comió ese evento, así que medir después daría un número falso."""
    inicio_total = time.perf_counter()
    # Cache-first, idéntico a generate_estimation: la clave es determinista
    # sobre las ENTRADAS y un hit no arma prompt ni toca proveedor.
    cache_key = build_cache_key(transcription, settings) if cache is not None else None
    latencia_cache_ms: float | None = None
    if cache is not None and cache_key is not None:
        inicio_cache = time.perf_counter()
        cached = await get_cached_estimation(cache, cache_key)
        latencia_cache_ms = (time.perf_counter() - inicio_cache) * 1000
        if cached is not None:
            resultado = EstimationResult(**cached)
            _log_estimacion_final(
                camino="cache_hit",
                cache_hit=True,
                uso_fallback=False,
                intentos_proveedor=0,
                proveedor=resultado.provider,
                modelo=resultado.model,
                truncated=resultado.truncated,
                uso=resultado.usage,
                costo_usd=resultado.cost_usd,
                costo_nota=resultado.cost_note,
                latencia_total_ms=(time.perf_counter() - inicio_total) * 1000,
                latencia_cache_ms=latencia_cache_ms,
                latencia_llm_ms=None,
            )
            yield StreamFinal(resultado=resultado)
            return

    # Stopwatch starts BEFORE the stream is opened. Opening the stream consumes
    # the first provider chunk, so a stopwatch started after this block would
    # only measure a no-op gap (the chunk is already in `primero`) and report a
    # fake `ttft_ms` of ~0.5ms while the provider actually spent ~1.4s. From
    # here `ttft_ms` means request-submission -> first content chunk (network
    # + provider processing included), and `latencia_llm_ms` means
    # request -> stream close. If the primary fails and the fallback answers,
    # BOTH include the failed primary attempt: that is the latency the user
    # actually experienced, and the per-attempt breakdown stays in
    # `intento_proveedor`. Do not "fix" this back to a post-open stopwatch.
    inicio_llm = time.perf_counter()

    # Pre-arranque del primario: abrir el stream consume el primer evento.
    # Solo si ese adelanto termina bien, el proveedor "arrancó" y el camino
    # queda decidido; si lanza con can_fallback, un UN salto al secundario.
    try:
        stream, primero, mensajes = await _abrir_stream_con_provider(
            transcription, settings, settings.llm_provider
        )
        proveedor = settings.llm_provider
        used_fallback = False
    except LLMProviderError as exc:
        if not settings.llm_fallback or not exc.can_fallback:
            raise LLMServiceError(f"Fallo del proveedor '{settings.llm_provider}'") from exc
        try:
            stream, primero, mensajes = await _abrir_stream_con_provider(
                transcription, settings, settings.llm_fallback
            )
            proveedor = settings.llm_fallback
            used_fallback = True
        except LLMProviderError as exc2:
            raise LLMServiceError(
                f"Fallo del proveedor '{settings.llm_provider}' "
                f"y del fallback '{settings.llm_fallback}'"
            ) from exc2
    # NO se captura Exception: los bugs internos deben salir como 500,
    # no disfrazados de fallo del proveedor.

    camino = "fallback" if used_fallback else "primario"
    rendido_meta = False
    deltas: list[str] = []
    ttft_ms: float | None = None
    done: StreamDone | None = None
    evento = primero
    # El generador del proveedor ya avanzó un evento (el `primero`); este
    # loop procesa ese evento y sigue pidiendo `__anext__()` hasta el cierre.
    while done is None:
        if isinstance(evento, StreamChunk):
            if not rendido_meta:
                yield StreamMeta(proveedor=proveedor, camino=camino)
                rendido_meta = True
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - inicio_llm) * 1000
            deltas.append(evento.delta)
            yield evento
        elif isinstance(evento, StreamDone):
            if not rendido_meta:
                yield StreamMeta(proveedor=proveedor, camino=camino)
                rendido_meta = True
            done = evento
            break
        try:
            evento = await stream.__anext__()
        except StopAsyncIteration:
            break
        except LLMProviderError as exc:
            # El proveedor murió DESPUÉS de empezar a escribir. Sin fallback:
            # el cliente ya recibió fragmentos de este proveedor.
            raise LLMServiceError(f"Stream del proveedor '{proveedor}' cortado a mitad") from exc
    if done is None:
        raise LLMServiceError(f"Stream del proveedor '{proveedor}' terminó sin cierre")

    respuesta = "".join(deltas)
    cost_usd, cost_note = estimate_cost(done.model, done.usage)
    resultado = EstimationResult(
        estimation=respuesta,
        truncated=done.truncated,
        model=done.model,
        provider=proveedor,
        used_fallback=used_fallback,
        usage=done.usage,
        cost_usd=cost_usd,
        cost_note=cost_note,
    )
    # Misma regla que generate_estimation: solo el PRIMARIO alimenta el cache.
    if cache is not None and cache_key is not None and not used_fallback:
        await set_cached_estimation(cache, cache_key, asdict(resultado), settings.cache_ttl)

    latencia_llm_ms = (time.perf_counter() - inicio_llm) * 1000
    # Dimensión 1 completa del streaming: la respuesta recién existe ahora.
    emitir(
        __name__,
        "contenido_intento",
        nivel="debug",
        provider=proveedor,
        streaming=True,
        mensajes=[asdict(m) for m in mensajes],
        respuesta_modelo=respuesta,
    )
    _log_estimacion_final(
        camino=camino,
        cache_hit=False,
        uso_fallback=used_fallback,
        intentos_proveedor=2 if used_fallback else 1,
        proveedor=proveedor,
        modelo=done.model,
        truncated=done.truncated,
        uso=done.usage,
        costo_usd=cost_usd,
        costo_nota=cost_note,
        latencia_total_ms=(time.perf_counter() - inicio_total) * 1000,
        latencia_cache_ms=latencia_cache_ms,
        latencia_llm_ms=latencia_llm_ms,
        ttft_ms=ttft_ms,
    )
    yield StreamFinal(resultado=resultado)
