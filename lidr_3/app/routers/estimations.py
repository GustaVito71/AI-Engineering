"""Router de estimaciones. Solo hace dos cosas: validar la entrada (en el
borde, antes de gastar un token) y traducir errores de dominio a HTTP.

No conoce el SDK de ningún proveedor: el servicio le devuelve `EstimationResult`
o le lanza sus propias excepciones. La traducción a códigos HTTP es la única
responsabilidad de esta capa, y los modelos del contrato viven en
`app.schemas.estimation` (la validación con Settings incluida).

Endpoint SSE (`POST /estimate/stream`): el handler hace un PRE-ARranque del
generador del servicio — consume su primer evento antes de construir la
respuesta. Así un fallo de configuración (503) o de proveedores (502) que
ocurre ANTES del primer fragmento se responde con status HTTP real, no como
un body roto de un 200. Después traduce los eventos de dominio a líneas SSE
(meta/delta/estimation); un fallo a MITAD de stream (el status ya es 200 y no
se puede cambiar) baja como un evento `error` SSE honesto."""

from __future__ import annotations

import json
from dataclasses import asdict

import structlog
from fastapi import APIRouter, Depends, HTTPException
from starlette.responses import StreamingResponse

from ..cache import get_cache_client
from ..config import Settings, get_settings
from ..schemas.estimation import EstimationRequest, EstimationResponse
from ..services.llm_service import (
    CacheClient,
    EstimationResult,
    LLMConfigurationError,
    LLMServiceError,
    StreamChunk,
    StreamFinal,
    StreamMeta,
    generate_estimation,
    stream_estimation,
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1")


@router.post("/estimate", response_model=EstimationResponse)
async def create_estimation(
    body: EstimationRequest,
    settings: Settings = Depends(get_settings),
    cache: CacheClient | None = Depends(get_cache_client),
) -> EstimationResult:
    # Handler async porque la cadena completa es async (cliente async del SDK):
    # el event loop queda libre durante la llamada al proveedor y las requests
    # concurrentes (incluido /health) comparten el mismo loop.
    try:
        return await generate_estimation(body.transcription, settings, cache)
    except LLMConfigurationError as exc:
        # 503: el problema es local y el mensaje nombra la variable que falta.
        # El operador lo lee en /estimate sin tocar los logs de arranque.
        logger.error(f"Estimación rechazada por configuración: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMServiceError as exc:
        # 502: falló el proveedor. La traza completa se queda en el log del
        # servidor; el cliente solo recibe un mensaje fijo. Un `str(exc)` de
        # un error del SDK arrastra fragmentos de la API key y URLs internas.
        logger.exception("Fallo al generar la estimación")
        raise HTTPException(
            status_code=502,
            detail="No se pudo generar la estimación. Inténtalo de nuevo más tarde.",
        ) from exc


def _evento_sse(tipo: str, datos: dict) -> str:
    """Una línea `event:` + `data:` de Server-Sent Events (formato estándar)."""
    return f"event: {tipo}\ndata: {json.dumps(datos, ensure_ascii=False)}\n\n"


def _render_evento(evento: StreamMeta | StreamChunk | StreamFinal) -> str:
    """Traducción de los eventos de dominio a nombres y JSON de SSE."""
    if isinstance(evento, StreamMeta):
        return _evento_sse("meta", {"proveedor": evento.proveedor, "camino": evento.camino})
    if isinstance(evento, StreamChunk):
        return _evento_sse("delta", {"texto": evento.delta})
    if isinstance(evento, StreamFinal):
        return _evento_sse("estimation", asdict(evento.resultado))
    # La firma cubre los tres tipos del dominio; cualquier otro es un bug.
    raise TypeError(f"Evento de dominio desconocido: {type(evento).__name__}")


@router.post("/estimate/stream")
async def create_estimation_stream(
    body: EstimationRequest,
    settings: Settings = Depends(get_settings),
    cache: CacheClient | None = Depends(get_cache_client),
) -> StreamingResponse:
    agen = stream_estimation(body.transcription, settings, cache)
    # PRE-ARRANQUE: el primer evento se consume acá. La cache lee, el provider
    # abre y el camino primario/fallback se decide ANTES de escribir el
    # status 200; los errores de esa fase bajan como HTTP real (503/502).
    try:
        primero = await agen.__anext__()
    except LLMConfigurationError as exc:
        logger.error(f"Estimación en stream rechazada por configuración: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMServiceError as exc:
        # Ambos proveedores fallaron antes del primer fragmento, o el stream
        # del proveedor arrancado no produjo eventos.
        logger.exception("Fallo al abrir el stream de estimación")
        raise HTTPException(
            status_code=502,
            detail="No se pudo generar la estimación. Inténtalo de nuevo más tarde.",
        ) from exc

    async def sse():
        # El pre-arranque ya avanzó el generador un evento; el async for
        # continúa desde el segundo. En un cache hit `primero` ES el final y
        # el loop no rinde nada más (contrato: cache = solo `estimation`).
        try:
            yield _render_evento(primero)
            async for evento in agen:
                yield _render_evento(evento)
        except LLMServiceError:
            # El status 200 ya salió con los fragmentos previos: no se puede
            # cambiar. El error baja como evento SSE y la conexión se cierra.
            logger.exception("Stream de estimación cortado a mitad")
            yield _evento_sse(
                "error",
                {"detail": "No se pudo completar la estimación. Inténtalo de nuevo."},
            )

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
