"""Router de estimaciones. Solo hace dos cosas: validar la entrada (en el
borde, antes de gastar un token) y traducir errores de dominio a HTTP.

No conoce el SDK de ningún proveedor: el servicio le devuelve `EstimationResult`
o le lanza sus propias excepciones. La traducción a códigos HTTP es la única
responsabilidad de esta capa, y los modelos del contrato viven en
`app.schemas.estimation` (la validación con Settings incluida)."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException

from ..config import Settings, get_settings
from ..schemas.estimation import EstimationRequest, EstimationResponse
from ..services.llm_service import (
    EstimationResult,
    LLMConfigurationError,
    LLMServiceError,
    generate_estimation,
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1")


@router.post("/estimate", response_model=EstimationResponse)
async def create_estimation(
    body: EstimationRequest,
    settings: Settings = Depends(get_settings),
) -> EstimationResult:
    # Handler async porque la cadena completa es async (cliente async del SDK):
    # el event loop queda libre durante la llamada al proveedor y las requests
    # concurrentes (incluido /health) comparten el mismo loop.
    try:
        return await generate_estimation(body.transcription, settings)
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
