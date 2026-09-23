"""Router de estimaciones. Solo hace dos cosas: validar la entrada (en el
borde, antes de gastar un token) y traducir errores de dominio a HTTP.

No conoce el SDK de ningún proveedor: el servicio le devuelve `EstimationResult`
o le lanza sus propias excepciones. La traducción a códigos HTTP es la única
responsabilidad de esta capa."""
from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, model_validator

from ..config import Settings, get_settings
from ..services.llm_service import (
    EstimationResult,
    LLMConfigurationError,
    LLMServiceError,
    generate_estimation,
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1")


class EstimationRequest(BaseModel):
    transcription: str = ""

    @model_validator(mode="after")
    def _validar_limites(self) -> EstimationRequest:
        """Valida cantidad con los límites de Settings (configurables via .env).

        Se valida en el borde del sistema: una entrada inválida se rechaza con
        422 sin tocar el proveedor, es decir, sin gastar un token. Poner solo
        min_length es validar lo que enseñan los tutoriales; el `max` es el
        que protege el coste (el tamaño de la entrada es la factura)."""
        cfg = get_settings()
        longitud = len(self.transcription)
        if longitud < cfg.estimation_min_chars:
            raise ValueError(
                f"La transcripción debe tener al menos {cfg.estimation_min_chars} caracteres "
                f"(recibidos {longitud})."
            )
        if longitud > cfg.estimation_max_chars:
            raise ValueError(
                f"La transcripción no puede superar {cfg.estimation_max_chars} caracteres "
                f"(recibidos {longitud})."
            )
        return self


class EstimationResponse(BaseModel):
    estimation: str
    truncated: bool
    model: str
    provider: str
    usage: dict[str, int | None] | None
    cost_usd: float | None
    cost_note: str | None


@router.post("/estimate", response_model=EstimationResponse)
def create_estimation(
    body: EstimationRequest,
    settings: Settings = Depends(get_settings),
) -> EstimationResult:
    # Handler síncrono a propósito: FastAPI lo ejecuta en un threadpool.
    # Si fuera `async` con un cliente síncrono dentro, cada llamada lenta
    # congelaría el event loop y /health dejaría de responder.
    try:
        return generate_estimation(body.transcription, settings)
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