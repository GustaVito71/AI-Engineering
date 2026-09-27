"""Schemas del contrato HTTP de estimaciones.

La validación de la entrada vive acá (en el borde del sistema, antes de
gastar un token) y usa los límites de Settings: configurables vía .env, no
números mágicos. Swagger documenta esos mismos límites porque create_app()
deriva el OpenAPI de la instancia activa (app.main._completar_schema_openapi):
Pydantic no puede volcar a un JSON Schema estático un model_validator que lee
configuración."""

from __future__ import annotations

from pydantic import BaseModel, model_validator

from ..config import get_settings


class EstimationRequest(BaseModel):
    transcription: str = ""

    @model_validator(mode="after")
    def _validar_limites(self) -> EstimationRequest:
        """Valida la cantidad con los límites de Settings (configurables vía .env).

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
    used_fallback: bool
    usage: dict[str, int | None] | None
    cost_usd: float | None
    cost_note: str | None
