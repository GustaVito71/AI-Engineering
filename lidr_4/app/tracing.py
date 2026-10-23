"""Emisión de eventos de trazabilidad (punto único de creación del logger).

`emit` usa `structlog.get_logger` FRESCO por evento, NO un logger
module-level congelado. Por qué: con `cache_logger_on_first_use` (que la
app configura), un logger module-level se congela en su PRIMER uso con el
backend vigente en ese instante. Si un test toca el módulo antes de que
corra el lifespan (`configure_logging`), el logger queda atado al backend
por defecto (print) y los tests de captura dependerían del orden de la
suite: un test que corre antes rompería la captura de los que corren
después.

Un proxy fresco por evento se congela con la configuración vigente en el
momento del evento: en producción el pipeline stdlib ya está configurado
antes del primer request (lifespan), y en tests `capture_logs` lo captura
sin importar el orden. Emitir un proxy es barato y la frecuencia es baja
(unos pocos eventos por estimación): no vale la pena el cache del bound
logger contra la fragilidad en tests.
"""

from __future__ import annotations

import structlog


def emit(
    logger_name: str,
    event: str,
    *,
    level: str = "info",
    **fields: object,
) -> None:
    """Registra un evento de trazabilidad con la configuración vigente."""
    getattr(structlog.get_logger(logger_name), level)(event, **fields)
