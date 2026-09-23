"""context: las fuentes de contexto para el prompt (el cache de CAG)."""

from .examples import (
    ESTIMATION_EXAMPLES,
    EstimationExample,
    build_system_prompt,
    get_examples_as_text,
    transcription_delimiter,
)

__all__ = [
    "ESTIMATION_EXAMPLES",
    "EstimationExample",
    "build_system_prompt",
    "get_examples_as_text",
    "transcription_delimiter",
]