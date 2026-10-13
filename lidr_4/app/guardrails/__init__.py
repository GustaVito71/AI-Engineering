"""Guardrails del servicio: lo que se comprueba antes y después del modelo.

- `input.py`: sobre la descripción del usuario, antes de gastar nada (inyección de
  prompt, datos personales, moderación).
- `detectors.py`: funciones puras que comparten los dos (normalizar texto, datos
  personales, etiquetas del prompt).
- `output.py`: sobre la estimación que devuelve el modelo (fuera de alcance,
  totales que no cuadran, confianza baja). Las reglas que rechazan y re-preguntan
  viven en el schema (`app/schemas/structured_estimation.py`), porque ahí las
  ejecuta Instructor; `output.py` documenta cuál es cuál.

Este paquete reexporta lo que usan el servicio, el router y las dependencias.
"""

from app.guardrails.input import (
    INJECTION_MESSAGE,
    MODERATION_MESSAGE,
    InputGuardrails,
    InputGuardrailViolation,
    create_input_guardrails,
)
from app.guardrails.output import (
    DEFAULT_LOW_CONFIDENCE_WARN_PCT,
    OutputGuardrails,
    create_output_guardrails,
)

__all__ = [
    "DEFAULT_LOW_CONFIDENCE_WARN_PCT",
    "INJECTION_MESSAGE",
    "MODERATION_MESSAGE",
    "InputGuardrailViolation",
    "InputGuardrails",
    "OutputGuardrails",
    "create_input_guardrails",
    "create_output_guardrails",
]
