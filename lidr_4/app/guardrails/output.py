"""Guardrails de salida (WU9 B): lo que se comprueba de la estimación del modelo.

Viven en `app/guardrails/`, con los de entrada (`input.py`). Tienen dos partes,
con consecuencias distintas:

- **Rechazan** (se re-pregunta al modelo). Las reglas viven en el schema
  (`app/schemas/structured_estimation.py`), no acá, porque Instructor las ejecuta
  como validadores y le devuelve el mensaje al modelo; una comprobación posterior
  rechazaría sin re-preguntar. Son las que hacen que una estimación incoherente
  no llegue al servicio ni a la caché:

  | Regla (validador) | Qué rechaza |
  |---|---|
  | `_hours_fit_team` | horas de una fase o del total por encima de lo que el equipo trabaja (40 h por persona y semana) |
  | `_totals_close_to_phases` | horas o coste totales a más de 5 % de la suma de las fases |
  | `_confidence_matches_refusal`, `_shape_matches_outcome` | un rechazo sin «Fuera de alcance:», o una estimación con cifras en cero |
  | `_rendered_matches_format` | un `rendered` (v4) que no sigue el formato o cuyas cifras no son las de los campos |

  Una entrada de la caché que ya no las cumple tampoco sirve: el servicio la
  valida otra vez al leerla.
- **Avisan** (la estimación se devuelve igual): lo que `OutputGuardrails` hace acá.
  Son cosas que el usuario tiene que saber pero que no hacen inservible la
  estimación: una confianza baja pero por encima del umbral de rechazo, y los
  totales que no cuadran dentro de la tolerancia. También deja el evento de la
  estimación fuera de alcance, que no es un fallo.

`OutputGuardrails.check` corre sobre la estimación completa, también cuando sale de la
caché: los avisos no se guardan con ella, se calculan siempre. Ningún evento lleva
el texto de la estimación.
"""

from __future__ import annotations

from app.config import DEFAULT_LOW_CONFIDENCE_WARN_PCT, Settings
from app.schemas.structured_estimation import StructuredResult
from app.tracing import emit

__all__ = ["DEFAULT_LOW_CONFIDENCE_WARN_PCT", "OutputGuardrails", "create_output_guardrails"]


class OutputGuardrails:
    """Los avisos y eventos sobre la estimación validada. No guarda estado entre llamadas."""

    def __init__(self, *, low_confidence_warn_pct: int = DEFAULT_LOW_CONFIDENCE_WARN_PCT) -> None:
        """`low_confidence_warn_pct`: por debajo de esta confianza global se avisa."""
        self._low_confidence_warn_pct = low_confidence_warn_pct

    def check(self, estimation: StructuredResult, *, prompt_version: str, model: str) -> list[str]:
        """Avisos para el usuario sobre una estimación ya validada. Lista vacía si no hay.

        Deja los eventos de negocio: `estimacion_fuera_de_alcance` (el modelo dijo que
        la descripción no alcanza), `totales_no_cuadran` y `confianza_baja`. Los avisos
        salen en ese orden: primero los totales, después la confianza.
        """
        if estimation.out_of_scope:
            # No es un fallo: queda en el log para medir cuántas descripciones llegan
            # sin información suficiente. Una estimación rechazada no lleva más avisos.
            emit(
                __name__,
                "estimacion_fuera_de_alcance",
                prompt_version=prompt_version,
                modelo=model,
                confianza_pct=estimation.confidence_pct,
            )
            return []

        warnings = estimation.total_discrepancies()
        if warnings:
            emit(
                __name__,
                "totales_no_cuadran",
                level="warning",
                prompt_version=prompt_version,
                modelo=model,
                detalle=warnings,
            )
        if estimation.confidence_pct < self._low_confidence_warn_pct:
            emit(
                __name__,
                "confianza_baja",
                level="warning",
                prompt_version=prompt_version,
                modelo=model,
                confianza_pct=estimation.confidence_pct,
            )
            warnings.append(
                f"Confianza baja ({estimation.confidence_pct} %): la descripción deja dudas "
                "y las cifras pueden variar mucho. Conviene detallar el alcance y volver a "
                "estimar."
            )
        return warnings


def create_output_guardrails(settings: Settings) -> OutputGuardrails | None:
    """Los guardrails de salida que pide la configuración, o None si están apagados.

    Apagarlos deja un evento, para que no sea un estado invisible. No apaga las reglas
    del schema que rechazan y re-preguntan.
    """
    if not settings.output_guardrails_enabled:
        emit(
            __name__,
            "guardrails_desactivados",
            level="warning",
            motivo="OUTPUT_GUARDRAILS_ENABLED=false",
        )
        return None
    return OutputGuardrails(low_confidence_warn_pct=settings.low_confidence_warn_pct)
