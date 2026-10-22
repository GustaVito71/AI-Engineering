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
- **Descartan sin re-preguntar** (`OutputGuardrails.inspect_*`, 502): datos personales
  en la salida (email, teléfono, IBAN, tarjeta, documento) y el prompt de sistema
  volcado (etiquetas de las plantillas o frases fijas de las instrucciones). Volver
  a preguntar con el mismo prompt no cambia el riesgo y cada intento se paga; se
  descarta, se deja el evento `salida_rechazada` y no se guarda en la caché. Son
  heurísticas, no una garantía: un dato con otro formato, o instrucciones
  parafraseadas, pasan.
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

from collections.abc import Iterator

from app.config import DEFAULT_LOW_CONFIDENCE_WARN_PCT, Settings
from app.guardrails.detectors import find_prompt_leak, mask_personal_data
from app.schemas.structured_estimation import StructuredResult
from app.tracing import emit

__all__ = [
    "DEFAULT_LOW_CONFIDENCE_WARN_PCT",
    "OUTPUT_LEAK_MESSAGE",
    "OUTPUT_PII_MESSAGE",
    "OutputGuardrailViolation",
    "OutputGuardrails",
    "create_output_guardrails",
]

OUTPUT_PII_MESSAGE = (
    "La respuesta del modelo incluía datos personales y se descartó. "
    "Vuelve a intentarlo; si se repite, revisa la descripción."
)
OUTPUT_LEAK_MESSAGE = (
    "La respuesta del modelo no es válida y se descartó. Vuelve a intentarlo en unos minutos."
)


class OutputGuardrailViolation(Exception):
    """Una regla de salida descartó la respuesta del modelo (el router responde 502).

    `rule` va al log; `message` es el texto para el cliente, fijo por regla: no repite
    nada de lo que escribió el modelo.
    """

    def __init__(self, rule: str, message: str) -> None:
        super().__init__(f"{rule}: {message}")
        self.rule = rule
        self.message = message


def _strings(value: object) -> Iterator[str]:
    """Todos los textos de un valor anidado (dicts, listas), sin las claves."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


class OutputGuardrails:
    """Los avisos y eventos sobre la estimación validada. No guarda estado entre llamadas."""

    def __init__(self, *, low_confidence_warn_pct: int = DEFAULT_LOW_CONFIDENCE_WARN_PCT) -> None:
        """`low_confidence_warn_pct`: por debajo de esta confianza global se avisa."""
        self._low_confidence_warn_pct = low_confidence_warn_pct

    def inspect_text(self, text: str, *, prompt_version: str, model: str) -> None:
        """Descarta una salida de texto con datos personales o con el prompt volcado."""
        self._inspect(text, prompt_version=prompt_version, model=model)

    def inspect_estimation(
        self, estimation: StructuredResult, *, prompt_version: str, model: str
    ) -> None:
        """Igual que `inspect_text`, sobre todos los textos de la estimación (y `rendered`)."""
        text = "\n".join(_strings(estimation.model_dump()))
        self._inspect(text, prompt_version=prompt_version, model=model)

    def _inspect(self, text: str, *, prompt_version: str, model: str) -> None:
        """Lanza `OutputGuardrailViolation`; el evento no lleva el texto ni el dato."""
        if (leak := find_prompt_leak(text)) is not None:
            self._reject(
                "prompt_filtrado", OUTPUT_LEAK_MESSAGE, prompt_version, model, detalle=leak
            )
        if kinds := mask_personal_data(text)[1]:
            self._reject(
                "datos_personales",
                OUTPUT_PII_MESSAGE,
                prompt_version,
                model,
                tipos=list(kinds),
            )

    @staticmethod
    def _reject(rule: str, message: str, prompt_version: str, model: str, **fields: object) -> None:
        """Registra en el log por qué se descartó la respuesta del modelo y lanza
        `OutputGuardrailViolation`.

        La regla que saltó queda en el log; al cliente solo le llega un mensaje fijo.
        """
        emit(
            __name__,
            "salida_rechazada",
            level="warning",
            regla=rule,
            prompt_version=prompt_version,
            modelo=model,
            **fields,
        )
        raise OutputGuardrailViolation(rule, message)

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
