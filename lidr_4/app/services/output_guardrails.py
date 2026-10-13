"""Guardrails de salida (WU9 B): lo que se comprueba de la estimación ya validada.

Tienen dos partes, con consecuencias distintas:

- **Rechazan** (se re-pregunta al modelo): las reglas viven en el schema, en
  `StructuredResult`, porque Instructor las ejecuta como validadores y le devuelve
  el mensaje al modelo. Son las que hacen que una estimación incoherente no
  llegue al servicio ni a la caché: las horas contra el tamaño del equipo
  (`_hours_fit_team`), los totales que se alejan de la suma de las fases y la
  coherencia del rechazo («Fuera de alcance:»). Una entrada de la caché que ya no
  las cumple tampoco sirve: el servicio la valida otra vez al leerla.
- **Avisan** (la estimación se devuelve igual): lo que acá se comprueba. Una
  confianza baja, pero por encima del umbral de rechazo, no hace inservible la
  estimación: el usuario tiene que saberlo, no recibir un error.

Las dos partes corren sobre la estimación completa, también cuando sale de la
caché: el aviso no se guarda con ella, se calcula siempre.
"""

from __future__ import annotations

from app.schemas.structured_estimation import StructuredResult
from app.tracing import emit

# Por debajo de esta confianza global (y por encima del rechazo) se avisa al usuario.
LOW_CONFIDENCE_WARN_PCT = 70


def check_output(estimation: StructuredResult, *, prompt_version: str, model: str) -> list[str]:
    """Avisos para el usuario sobre una estimación ya validada. Lista vacía si no hay."""
    if estimation.out_of_scope or estimation.confidence_pct >= LOW_CONFIDENCE_WARN_PCT:
        return []
    emit(
        __name__,
        "confianza_baja",
        level="warning",
        prompt_version=prompt_version,
        modelo=model,
        confianza_pct=estimation.confidence_pct,
    )
    aviso = (
        f"Confianza baja ({estimation.confidence_pct} %): la descripción deja dudas y las "
        "cifras pueden variar mucho. Conviene detallar el alcance y volver a estimar."
    )
    return [aviso]
