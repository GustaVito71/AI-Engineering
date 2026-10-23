"""Traza del patrón Actor-Crítico-Jefe (Paso 8) y respuesta del endpoint ACB.

QUÉ HACE ESTE ARCHIVO

Define lo que el endpoint `POST /sessions/{id}/estimate-acb` devuelve además de la
estimación: la traza de lo que pasó en cada vuelta (qué dijo el crítico, qué decidió
el jefe) para poder auditar el proceso. La traza nunca lleva el contenido de la
conversación ni de los adjuntos, solo veredictos, categorías y rutas de campo.

EL ESTADO DE LA REVISIÓN (`review_status`)

El crítico debería ser un modelo de otro proveedor que el actor, para que no se
revise a sí mismo. No siempre se puede, y la traza lo dice con `review_status`:

- `independent`: lo revisó un proveedor distinto del que escribió la estimación.
- `same_model`: el otro proveedor falló y revisó el mismo proveedor del actor.
- `unavailable`: ningún proveedor pudo revisar; la estimación se entrega sin revisar.
- `skipped`: no había nada que revisar (el actor rechazó la estimación por falta de
  información, ver `StructuredResult.out_of_scope`).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.critic import MAX_ISSUES
from app.schemas.structured_estimation import StructuredEstimationResponse, StructuredResult
from app.schemas.tier_views import TierView

# Cuando el jefe entrega un borrador con salvedades antepone una nota al resumen. La nota
# empieza con esta marca y termina con una línea en blanco; después sigue el resumen que
# escribió el modelo. Las usan el jefe (para escribirla) y `ReviewedResult` (para saltarla).
REVIEW_NOTE_MARK = "⚠"
REVIEW_NOTE_SEPARATOR = "\n\n"


def split_review_note(summary: str) -> tuple[str, str]:
    """Separa la nota de salvedades del resumen: devuelve `(nota, resumen_del_modelo)`.

    Si el resumen no empieza con la nota del jefe, la nota es una cadena vacía y el
    resumen se devuelve entero.
    """
    if summary.startswith(REVIEW_NOTE_MARK) and REVIEW_NOTE_SEPARATOR in summary:
        note, _, body = summary.partition(REVIEW_NOTE_SEPARATOR)
        return note + REVIEW_NOTE_SEPARATOR, body
    return "", summary


# Quién revisó (ver el docstring del módulo).
ReviewStatus = Literal["independent", "same_model", "unavailable", "skipped"]

# Lo que decide el jefe después de cada revisión: aceptar tal cual, volver a llamar al
# actor con el feedback, o cerrar entregando el último borrador con salvedades.
BossDecision = Literal["accept", "iterate", "synthesize"]

# Veredicto que muestra la traza: los del crítico más `unavailable` (no hubo crítico)
# y `skipped` (no hubo nada que revisar).
TraceVerdict = Literal["accept", "needs_iteration", "reject", "unavailable", "skipped"]


class ACBIteration(BaseModel):
    """Una vuelta del bucle: un borrador del actor, la revisión y la decisión del jefe."""

    iteration: int = Field(ge=0, description="Número de la vuelta, desde 0.")
    review_status: ReviewStatus = Field(description="Quién revisó el borrador.")
    critic_provider: str | None = Field(
        default=None, description="Proveedor del crítico; None si no hubo revisión."
    )
    critic_verdict: TraceVerdict = Field(description="Veredicto del crítico.")
    critic_confidence: int = Field(
        ge=0, le=100, description="Confianza del crítico en su revisión (0 si no hubo)."
    )
    decision_after: BossDecision = Field(description="Decisión del jefe tras esta revisión.")
    issue_summary: list[str] = Field(
        max_length=MAX_ISSUES,
        description="Problemas encontrados, uno por línea: [severidad] categoría @ campo.",
    )


class BossTrace(BaseModel):
    """Resumen de todo el bucle: las vueltas y cómo terminó."""

    final_decision: Literal["accept", "synthesize"] = Field(
        description="accept: se entregó el borrador tal cual. synthesize: con salvedades."
    )
    iterations_run: int = Field(ge=0, description="Cuántas vueltas se ejecutaron.")
    iterations: list[ACBIteration] = Field(
        default_factory=list, description="Detalle de cada vuelta, en orden."
    )


class ReviewedResult(StructuredResult):
    """Una estimación que pasó por el jefe: igual que `StructuredResult`, con una excepción.

    El schema original comprueba que las cifras con unidad que cita el resumen («250 EUR»)
    existan en la estimación, para frenar cifras inventadas por el modelo. Pero al cerrar
    con salvedades el jefe antepone al resumen las observaciones del crítico, que citan
    cifras que pueden no estar en la estimación (justamente porque las critica). Esa nota
    la escribe el código y no el modelo, así que aquí la comprobación se aplica solo al
    resumen del modelo, sin la nota. Sin esto, la respuesta fallaba con un 500.
    """

    @model_validator(mode="after")
    def _summary_figures_exist(self) -> ReviewedResult:  # mismo nombre: reemplaza al original
        """Revisa las cifras del resumen del modelo, ignorando la nota de salvedades."""
        _note, body = split_review_note(self.summary)
        # Se vuelve a validar con el schema original, pero con el resumen sin la nota.
        StructuredResult.model_validate({**self.model_dump(), "summary": body})
        return self


class ACBResponse(StructuredEstimationResponse):
    """Respuesta de `POST /sessions/{id}/estimate-acb`: la estimación y la traza del ACB."""

    estimation: ReviewedResult = Field(description="Estimación validada contra el schema.")

    @field_validator("estimation", mode="before")
    @classmethod
    def _accept_plain_result(cls, value: object) -> object:
        """Acepta también un `StructuredResult` común (lo que devuelve el jefe)."""
        if isinstance(value, StructuredResult) and not isinstance(value, ReviewedResult):
            return value.model_dump()
        return value

    acb: BossTrace = Field(description="Qué hizo el crítico y qué decidió el jefe en cada vuelta.")
    tier_view: TierView | None = Field(
        default=None,
        description=(
            "Vista de salida propia del tier resuelto para este turno "
            "(`DeveloperView`, `PmView` o `ExecutiveView`); `None` con tier "
            "`default` o estimación fuera de alcance."
        ),
    )
