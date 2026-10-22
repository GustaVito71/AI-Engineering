"""Contrato de la revisión del crítico (Paso 8, patrón Actor-Crítico-Jefe).

QUÉ HACE ESTE ARCHIVO

Define la forma exacta de lo que el crítico le devuelve al jefe: una lista de
problemas («issues») y un veredicto. El crítico NO escribe una estimación nueva ni
texto libre: solo señala defectos de la estimación que revisó. Que el feedback sea
un schema (y no prosa) es lo que permite al jefe decidir por código, sin interpretar
texto, y lo que permite pasarle al actor, en la iteración siguiente, qué campo
concreto tiene que corregir.

CÓMO SE USA

`CriticFeedback` se le exige al modelo con Instructor, igual que `StructuredResult`
con la estimación: si la respuesta no cumple el schema (incluidas las reglas de
`CriticFeedback._verdict_matches_issues`), Instructor se la devuelve al modelo con el
error y reintenta.

El schema que ve el proveedor es plano a propósito (sin campos opcionales ni `null`):
`suggested_fix` es un texto que puede ir vacío.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Qué tipo de defecto es. Las siete primeras categorías vienen de la solución de
# referencia (session_05_live); `missing_component` e `inconsistency_with_metadata`
# las pide el enunciado del ACB (componentes que la transcripción menciona y la
# estimación no tiene; contradicciones con la memoria del proyecto).
CriticIssueCategory = Literal[
    "math_error",
    "hallucination",
    "scope_mismatch",
    "phase_imbalance",
    "missing_assumption",
    "unrealistic_estimate",
    "tier_mismatch",
    "missing_component",
    "inconsistency_with_metadata",
]

# Cuánto pesa: `critical` y `major` bloquean (justifican otra iteración); `minor` es
# solo advertencia.
CriticIssueSeverity = Literal["critical", "major", "minor"]

# Qué concluye el crítico: aceptar, pedir otra iteración o rechazar de plano.
CriticVerdict = Literal["accept", "needs_iteration", "reject"]

# Cuántos problemas, como máximo, puede listar el crítico en una revisión.
MAX_ISSUES = 12


class CriticIssue(BaseModel):
    """Un defecto concreto de la estimación revisada."""

    category: CriticIssueCategory = Field(description="Tipo de defecto.")
    severity: CriticIssueSeverity = Field(
        description="critical o major bloquean la aceptación; minor es solo advertencia."
    )
    field_path: str = Field(
        min_length=1,
        max_length=120,
        description=(
            "Campo de la estimación afectado, por ejemplo summary, totals.cost_eur "
            "o phases[2].hours."
        ),
    )
    description: str = Field(
        min_length=5, max_length=500, description="Qué está mal, en una o dos frases."
    )
    suggested_fix: str = Field(
        max_length=300,
        description="Pista de corrección en una frase. Cadena vacía si no hay una concreta.",
    )


class CriticFeedback(BaseModel):
    """Revisión completa: veredicto, problemas encontrados y confianza del crítico."""

    verdict: CriticVerdict = Field(description="accept, needs_iteration o reject.")
    issues: list[CriticIssue] = Field(
        max_length=MAX_ISSUES, description="Problemas encontrados. Vacía si no hay ninguno."
    )
    confidence_in_review: int = Field(
        ge=0, le=100, description="Confianza del crítico en su propia revisión, de 0 a 100."
    )

    @model_validator(mode="after")
    def _verdict_matches_issues(self) -> CriticFeedback:
        """El veredicto tiene que ser coherente con los problemas que lo justifican.

        - `needs_iteration` pide al menos un problema `critical` o `major`: con solo
          `minor` no vale la pena gastar otra llamada del actor.
        - `reject` pide al menos un problema que explique por qué.
        """
        if self.verdict == "needs_iteration" and not self.blocking_issues:
            raise ValueError(
                "El veredicto 'needs_iteration' requiere al menos un problema con severidad "
                "critical o major; si solo hay problemas minor, el veredicto es 'accept'."
            )
        if self.verdict == "reject" and not self.issues:
            raise ValueError("El veredicto 'reject' requiere al menos un problema que lo explique.")
        return self

    @property
    def blocking_issues(self) -> list[CriticIssue]:
        """Los problemas `critical` o `major`, los que impiden aceptar sin más."""
        return [issue for issue in self.issues if issue.severity in {"critical", "major"}]
