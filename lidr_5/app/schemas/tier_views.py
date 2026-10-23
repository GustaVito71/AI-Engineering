"""Vistas de salida por tier: la misma estimación, con forma distinta por audiencia.

QUÉ HACE ESTE MÓDULO

Hasta aquí el `tier` (Paso 9) solo cambiaba el TONO de la respuesta vía el bloque
`<audience>` del prompt: el modelo escribía distinto, pero siempre devolvía los mismos
campos (`StructuredResult`), fueran para un ejecutivo o para un developer.

Estas clases son la adaptación ESTRUCTURAL: cada una tiene solo los campos que le
sirven a esa audiencia, no un superconjunto con campos vacíos u opcionales. No las
genera el LLM ni las valida Instructor: se construyen en `app/services/tier_projection.py`
a partir de un `StructuredResult` que el Jefe/Crítico (ACB) ya validó. Por eso el ACB,
la caché y el resumen/anclas del Paso 9 no cambian: siguen trabajando con la forma
única de siempre; la vista por tier es solo cómo se presenta el resultado al final.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.structured_estimation import Phase, StructuredEstimationResponse, TeamMember

# Niveles de confianza en castellano para la vista ejecutiva: un ejecutivo lee "alta"
# más rápido que un porcentaje, y evita la falsa precisión de un número exacto.
ConfidenceLevel = Literal["baja", "media", "alta"]


class DeveloperView(BaseModel):
    """Lo que un perfil técnico necesita: fases completas (supuestos y riesgos
    técnicos incluidos) y equipo, sin resumen ejecutivo ni cifras de negocio."""

    phases: list[Phase] = Field(description="Fases con su detalle técnico completo.")
    team: list[TeamMember] = Field(description="Composición del equipo.")
    total_hours: int = Field(ge=0, description="Horas totales de la estimación.")


class Milestone(BaseModel):
    """Un hito de calendario para la vista de PM: cuándo y con qué confianza."""

    phase_name: str = Field(description="Nombre de la fase que da origen al hito.")
    duration_weeks: float = Field(ge=0, description="Duración de la fase, en semanas.")
    confidence_pct: int = Field(ge=0, le=100, description="Confianza en esa fase.")


class PmView(BaseModel):
    """Lo que un jefe de proyecto necesita: hitos, equipo, duración y bloqueos,
    sin el detalle técnico de cada fase ni cifras de coste."""

    milestones: list[Milestone] = Field(description="Un hito por fase, con su duración.")
    team: list[TeamMember] = Field(description="Composición del equipo.")
    duration_weeks: float = Field(ge=0, description="Duración total del proyecto.")
    blockers: list[str] = Field(
        description="Riesgos de todas las fases, aplanados en una sola lista."
    )


class ExecutiveView(BaseModel):
    """Lo que un ejecutivo necesita: tres cifras, el nivel de confianza, los riesgos
    de negocio principales y una recomendación — nada de fases ni de equipo."""

    headline_cost_eur: float = Field(ge=0, description="Coste total, en EUR.")
    headline_duration_weeks: float = Field(ge=0, description="Duración total, en semanas.")
    confidence_level: ConfidenceLevel = Field(description="Confianza global, en tres niveles.")
    top_risks: list[str] = Field(
        max_length=3, description="Hasta 3 riesgos principales, sin su mitigación."
    )
    recommendation: str = Field(
        min_length=1, description="Recomendación de negocio en una frase (go / no-go)."
    )


# Lo que devuelve `project_to_tier` para cada tier; `default` no tiene vista propia
# (sigue mostrándose solo el `StructuredResult` de siempre).
TierView = DeveloperView | PmView | ExecutiveView


class TieredEstimationResponse(StructuredEstimationResponse):
    """Respuesta de `POST /sessions/{id}/estimate`: la estimación de siempre, más su
    vista propia del tier resuelto para este turno."""

    tier_view: TierView | None = Field(
        default=None,
        description=(
            "Vista de salida propia del tier resuelto para este turno "
            "(`DeveloperView`, `PmView` o `ExecutiveView`); `None` con tier "
            "`default` o estimación fuera de alcance."
        ),
    )
