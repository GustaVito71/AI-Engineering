"""Proyecta una estimación ya validada a la vista de salida de cada tier (Paso 9+).

QUÉ HACE ESTE MÓDULO

El Jefe/Crítico (ACB) valida la estimación UNA sola vez, siempre contra el mismo
`StructuredResult` (ver `app/schemas/structured_estimation.py`): esto no cambia. Lo que
hace este módulo es transformar, recién al final y solo para la respuesta HTTP, ese
mismo resultado ya validado en la vista propia de cada tier (`app/schemas/tier_views.py`).

Por qué así y no con un schema/prompt por tier (como hace `session_05_live`): acá el
`tier` se infiere por turno dentro de la misma sesión (puede cambiar de un mensaje a
otro), así que fijar el schema de GENERACIÓN por tier complicaría el ACB, la caché y el
resumen/anclas del Paso 9, que hoy asumen una forma única por sesión. Proyectando solo
en la salida, nada de eso cambia: se gana la adaptación estructural real (campos propios
de cada audiencia, no un superconjunto con huecos) sin tocar el resto del pipeline.

Con `tier="default"` (o uno sin proyección registrada) no hay vista: `project_to_tier`
devuelve `None` y la respuesta sigue mostrando solo el `StructuredResult` de siempre.
"""

from __future__ import annotations

from collections.abc import Callable

from app.schemas.structured_estimation import StructuredResult
from app.schemas.tier_views import (
    ConfidenceLevel,
    DeveloperView,
    ExecutiveView,
    Milestone,
    PmView,
    TierView,
)

# Umbrales de `confidence_pct` (0-100) que separan los tres niveles de la vista ejecutiva.
_CONFIDENCE_MEDIUM_MIN = 50
_CONFIDENCE_HIGH_MIN = 75

# Cuántos riesgos como máximo entran en `ExecutiveView.top_risks` (coincide con su límite
# de schema): a un ejecutivo no le sirve una lista larga, le sirve la que importa.
_MAX_EXECUTIVE_RISKS = 3

# Confianza global a partir de la cual la recomendación es "go" en la vista ejecutiva;
# coincide con el umbral "alta" para que la recomendación y el nivel no se contradigan.
_GO_CONFIDENCE_MIN = _CONFIDENCE_HIGH_MIN


def _confidence_level(confidence_pct: int) -> ConfidenceLevel:
    """Traduce un porcentaje a los tres niveles que lee un ejecutivo."""
    if confidence_pct < _CONFIDENCE_MEDIUM_MIN:
        return "baja"
    if confidence_pct < _CONFIDENCE_HIGH_MIN:
        return "media"
    return "alta"


def _project_developer(result: StructuredResult) -> DeveloperView:
    """Fases completas (supuestos y riesgos técnicos incluidos) y equipo, sin plata."""
    return DeveloperView(
        phases=result.phases,
        team=result.team,
        total_hours=result.totals.hours,
    )


def _project_pm(result: StructuredResult) -> PmView:
    """Un hito por fase y los riesgos de todas las fases, aplanados como bloqueos."""
    blockers = [risk.risk for phase in result.phases for risk in phase.risks]
    return PmView(
        milestones=[
            Milestone(
                phase_name=phase.name,
                duration_weeks=phase.duration_weeks,
                confidence_pct=phase.confidence_pct,
            )
            for phase in result.phases
        ],
        team=result.team,
        duration_weeks=result.totals.duration_weeks,
        blockers=blockers,
    )


def _project_executive(result: StructuredResult) -> ExecutiveView:
    """Tres cifras, confianza en tres niveles, los riesgos principales y una recomendación."""
    all_risks = [risk.risk for phase in result.phases for risk in phase.risks]
    recommendation = (
        "Go: el alcance y el presupuesto son razonables para la confianza declarada."
        if result.confidence_pct >= _GO_CONFIDENCE_MIN
        else "No-go todavía: conviene acotar alcance o revisar supuestos antes de "
        "comprometer presupuesto."
    )
    return ExecutiveView(
        headline_cost_eur=result.totals.cost_eur,
        headline_duration_weeks=result.totals.duration_weeks,
        confidence_level=_confidence_level(result.confidence_pct),
        top_risks=all_risks[:_MAX_EXECUTIVE_RISKS],
        recommendation=recommendation,
    )


# Un proyector por tier; `default` no tiene entrada a propósito (sin vista propia).
_PROJECTORS: dict[str, Callable[[StructuredResult], TierView]] = {
    "developer": _project_developer,
    "pm": _project_pm,
    "executive": _project_executive,
}


def project_to_tier(result: StructuredResult, tier: str) -> TierView | None:
    """La vista de `result` para `tier`, o `None` si el tier no tiene vista propia.

    `result` ya fue validado por el Jefe/Crítico: esta función no valida de nuevo, solo
    reorganiza campos que ya existen. Una estimación fuera de alcance
    (`result.out_of_scope`) no tiene fases con datos, así que tampoco tiene vista: el
    cliente sigue mostrando el resumen de `StructuredResult` (el motivo del rechazo).
    """
    if result.out_of_scope:
        return None
    projector = _PROJECTORS.get(tier)
    if projector is None:
        return None
    return projector(result)
