"""Tests de `project_to_tier` (adaptación estructural por tier, sobre el Paso 9).

QUÉ SE COMPRUEBA

- Cada tier con vista propia (`executive`, `pm`, `developer`) devuelve el tipo correcto,
  con los campos que le corresponden a esa audiencia.
- `default` y un tier desconocido no tienen vista: `None`.
- Una estimación fuera de alcance (`StructuredResult.out_of_scope`) tampoco tiene vista,
  sea cual sea el tier: no hay fases con datos para proyectar.
- Los valores de cada vista se calculan a partir del `StructuredResult` de origen (no se
  inventan): horas totales, riesgos aplanados, nivel de confianza y recomendación.
"""

from __future__ import annotations

import pytest

from app.schemas.structured_estimation import (
    OUT_OF_SCOPE_PREFIX,
    UNESTIMATED_PHASE,
    Phase,
    Risk,
    StructuredResult,
    TeamMember,
    Totals,
)
from app.schemas.tier_views import DeveloperView, ExecutiveView, PmView
from app.services.tier_projection import project_to_tier


def _estimacion(*, confidence_pct: int = 76) -> StructuredResult:
    """Una estimación de ejemplo con 2 fases, cada una con un riesgo."""
    return StructuredResult(
        phases=[
            Phase(
                name="Descubrimiento",
                summary="Relevamiento de requisitos y alcance con el cliente.",
                duration_weeks=2,
                hours=80,
                cost_eur=4800,
                confidence_pct=85,
                assumptions=["El cliente provee acceso a usuarios clave"],
                risks=[Risk(risk="Alcance ambiguo", mitigation="Taller de alcance en semana 1")],
            ),
            Phase(
                name="Implementación",
                summary="Desarrollo del backend y la integración con el proveedor de pagos.",
                duration_weeks=8,
                hours=640,
                cost_eur=38400,
                confidence_pct=70,
                assumptions=[],
                risks=[Risk(risk="Cambios de API del proveedor", mitigation="Adaptador propio")],
            ),
        ],
        team=[TeamMember(role="Desarrollador backend", headcount=2)],
        totals=Totals(hours=720, cost_eur=43200, duration_weeks=10),
        summary="Proyecto con integración de pagos, 10 semanas y equipo de 2 personas.",
        confidence_pct=confidence_pct,
    )


def _fuera_de_alcance() -> StructuredResult:
    """Una estimación rechazada: sin fases con datos, team vacío, todo en 0."""
    return StructuredResult(
        phases=[
            Phase(
                name=UNESTIMATED_PHASE,
                summary="No hay información suficiente para estimar el alcance pedido.",
                duration_weeks=0,
                hours=0,
                cost_eur=0,
                confidence_pct=0,
                assumptions=[],
                risks=[],
            )
        ],
        team=[],
        totals=Totals(hours=0, cost_eur=0, duration_weeks=0),
        summary=f"{OUT_OF_SCOPE_PREFIX} falta saber el tamaño del equipo disponible.",
        confidence_pct=10,
    )


def test_developer_incluye_fases_completas_y_horas_totales() -> None:
    vista = project_to_tier(_estimacion(), "developer")
    assert isinstance(vista, DeveloperView)
    assert [f.name for f in vista.phases] == ["Descubrimiento", "Implementación"]
    assert vista.phases[0].risks[0].risk == "Alcance ambiguo"  # el detalle técnico se conserva
    assert vista.total_hours == 720
    assert vista.team == [TeamMember(role="Desarrollador backend", headcount=2)]


def test_pm_tiene_un_hito_por_fase_y_los_riesgos_aplanados() -> None:
    vista = project_to_tier(_estimacion(), "pm")
    assert isinstance(vista, PmView)
    assert [m.phase_name for m in vista.milestones] == ["Descubrimiento", "Implementación"]
    assert vista.milestones[1].duration_weeks == 8
    assert vista.duration_weeks == 10
    assert vista.blockers == ["Alcance ambiguo", "Cambios de API del proveedor"]


@pytest.mark.parametrize(
    "confidence_pct, nivel_esperado",
    [(40, "baja"), (60, "media"), (80, "alta")],
)
def test_executive_traduce_la_confianza_a_tres_niveles(
    confidence_pct: int, nivel_esperado: str
) -> None:
    vista = project_to_tier(_estimacion(confidence_pct=confidence_pct), "executive")
    assert isinstance(vista, ExecutiveView)
    assert vista.confidence_level == nivel_esperado


def test_executive_recorta_a_3_riesgos_y_usa_las_cifras_de_totals() -> None:
    vista = project_to_tier(_estimacion(), "executive")
    assert isinstance(vista, ExecutiveView)
    assert vista.headline_cost_eur == 43200
    assert vista.headline_duration_weeks == 10
    assert len(vista.top_risks) <= 3
    assert vista.top_risks == ["Alcance ambiguo", "Cambios de API del proveedor"]


def test_executive_recomienda_go_con_confianza_alta_y_no_go_con_confianza_baja() -> None:
    alta = project_to_tier(_estimacion(confidence_pct=80), "executive")
    baja = project_to_tier(_estimacion(confidence_pct=60), "executive")
    assert isinstance(alta, ExecutiveView) and isinstance(baja, ExecutiveView)
    assert alta.recommendation.startswith("Go")
    assert baja.recommendation.startswith("No-go")


@pytest.mark.parametrize("tier", ["default", "no_existe", ""])
def test_sin_vista_propia_devuelve_none(tier: str) -> None:
    assert project_to_tier(_estimacion(), tier) is None


@pytest.mark.parametrize("tier", ["developer", "pm", "executive", "default"])
def test_estimacion_fuera_de_alcance_nunca_tiene_vista(tier: str) -> None:
    assert project_to_tier(_fuera_de_alcance(), tier) is None
