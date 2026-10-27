"""Tests de `evals/metrics.py` (Bloque 1, LIDR_6): las tres métricas binarias del
framework de evals.

Son tests de lógica pura: arman un `CaseResult` a mano (sin llamar al servicio ni al
LLM) y comprueban que cada métrica aprueba o reprueba como corresponde. `evals/run.py`
(el que sí llama al servicio real) no se prueba acá: cuesta dinero real y no es
determinista, así que se corre a mano, no en la suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.metrics import (
    OUT_OF_SCOPE_PREFIX,
    CaseResult,
    ContentRecallMetric,
    CostBoundsMetric,
    SchemaAdherenceMetric,
    TurnResult,
)

# Una estimación válida mínima, con la forma de `StructuredResult`, para no repetirla
# en cada caso de prueba.
_ESTIMACION_OK = {
    "phases": [{"name": "Diseño"}, {"name": "Desarrollo"}],
    "team": [{"role": "Desarrollador", "headcount": 2}],
    "totals": {"hours": 100, "cost_eur": 5000, "duration_weeks": 4},
    "summary": "Portal con SSO para empleados, listo en cuatro semanas.",
    "confidence_pct": 80,
}


def _turno_ok(*, estimation=None, tier_view=None, cost_usd=0.001, latency_s=0.1) -> TurnResult:
    """Un turno que respondió 200 con el cuerpo de `/estimate`."""
    return TurnResult(
        status_code=200,
        body={"estimation": estimation or _ESTIMACION_OK, "tier_view": tier_view},
        cost_usd=cost_usd,
        latency_s=latency_s,
    )


def _caso(**overrides) -> dict:
    """Un caso golden mínimo; cada test sobreescribe lo que necesita."""
    base = {
        "id": "caso-prueba",
        "expected_phase_range": [1, 8],
        "expected_tier_view_keys": None,
        "max_cost_usd_per_turn": 0.05,
        "max_total_cost_usd": 0.05,
        "expected_facts": [],
    }
    base.update(overrides)
    return base


# --- SchemaAdherenceMetric -----------------------------------------------------------------


def test_schema_aprueba_una_respuesta_dentro_del_rango_de_fases() -> None:
    caso = _caso(expected_phase_range=[1, 4])
    resultado = CaseResult(case_id="x", session_id="s", turns=[_turno_ok()])
    aprobado, _ = SchemaAdherenceMetric().evaluate(caso, resultado)
    assert aprobado is True


def test_schema_reprueba_fuera_del_rango_de_fases() -> None:
    caso = _caso(expected_phase_range=[5, 8])  # la estimación de prueba tiene 2 fases
    resultado = CaseResult(case_id="x", session_id="s", turns=[_turno_ok()])
    aprobado, motivo = SchemaAdherenceMetric().evaluate(caso, resultado)
    assert aprobado is False
    assert "fases" in motivo


def test_schema_reprueba_si_algun_turno_no_fue_200() -> None:
    caso = _caso()
    resultado = CaseResult(
        case_id="x",
        session_id="s",
        turns=[_turno_ok(), TurnResult(status_code=422, body=None, cost_usd=0.0, latency_s=0.0)],
    )
    aprobado, motivo = SchemaAdherenceMetric().evaluate(caso, resultado)
    assert aprobado is False
    assert "200" in motivo


def test_schema_fuera_de_alcance_aprueba_con_el_prefijo() -> None:
    caso = _caso(expect_out_of_scope=True)
    estimacion = {**_ESTIMACION_OK, "summary": f"{OUT_OF_SCOPE_PREFIX} falta alcance."}
    resultado = CaseResult(case_id="x", session_id="s", turns=[_turno_ok(estimation=estimacion)])
    aprobado, _ = SchemaAdherenceMetric().evaluate(caso, resultado)
    assert aprobado is True


def test_schema_fuera_de_alcance_reprueba_sin_el_prefijo() -> None:
    caso = _caso(expect_out_of_scope=True)
    resultado = CaseResult(case_id="x", session_id="s", turns=[_turno_ok()])  # summary normal
    aprobado, _ = SchemaAdherenceMetric().evaluate(caso, resultado)
    assert aprobado is False


def test_schema_tier_view_aprueba_con_las_claves_esperadas() -> None:
    caso = _caso(expected_tier_view_keys=["phases", "team", "total_hours"])
    tier_view = {"phases": [], "team": [], "total_hours": 10}
    resultado = CaseResult(case_id="x", session_id="s", turns=[_turno_ok(tier_view=tier_view)])
    aprobado, _ = SchemaAdherenceMetric().evaluate(caso, resultado)
    assert aprobado is True


def test_schema_tier_view_reprueba_con_claves_distintas() -> None:
    caso = _caso(expected_tier_view_keys=["phases", "team", "total_hours"])
    resultado = CaseResult(case_id="x", session_id="s", turns=[_turno_ok(tier_view=None)])
    aprobado, motivo = SchemaAdherenceMetric().evaluate(caso, resultado)
    assert aprobado is False
    assert "tier_view" in motivo


def test_schema_sin_expected_tier_view_keys_no_mira_el_tier() -> None:
    """`expected_tier_view_keys=None` significa "este caso no lo comprueba": da igual
    qué tier_view llegue."""
    caso = _caso(expected_tier_view_keys=None)
    resultado = CaseResult(case_id="x", session_id="s", turns=[_turno_ok(tier_view={"algo": 1})])
    aprobado, _ = SchemaAdherenceMetric().evaluate(caso, resultado)
    assert aprobado is True


# --- CostBoundsMetric -----------------------------------------------------------------------


def test_cost_aprueba_dentro_de_los_topes() -> None:
    caso = _caso(max_cost_usd_per_turn=0.01, max_total_cost_usd=0.02)
    resultado = CaseResult(
        case_id="x", session_id="s", turns=[_turno_ok(cost_usd=0.005), _turno_ok(cost_usd=0.005)]
    )
    aprobado, _ = CostBoundsMetric().evaluate(caso, resultado)
    assert aprobado is True


def test_cost_reprueba_un_turno_por_encima_del_tope_por_turno() -> None:
    caso = _caso(max_cost_usd_per_turn=0.01, max_total_cost_usd=1.0)
    resultado = CaseResult(case_id="x", session_id="s", turns=[_turno_ok(cost_usd=0.5)])
    aprobado, motivo = CostBoundsMetric().evaluate(caso, resultado)
    assert aprobado is False
    assert "turno" in motivo


def test_cost_reprueba_el_total_por_encima_del_tope() -> None:
    caso = _caso(max_cost_usd_per_turn=1.0, max_total_cost_usd=0.01)
    resultado = CaseResult(
        case_id="x", session_id="s", turns=[_turno_ok(cost_usd=0.006), _turno_ok(cost_usd=0.006)]
    )
    aprobado, motivo = CostBoundsMetric().evaluate(caso, resultado)
    assert aprobado is False
    assert "total" in motivo


def test_cost_reprueba_si_algun_turno_no_fue_200() -> None:
    caso = _caso()
    resultado = CaseResult(
        case_id="x",
        session_id="s",
        turns=[TurnResult(status_code=500, body=None, cost_usd=0.0, latency_s=0.0)],
    )
    aprobado, _ = CostBoundsMetric().evaluate(caso, resultado)
    assert aprobado is False


# --- ContentRecallMetric --------------------------------------------------------------------


def test_recall_aprueba_cuando_todos_los_hechos_aparecen() -> None:
    caso = _caso(expected_facts=["sso", "empleados"])
    resultado = CaseResult(case_id="x", session_id="s", turns=[_turno_ok()])
    aprobado, _ = ContentRecallMetric().evaluate(caso, resultado)
    assert aprobado is True


def test_recall_reprueba_cuando_falta_un_hecho() -> None:
    caso = _caso(expected_facts=["sso", "kubernetes"])
    resultado = CaseResult(case_id="x", session_id="s", turns=[_turno_ok()])
    aprobado, motivo = ContentRecallMetric().evaluate(caso, resultado)
    assert aprobado is False
    assert "kubernetes" in motivo


def test_recall_busca_tambien_en_la_metadata_de_sesion() -> None:
    """Un hecho que no está en la estimación pero sí en `session_metadata` también cuenta."""
    caso = _caso(expected_facts=["presupuesto-ajustado"])
    resultado = CaseResult(
        case_id="x",
        session_id="s",
        turns=[_turno_ok()],
        session_metadata={"other_facts": ["presupuesto-ajustado"]},
    )
    aprobado, _ = ContentRecallMetric().evaluate(caso, resultado)
    assert aprobado is True


def test_recall_sin_hechos_aprueba_sin_mirar_nada_mas() -> None:
    """Una lista vacía de hechos (los casos fuera de alcance) aprueba siempre, incluso
    si ningún turno respondió 200."""
    caso = _caso(expected_facts=[])
    resultado = CaseResult(
        case_id="x",
        session_id="s",
        turns=[TurnResult(status_code=500, body=None, cost_usd=0.0, latency_s=0.0)],
    )
    aprobado, _ = ContentRecallMetric().evaluate(caso, resultado)
    assert aprobado is True


def test_recall_es_insensible_a_mayusculas() -> None:
    caso = _caso(expected_facts=["SSO"])
    resultado = CaseResult(case_id="x", session_id="s", turns=[_turno_ok()])
    aprobado, _ = ContentRecallMetric().evaluate(caso, resultado)
    assert aprobado is True
