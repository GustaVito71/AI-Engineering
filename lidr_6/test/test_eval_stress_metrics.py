"""Tests de `evals/stress/metrics.py` (Bloque 4, LIDR_6): las tres métricas nuevas
sobre presupuestos y memoria entre turnos.

Son tests de lógica pura: arman la observación o la foto de sesión a mano, sin llamar
al servicio ni al LLM, igual que `test/test_eval_metrics.py` hace para el Bloque 1.
Incluyen un caso con los modelos REALES de `app.sessions.models` (no solo dicts a
medida) para comprobar que `MemoryDriftMetric` funciona con la forma que de verdad
tendría una `Session` sacada del almacén en memoria, no solo con un atajo de test.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.sessions.models import ConversationHistory, Message, ProjectMetadata, Session
from evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric, MemoryDriftMetric, MetricResult


# --- LatencyBudgetMetric ---------------------------------------------------------------------


def test_latencia_aprueba_dentro_del_presupuesto() -> None:
    resultado = LatencyBudgetMetric(budget_ms=5000).evaluate({"latency_ms": 3000})
    assert resultado == MetricResult(
        name="latency_budget_5000ms", score=1.0, passed=True, details="3000 ms, dentro del tope de 5000 ms"
    )


def test_latencia_reprueba_por_encima_del_presupuesto() -> None:
    resultado = LatencyBudgetMetric(budget_ms=5000).evaluate({"latency_ms": 7500})
    assert resultado.passed is False
    assert resultado.score == 0.0
    assert "supera" in resultado.details


def test_latencia_en_el_limite_exacto_aprueba() -> None:
    """`<=`, no `<`: el presupuesto es un tope que SÍ se puede tocar."""
    resultado = LatencyBudgetMetric(budget_ms=5000).evaluate({"latency_ms": 5000})
    assert resultado.passed is True


def test_latencia_se_calcula_desde_latency_s_en_segundos() -> None:
    """Como `TurnResult` de `evals/metrics.py`: segundos, no milisegundos."""
    resultado = LatencyBudgetMetric(budget_ms=5000).evaluate({"latency_s": 4.0})
    assert resultado.passed is True
    assert "4000 ms" in resultado.details


def test_latencia_se_calcula_desde_latencia_s_en_castellano() -> None:
    """Como el dict que devuelve `correr_punto` de `evals/stress/attachments.py`."""
    resultado = LatencyBudgetMetric(budget_ms=5000).evaluate({"latencia_s": 6.0})
    assert resultado.passed is False


def test_latencia_acepta_un_objeto_con_atributos_no_solo_un_dict() -> None:
    @dataclass
    class Observacion:
        latency_ms: float

    resultado = LatencyBudgetMetric(budget_ms=1000).evaluate(Observacion(latency_ms=500))
    assert resultado.passed is True


def test_latencia_sin_ningun_campo_conocido_lanza_keyerror() -> None:
    with pytest.raises(KeyError):
        LatencyBudgetMetric(budget_ms=1000).evaluate({"otra_cosa": 1})


def test_latencia_budget_no_positivo_lanza_valueerror() -> None:
    with pytest.raises(ValueError):
        LatencyBudgetMetric(budget_ms=0)


# --- CostBudgetMetric -------------------------------------------------------------------------


def test_coste_aprueba_dentro_del_presupuesto() -> None:
    resultado = CostBudgetMetric(budget_usd=0.01).evaluate({"cost_usd": 0.005})
    assert resultado.passed is True
    assert resultado.score == 1.0


def test_coste_reprueba_por_encima_del_presupuesto() -> None:
    resultado = CostBudgetMetric(budget_usd=0.01).evaluate({"cost_usd": 0.02})
    assert resultado.passed is False
    assert resultado.score == 0.0


def test_coste_en_el_limite_exacto_aprueba() -> None:
    resultado = CostBudgetMetric(budget_usd=0.01).evaluate({"cost_usd": 0.01})
    assert resultado.passed is True


def test_coste_acepta_el_nombre_de_campo_en_castellano() -> None:
    """Como el dict que devuelve `correr_punto` de `evals/stress/attachments.py`."""
    resultado = CostBudgetMetric(budget_usd=0.01).evaluate({"coste_usd": 0.02})
    assert resultado.passed is False


def test_coste_budget_no_positivo_lanza_valueerror() -> None:
    with pytest.raises(ValueError):
        CostBudgetMetric(budget_usd=0.0)


# --- MemoryDriftMetric: con dicts a medida ----------------------------------------------------


def _snapshot(*, summary=None, anchors=(), metadata=None) -> dict:
    return {"summary": summary, "anchors": list(anchors), "metadata": metadata or {}}


def test_drift_aprueba_si_el_hecho_esta_en_el_resumen() -> None:
    snapshot = _snapshot(summary="El proyecto se llama Portal Unificado de Clientes.")
    resultado = MemoryDriftMetric("Portal Unificado de Clientes").evaluate(snapshot)
    assert resultado.passed is True
    assert "summary" in resultado.details


def test_drift_aprueba_si_el_hecho_esta_en_las_anclas() -> None:
    snapshot = _snapshot(anchors=["Presupuesto cerrado: 80000 euros."])
    resultado = MemoryDriftMetric("80000").evaluate(snapshot)
    assert resultado.passed is True
    assert "anchors" in resultado.details


def test_drift_aprueba_si_el_hecho_esta_en_la_metadata() -> None:
    snapshot = _snapshot(metadata={"mentioned_technologies": ["flutter"]})
    resultado = MemoryDriftMetric("flutter").evaluate(snapshot)
    assert resultado.passed is True
    assert "metadata" in resultado.details


def test_drift_reprueba_si_el_hecho_no_aparece_en_ningun_lado() -> None:
    snapshot = _snapshot(summary="otra cosa", anchors=["otra cosa más"], metadata={"other_facts": []})
    resultado = MemoryDriftMetric("kubernetes").evaluate(snapshot)
    assert resultado.passed is False
    assert resultado.score == 0.0


def test_drift_es_insensible_a_mayusculas() -> None:
    snapshot = _snapshot(summary="Stack: REACT")
    resultado = MemoryDriftMetric("react").evaluate(snapshot)
    assert resultado.passed is True


def test_drift_respeta_un_where_acotado() -> None:
    """El hecho está en `summary`, pero `where` solo mira `anchors`: no cuenta."""
    snapshot = _snapshot(summary="react", anchors=[])
    resultado = MemoryDriftMetric("react", where=["anchors"]).evaluate(snapshot)
    assert resultado.passed is False


def test_drift_where_con_zona_desconocida_lanza_valueerror() -> None:
    with pytest.raises(ValueError):
        MemoryDriftMetric("x", where=["summary", "no_existe"])


def test_drift_fact_vacio_lanza_valueerror() -> None:
    with pytest.raises(ValueError):
        MemoryDriftMetric("   ")


def test_drift_instancias_distintas_no_comparten_el_where_por_defecto() -> None:
    """Regresión: `where` por defecto no puede ser una lista mutable compartida entre
    instancias (el error clásico de Python del argumento por defecto mutable)."""
    a = MemoryDriftMetric("x")
    b = MemoryDriftMetric("y")
    a.where.append("rompiendo_b")
    assert "rompiendo_b" not in b.where


# --- MemoryDriftMetric: con los modelos reales de `app.sessions.models` ----------------------


def test_drift_funciona_con_una_session_real_del_almacen() -> None:
    """La forma que de verdad tendría el `session_snapshot` en el runner del Bloque 4:
    una `Session` tal como la devuelve `SessionStore`, no un dict armado para el test.
    """
    session = Session(
        history=ConversationHistory(
            summary="Resumen acumulado de los primeros turnos.",
            anchors=[Message(role="user", content="El presupuesto final es de 80000 euros.")],
        ),
        metadata=ProjectMetadata(project_name="Portal Unificado de Clientes", mentioned_technologies=["flutter"]),
    )

    assert MemoryDriftMetric("80000").evaluate(session).passed is True
    assert MemoryDriftMetric("flutter").evaluate(session).passed is True
    assert MemoryDriftMetric("Portal Unificado de Clientes").evaluate(session).passed is True
    assert MemoryDriftMetric("kubernetes").evaluate(session).passed is False


def test_drift_con_session_real_distingue_resumen_de_anclas() -> None:
    session = Session(
        history=ConversationHistory(
            summary="react quedó mencionado acá, no como ancla.",
            anchors=[],
        ),
        metadata=ProjectMetadata(),
    )
    resultado_resumen = MemoryDriftMetric("react", where=["summary"]).evaluate(session)
    resultado_anclas = MemoryDriftMetric("react", where=["anchors"]).evaluate(session)
    assert resultado_resumen.passed is True
    assert resultado_anclas.passed is False
