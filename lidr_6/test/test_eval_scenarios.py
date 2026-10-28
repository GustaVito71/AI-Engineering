"""Tests de `evals/stress/scenarios.py` (Bloque 2, LIDR_6): el generador de
conversaciones sintéticas y su fact-tracker.

Son tests de lógica pura, sin red ni LLM: no corren el servicio, solo comprueban que
los escenarios generados tienen la forma que describe el módulo (turnos consecutivos
sin huecos, el hito de cada perfil en el turno que toca, y que `facts_expected_after`
resuelve bien lo que debería seguir vigente).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# `evals/` vive en la raíz del proyecto, un nivel por encima de `test/`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.stress import scenarios as s


def test_son_tres_perfiles_por_cinco_n() -> None:
    """`all_scenarios` devuelve exactamente 3 × 5 = 15 escenarios."""
    todos = s.all_scenarios()
    assert len(todos) == len(s.PROFILES) * len(s.TURN_COUNTS)
    assert {esc.profile for esc in todos} == set(s.PROFILES)


@pytest.mark.parametrize("profile", list(s.PROFILES))
@pytest.mark.parametrize("n_turns", list(s.TURN_COUNTS))
def test_los_turnos_son_consecutivos_sin_huecos(profile, n_turns) -> None:
    """Cada escenario tiene exactamente `n_turns` turnos, con índices 1..n_turns, sin
    repetir ninguno y sin saltarse ninguno."""
    escenario = s.build_scenario(profile, n_turns)
    assert escenario.n_turns == n_turns
    assert [t.index for t in escenario.turns] == list(range(1, n_turns + 1))


def test_perfil_desconocido_lanza_value_error() -> None:
    """Un nombre de perfil que no existe no genera un escenario vacío: falla."""
    with pytest.raises(ValueError):
        s.build_scenario("inexistente", 3)


def test_n_turns_menor_a_uno_lanza_value_error() -> None:
    """Una conversación de cero o menos turnos no tiene sentido."""
    with pytest.raises(ValueError):
        s.build_scenario("growing", 0)


# --- Perfil growing ------------------------------------------------------------------------


@pytest.mark.parametrize("n_turns", list(s.TURN_COUNTS))
def test_growing_el_nombre_del_proyecto_sobrevive_a_cualquier_turno(n_turns) -> None:
    """El nombre fijado en el turno 1 nunca se marca como superado: tiene que seguir
    vigente después de cualquier turno, incluido el último."""
    escenario = s.build_scenario("growing", n_turns)
    nombre = escenario.turns[0].expected_facts[0]
    for k in range(1, n_turns + 1):
        assert nombre in escenario.facts_expected_after(k)


def test_growing_cada_turno_agrega_un_hecho_nuevo() -> None:
    """A partir del turno 2, cada turno suma exactamente un hecho más a los vigentes."""
    escenario = s.build_scenario("growing", 10)
    anteriores = len(escenario.facts_expected_after(1))
    for k in range(2, 11):
        actuales = len(escenario.facts_expected_after(k))
        assert actuales == anteriores + 1
        anteriores = actuales


# --- Perfil pivoting -----------------------------------------------------------------------


def test_pivoting_con_un_turno_no_hay_pivote() -> None:
    """Con N=1 no hay "antes": el único turno ya declara el stack de destino, no el
    inicial, y no hay ningún hecho superado."""
    escenario = s.build_scenario("pivoting", 1)
    assert len(escenario.turns) == 1
    assert not escenario.turns[0].superseded_facts
    assert s._STACK_NUEVO in escenario.turns[0].expected_facts
    assert s._STACK_INICIAL not in escenario.turns[0].expected_facts


@pytest.mark.parametrize("n_turns", [3, 6, 10, 20])
def test_pivoting_el_stack_viejo_desaparece_tras_el_pivote(n_turns) -> None:
    """Antes del pivote, el stack inicial está vigente; desde el turno del pivote en
    adelante, el vigente es el nuevo y el viejo ya no aparece."""
    escenario = s.build_scenario("pivoting", n_turns)
    pivote = s._pivot_turn(n_turns)
    assert pivote is not None

    assert s._STACK_INICIAL in escenario.facts_expected_after(pivote - 1)
    for k in range(pivote, n_turns + 1):
        vigentes = escenario.facts_expected_after(k)
        assert s._STACK_NUEVO in vigentes
        assert s._STACK_INICIAL not in vigentes


@pytest.mark.parametrize("n_turns", list(s.TURN_COUNTS))
def test_pivoting_turno_5_para_conversaciones_que_llegan_ahi(n_turns) -> None:
    """El enunciado pide el pivote en el turno 5: para N>=5 tiene que ser exactamente
    ese, sin escalarlo."""
    pivote = s._pivot_turn(n_turns)
    if n_turns >= 5:
        assert pivote == 5
    elif n_turns >= 3:
        assert pivote == n_turns - 1
    else:
        assert pivote is None


# --- Perfil contradicting ------------------------------------------------------------------


@pytest.mark.parametrize("n_turns", [1, 2])
def test_contradicting_con_pocos_turnos_no_hay_correccion(n_turns) -> None:
    """Con 1 o 2 turnos no hay espacio para un turno de declaración y otro,
    DISTINTO, de corrección: solo se declara un presupuesto, sin contradecirlo."""
    escenario = s.build_scenario("contradicting", n_turns)
    for turno in escenario.turns:
        assert not turno.superseded_facts
    assert s._PRESUPUESTO_INICIAL in escenario.facts_expected_after(n_turns)
    assert s._PRESUPUESTO_CORREGIDO not in escenario.facts_expected_after(n_turns)


@pytest.mark.parametrize("n_turns", [3, 6, 10, 20])
def test_contradicting_el_presupuesto_corregido_reemplaza_al_inicial(n_turns) -> None:
    """Antes de la corrección, el presupuesto vigente es el inicial; desde el turno de
    la corrección en adelante, el vigente es el corregido y el inicial ya no aparece."""
    escenario = s.build_scenario("contradicting", n_turns)
    _declaracion, correccion = s._contradiction_turns(n_turns)

    assert s._PRESUPUESTO_INICIAL in escenario.facts_expected_after(correccion - 1)
    for k in range(correccion, n_turns + 1):
        vigentes = escenario.facts_expected_after(k)
        assert s._PRESUPUESTO_CORREGIDO in vigentes
        assert s._PRESUPUESTO_INICIAL not in vigentes
    # El nombre del proyecto, declarado siempre en el turno 1, nunca se supera.
    assert escenario.turns[0].expected_facts
    nombre_proyecto = escenario.turns[0].expected_facts[0]
    assert nombre_proyecto in escenario.facts_expected_after(n_turns)


@pytest.mark.parametrize("n_turns", list(s.TURN_COUNTS))
def test_contradicting_turnos_3_y_8_para_conversaciones_que_llegan_ahi(n_turns) -> None:
    """El enunciado pide declaración en el turno 3 y corrección en el 8: para N>=8
    tiene que ser exactamente eso, sin escalarlo."""
    puntos = s._contradiction_turns(n_turns)
    if n_turns >= 8:
        assert puntos == (3, 8)
    elif n_turns >= 6:
        assert puntos == (2, n_turns)
    elif n_turns >= 3:
        assert puntos == (1, n_turns)
    else:
        assert puntos is None
