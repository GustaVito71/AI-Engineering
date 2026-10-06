"""Tests del contrato de salida estructurada (`StructuredResult`).

Dos clases de comprobaciones: las del schema son errores (Instructor re-pregunta
al modelo) y la coherencia de los totales es un aviso (la estimación se devuelve
igual). Estos tests fijan qué cae de cada lado.
"""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from app.schemas.structured_estimation import (
    MIN_CONFIDENCE_PCT,
    OUT_OF_SCOPE_PREFIX,
    UNESTIMATED_PHASE,
    StructuredResult,
)

VALIDA = {
    "phases": [
        {
            "name": "Descubrimiento",
            "summary": "Trabajo de la fase de Descubrimiento.",
            "duration_weeks": 1,
            "hours": 70,
            "cost_eur": 3950,
            "confidence_pct": 85,
            "assumptions": [],
            "risks": [],
        },
        {
            "name": "Implementación",
            "summary": "Trabajo de la fase de Implementación.",
            "duration_weeks": 5,
            "hours": 320,
            "cost_eur": 20000,
            "confidence_pct": 70,
            "assumptions": ["Sin dependencias externas"],
            "risks": [{"risk": "Cambios de alcance", "mitigation": "Cerrar el alcance antes"}],
        },
    ],
    "team": [{"role": "Desarrollador", "headcount": 2}],
    "totals": {"hours": 390, "cost_eur": 23950, "duration_weeks": 6},
    "summary": "Estimación de prueba con las fases y el equipo indicados.",
    "confidence_pct": 70,
}


def _con(**cambios) -> dict:
    """Copia de VALIDA con cambios en la primera fase (`fase_*`) o en los totales."""
    datos = copy.deepcopy(VALIDA)
    for clave, valor in cambios.items():
        if clave.startswith("fase_"):
            datos["phases"][0][clave.removeprefix("fase_")] = valor
        elif clave.startswith("total_"):
            datos["totals"][clave.removeprefix("total_")] = valor
        else:
            datos[clave] = valor
    return datos


def test_una_estimacion_valida_se_acepta() -> None:
    estimacion = StructuredResult.model_validate(VALIDA)
    assert [f.name for f in estimacion.phases] == ["Descubrimiento", "Implementación"]
    assert estimacion.total_discrepancies() == []


def test_phases_va_primero_y_el_resumen_al_final() -> None:
    """WU0: el modelo emite en el orden del schema. Las fases salen antes, los
    totales se declaran después de decidirlas, y el resumen y la confianza al final."""
    assert list(StructuredResult.model_json_schema()["properties"]) == [
        "phases",
        "team",
        "totals",
        "summary",
        "confidence_pct",
    ]


@pytest.mark.parametrize(
    "cambio",
    [
        {"fase_hours": 0},
        {"fase_cost_eur": -1},
        {"fase_duration_weeks": 0},
        {"fase_confidence_pct": 101},
        {"fase_confidence_pct": -5},
        {"fase_name": ""},
        {"total_hours": 0},
        {"phases": []},
        {"team": []},
        {"team": [{"role": "QA", "headcount": 0}]},
        # Topes: una estimación absurda también se vuelve a pedir.
        {"fase_duration_weeks": 53},
        {"fase_hours": 10_001},
        {"fase_cost_eur": 1_000_001},
        {"total_duration_weeks": 105},
        {"phases": [VALIDA["phases"][0] | {"name": f"Fase {i}"} for i in range(9)]},
        {"team": [{"role": "QA", "headcount": 51}]},
        {"fase_assumptions": ["Supuesto"] * 7},
        # Textos: resumen general y de fase obligatorios y con un mínimo.
        {"summary": "Corto."},
        {"fase_summary": "Corto."},
        {"confidence_pct": 101},
    ],
    ids=lambda c: next(iter(c)),
)
def test_lo_que_no_cumple_el_schema_es_un_error(cambio) -> None:
    with pytest.raises(ValidationError):
        StructuredResult.model_validate(_con(**cambio))


def test_supuestos_y_riesgos_son_obligatorios_aunque_vayan_vacios() -> None:
    """Sin default: el modelo tiene que decidir explícitamente que no hay ninguno."""
    datos = copy.deepcopy(VALIDA)
    del datos["phases"][0]["assumptions"]
    with pytest.raises(ValidationError, match="assumptions"):
        StructuredResult.model_validate(datos)


def test_nombres_de_fase_repetidos_es_un_error() -> None:
    datos = copy.deepcopy(VALIDA)
    datos["phases"][1]["name"] = " descubrimiento "
    with pytest.raises(ValidationError, match="se repite:  descubrimiento"):
        StructuredResult.model_validate(datos)


def test_un_total_que_no_cuadra_no_es_un_error_sino_un_aviso() -> None:
    """La prueba real del 3/10: las fases sumaban 29.750 y el modelo escribió 29.850."""
    estimacion = StructuredResult.model_validate(_con(total_hours=400, total_cost_eur=24050))

    assert estimacion.total_discrepancies() == [
        "El total de horas declarado (400) no coincide con la suma de las fases (390).",
        "El coste total declarado (24.050 EUR) no coincide con la suma de las fases (23.950 EUR).",
    ]


def test_la_duracion_total_no_se_compara_con_la_suma() -> None:
    """Dos fases pueden solaparse en el calendario: una duración menor no es un error."""
    estimacion = StructuredResult.model_validate(_con(total_duration_weeks=5))
    assert estimacion.total_discrepancies() == []


def test_coste_con_decimales_se_compara_sin_ruido_de_coma_flotante() -> None:
    datos = copy.deepcopy(VALIDA)
    datos["phases"][0]["cost_eur"] = 0.1
    datos["phases"][1]["cost_eur"] = 0.2
    datos["totals"]["cost_eur"] = 0.3  # 0.1 + 0.2 == 0.30000000000000004
    assert StructuredResult.model_validate(datos).total_discrepancies() == []


# --- Rechazo: «Fuera de alcance:» ---------------------------------------------------------

RECHAZO = {
    "phases": [
        {
            "name": UNESTIMATED_PHASE,
            "summary": "No se puede dimensionar sin más información.",
            "duration_weeks": 0,
            "hours": 0,
            "cost_eur": 0,
            "confidence_pct": 0,
            "assumptions": [],
            "risks": [],
        }
    ],
    "team": [],
    "totals": {"hours": 0, "cost_eur": 0, "duration_weeks": 0},
    "summary": f"{OUT_OF_SCOPE_PREFIX} la descripción no dice qué hay que construir.",
    "confidence_pct": 10,
}


def test_un_rechazo_bien_formado_se_acepta() -> None:
    estimacion = StructuredResult.model_validate(RECHAZO)
    assert estimacion.out_of_scope
    assert estimacion.total_discrepancies() == []  # todo en cero: nada que avisar


def test_una_estimacion_normal_no_es_un_rechazo() -> None:
    assert not StructuredResult.model_validate(VALIDA).out_of_scope


def test_confianza_baja_sin_el_prefijo_es_un_error() -> None:
    """El mensaje es para el modelo: Instructor se lo devuelve al re-preguntar."""
    with pytest.raises(ValidationError, match=f"tiene que empezar con «{OUT_OF_SCOPE_PREFIX}»"):
        StructuredResult.model_validate(_con(confidence_pct=MIN_CONFIDENCE_PCT - 1))


def test_el_umbral_de_confianza_no_exige_rechazo() -> None:
    assert StructuredResult.model_validate(_con(confidence_pct=MIN_CONFIDENCE_PCT))


def test_el_prefijo_con_confianza_suficiente_es_un_error() -> None:
    datos = copy.deepcopy(RECHAZO) | {"confidence_pct": MIN_CONFIDENCE_PCT}
    with pytest.raises(ValidationError, match="exige confidence_pct menor"):
        StructuredResult.model_validate(datos)


@pytest.mark.parametrize(
    "cambio",
    [
        {"team": [{"role": "QA", "headcount": 1}]},
        {"totals": {"hours": 10, "cost_eur": 0, "duration_weeks": 0}},
        {"phases": [RECHAZO["phases"][0] | {"cost_eur": 500}]},
        {"phases": [RECHAZO["phases"][0] | {"name": "Descubrimiento"}]},
        {"phases": [RECHAZO["phases"][0], RECHAZO["phases"][0] | {"name": "Otra"}]},
    ],
    ids=["con-equipo", "totales", "coste", "nombre", "dos-fases"],
)
def test_un_rechazo_con_cifras_es_un_error(cambio) -> None:
    with pytest.raises(ValidationError, match="una sola fase"):
        StructuredResult.model_validate(copy.deepcopy(RECHAZO) | cambio)


@pytest.mark.parametrize(
    "cambio",
    [{"fase_hours": 0}, {"fase_cost_eur": 0}, {"fase_duration_weeks": 0}],
    ids=lambda c: next(iter(c)),
)
def test_una_fase_estimada_en_cero_es_un_error(cambio) -> None:
    with pytest.raises(ValidationError, match="mayores que 0; no los tiene: Descubrimiento"):
        StructuredResult.model_validate(_con(**cambio))


def test_una_estimacion_sin_equipo_es_un_error() -> None:
    with pytest.raises(ValidationError, match="al menos un rol"):
        StructuredResult.model_validate(_con(team=[]))
