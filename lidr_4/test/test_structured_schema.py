"""Tests del contrato de salida estructurada (`StructuredResult`).

Dos clases de comprobaciones: las del schema son errores (Instructor re-pregunta
al modelo) y la coherencia de los totales es un aviso (la estimación se devuelve
igual). Estos tests fijan qué cae de cada lado.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.schemas.structured_estimation import (
    MAX_HOURLY_RATE_EUR,
    MIN_CONFIDENCE_PCT,
    MIN_HOURLY_RATE_EUR,
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
    "team": [{"role": "Desarrollador", "headcount": 3}],
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
    datos["phases"][0]["cost_eur"] = 3500.1
    datos["phases"][1]["cost_eur"] = 16000.3
    datos["totals"]["cost_eur"] = 19500.4  # 3500.1 + 16000.3 == 19500.399999999998
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


@pytest.mark.parametrize("campo", ["duration_weeks", "hours", "cost_eur"])
def test_un_total_en_cero_se_nombra_para_que_el_modelo_lo_corrija(campo) -> None:
    """El mensaje de la re-pregunta dice cuál total está en cero, no solo que alguno lo está."""
    with pytest.raises(ValidationError) as error:
        StructuredResult.model_validate(_con(**{f"total_{campo}": 0}))
    assert f"en cero: totals.{campo}." in str(error.value)


def test_varios_totales_en_cero_se_nombran_todos() -> None:
    with pytest.raises(ValidationError, match=r"en cero: totals.duration_weeks, totals.hours\."):
        StructuredResult.model_validate(_con(total_duration_weeks=0, total_hours=0))


@pytest.mark.parametrize(
    ("cambio", "campo"),
    [({"fase_hours": 0}, "hours"), ({"fase_cost_eur": 0}, "cost_eur")],
)
def test_una_fase_en_cero_dice_que_campo(cambio, campo) -> None:
    with pytest.raises(ValidationError, match=rf"no los tiene: Descubrimiento \({campo}\)"):
        StructuredResult.model_validate(_con(**cambio))


def test_una_estimacion_sin_equipo_es_un_error() -> None:
    with pytest.raises(ValidationError, match="al menos un rol"):
        StructuredResult.model_validate(_con(team=[]))


# --- Totals tolerance -------------------------------------------------------
# Hasta TOTALS_MAX_DEVIATION_PCT de diferencia con la suma de las fases es un aviso;
# por encima, un error que se re-pregunta. VALIDA suma 390 horas y 23.950 EUR.


def test_la_tolerancia_es_de_cinco_por_ciento() -> None:
    from app.schemas.structured_estimation import TOTALS_MAX_DEVIATION_PCT

    assert TOTALS_MAX_DEVIATION_PCT == 5


@pytest.mark.parametrize(
    ("cambio", "campo"),
    [
        ({"total_hours": 410}, "totals.hours (410)"),  # +5,1 %
        ({"total_hours": 370}, "totals.hours (370)"),  # −5,1 %
        ({"total_cost_eur": 25200}, "totals.cost_eur (25.200)"),  # +5,2 %
    ],
)
def test_un_total_muy_desviado_es_un_error(cambio, campo) -> None:
    with pytest.raises(ValidationError) as error:
        StructuredResult.model_validate(_con(**cambio))
    mensaje = error.value.errors()[0]["msg"]
    assert f"{campo} se aleja más de 5 % de la suma de las fases" in mensaje
    assert "Calcula cada total sumando las fases." in mensaje


def test_los_dos_totales_desviados_se_nombran_en_un_solo_error() -> None:
    with pytest.raises(ValidationError) as error:
        StructuredResult.model_validate(_con(total_hours=500, total_cost_eur=30000))
    mensaje = error.value.errors()[0]["msg"]
    assert "totals.hours (500)" in mensaje
    assert "totals.cost_eur (30.000)" in mensaje


def test_justo_cinco_por_ciento_sigue_siendo_un_aviso() -> None:
    """El borde es inclusivo: 409,5 no es un número de horas, pero 25.147,50 EUR sí."""
    estimacion = StructuredResult.model_validate(_con(total_cost_eur=25147.5))  # +5,0 %
    assert len(estimacion.total_discrepancies()) == 1


def test_la_duracion_total_no_tiene_tolerancia() -> None:
    """Las fases pueden solaparse: ni aviso ni error por la duración, mientras las horas quepan."""
    estimacion = StructuredResult.model_validate(_con(total_duration_weeks=5))
    assert estimacion.total_discrepancies() == []


# --- Guardrail de salida: las horas caben en el equipo ----------------------------------


def test_las_horas_de_una_fase_tienen_que_caber_en_el_equipo() -> None:
    """3 personas en 1 semana: como mucho 120 h. 200 h no las trabaja nadie."""
    with pytest.raises(ValidationError) as error:
        StructuredResult.model_validate(_con(fase_hours=200, total_hours=520))
    mensaje = str(error.value)
    assert "no caben en el equipo" in mensaje
    assert "Descubrimiento (200 h en 1 semanas)" in mensaje
    assert "3 personas" in mensaje


def test_el_borde_del_tope_se_acepta() -> None:
    """120 h = 3 personas × 1 semana × 40 h."""
    StructuredResult.model_validate(
        _con(fase_hours=120, fase_cost_eur=6000, total_hours=440, total_cost_eur=26000)
    )
    with pytest.raises(ValidationError):
        StructuredResult.model_validate(
            _con(fase_hours=121, fase_cost_eur=6050, total_hours=441, total_cost_eur=26050)
        )


def test_el_total_tambien_tiene_que_caber_en_el_equipo() -> None:
    """390 h en 1 semana con 3 personas: las fases se solapan, pero no tanto."""
    with pytest.raises(ValidationError) as error:
        StructuredResult.model_validate(_con(total_duration_weeks=1))
    assert "totals (390 h en 1 semanas)" in str(error.value)


def test_el_tope_crece_con_el_equipo() -> None:
    una_persona = _con(team=[{"role": "Desarrollador", "headcount": 1}])
    with pytest.raises(ValidationError, match="1 personas"):
        StructuredResult.model_validate(una_persona)  # 320 h en 5 semanas: tope 200 h
    StructuredResult.model_validate(_con(team=[{"role": "Desarrollador", "headcount": 2}]))


def test_el_tope_suma_todos_los_roles() -> None:
    equipo = [{"role": "Desarrollador", "headcount": 1}, {"role": "QA", "headcount": 1}]
    StructuredResult.model_validate(_con(team=equipo))  # 2 personas: 5 semanas, tope 400 h


def test_una_estimacion_rechazada_no_se_compara_con_el_equipo() -> None:
    """En cero y sin equipo: no hay horas que comparar."""
    StructuredResult.model_validate(RECHAZO)


# --- Alucinación de bajo nivel (WU9 B, paso 3) ----------------------------------------


def test_las_tarifas_del_schema_son_las_de_las_plantillas() -> None:
    """Si una versión nueva cambia las tarifas, el rango de `_cost_per_hour_in_rates` miente."""
    raiz = Path(__file__).resolve().parents[1] / "app" / "prompts" / "estimation"
    tarifas = [
        rol["eur_per_hour"]
        for archivo in raiz.glob("*/examples.yaml")
        for rol in yaml.safe_load(archivo.read_text(encoding="utf-8"))["rates"].values()
    ]
    assert tarifas
    assert (min(tarifas), max(tarifas)) == (MIN_HOURLY_RATE_EUR, MAX_HOURLY_RATE_EUR)


def test_el_coste_por_hora_admite_el_margen_de_redondeo() -> None:
    """70 h: el rango aceptado va de 47,50 EUR/h (3.325) a 65,63 EUR/h (4.593,75)."""
    for coste in (3325, 3950, 4593):
        StructuredResult.model_validate(_con(fase_cost_eur=coste, total_cost_eur=coste + 20000))
    for coste in (3300, 4600):
        with pytest.raises(ValidationError, match="coste por hora"):
            StructuredResult.model_validate(_con(fase_cost_eur=coste, total_cost_eur=coste + 20000))


def test_el_coste_por_hora_bajo_el_minimo_se_rechaza_con_la_cifra() -> None:
    with pytest.raises(ValidationError) as error:
        StructuredResult.model_validate(_con(fase_cost_eur=700, total_cost_eur=20700))
    mensaje = str(error.value)
    assert "Descubrimiento (700 EUR en 70 h son 10 EUR/h)" in mensaje
    assert "entre 50 y 62,50 EUR/h" in mensaje


def test_el_coste_por_hora_sobre_el_maximo_se_rechaza() -> None:
    with pytest.raises(ValidationError, match="coste por hora"):
        StructuredResult.model_validate(_con(fase_cost_eur=7000, total_cost_eur=27000))


def test_el_coste_por_hora_nombra_todas_las_fases_fuera_de_rango() -> None:
    datos = _con(fase_cost_eur=700, total_cost_eur=700 + 100)
    datos["phases"][1]["cost_eur"] = 100
    with pytest.raises(ValidationError) as error:
        StructuredResult.model_validate(datos)
    assert "Descubrimiento" in str(error.value) and "Implementación" in str(error.value)


def test_el_rechazo_no_se_mide_contra_las_tarifas() -> None:
    StructuredResult.model_validate(RECHAZO)


def test_una_duracion_total_mayor_que_la_suma_se_rechaza() -> None:
    with pytest.raises(ValidationError) as error:
        StructuredResult.model_validate(_con(total_duration_weeks=9))
    assert "totals.duration_weeks (9) es mayor que la suma de las fases (6)" in str(error.value)


def test_una_duracion_total_menor_que_la_fase_mas_larga_se_rechaza() -> None:
    with pytest.raises(ValidationError) as error:
        StructuredResult.model_validate(_con(total_duration_weeks=4, team=_EQUIPO_GRANDE))
    assert "menor que la fase más larga (5)" in str(error.value)


_EQUIPO_GRANDE = [{"role": "Desarrollador", "headcount": 10}]


def test_las_fases_solapadas_y_el_margen_de_la_duracion_se_aceptan() -> None:
    for semanas in (5, 5.5, 6, 6.3):  # 6 + 5 % = 6,3
        StructuredResult.model_validate(_con(total_duration_weeks=semanas, team=_EQUIPO_GRANDE))
    for semanas in (4.7, 6.4):
        with pytest.raises(ValidationError, match="totals.duration_weeks"):
            StructuredResult.model_validate(_con(total_duration_weeks=semanas, team=_EQUIPO_GRANDE))


def _con_resumen(texto: str) -> dict:
    return _con(summary=texto)


@pytest.mark.parametrize(
    "texto",
    [
        "Proyecto de 390 horas y 23.950 EUR en 6 semanas con tres fases.",
        "Unas 395 h, 24.000 euros y 6,2 semanas.",
        "Total 23.950 € (62,50 EUR/hora de tope), jornadas de 32 h por persona.",
        "La fase de Implementación pesa 320 horas de las 390.",
        "Hay 3 sprints, 15 % de contingencia y 2 integraciones externas.",
        "Unas 40 horas semanales por persona.",
    ],
)
def test_las_cifras_del_resumen_que_estan_en_los_campos_se_aceptan(texto) -> None:
    StructuredResult.model_validate(_con_resumen(texto))


@pytest.mark.parametrize(
    ("texto", "cita"),
    [
        ("Proyecto de 500 horas en 6 semanas.", "500 horas"),
        ("Proyecto de 390 horas y 30.000 EUR.", "30.000 EUR"),
        ("Proyecto de 390 horas en 12 semanas.", "12 semanas"),
        ("Costará 23.950 € o 99.999 euros.", "99.999 euros"),
        ("Entre 3,5 semanas y 6 semanas.", "3,5 semanas"),
    ],
)
def test_una_cifra_inventada_en_el_resumen_se_rechaza_y_se_cita(texto, cita) -> None:
    with pytest.raises(ValidationError) as error:
        StructuredResult.model_validate(_con_resumen(texto))
    assert cita in str(error.value)


def test_el_resumen_de_una_fase_tambien_se_comprueba() -> None:
    datos = _con()
    datos["phases"][0]["summary"] = "Descubrimiento con 4.000 horas de entrevistas."
    with pytest.raises(ValidationError, match=r"4\.000 horas"):
        StructuredResult.model_validate(datos)


def test_el_resumen_de_un_rechazo_no_se_comprueba() -> None:
    datos = copy.deepcopy(RECHAZO)
    datos["summary"] = f"{OUT_OF_SCOPE_PREFIX} faltan datos, pensábamos en 500 horas."
    StructuredResult.model_validate(datos)


def test_la_suma_de_las_semanas_de_las_fases_es_una_cifra_conocida() -> None:
    """Total de 5 semanas con fases de 1 y 5: «6 semanas» es la suma, no un invento."""
    StructuredResult.model_validate(
        _con(total_duration_weeks=5, summary="Dura 5 semanas en total, 6 semanas si no se solapan.")
    )
    with pytest.raises(ValidationError, match="8 semanas"):
        StructuredResult.model_validate(
            _con(total_duration_weeks=5, summary="Dura 5 semanas, o hasta 8 semanas.")
        )


def test_una_semana_en_singular_tambien_es_una_cifra() -> None:
    datos = _con(fase_duration_weeks=2, total_duration_weeks=7, summary="Ocupa 4 semana.")
    with pytest.raises(ValidationError, match="4 semana"):
        StructuredResult.model_validate(datos)
