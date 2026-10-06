"""Tests de `RenderedResult`: la salida de `v4`, con la presentación en `rendered`.

El validador de `rendered` recibe `output_format` por el contexto de validación
y exige dos cosas: la forma pedida (tabla, lista numerada o prosa) y que las
cifras del texto sean las de los campos. Un rechazo es solo texto.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.structured_estimation import (
    OUT_OF_SCOPE_PREFIX,
    UNESTIMATED_PHASE,
    RenderedEstimationResponse,
    RenderedResult,
    StructuredResult,
    format_number,
)

BASE = {
    "phases": [
        {
            "name": "Diseño",
            "summary": "Wireframes y prototipo navegable.",
            "duration_weeks": 2,
            "hours": 130,
            "cost_eur": 6500,
            "confidence_pct": 80,
            "assumptions": [],
            "risks": [],
        },
        {
            "name": "Implementación",
            "summary": "Desarrollo de la app y del backend.",
            "duration_weeks": 6,
            "hours": 1320,
            "cost_eur": 82500,
            "confidence_pct": 70,
            "assumptions": [],
            "risks": [],
        },
    ],
    "team": [{"role": "Desarrollador", "headcount": 2}],
    "totals": {"hours": 1450, "cost_eur": 89000, "duration_weeks": 8},
    "summary": "Estimación de prueba con dos fases y un rol.",
    "confidence_pct": 70,
}
TABLA = (
    "| Fase | Semanas | Horas | Coste (EUR) | Confianza (%) |\n"
    "|---|---|---|---|---|\n"
    "| Diseño | 2 | 130 | 6.500 | 80 |\n"
    "| Implementación | 6 | 1.320 | 82.500 | 70 |\n"
    "| **Total** | 8 | 1.450 | 89.000 | 70 |"
)
PARTIDAS = (
    "1. **Diseño** — 2 semanas — 130 horas — 6.500 EUR. Wireframes.\n"
    "2. **Implementación** — 6 semanas — 1.320 horas — 82.500 EUR. Desarrollo.\n"
    "\n"
    "**Total:** 8 semanas, 1.450 horas y 89.000 EUR."
)
NARRATIVA = (
    "**Diseño.** Dura 2 semanas, con 130 horas y 6.500 EUR.\n\n"
    "**Implementación.** Dura 6 semanas, con 1.320 horas y 82.500 EUR.\n\n"
    "En total, 8 semanas, 1.450 horas y 89.000 EUR."
)
POR_FORMATO = {"phases_table": TABLA, "line_items": PARTIDAS, "narrative": NARRATIVA}


def _validar(rendered: str, output_format: str | None, **cambios) -> RenderedResult:
    contexto = None if output_format is None else {"output_format": output_format}
    return RenderedResult.model_validate(
        {**BASE, **cambios, "rendered": rendered}, context=contexto
    )


def _error(rendered: str, output_format: str, **cambios) -> str:
    with pytest.raises(ValidationError) as exc:
        _validar(rendered, output_format, **cambios)
    return exc.value.errors()[0]["msg"]


# --- Forma ---------------------------------------------------------------------------


@pytest.mark.parametrize("output_format", list(POR_FORMATO))
def test_cada_formato_acepta_su_forma(output_format) -> None:
    resultado = _validar(POR_FORMATO[output_format], output_format)
    assert resultado.rendered == POR_FORMATO[output_format]


@pytest.mark.parametrize(
    ("output_format", "otro"),
    [(f, o) for f in POR_FORMATO for o in POR_FORMATO if f != o],
)
def test_cada_formato_rechaza_la_forma_de_otro(output_format, otro) -> None:
    mensaje = _error(POR_FORMATO[otro], output_format)
    assert f"output_format {output_format}" in mensaje


def test_la_tabla_necesita_la_fila_separadora() -> None:
    sin_separador = "\n".join(x for x in TABLA.splitlines() if "---" not in x)
    assert "fila separadora" in _error(sin_separador, "phases_table")


def test_la_narrativa_no_admite_viñetas() -> None:
    con_vinetas = NARRATIVA + "\n\n- Un supuesto en viñeta."
    assert "sin tablas ni listas" in _error(con_vinetas, "narrative")


def test_una_partida_puede_escribir_la_cifra_sin_separador_de_miles() -> None:
    _validar(PARTIDAS.replace("1.320 horas", "1320 horas"), "line_items")


# --- Cifras --------------------------------------------------------------------------


def test_una_cifra_de_fase_distinta_de_los_campos_se_rechaza() -> None:
    mensaje = _error(TABLA.replace("1.320", "1.330"), "phases_table")
    assert "falta o no coincide: Implementación" in mensaje


def test_una_cifra_dentro_de_otra_no_cuenta() -> None:
    """130 no aparece en «1.130»: la fila de Diseño con 1.130 horas no coincide."""
    tabla = TABLA.replace("| Diseño | 2 | 130 |", "| Diseño | 2 | 1.130 |")
    assert "falta o no coincide: Diseño" in _error(tabla, "phases_table")


def test_una_fase_que_falta_se_nombra() -> None:
    sin_diseño = "\n".join(x for x in TABLA.splitlines() if "Diseño" not in x)
    assert "falta o no coincide: Diseño" in _error(sin_diseño, "phases_table")


def test_los_totales_tienen_que_aparecer() -> None:
    sin_total = NARRATIVA.rsplit("\n\n", 1)[0]
    mensaje = _error(sin_total, "narrative")
    assert "1.450 horas y 89.000 EUR" in mensaje


def test_un_total_que_no_cuadra_se_muestra_como_se_declaro() -> None:
    """El validador compara rendered con los campos, no con la suma de las fases:
    un total mal sumado sigue siendo un aviso (total_discrepancies), no un error."""
    resultado = _validar(
        TABLA.replace("1.450", "1.500"),
        "phases_table",
        totals={"hours": 1500, "cost_eur": 89000, "duration_weeks": 8},
    )
    assert resultado.total_discrepancies()


# --- Rechazo -------------------------------------------------------------------------

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
    "summary": f"{OUT_OF_SCOPE_PREFIX} falta el alcance.",
    "confidence_pct": 10,
}


@pytest.mark.parametrize("output_format", list(POR_FORMATO))
def test_un_rechazo_es_solo_texto_con_el_prefijo(output_format) -> None:
    _validar(f"{OUT_OF_SCOPE_PREFIX} falta el alcance.", output_format, **RECHAZO)


def test_un_rechazo_sin_prefijo_se_rechaza() -> None:
    mensaje = _error("No se puede estimar.", "phases_table", **RECHAZO)
    assert OUT_OF_SCOPE_PREFIX in mensaje


def test_un_rechazo_con_tabla_se_rechaza() -> None:
    rendered = f"{OUT_OF_SCOPE_PREFIX} falta el alcance.\n\n| a | b |\n|---|---|\n| 0 | 0 |"
    assert "sin tabla ni lista" in _error(rendered, "phases_table", **RECHAZO)


# --- Contexto y contrato -------------------------------------------------------------


def test_sin_contexto_solo_se_validan_los_campos() -> None:
    assert _validar("Texto cualquiera sin cifras.", None).rendered


def test_un_formato_desconocido_es_un_error_de_programacion() -> None:
    """No es un error del modelo: no se convierte en ValidationError ni se re-pregunta."""
    with pytest.raises(TypeError, match="no es un formato de rendered"):
        _validar(TABLA, "pdf")


def test_rendered_va_al_final_del_schema() -> None:
    """El modelo emite los campos en orden: decide las cifras antes de escribir el texto."""
    propiedades = list(RenderedResult.model_json_schema()["properties"])
    assert propiedades[-1] == "rendered"
    assert propiedades[:-1] == list(StructuredResult.model_json_schema()["properties"])
    assert "rendered" in RenderedResult.model_json_schema()["required"]


def test_la_respuesta_lleva_el_schema_renderizado() -> None:
    schema = RenderedEstimationResponse.model_json_schema()
    assert schema["properties"]["estimation"]["$ref"].endswith("/RenderedResult")


def test_formato_de_numeros() -> None:
    assert format_number(89000) == "89.000"
    assert format_number(62.5) == "62,50"
