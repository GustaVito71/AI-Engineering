"""Tests de `ReviewedResult` y `split_review_note` (`app/schemas/acb.py`).

QUÉ SE COMPRUEBA

- La nota de salvedades que el jefe antepone al resumen se separa bien del resumen del modelo.
- Una nota con cifras que no están en la estimación (las que cita el crítico) NO hace fallar
  a `ReviewedResult`: es la causa del 500 que se vio en la prueba real.
- La protección contra cifras inventadas sigue activa para el resumen del modelo, aun con
  nota delante.
- El schema original (`StructuredResult`, el de lidr_4) no cambió: sigue rechazando un
  resumen con cifras que no existen.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from test_estimate_structured_endpoint import ESTIMACION

from app.schemas.acb import (
    REVIEW_NOTE_MARK,
    REVIEW_NOTE_SEPARATOR,
    ReviewedResult,
    split_review_note,
)
from app.schemas.structured_estimation import StructuredResult

NOTA_CON_CIFRAS = (
    f"{REVIEW_NOTE_MARK} Salvedades abiertas de la revisión independiente:\n"
    "- [major] unrealistic_estimate (phases[0].cost_eur): el coste de 250 EUR es bajo."
    f"{REVIEW_NOTE_SEPARATOR}"
)


def _con_resumen(resumen: str) -> dict:
    """La estimación de prueba con el resumen indicado."""
    return {**ESTIMACION, "summary": resumen}


def test_split_separa_la_nota_del_resumen_del_modelo() -> None:
    """La nota (con su separador) y el resumen del modelo se devuelven por separado."""
    nota, cuerpo = split_review_note(NOTA_CON_CIFRAS + "Resumen del modelo.")
    assert nota == NOTA_CON_CIFRAS
    assert cuerpo == "Resumen del modelo."


def test_split_sin_nota_devuelve_el_resumen_entero() -> None:
    """Un resumen que no empieza con la marca no tiene nota."""
    assert split_review_note("Resumen normal.\n\nSegundo párrafo.") == (
        "",
        "Resumen normal.\n\nSegundo párrafo.",
    )


def test_una_nota_con_cifras_ajenas_no_hace_fallar_a_reviewed_result() -> None:
    """Las cifras que cita el crítico en la nota no se comparan con la estimación."""
    datos = _con_resumen(NOTA_CON_CIFRAS + ESTIMACION["summary"])
    assert ReviewedResult.model_validate(datos).summary.startswith(REVIEW_NOTE_MARK)


def test_una_cifra_inventada_por_el_modelo_se_sigue_rechazando_con_nota() -> None:
    """La comprobación sigue vigente para el resumen que escribió el modelo."""
    datos = _con_resumen(NOTA_CON_CIFRAS + "El proyecto cuesta 999.999 EUR en total.")
    with pytest.raises(ValidationError, match="cita cifras que no están"):
        ReviewedResult.model_validate(datos)


def test_el_schema_original_no_cambio() -> None:
    """`StructuredResult` (el de lidr_4) sigue rechazando un resumen con cifras ajenas,
    también si empieza con la marca: la excepción es solo de `ReviewedResult`.
    """
    datos = _con_resumen(NOTA_CON_CIFRAS + ESTIMACION["summary"])
    with pytest.raises(ValidationError, match="cita cifras que no están"):
        StructuredResult.model_validate(datos)
