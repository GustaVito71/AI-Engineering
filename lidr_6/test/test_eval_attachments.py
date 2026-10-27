"""Tests de `evals/stress/attachments.py` (Bloque 3, LIDR_6): el generador de PDFs
sintéticos y su reporte local de truncamiento.

Son tests de lógica pura y extracción real de PDF (con `pypdf`, la misma librería que
usa `app/services/attachments.py`), sin red ni LLM: no llaman al servicio, así que no
cuestan nada y corren en cualquier máquina. El runner HTTP del mismo módulo (el que sí
llama al servicio real) no se prueba acá por la misma razón que `evals/run.py`: cuesta
dinero real y no es determinista.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# `evals/` vive en la raíz del proyecto, un nivel por encima de `test/`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.attachments import MAX_TEXT_CHARS
from evals.stress import attachments as a


@pytest.mark.parametrize("size_kb", list(a.SIZE_POINTS_KB))
def test_los_cinco_puntos_son_los_del_enunciado(size_kb) -> None:
    """0, 5, 20, 50 y 100 KB: los cinco puntos que pide el bloque 3."""
    assert size_kb in (0, 5, 20, 50, 100)


@pytest.mark.parametrize("size_kb", [5, 20, 50, 100])
def test_el_pdf_calibrado_pesa_cerca_del_tamano_pedido(size_kb) -> None:
    """Cada PDF generado pesa su tamaño objetivo con un margen chico (no hace falta el
    byte exacto: este es un escenario de estrés, no un test de bytes)."""
    attachment = a.build_attachment(size_kb)
    objetivo = size_kb * 1000
    margen = max(300, round(objetivo * 0.01))  # 1% o 300 bytes, lo que sea más generoso
    assert abs(len(attachment.pdf_bytes) - objetivo) <= margen


@pytest.mark.parametrize("size_kb", [5, 20, 50])
def test_con_poco_texto_no_hay_recorte_y_sobreviven_los_dos_marcadores(size_kb) -> None:
    """A 5, 20 y 50 KB el texto del PDF no llega a los 60.000 caracteres de
    `MAX_TEXT_CHARS`: el servicio no tiene que recortar nada."""
    attachment = a.build_attachment(size_kb)
    reporte = a.local_truncation_report(attachment)
    assert reporte["chars_del_pdf"] < MAX_TEXT_CHARS
    assert reporte["truncado"] is False
    assert reporte["marcador_inicio_sobrevive"] is True
    assert reporte["marcador_final_sobrevive"] is True
    assert reporte["avisos"] == []


def test_con_100kb_el_texto_se_recorta_y_el_marcador_final_se_pierde() -> None:
    """A 100 KB el texto del PDF supera los 60.000 caracteres: el servicio lo recorta
    a `MAX_TEXT_CHARS` y, como el marcador final es literalmente lo último del texto,
    es el que se pierde. El de inicio, al principio, sobrevive siempre."""
    attachment = a.build_attachment(100)
    reporte = a.local_truncation_report(attachment)
    assert reporte["chars_del_pdf"] > MAX_TEXT_CHARS
    assert reporte["truncado"] is True
    assert reporte["chars_enviados_al_modelo"] == MAX_TEXT_CHARS
    assert reporte["marcador_inicio_sobrevive"] is True
    assert reporte["marcador_final_sobrevive"] is False
    assert reporte["avisos"]  # el extractor real avisa del recorte


def test_el_marcador_final_es_lo_ultimo_del_texto_incrustado() -> None:
    """Condición necesaria para que el test anterior pruebe lo que dice probar: si el
    marcador final no fuera lo último del texto, "se pierde al recortar" no diría nada
    sobre DÓNDE cae el corte."""
    attachment = a.build_attachment(100)
    assert attachment.embedded_text.endswith(a.MARKER_END)


def test_size_kb_cero_no_se_calibra_como_adjunto() -> None:
    """El punto de 0 KB es la línea base sin adjunto: el runner HTTP no llama a
    `build_attachment` para él (lo comprueba no generando nada, no corriendo el
    servicio: ver `correr_punto`, que solo construye el adjunto si `size_kb > 0`)."""
    assert a.SIZE_POINTS_KB[0] == 0
