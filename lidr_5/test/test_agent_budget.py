"""Tests de `app/agent/budget.py` (Paso 10): medir tokens y recortar por prioridad.

QUÉ SE COMPRUEBA

- `count_tokens` mide con LiteLLM y, si el contador falla, usa la aproximación de respaldo.
- `truncate_to_tokens` deja el texto dentro del máximo y avisa del recorte; no toca lo que cabe.
- `fit_blocks` conserva las piezas protegidas, recorta primero las últimas de la lista y deja
  vacías las que ya no tienen presupuesto.
"""

from __future__ import annotations

import pytest

from app.agent import budget
from app.agent.budget import (
    TRUNCATION_MARK,
    Block,
    count_tokens,
    fit_blocks,
    truncate_to_tokens,
)

MODELO = "openai/gpt-4o-mini"
TEXTO_LARGO = "palabra " * 4000


def test_count_tokens_de_un_texto_vacio_es_cero() -> None:
    """Un texto vacío no cuesta nada ni llama al contador."""
    assert count_tokens("", MODELO) == 0


def test_count_tokens_mide_con_litellm() -> None:
    """Una frase corta da unos pocos tokens; un texto 100 veces más largo, muchos más."""
    corto = count_tokens("hola mundo", MODELO)
    assert 1 <= corto <= 5
    assert count_tokens("hola mundo " * 100, MODELO) > corto * 50


def test_count_tokens_usa_la_aproximacion_si_el_contador_falla(monkeypatch) -> None:
    """Medir tokens no debe romper un turno: sin contador, 4 caracteres por token."""

    def roto(**_kwargs):
        """Un contador que siempre falla."""
        raise RuntimeError("modelo desconocido")

    monkeypatch.setattr(budget.litellm, "token_counter", roto)
    assert count_tokens("a" * 40, "modelo/raro") == 10


def test_truncate_no_toca_un_texto_que_cabe() -> None:
    """Si ya cabe, se devuelve igual y sin marcar recorte."""
    assert truncate_to_tokens("hola", 100, MODELO) == ("hola", False)


def test_truncate_deja_el_texto_dentro_del_maximo_con_el_aviso() -> None:
    """El texto recortado cabe en el máximo, termina con el aviso y es más corto que el original."""
    recortado, se_recorto = truncate_to_tokens(TEXTO_LARGO, 100, MODELO)
    assert se_recorto
    assert count_tokens(recortado, MODELO) <= 100
    assert recortado.endswith(TRUNCATION_MARK)
    assert len(recortado) < len(TEXTO_LARGO)


@pytest.mark.parametrize("maximo", [0, -5])
def test_truncate_con_maximo_cero_o_negativo_deja_vacio(maximo) -> None:
    """Sin presupuesto no queda nada, pero se avisa de que se recortó."""
    assert truncate_to_tokens("algo de texto", maximo, MODELO) == ("", True)


def test_fit_conserva_lo_protegido_y_recorta_primero_lo_ultimo() -> None:
    """Con poco presupuesto, los adjuntos se recortan antes que la transcripción; las
    herramientas, que van al final, se llevan lo que sobre (aquí nada).
    """
    resultados = fit_blocks(
        [
            Block("transcripcion", "descripcion del proyecto " * 20, protected=True),
            Block("adjuntos", TEXTO_LARGO),
            Block("herramientas", TEXTO_LARGO),
        ],
        budget=600,
        model=MODELO,
    )
    por_nombre = {r.name: r for r in resultados}
    assert not por_nombre["transcripcion"].truncated
    assert por_nombre["transcripcion"].text.startswith("descripcion del proyecto")
    assert por_nombre["adjuntos"].truncated
    assert por_nombre["herramientas"].text == ""
    assert por_nombre["herramientas"].truncated
    total = sum(r.tokens for r in resultados)
    assert total <= 600


def test_fit_no_recorta_nada_si_todo_cabe() -> None:
    """Con presupuesto de sobra, ninguna pieza se toca."""
    resultados = fit_blocks(
        [Block("a", "uno dos tres"), Block("b", "cuatro cinco")],
        budget=10_000,
        model=MODELO,
    )
    assert [r.truncated for r in resultados] == [False, False]
    assert [r.text for r in resultados] == ["uno dos tres", "cuatro cinco"]


def test_fit_descuenta_la_reserva_del_prompt_del_sistema() -> None:
    """La reserva reduce lo disponible: con ella, una pieza que antes cabía se recorta."""
    bloques = [Block("adjuntos", TEXTO_LARGO)]
    necesarios = count_tokens(TEXTO_LARGO, MODELO)
    sin_reserva = fit_blocks(bloques, budget=necesarios + 10, model=MODELO)
    con_reserva = fit_blocks(bloques, budget=necesarios + 10, model=MODELO, reserved=500)
    assert not sin_reserva[0].truncated
    assert con_reserva[0].truncated


def test_fit_conserva_lo_protegido_aunque_solo_ya_supere_el_presupuesto() -> None:
    """Lo protegido nunca se recorta: si ya pasa el presupuesto, lo demás queda vacío."""
    resultados = fit_blocks(
        [Block("transcripcion", TEXTO_LARGO, protected=True), Block("adjuntos", "algo")],
        budget=50,
        model=MODELO,
    )
    assert resultados[0].text == TEXTO_LARGO
    assert resultados[1].text == ""
