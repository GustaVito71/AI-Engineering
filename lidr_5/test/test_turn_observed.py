"""Tests del evento `turn_observed`: un único registro del log por turno, con todo lo que lo describe.

QUÉ SE COMPRUEBA

- Cada turno exitoso de `POST /sessions/{id}/estimate` emite exactamente un `turn_observed`.
- `turn_index` cuenta los turnos de la sesión (1, 2, 3...) y `session_id` es el de la sesión.
- La ventana, las anclas y el resumen que se registran son los de después del turno.
- Los caracteres de los adjuntos suman a `enriched_transcript_chars` y se reportan aparte.
- `cache_hit_kind` es siempre "none": el camino conversacional no usa caché.
- El evento lleva cantidades, nunca el contenido de la conversación ni de los adjuntos.

Reutiliza la fixture `entorno` y el helper `estimar` de `test_sessions_estimate_endpoint.py`:
sin red ni Redis, con el wrapper falso.
"""

from __future__ import annotations

from structlog.testing import capture_logs
from test_sessions_estimate_endpoint import TRANSCRIPCION, entorno, estimar  # noqa: F401

# Nombre del evento que se busca en los logs capturados.
EVENTO = "turn_observed"


def _eventos(logs: list[dict]) -> list[dict]:
    """Solo los registros `turn_observed` de la lista de logs capturados."""
    return [registro for registro in logs if registro.get("event") == EVENTO]


def test_cada_turno_emite_un_evento(entorno) -> None:
    """Un turno exitoso deja exactamente un `turn_observed` con los campos esperados."""
    cliente, _wrapper, sid = entorno
    with capture_logs() as logs:
        assert estimar(cliente, sid).status_code == 200

    eventos = _eventos(logs)
    assert len(eventos) == 1
    evento = eventos[0]
    assert evento["turn_index"] == 1
    assert evento["session_id"] == sid
    assert evento["cache_hit_kind"] == "none"
    assert evento["last_resolved_tier"] == "default"
    # Tras el primer turno la ventana tiene el par usuario + asistente, sin anclas ni resumen.
    assert evento["messages_in_window"] == 2
    assert evento["anchors_count"] == 0
    assert evento["summary_chars"] == 0
    assert evento["tokens_in"] >= 0 and evento["tokens_out"] >= 0
    assert evento["cost_usd"] >= 0
    assert evento["latency_ms"] >= 0


def test_el_indice_avanza_turno_a_turno(entorno) -> None:
    """Tres turnos seguidos numeran 1, 2 y 3, con la misma sesión y la ventana creciendo."""
    cliente, _wrapper, sid = entorno
    with capture_logs() as logs:
        for _ in range(3):
            assert estimar(cliente, sid).status_code == 200

    eventos = _eventos(logs)
    assert [e["turn_index"] for e in eventos] == [1, 2, 3]
    assert {e["session_id"] for e in eventos} == {sid}
    assert [e["messages_in_window"] for e in eventos] == [2, 4, 6]


def test_la_ventana_se_recorta_pero_el_indice_sigue(entorno) -> None:
    """Con más turnos que la ventana (MAX_TURNS=6), la ventana se queda en 12 mensajes y
    el índice del turno sigue creciendo: el contador no depende de lo que se recorta.
    """
    cliente, _wrapper, sid = entorno
    with capture_logs() as logs:
        for _ in range(7):
            assert estimar(cliente, sid).status_code == 200

    ultimo = _eventos(logs)[-1]
    assert ultimo["turn_index"] == 7
    assert ultimo["messages_in_window"] == 12


def test_adjuntos_suman_en_los_caracteres_del_turno(entorno) -> None:
    """El texto del adjunto aparece en `attachments_total_chars` y se suma a la transcripción."""
    cliente, _wrapper, sid = entorno
    texto = "alcance-de-prueba"
    with capture_logs() as logs:
        assert estimar(cliente, sid, archivos=[("a.txt", texto.encode())]).status_code == 200

    evento = _eventos(logs)[0]
    assert evento["attachments_total_chars"] == len(texto)
    assert evento["enriched_transcript_chars"] == len(TRANSCRIPCION) + len(texto)


def test_sin_adjuntos_el_total_es_cero(entorno) -> None:
    """Sin adjuntos, `attachments_total_chars` vale 0 y el turno cuenta solo la transcripción."""
    cliente, _wrapper, sid = entorno
    with capture_logs() as logs:
        assert estimar(cliente, sid).status_code == 200

    evento = _eventos(logs)[0]
    assert evento["attachments_total_chars"] == 0
    assert evento["enriched_transcript_chars"] == len(TRANSCRIPCION)


def test_el_evento_no_lleva_contenido(entorno) -> None:
    """El `turn_observed` trae cantidades: ni la transcripción ni el texto de los adjuntos."""
    cliente, _wrapper, sid = entorno
    with capture_logs() as logs:
        assert (
            estimar(cliente, sid, archivos=[("a.txt", b"secreto-del-adjunto")]).status_code == 200
        )

    eventos = str(_eventos(logs))
    assert "secreto-del-adjunto" not in eventos
    assert TRANSCRIPCION not in eventos
