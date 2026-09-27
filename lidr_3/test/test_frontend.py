"""Contract tests between the Streamlit frontend and the HTTP API.

The frontend's SSE parser is a pure function precisely so it can be checked
against the EXACT bytes the backend produces: we build the response body with
the router's own `_evento_sse` helper and feed `_parsear_sse` those lines.
A divergent rename (event name, JSON key) fails this test without needing a
running Streamlit runtime or a real network — httpx.MockTransport stands in
for the server.
"""

from __future__ import annotations

from collections.abc import Iterable

import httpx
import pytest
from httpx._content import IteratorByteStream

from app.routers.estimations import _evento_sse
from streamlit_app import (
    DEFAULT_API_BASE,
    _ApiError,
    _parsear_sse,
    _resumen,
    _stream_deltas,
)


def _sse_cuerpo(*eventos: tuple[str, dict]) -> str:
    """Construye el body SSE igual que lo haría el router (event/data en pares)."""
    return "".join(_evento_sse(tipo, datos) for tipo, datos in eventos)


def test_parsear_sse_entiende_el_formato_exacto_del_router() -> None:
    """El parser del front entiende lo que `_evento_sse` del router emite."""
    cuerpo = _sse_cuerpo(
        ("meta", {"proveedor": "openai", "camino": "primario"}),
        ("delta", {"texto": "ho"}),
        ("delta", {"texto": "la"}),
        (
            "estimation",
            {
                "estimation": "hola",
                "truncated": False,
                "model": "gpt-4o-mini",
                "provider": "openai",
                "used_fallback": False,
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "cost_usd": 0.00012,
                "cost_note": None,
            },
        ),
    )
    pares = list(_parsear_sse(cuerpo.splitlines()))
    assert [tipo for tipo, _ in pares] == ["meta", "delta", "delta", "estimation"]
    assert pares[0][1] == {"proveedor": "openai", "camino": "primario"}
    assert pares[2][1] == {"texto": "la"}
    assert pares[3][1]["model"] == "gpt-4o-mini"
    assert pares[3][1]["cost_usd"] == 0.00012


def test_stream_deltas_consume_un_stream_real_del_endpoint() -> None:
    """Un miss real: meta, deltas y estimation se traducen a texto + captura."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=_sse_cuerpo(
                ("meta", {"proveedor": "openai", "camino": "primario"}),
                ("delta", {"texto": "3 "}),
                ("delta", {"texto": "meses"}),
                (
                    "estimation",
                    {
                        "estimation": "3 meses",
                        "truncated": False,
                        "model": "gpt-4o-mini",
                        "provider": "openai",
                        "used_fallback": False,
                        "usage": {"input_tokens": 9, "output_tokens": 4},
                        "cost_usd": 0.0001,
                        "cost_note": None,
                    },
                ),
            ),
            headers={"content-type": "text/event-stream"},
        )
    )
    cliente = httpx.Client(transport=transport, base_url="http://test")
    captura: dict = {}
    deltas = list(_stream_deltas(DEFAULT_API_BASE, "x" * 60, captura, client=cliente))
    assert "".join(deltas) == "3 meses"
    assert captura["meta"] == {"proveedor": "openai", "camino": "primario"}
    assert captura["final"]["estimation"] == "3 meses"


def test_stream_deltas_eleva_api_error_en_502() -> None:
    """Fallo pre-primer-byte (ambos proveedores): el front ve el status real.

    El body viaja como STREAM sin leer (igual que en el cable real): httpx
    lanza ResponseNotRead si el cliente accede a .json() sin read() previo.
    MockTransport con `json=` entrega el body ya leído y NO reproduce ese
    bug; `stream=` (iterable de bytes) sí lo reproduce."""

    def _body() -> Iterable[bytes]:
        yield '{"detail": "No se pudo generar la estimación. Inténtalo de nuevo más tarde."}'.encode()

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            502,
            stream=IteratorByteStream(_body()),
            headers={"content-type": "application/json"},
        )
    )
    cliente = httpx.Client(transport=transport, base_url="http://test")
    with pytest.raises(_ApiError) as excinfo:
        list(_stream_deltas(DEFAULT_API_BASE, "x" * 60, {}, client=cliente))
    assert excinfo.value.status == 502
    assert "Inténtalo de nuevo" in str(excinfo.value.detail)


def test_stream_deltas_eleva_api_error_en_evento_error_a_mitad() -> None:
    """Muerte a mitad de stream: los deltas ya rendidos y luego _ApiError."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text=_sse_cuerpo(
                ("meta", {"proveedor": "openai", "camino": "primario"}),
                ("delta", {"texto": "3 "}),
                ("error", {"detail": "No se pudo completar la estimación. Inténtalo de nuevo."}),
            ),
            headers={"content-type": "text/event-stream"},
        )
    )
    cliente = httpx.Client(transport=transport, base_url="http://test")
    generador = _stream_deltas(DEFAULT_API_BASE, "x" * 60, {}, client=cliente)
    # El primer next() rinde el delta ya emitido antes de la muerte del stream...
    assert next(generador) == "3 "
    # ...y el siguiente next() encuentra el evento `error` del cierre.
    with pytest.raises(_ApiError) as excinfo:
        next(generador)
    assert excinfo.value.status == 502


def test_resumen_formatea_meta_y_costo() -> None:
    captura = {
        "meta": {"proveedor": "openai", "camino": "fallback"},
        "final": {
            "model": "claude-haiku-4-5",
            "used_fallback": True,
            "truncated": False,
            "cost_usd": 0.000123,
        },
    }
    resumen = _resumen(captura)
    assert "Camino: fallback" in resumen
    assert "claude-haiku-4-5" in resumen
    assert "Costo: $0.000123" in resumen
    assert "Truncado: no" in resumen
