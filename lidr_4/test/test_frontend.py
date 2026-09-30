"""Tests del cliente HTTP de `streamlit_app.py`.

El módulo de Streamlit importa `st` pero solo lo usa dentro de `main()`, que
Streamlit ejecuta como `__main__`. Importar el módulo desde pytest define las
funciones puras (`_estimar`, `_detail_texto`) y no ejecuta nada de la UI: por
eso estos tests corren sin runtime de Streamlit y sin red.

`_estimar` recibe un `client` inyectado (httpx.MockTransport), así que el test
no necesita servidor. Lo que se verifica es el contrato del cliente contra el
de `EstimationRequest`/`EstimationResponse` del backend, que es la parte que
se rompe en silencio: un `payload` con una clave mal escrita no falla en el
cliente, aparece como un 422 del servidor con un mensaje que no ajuda.
"""

from __future__ import annotations

import httpx
import pytest

from streamlit_app import _ApiError, _detail_texto, _estimar

PAYLOAD_VALIDO = {
    "description": "A small B2B SaaS to manage employee equipment loans across teams.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}


def _mock_transport(*, status: int = 200, cuerpo: object) -> httpx.MockTransport:
    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=cuerpo)

    return httpx.MockTransport(_handler)


def _cliente(transport: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=transport)


def test_devuelve_text_y_prompt_version():
    transport = _mock_transport(cuerpo={"text": "3 fases, 8 semanas", "prompt_version": "v1"})

    texto, version = _estimar("http://api:8001", PAYLOAD_VALIDO, _cliente(transport))

    assert texto == "3 fases, 8 semanas"
    assert version == "v1"


def test_el_endpoint_es_estimate_y_no_stream():
    """El endpoint de `lidr_4` no hace streaming: `lidr_3` usaba
    `/api/v1/estimate/stream`. Si esto falla, el cliente volvió atrás."""
    vistos: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        vistos.append(request.url.path)
        return httpx.Response(200, json={"text": "x", "prompt_version": "v1"})

    _estimar("http://api:8001", PAYLOAD_VALIDO, _cliente(httpx.MockTransport(_handler)))

    assert vistos == ["/api/v1/estimate"]


def test_el_payload_viaja_como_json_exacto():
    """Las cuatro claves del contrato, sin renombrar ni agregar."""
    cuerpos: list[dict] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        import json

        cuerpos.append(json.loads(request.content))
        return httpx.Response(200, json={"text": "x", "prompt_version": "v1"})

    _estimar("http://api:8001", PAYLOAD_VALIDO, _cliente(httpx.MockTransport(_handler)))

    assert cuerpos == [PAYLOAD_VALIDO]


def test_422_de_fastapi_llega_como_lista_de_msg():
    transport = _mock_transport(
        status=422,
        cuerpo={"detail": [{"msg": "String should have at least 20 characters"}]},
    )

    with pytest.raises(_ApiError) as exc:
        _estimar("http://api:8001", {"description": "corto"}, _cliente(transport))

    assert exc.value.status == 422
    assert "at least 20 characters" in _detail_texto(exc.value.detail)


def test_500_con_detail_texto_plano():
    transport = _mock_transport(status=500, cuerpo={"detail": "upstream timeout"})

    with pytest.raises(_ApiError) as exc:
        _estimar("http://api:8001", PAYLOAD_VALIDO, _cliente(transport))

    assert exc.value.status == 500
    assert _detail_texto(exc.value.detail) == "upstream timeout"


def test_body_que_no_es_json_no_revienta():
    """Un 502 de un proxy devuelve HTML. `respuesta.json()` lanza `ValueError`,
    y sin el fallback el cliente muestra un traceback en vez del error."""

    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>Bad Gateway</html>")

    with pytest.raises(_ApiError) as exc:
        _estimar("http://api:8001", PAYLOAD_VALIDO, _cliente(httpx.MockTransport(_handler)))

    assert exc.value.status == 502
    assert "Bad Gateway" in str(exc.value.detail)


def test_el_timeout_del_cliente_no_acota_la_llamada():
    """El backend puede tardar: la estimación es una llamada de inferencia, no
    un query. `timeout=None` deja que httpx espere en vez de cortar a los 5s."""
    timeouts: list[dict] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions["timeout"])
        return httpx.Response(200, json={"text": "x", "prompt_version": "v1"})

    _estimar("http://api:8001", PAYLOAD_VALIDO, _cliente(httpx.MockTransport(_handler)))

    # httpx expande `timeout=None` a las cuatro Dimensions.
    assert timeouts == [{"connect": None, "read": None, "write": None, "pool": None}]


def test_detail_texto_aplana_una_lista_de_msg():
    detalle = [{"msg": "primer error"}, {"msg": "segundo error"}]

    assert _detail_texto(detalle) == "primer error\nsegundo error"


def test_detail_texto_pasa_escalares():
    assert _detail_texto("texto plano") == "texto plano"
