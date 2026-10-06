"""Tests del cliente HTTP de `streamlit_app.py`.

El módulo de Streamlit importa `st` pero solo lo usa dentro de `main()`, que
Streamlit ejecuta como `__main__`. Importar el módulo desde pytest define las
funciones puras (`_estimate`, `_detail_text`, `_length_error`) y no
ejecuta nada de la UI: por eso estos tests corren sin runtime de Streamlit y
sin red.

`_estimate` recibe un `client` inyectado (httpx.MockTransport), así que el test
no necesita servidor. Lo que se verifica es el contrato del cliente contra el
de `EstimationRequest`/`EstimationResponse` del backend, que es la parte que
se rompe en silencio: un `payload` con una clave mal escrita no falla en el
cliente, aparece como un 422 del servidor con un mensaje que no ayuda.
"""

from __future__ import annotations

import httpx
import pytest

from streamlit_app import (
    DESCRIPTION_MAX_CHARS,
    DESCRIPTION_MIN_CHARS,
    _ApiError,
    _detail_text,
    _estimate,
    _length_error,
    _read_health,
    _version_options,
)

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

    texto, version, avisos = _estimate("http://api:8001", PAYLOAD_VALIDO, _cliente(transport))

    assert texto == "3 fases, 8 semanas"
    assert version == "v1"
    assert avisos == []  # una API sin el campo `warnings` sigue funcionando


def test_devuelve_los_avisos_de_la_api():
    aviso = "El modelo de respaldo no está disponible porque falta ANTHROPIC_API_KEY."
    transport = _mock_transport(
        cuerpo={"text": "3 fases, 8 semanas", "prompt_version": "v1", "warnings": [aviso]}
    )

    _texto, _version, avisos = _estimate("http://api:8001", PAYLOAD_VALIDO, _cliente(transport))

    assert avisos == [aviso]


def test_el_endpoint_es_estimate_y_no_stream():
    """El endpoint de `lidr_4` no hace streaming: `lidr_3` usaba
    `/api/v1/estimate/stream`. Si esto falla, el cliente volvió atrás."""
    vistos: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        vistos.append(request.url.path)
        return httpx.Response(200, json={"text": "x", "prompt_version": "v1"})

    _estimate("http://api:8001", PAYLOAD_VALIDO, _cliente(httpx.MockTransport(_handler)))

    assert vistos == ["/api/v1/estimate"]


def test_el_payload_viaja_como_json_exacto():
    """Las cuatro claves del contrato, sin renombrar ni agregar."""
    cuerpos: list[dict] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        import json

        cuerpos.append(json.loads(request.content))
        return httpx.Response(200, json={"text": "x", "prompt_version": "v1"})

    _estimate("http://api:8001", PAYLOAD_VALIDO, _cliente(httpx.MockTransport(_handler)))

    assert cuerpos == [PAYLOAD_VALIDO]


def test_422_de_fastapi_llega_como_lista_de_msg():
    transport = _mock_transport(
        status=422,
        cuerpo={"detail": [{"msg": "String should have at least 20 characters"}]},
    )

    with pytest.raises(_ApiError) as exc:
        _estimate("http://api:8001", {"description": "corto"}, _cliente(transport))

    assert exc.value.status == 422
    assert "at least 20 characters" in _detail_text(exc.value.detail)


def test_500_con_detail_texto_plano():
    transport = _mock_transport(status=500, cuerpo={"detail": "upstream timeout"})

    with pytest.raises(_ApiError) as exc:
        _estimate("http://api:8001", PAYLOAD_VALIDO, _cliente(transport))

    assert exc.value.status == 500
    assert _detail_text(exc.value.detail) == "upstream timeout"


def test_body_que_no_es_json_no_revienta():
    """Un 502 de un proxy devuelve HTML. `respuesta.json()` lanza `ValueError`,
    y sin el fallback el cliente muestra un traceback en vez del error."""

    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>Bad Gateway</html>")

    with pytest.raises(_ApiError) as exc:
        _estimate("http://api:8001", PAYLOAD_VALIDO, _cliente(httpx.MockTransport(_handler)))

    assert exc.value.status == 502
    assert "Bad Gateway" in str(exc.value.detail)


def test_el_timeout_del_cliente_no_acota_la_llamada():
    """El backend puede tardar: la estimación es una llamada de inferencia, no
    un query. `timeout=None` deja que httpx espere en vez de cortar a los 5s."""
    timeouts: list[dict] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions["timeout"])
        return httpx.Response(200, json={"text": "x", "prompt_version": "v1"})

    _estimate("http://api:8001", PAYLOAD_VALIDO, _cliente(httpx.MockTransport(_handler)))

    # httpx expande `timeout=None` a las cuatro Dimensions.
    assert timeouts == [{"connect": None, "read": None, "write": None, "pool": None}]


def test_detail_texto_aplana_una_lista_de_msg():
    detalle = [{"msg": "primer error"}, {"msg": "segundo error"}]

    assert _detail_text(detalle) == "primer error\nsegundo error"


def test_detail_texto_pasa_escalares():
    assert _detail_text("texto plano") == "texto plano"


# --- Longitud de la descripción ---------------------------------------------------


@pytest.mark.parametrize("largo", [DESCRIPTION_MIN_CHARS, 500, DESCRIPTION_MAX_CHARS])
def test_longitud_dentro_de_los_limites_no_da_error(largo):
    assert _length_error("x" * largo) is None


def test_descripcion_corta_avisa_el_minimo():
    error = _length_error("x" * (DESCRIPTION_MIN_CHARS - 1))
    assert error is not None
    assert f"al menos {DESCRIPTION_MIN_CHARS}" in error
    assert f"tiene {DESCRIPTION_MIN_CHARS - 1}" in error


def test_descripcion_larga_avisa_el_maximo():
    """Un texto pegado de más de 2000 caracteres llega entero al formulario (sin
    `max_chars`) y el usuario ve cuánto sobra, en vez de un campo vacío."""
    error = _length_error("x" * 3150)
    assert error is not None
    assert f"hasta {DESCRIPTION_MAX_CHARS}" in error
    assert "tiene 3150" in error


# --- Versión del prompt -----------------------------------------------------------


def _transport_que_registra(urls: list[httpx.URL]) -> httpx.MockTransport:
    def _handler(request: httpx.Request) -> httpx.Response:
        urls.append(request.url)
        return httpx.Response(200, json={"text": "x", "prompt_version": "v1"})

    return httpx.MockTransport(_handler)


def test_con_version_elegida_va_en_la_query():
    urls: list[httpx.URL] = []
    _estimate("http://api:8001", PAYLOAD_VALIDO, _cliente(_transport_que_registra(urls)), "v1")
    assert urls[0].params["prompt_version"] == "v1"


def test_sin_version_elegida_no_se_manda_el_parametro():
    """Predeterminada: decide PROMPT_VERSION en el servicio."""
    urls: list[httpx.URL] = []
    _estimate("http://api:8001", PAYLOAD_VALIDO, _cliente(_transport_que_registra(urls)))
    assert "prompt_version" not in urls[0].params


def test_opciones_salen_de_health():
    health = {"prompt_version": "v2", "prompt_versions": ["v1", "v2"]}
    assert _version_options(health) == [
        ("Predeterminada (v2)", None),
        ("v1", "v1"),
        ("v2", "v2"),
    ]


@pytest.mark.parametrize("health", [None, {}], ids=["api-caida", "health-vacio"])
def test_sin_health_queda_solo_la_predeterminada(health):
    assert _version_options(health) == [("Predeterminada", None)]


def test_health_de_una_api_anterior_sin_prompt_versions():
    """Una API sin `prompt_versions` deja solo la predeterminada, con su versión."""
    assert _version_options({"prompt_version": "v1"}) == [("Predeterminada (v1)", None)]


def test_leer_health_devuelve_none_si_la_api_no_responde():
    def _caida(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sin conexión")

    assert _read_health("http://api:8001", _cliente(httpx.MockTransport(_caida))) is None


def test_leer_health_devuelve_none_si_no_es_json():
    transport = httpx.MockTransport(lambda _r: httpx.Response(502, text="<html>Bad Gateway</html>"))
    assert _read_health("http://api:8001", _cliente(transport)) is None
