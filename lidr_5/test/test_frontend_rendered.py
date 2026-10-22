"""Tests del cliente de salida renderizada de `streamlit_app.py`.

Las funciones puras se prueban como en test_frontend_structured.py. El selector único
de salida y versión (v2 texto, v3 estructurada, v4 renderizada) se prueba además con el
runtime de pruebas de Streamlit (`AppTest`): que cada versión cambie el modo depende de
cómo se arma la pantalla, y eso solo se ve ejecutando la app.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from streamlit_app import (
    OUTPUT_CHOICES,
    _estimate_rendered,
    _output_choices,
    _rendered_version_options,
)

APP = Path(__file__).resolve().parent.parent / "streamlit_app.py"
PAYLOAD = {
    "description": "Un SaaS B2B pequeño para gestionar el préstamo de equipos a empleados.",
    "project_type": "web_saas",
    "detail_level": "summary",
    "output_format": "narrative",
}
ESTIMACION = {"summary": "Resumen.", "rendered": "**Diseño.** Texto."}


def _cliente(status: int, cuerpo: object, vistos: list | None = None) -> httpx.Client:
    """Crea un cliente HTTP con un servidor falso que responde con el estado y cuerpo dados
    y anota las peticiones recibidas.
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        """Anota la petición recibida (si se pidió) y responde con el estado y el cuerpo
        preparados.
        """
        if vistos is not None:
            vistos.append(request)
        return httpx.Response(status, json=cuerpo)

    return httpx.Client(transport=httpx.MockTransport(_handler))


# --- Cliente HTTP ------------------------------------------------------------------------


def test_llama_al_endpoint_renderizado() -> None:
    """El cliente llama al endpoint de salida renderizada y devuelve estimación, versión,
    avisos e indicador de caché.
    """
    vistos: list[httpx.Request] = []
    cuerpo = {"estimation": ESTIMACION, "prompt_version": "v4", "cached": True}

    estimacion, version, avisos, cached = _estimate_rendered(
        "http://api:8001", PAYLOAD, _cliente(200, cuerpo, vistos), prompt_version="v4"
    )

    assert (estimacion, version, avisos, cached) == (ESTIMACION, "v4", [], True)
    [request] = vistos
    assert request.url.path == "/api/v1/estimate/rendered"
    assert request.url.params["prompt_version"] == "v4"


# --- Selector de salida y versión -------------------------------------------------------


def test_sin_health_se_ofrecen_las_tres_versiones() -> None:
    """Si la API no respondió, el selector ofrece v2, v3 y v4 y el formulario funciona."""
    assert [version for version, _modo, _etiqueta in _output_choices(None)] == ["v2", "v3", "v4"]
    assert _output_choices(None) == list(OUTPUT_CHOICES)


def test_cada_version_pertenece_a_un_modo() -> None:
    """v2 es texto libre, v3 estructurada y v4 renderizada."""
    assert {version: modo for version, modo, _etiqueta in OUTPUT_CHOICES} == {
        "v2": "text",
        "v3": "structured",
        "v4": "rendered",
    }


def test_solo_se_ofrecen_las_versiones_que_informa_la_api() -> None:
    """Una versión que `/health` no lista para su modo no aparece en el selector."""
    health = {
        "prompt_versions": ["v1", "v2"],
        "structured_prompt_versions": ["v3", "v5"],
        "rendered_prompt_versions": [],
    }
    assert [version for version, _m, _e in _output_choices(health)] == ["v2", "v3"]


def test_un_health_sin_ninguna_de_las_tres_ofrece_todas() -> None:
    """Si el servicio no lista ninguna, no se deja el selector vacío."""
    assert _output_choices({"prompt_versions": ["v1"]}) == list(OUTPUT_CHOICES)


def test_el_selector_renderizado_usa_las_versiones_renderizadas() -> None:
    """El selector de versión de la salida renderizada ofrece las versiones renderizadas
    que informa `/health`.
    """
    health = {"rendered_prompt_version": "v4", "rendered_prompt_versions": ["v4"]}
    assert _rendered_version_options(health) == [("Predeterminada (v4)", None), ("v4", "v4")]
    assert _rendered_version_options(None) == [("Predeterminada", None)]


def _app() -> AppTest:
    # /health ya leído: la app no consulta la API y arma los selectores con esto.
    """Crea la pantalla de Streamlit para pruebas, con la respuesta de `/health` ya cargada
    (sin consultar la API).
    """
    app = AppTest.from_file(str(APP), default_timeout=10)
    app.session_state["health_de"] = "http://localhost:8001"
    app.session_state["health"] = {
        "prompt_version": "v2",
        "prompt_versions": ["v1", "v2"],
        "structured_prompt_version": "v3",
        "structured_prompt_versions": ["v3"],
        "rendered_prompt_version": "v4",
        "rendered_prompt_versions": ["v4"],
    }
    return app.run()


def _selector(app: AppTest):
    """El selector de salida y versión: el primero de la pestaña «Estimación»."""
    return app.tabs[0].selectbox[0]


def test_en_la_app_el_selector_ofrece_v2_v3_y_v4() -> None:
    """En la pantalla hay un único selector con las tres versiones, empezando por v2, y
    ya no hay interruptores.
    """
    app = _app()
    assert not app.exception
    assert _selector(app).options == [etiqueta for _v, _m, etiqueta in OUTPUT_CHOICES]
    assert _selector(app).value == ("v2", "text", "Texto libre (v2)")
    # Solo la pestaña «Estimación»: la de conversación tiene su propio interruptor ACB,
    # que aparece si la API local responde (por eso no se cuenta en toda la app).
    assert len(app.tabs[0].toggle) == 0


def test_elegir_una_version_cambia_el_modo_de_salida() -> None:
    """Elegir v3 o v4 en el selector deja esa versión como la elegida."""
    app = _app()
    _selector(app).set_value(OUTPUT_CHOICES[1]).run()
    assert app.session_state["output_choice"][:2] == ("v3", "structured")
    _selector(app).set_value(OUTPUT_CHOICES[2]).run()
    assert app.session_state["output_choice"][:2] == ("v4", "rendered")
    assert not app.exception


@pytest.mark.parametrize(
    ("indice", "ruta", "version"),
    [
        (0, "/api/v1/estimate", "v2"),
        (1, "/api/v1/estimate/structured", "v3"),
        (2, "/api/v1/estimate/rendered", "v4"),
    ],
)
def test_el_envio_va_al_endpoint_de_la_version_elegida(monkeypatch, indice, ruta, version) -> None:
    """Cada versión del selector envía el formulario a su endpoint, con esa versión explícita."""
    vistos: list[tuple[str, dict]] = []

    def post(_cliente, url, **kwargs):
        """Registra la petición y responde 422 (basta para saber a dónde fue). La pestaña
        dinámica también crea su sesión al cargar: esa llamada no interesa acá.
        """
        if not url.endswith("/sessions"):
            vistos.append((url, kwargs.get("params") or {}))
        return httpx.Response(422, json={"detail": "prueba"})

    monkeypatch.setattr(httpx.Client, "post", post)
    app = _app()
    _selector(app).set_value(OUTPUT_CHOICES[indice]).run()
    app.text_area(key="description_entrada").set_value("Un SaaS B2B para gestionar préstamos.")
    app.button[[b.label for b in app.button].index("Estimar")].click()
    app.run()

    [(url, params)] = vistos
    assert url.endswith(ruta)
    assert params == {"prompt_version": version}
