"""Tests del cliente de salida renderizada de `streamlit_app.py`.

Las funciones puras se prueban como en test_frontend_structured.py. Los dos
interruptores se prueban además con el runtime de pruebas de Streamlit
(`AppTest`): que sean excluyentes depende de los callbacks, y eso solo se ve
ejecutando la app.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from streamlit_app import (
    RENDERED_TOGGLE,
    STRUCTURED_TOGGLE,
    _estimate_rendered,
    _output_mode,
    _rendered_version_options,
    _switch_off_other,
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
    def _handler(request: httpx.Request) -> httpx.Response:
        if vistos is not None:
            vistos.append(request)
        return httpx.Response(status, json=cuerpo)

    return httpx.Client(transport=httpx.MockTransport(_handler))


# --- Cliente HTTP ------------------------------------------------------------------------


def test_llama_al_endpoint_renderizado() -> None:
    vistos: list[httpx.Request] = []
    cuerpo = {"estimation": ESTIMACION, "prompt_version": "v4", "cached": True}

    estimacion, version, avisos, cached = _estimate_rendered(
        "http://api:8001", PAYLOAD, _cliente(200, cuerpo, vistos), prompt_version="v4"
    )

    assert (estimacion, version, avisos, cached) == (ESTIMACION, "v4", [], True)
    [request] = vistos
    assert request.url.path == "/api/v1/estimate/rendered"
    assert request.url.params["prompt_version"] == "v4"


# --- Interruptores -------------------------------------------------------------------


def test_encender_uno_apaga_el_otro() -> None:
    estado = {STRUCTURED_TOGGLE: True, RENDERED_TOGGLE: True}
    _switch_off_other(estado, RENDERED_TOGGLE, STRUCTURED_TOGGLE)
    assert estado == {STRUCTURED_TOGGLE: False, RENDERED_TOGGLE: True}


def test_apagar_uno_no_toca_el_otro() -> None:
    estado = {STRUCTURED_TOGGLE: False, RENDERED_TOGGLE: False}
    _switch_off_other(estado, STRUCTURED_TOGGLE, RENDERED_TOGGLE)
    assert estado == {STRUCTURED_TOGGLE: False, RENDERED_TOGGLE: False}


@pytest.mark.parametrize(
    ("estado", "modo"),
    [
        ({}, "text"),
        ({STRUCTURED_TOGGLE: True}, "structured"),
        ({RENDERED_TOGGLE: True}, "rendered"),
        ({STRUCTURED_TOGGLE: False, RENDERED_TOGGLE: False}, "text"),
    ],
)
def test_modo_de_salida(estado, modo) -> None:
    assert _output_mode(estado) == modo


def test_el_selector_renderizado_usa_las_versiones_renderizadas() -> None:
    health = {"rendered_prompt_version": "v4", "rendered_prompt_versions": ["v4"]}
    assert _rendered_version_options(health) == [("Predeterminada (v4)", None), ("v4", "v4")]
    assert _rendered_version_options(None) == [("Predeterminada", None)]


def _app() -> AppTest:
    # /health ya leído: la app no consulta la API y arma los selectores con esto.
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


def _interruptores(app: AppTest) -> tuple[bool, bool]:
    return app.toggle(key=STRUCTURED_TOGGLE).value, app.toggle(key=RENDERED_TOGGLE).value


def _versiones(app: AppTest) -> list[str]:
    return list(app.sidebar.selectbox[0].options)


def test_en_la_app_los_interruptores_son_excluyentes() -> None:
    app = _app()
    assert _interruptores(app) == (False, False)
    assert _versiones(app) == ["Predeterminada (v2)", "v1", "v2"]

    app.toggle(key=STRUCTURED_TOGGLE).set_value(True).run()
    assert _interruptores(app) == (True, False)
    assert _versiones(app) == ["Predeterminada (v3)", "v3"]

    app.toggle(key=RENDERED_TOGGLE).set_value(True).run()
    assert _interruptores(app) == (False, True)
    assert _versiones(app) == ["Predeterminada (v4)", "v4"]

    app.toggle(key=STRUCTURED_TOGGLE).set_value(True).run()
    assert _interruptores(app) == (True, False)

    app.toggle(key=STRUCTURED_TOGGLE).set_value(False).run()
    assert _interruptores(app) == (False, False)
    assert _versiones(app) == ["Predeterminada (v2)", "v1", "v2"]
    assert not app.exception
