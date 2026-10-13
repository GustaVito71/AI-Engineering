"""Tests del cliente de salida estructurada de `streamlit_app.py`.

Como test_frontend.py: sin runtime de Streamlit ni red. `_estimate_structured`
recibe un transporte simulado, y las funciones que arman lo que se muestra
(tabla, partidas, narrativa, equipo) son puras.
"""

from __future__ import annotations

import httpx
import pytest

from streamlit_app import (
    _ApiError,
    _estimate_structured,
    _format_number,
    _is_out_of_scope,
    _line_items,
    _narrative,
    _phase_rows,
    _structured_version_options,
    _team_summary,
    _version_options,
)

PAYLOAD = {
    "description": "Un SaaS B2B pequeño para gestionar el préstamo de equipos a empleados.",
    "project_type": "web_saas",
    "detail_level": "summary",
    "output_format": "phases_table",
}
ESTIMACION = {
    "phases": [
        {
            "name": "Descubrimiento",
            "summary": "Trabajo de la fase de Descubrimiento.",
            "duration_weeks": 1.0,
            "hours": 70,
            "cost_eur": 3950.0,
            "confidence_pct": 85,
            "assumptions": [],
            "risks": [],
        },
        {
            "name": "Implementación",
            "summary": "Trabajo de la fase de Implementación.",
            "duration_weeks": 2.5,
            "hours": 1320,
            "cost_eur": 82500.0,
            "confidence_pct": 70,
            "assumptions": [],
            "risks": [],
        },
    ],
    "team": [{"role": "Desarrollador", "headcount": 2}, {"role": "QA", "headcount": 1}],
    "totals": {"hours": 1390, "cost_eur": 86450.0, "duration_weeks": 3.5},
    "summary": "Estimación de prueba con las fases y el equipo indicados.",
    "confidence_pct": 70,
}


def _cliente(status: int, cuerpo: object, vistos: list | None = None) -> httpx.Client:
    def _handler(request: httpx.Request) -> httpx.Response:
        if vistos is not None:
            vistos.append(request)
        return httpx.Response(status, json=cuerpo)

    return httpx.Client(transport=httpx.MockTransport(_handler))


# --- Cliente HTTP ------------------------------------------------------------------------


def test_llama_al_endpoint_estructurado_y_devuelve_la_estimacion() -> None:
    vistos: list[httpx.Request] = []
    cuerpo = {"estimation": ESTIMACION, "prompt_version": "v3", "warnings": ["aviso"]}

    estimacion, version, avisos, cached = _estimate_structured(
        "http://api:8001", PAYLOAD, _cliente(200, cuerpo, vistos)
    )

    assert estimacion == ESTIMACION
    assert version == "v3"
    assert avisos == ["aviso"]
    assert cached is False  # una API sin el campo cuenta como no cacheada
    [request] = vistos
    assert request.url.path == "/api/v1/estimate/structured"
    assert request.url.params.get("prompt_version") is None


def test_manda_la_version_elegida() -> None:
    vistos: list[httpx.Request] = []
    cuerpo = {"estimation": ESTIMACION, "prompt_version": "v3"}

    _, _, avisos, _ = _estimate_structured(
        "http://api:8001", PAYLOAD, _cliente(200, cuerpo, vistos), prompt_version="v3"
    )

    assert vistos[0].url.params["prompt_version"] == "v3"
    assert avisos == []  # sin el campo, lista vacía


def test_un_error_de_la_api_llega_con_su_codigo_y_detalle() -> None:
    detalle = "La versión de prompt 'v2' es de texto libre: pedila en POST /api/v1/estimate."
    with pytest.raises(_ApiError) as error:
        _estimate_structured("http://api:8001", PAYLOAD, _cliente(422, {"detail": detalle}))
    assert error.value.status == 422
    assert error.value.detail == detalle


# --- Selector de versión -------------------------------------------------------------------


def test_el_selector_estructurado_usa_las_versiones_estructuradas() -> None:
    health = {
        "prompt_version": "v2",
        "prompt_versions": ["v1", "v2"],
        "structured_prompt_version": "v3",
        "structured_prompt_versions": ["v3"],
    }
    assert _structured_version_options(health) == [
        ("Predeterminada (v3)", None),
        ("v3", "v3"),
    ]
    # El selector de texto no cambia.
    assert _version_options(health) == [
        ("Predeterminada (v2)", None),
        ("v1", "v1"),
        ("v2", "v2"),
    ]


def test_una_api_sin_salida_estructurada_deja_solo_la_predeterminada() -> None:
    assert _structured_version_options({"prompt_version": "v2"}) == [("Predeterminada", None)]
    assert _structured_version_options(None) == [("Predeterminada", None)]


# --- Lo que se muestra --------------------------------------------------------------------


def test_numeros_en_formato_castellano() -> None:
    assert _format_number(86450.0) == "86.450"
    assert _format_number(1320) == "1.320"
    assert _format_number(2.5) == "2,5"
    assert _format_number(1234.56) == "1.234,56"


def test_tabla_de_fases() -> None:
    filas = _phase_rows(ESTIMACION)
    # La descripción va al final: es larga y taparía las cifras.
    assert list(filas[0]) == [
        "Fase",
        "Semanas",
        "Horas",
        "Coste (EUR)",
        "Confianza (%)",
        "Descripción",
    ]
    assert filas == [
        {
            "Fase": "Descubrimiento",
            "Descripción": "Trabajo de la fase de Descubrimiento.",
            "Semanas": "1",
            "Horas": "70",
            "Coste (EUR)": "3.950",
            "Confianza (%)": 85,
        },
        {
            "Fase": "Implementación",
            "Descripción": "Trabajo de la fase de Implementación.",
            "Semanas": "2,5",
            "Horas": "1.320",
            "Coste (EUR)": "82.500",
            "Confianza (%)": 70,
        },
    ]


def test_partidas_numeradas() -> None:
    assert _line_items(ESTIMACION) == [
        (
            "1. Descubrimiento — 1 semana — 70 horas — 3.950 EUR (confianza: 85 %). "
            "Trabajo de la fase de Descubrimiento."
        ),
        (
            "2. Implementación — 2,5 semanas — 1.320 horas — 82.500 EUR (confianza: 70 %). "
            "Trabajo de la fase de Implementación."
        ),
    ]


def test_narrativa_un_parrafo_por_fase() -> None:
    [primero, segundo] = _narrative(ESTIMACION)
    assert primero.startswith(
        "**Descubrimiento.** Trabajo de la fase de Descubrimiento. Dura 1 semana, con 70 horas"
    )
    assert "2,5 semanas" in segundo


def test_resumen_de_equipo() -> None:
    assert _team_summary(ESTIMACION) == "Desarrollador × 2, QA × 1"


def test_devuelve_cached_cuando_la_api_lo_informa() -> None:
    cuerpo = {"estimation": ESTIMACION, "prompt_version": "v3", "cached": True}
    *_, cached = _estimate_structured("http://api:8001", PAYLOAD, _cliente(200, cuerpo))
    assert cached is True


def test_reconoce_un_rechazo_por_el_prefijo_del_resumen() -> None:
    """El mismo prefijo que valida el servicio (OUT_OF_SCOPE_PREFIX del schema)."""
    from app.schemas.structured_estimation import OUT_OF_SCOPE_PREFIX
    from streamlit_app import OUT_OF_SCOPE_PREFIX as PREFIJO_DEL_FRONTEND

    assert PREFIJO_DEL_FRONTEND == OUT_OF_SCOPE_PREFIX
    assert _is_out_of_scope({"summary": "Fuera de alcance: falta el sector."})
    assert not _is_out_of_scope(ESTIMACION)
