"""Tests del backend de negocio simulado (`business_backend/main.py`, Paso 12).

QUÉ SE COMPRUEBA: autenticación por `X-API-Key`, el tope de resultados, que el ranking
priorice coincidencias y que falte `query` dé un error claro. Todo en memoria (TestClient),
sin levantar el contenedor.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "business_backend"))


@pytest.fixture
def cliente(monkeypatch):
    """Un `TestClient` del backend simulado, con una clave fija conocida por el test."""
    monkeypatch.setenv("BUSINESS_API_KEY", "clave-de-test")
    import main as business_main

    importlib.reload(business_main)
    return TestClient(business_main.app)


def test_sin_clave_responde_401(cliente) -> None:
    """Sin cabecera `X-API-Key` el backend rechaza la consulta."""
    r = cliente.get("/projects/similar", params={"query": "api"})
    assert r.status_code == 401


def test_con_clave_incorrecta_responde_401(cliente) -> None:
    """Con una clave distinta a la configurada, también rechaza."""
    r = cliente.get("/projects/similar", params={"query": "api"}, headers={"X-API-Key": "otra"})
    assert r.status_code == 401


def test_con_clave_correcta_devuelve_resultados(cliente) -> None:
    """Con la clave correcta devuelve una lista de proyectos con los campos esperados."""
    r = cliente.get(
        "/projects/similar",
        params={"query": "migración de API"},
        headers={"X-API-Key": "clave-de-test"},
    )
    assert r.status_code == 200
    resultados = r.json()["results"]
    assert 1 <= len(resultados) <= 3
    primero = resultados[0]
    assert set(primero) == {
        "project_type",
        "hours_estimated",
        "hours_actual",
        "deviation_pct",
        "risks_materialized",
    }


def test_el_mejor_match_queda_primero(cliente) -> None:
    """Una consulta que menciona el tipo exacto lo trae primero."""
    r = cliente.get(
        "/projects/similar",
        params={"query": "e-commerce"},
        headers={"X-API-Key": "clave-de-test"},
    )
    assert r.json()["results"][0]["project_type"] == "e-commerce"


def test_respeta_max_results(cliente) -> None:
    """`max_results=1` devuelve como máximo un proyecto."""
    r = cliente.get(
        "/projects/similar",
        params={"query": "proyecto", "max_results": 1},
        headers={"X-API-Key": "clave-de-test"},
    )
    assert len(r.json()["results"]) == 1


def test_max_results_tiene_un_tope_absoluto(cliente) -> None:
    """Pedir más de `MAX_RESULTS_CAP` (5) se rechaza, no se trunca en silencio."""
    r = cliente.get(
        "/projects/similar",
        params={"query": "proyecto", "max_results": 50},
        headers={"X-API-Key": "clave-de-test"},
    )
    assert r.status_code == 422


def test_sin_query_responde_422(cliente) -> None:
    """Falta el parámetro obligatorio `query`."""
    r = cliente.get("/projects/similar", headers={"X-API-Key": "clave-de-test"})
    assert r.status_code == 422


def test_health_no_requiere_clave(cliente) -> None:
    """`/health` no pide autenticación: solo confirma que el proceso arrancó."""
    assert cliente.get("/health").status_code == 200
