"""Slice vertical de la salida estructurada (WU8): del formulario a la tabla.

Como test_e2e.py, con todas las piezas reales:

    _build_payload + _estimate_structured (cliente de Streamlit)
      → FastAPI con su lifespan (Redis, caché)
      → EstimationRequest → render del prompt v3
      → LLMWrapper.estimate_structured → Instructor → Router de LiteLLM
      → StructuredResult validado → caché → StructuredEstimationResponse
      → _estimate_structured → _phase_rows

Lo único simulado es el proveedor (`mock_response` en cada llamada del Router,
una respuesta por intento) y Redis (fakeredis).
"""

from __future__ import annotations

import json

import litellm
import pytest
from fakeredis import FakeAsyncRedis, FakeServer
from fastapi.testclient import TestClient
from redis.asyncio import Redis
from structlog.testing import capture_logs

from app.config import get_settings
from app.main import create_app
from app.routers.estimations import INVALID_OUTPUT_MESSAGE
from streamlit_app import (
    _ApiError,
    _build_payload,
    _estimate_structured,
    _phase_rows,
    _read_health,
    _structured_version_options,
)

pytestmark = pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")

API = "http://testserver"
DESCRIPCION = "App móvil para reservar turnos en el gimnasio municipal, con avisos push."
ESTIMACION = {
    "phases": [
        {
            "name": "Diseño",
            "summary": "Trabajo de la fase de Diseño.",
            "duration_weeks": 2,
            "hours": 130,
            "cost_eur": 6500,
            "confidence_pct": 80,
            "assumptions": ["Marca y estilo ya definidos"],
            "risks": [],
        },
        {
            "name": "Implementación",
            "summary": "Trabajo de la fase de Implementación.",
            "duration_weeks": 6,
            "hours": 770,
            "cost_eur": 48150,
            "confidence_pct": 65,
            "assumptions": ["API de turnos existente"],
            "risks": [],
        },
    ],
    "team": [{"role": "Desarrollador", "headcount": 4}, {"role": "Diseñador", "headcount": 1}],
    "totals": {"hours": 900, "cost_eur": 54650, "duration_weeks": 8},
    "summary": "Estimación de prueba con las fases y el equipo indicados.",
    "confidence_pct": 70,
}
SIN_EQUIPO = json.dumps({**ESTIMACION, "team": []})


class _Proveedor:
    """Proveedor simulado detrás del Router real: una respuesta por llamada."""

    def __init__(self) -> None:
        self.respuestas: list[str | Exception] = [json.dumps(ESTIMACION)]
        self.llamadas: list[dict] = []

    def instalar(self, monkeypatch) -> None:
        original = litellm.Router.acompletion
        proveedor = self

        async def acompletion(router, **kwargs):
            proveedor.llamadas.append(kwargs)
            respuesta = proveedor.respuestas[len(proveedor.llamadas) - 1]
            return await original(router, **kwargs, mock_response=respuesta)

        monkeypatch.setattr(litellm.Router, "acompletion", acompletion)


@pytest.fixture
def proveedor(monkeypatch) -> _Proveedor:
    p = _Proveedor()
    p.instalar(monkeypatch)
    return p


@pytest.fixture
def stack(monkeypatch, proveedor):
    servidor = FakeServer()
    monkeypatch.setattr(
        Redis,
        "from_url",
        lambda *_a, **_k: FakeAsyncRedis(server=servidor, decode_responses=True),
    )

    def _arrancar(**entorno: str) -> TestClient:
        base = {
            "OPENAI_API_KEY": "sk-openai",
            "ANTHROPIC_API_KEY": "sk-anthropic",
            "REDIS_URL": "redis://redis-e2e:6379/0",
            "LLM_MAX_RETRIES": "0",
        }
        for variable, valor in {**base, **entorno}.items():
            monkeypatch.setenv(variable, valor)
        get_settings.cache_clear()
        return TestClient(create_app())

    yield _arrancar
    get_settings.cache_clear()


def _payload(detail_level: str = "medium") -> dict:
    return _build_payload(DESCRIPCION, "mobile_app", detail_level, "phases_table")


def test_del_formulario_a_la_tabla(stack, proveedor) -> None:
    with stack() as api, capture_logs() as logs:
        estimacion, version, avisos, cached = _estimate_structured(API, _payload(), api)

    assert estimacion["totals"] == {"hours": 900, "cost_eur": 54650.0, "duration_weeks": 8.0}
    assert version == "v3"
    assert avisos == []
    assert cached is False
    assert [f["Fase"] for f in _phase_rows(estimacion)] == ["Diseño", "Implementación"]

    # El proveedor recibió el prompt v3 y el schema como response_format.
    [llamada] = proveedor.llamadas
    sistema, usuario = (m["content"] for m in llamada["messages"])
    assert "un único objeto JSON" in sistema
    assert DESCRIPCION in usuario
    assert llamada["response_format"]["type"] == "json_schema"

    [evento] = [e for e in logs if e["event"] == "estimacion_completada"]
    assert evento["salida"] == "estructurada"
    assert evento["intentos"] == 1


def test_una_respuesta_invalida_se_corrige_con_un_reintento(stack, proveedor) -> None:
    proveedor.respuestas = [SIN_EQUIPO, json.dumps(ESTIMACION)]

    with stack() as api:
        estimacion, *_ = _estimate_structured(API, _payload(), api)

    assert len(proveedor.llamadas) == 2
    assert estimacion["team"][0]["role"] == "Desarrollador"


def test_agotados_los_intentos_el_formulario_recibe_un_502(stack, proveedor) -> None:
    proveedor.respuestas = [SIN_EQUIPO] * 3

    with stack(STRUCTURED_MAX_RETRIES="2") as api, pytest.raises(_ApiError) as error:
        _estimate_structured(API, _payload(), api)

    assert error.value.status == 502
    assert error.value.detail == INVALID_OUTPUT_MESSAGE
    assert len(proveedor.llamadas) == 3


def test_la_segunda_estimacion_sale_de_la_cache(stack, proveedor) -> None:
    with stack() as api:
        primera = _estimate_structured(API, _payload(), api)
        segunda = _estimate_structured(API, _payload(), api)

    assert len(proveedor.llamadas) == 1
    assert segunda[:3] == primera[:3]
    assert primera[3] is False
    assert segunda[3] is True  # el formulario sabe que salió de la caché


def test_texto_y_estructurada_no_comparten_cache(stack, proveedor) -> None:
    """Con la misma entrada, los prompts v2 y v3 son distintos: claves distintas."""
    proveedor.respuestas = [json.dumps(ESTIMACION), "| Fase | Semanas |"]

    with stack() as api:
        _estimate_structured(API, _payload(), api)
        r = api.post("/api/v1/estimate", json=_payload())

    assert r.status_code == 200
    assert r.json()["text"] == "| Fase | Semanas |"
    assert len(proveedor.llamadas) == 2


def test_un_total_que_no_cuadra_llega_al_formulario_como_aviso(stack, proveedor) -> None:
    descuadrada = {**ESTIMACION, "totals": {**ESTIMACION["totals"], "hours": 950}}
    proveedor.respuestas = [json.dumps(descuadrada)]

    with stack() as api:
        estimacion, _, avisos, _ = _estimate_structured(API, _payload(), api)

    assert estimacion["totals"]["hours"] == 950
    assert avisos == [
        "El total de horas declarado (950) no coincide con la suma de las fases (900)."
    ]


def test_el_selector_del_formulario_sale_de_health(stack) -> None:
    with stack() as api:
        health = _read_health(API, api)
    assert _structured_version_options(health) == [
        ("Predeterminada (v3)", None),
        ("v3", "v3"),
    ]
