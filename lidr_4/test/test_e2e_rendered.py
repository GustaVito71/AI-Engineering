"""Slice vertical de la salida renderizada (v4): del formulario a la presentación.

Como test_e2e_structured.py, con todas las piezas reales:

    _build_payload + _estimate_rendered (cliente de Streamlit)
      → FastAPI con su lifespan (Redis, caché)
      → EstimationRequest → render del prompt v4 (con output_format)
      → LLMWrapper.estimate_structured(RenderedResult, contexto) → Instructor
      → Router de LiteLLM → RenderedResult validado contra output_format
      → caché → RenderedEstimationResponse → _estimate_rendered

Lo único simulado es el proveedor (`mock_response`, una respuesta por intento) y
Redis (fakeredis). Lo que solo se ve con Instructor real: que el contexto llega
al validador, que un `rendered` inválido dispara el reintento con el error, y
que la caché se relee con el mismo contexto.
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
from streamlit_app import _ApiError, _build_payload, _estimate_rendered

pytestmark = pytest.mark.filterwarnings("ignore:You should not use the 'timeout' argument")

API = "http://testserver"
DESCRIPCION = "App móvil para reservar turnos en el gimnasio municipal, con avisos push."
TABLA = (
    "| Fase | Semanas | Horas | Coste (EUR) | Confianza (%) |\n"
    "|---|---|---|---|---|\n"
    "| Diseño | 2 | 130 | 6.500 | 80 |\n"
    "| Implementación | 6 | 770 | 48.150 | 65 |\n"
    "| **Total** | 8 | 900 | 54.650 | 70 |"
)
ESTIMACION = {
    "phases": [
        {
            "name": "Diseño",
            "summary": "Trabajo de la fase de Diseño.",
            "duration_weeks": 2,
            "hours": 130,
            "cost_eur": 6500,
            "confidence_pct": 80,
            "assumptions": [],
            "risks": [],
        },
        {
            "name": "Implementación",
            "summary": "Trabajo de la fase de Implementación.",
            "duration_weeks": 6,
            "hours": 770,
            "cost_eur": 48150,
            "confidence_pct": 65,
            "assumptions": [],
            "risks": [],
        },
    ],
    "team": [{"role": "Desarrollador", "headcount": 4}, {"role": "Diseñador", "headcount": 1}],
    "totals": {"hours": 900, "cost_eur": 54650, "duration_weeks": 8},
    "summary": "Estimación de prueba con las fases y el equipo indicados.",
    "confidence_pct": 70,
    "rendered": TABLA,
}
VALIDA = json.dumps(ESTIMACION)
# Cifras del texto distintas de las de los campos: el validador la rechaza.
CIFRA_DISTINTA = json.dumps({**ESTIMACION, "rendered": TABLA.replace("48.150", "48.000")})


class _Proveedor:
    """Proveedor simulado detrás del Router real: una respuesta por llamada."""

    def __init__(self) -> None:
        self.respuestas: list[str] = [VALIDA]
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


def _payload(output_format: str = "phases_table") -> dict:
    return _build_payload(DESCRIPCION, "mobile_app", "summary", output_format)


def test_del_formulario_a_la_tabla(stack, proveedor) -> None:
    with stack() as api, capture_logs() as logs:
        estimacion, version, avisos, cached = _estimate_rendered(API, _payload(), api)

    assert estimacion["rendered"] == TABLA
    assert version == "v4"
    assert avisos == []
    assert cached is False

    # El proveedor recibió el prompt v4 con el formato pedido y el schema con rendered.
    [llamada] = proveedor.llamadas
    sistema = llamada["messages"][0]["content"]
    assert 'output_format = "phases_table". En rendered:' in sistema
    schema = llamada["response_format"]["json_schema"]["schema"]
    assert "rendered" in schema["properties"]

    [evento] = [e for e in logs if e["event"] == "estimacion_completada"]
    assert evento["salida"] == "renderizada"
    assert evento["intentos"] == 1


def test_una_cifra_distinta_se_corrige_con_un_reintento(stack, proveedor) -> None:
    proveedor.respuestas = [CIFRA_DISTINTA, VALIDA]

    with stack() as api:
        estimacion, *_ = _estimate_rendered(API, _payload(), api)

    assert len(proveedor.llamadas) == 2
    assert "48.150" in estimacion["rendered"]
    # El reintento le devolvió al modelo el error del validador de rendered.
    reintento = json.dumps(proveedor.llamadas[1]["messages"], ensure_ascii=False)
    assert "falta o no coincide: Implementación" in reintento


def test_la_forma_depende_del_output_format_pedido(stack, proveedor) -> None:
    """La misma respuesta (una tabla) no sirve si se pidió narrativa: 3 intentos y 502."""
    proveedor.respuestas = [VALIDA] * 3

    with stack(STRUCTURED_MAX_RETRIES="2") as api, pytest.raises(_ApiError) as error:
        _estimate_rendered(API, _payload("narrative"), api)

    assert error.value.status == 502
    assert error.value.detail == INVALID_OUTPUT_MESSAGE
    assert len(proveedor.llamadas) == 3


def test_la_segunda_estimacion_sale_de_la_cache(stack, proveedor) -> None:
    with stack() as api:
        primera = _estimate_rendered(API, _payload(), api)
        segunda = _estimate_rendered(API, _payload(), api)

    assert len(proveedor.llamadas) == 1
    assert segunda[:3] == primera[:3]
    assert segunda[3] is True


def test_otro_formato_no_sale_de_la_cache(stack, proveedor) -> None:
    """El prompt de cada formato es distinto: la clave de caché también."""
    partidas = (
        "1. **Diseño** — 2 semanas — 130 horas — 6.500 EUR. Prototipo.\n"
        "2. **Implementación** — 6 semanas — 770 horas — 48.150 EUR. Desarrollo.\n\n"
        "**Total:** 8 semanas, 900 horas y 54.650 EUR."
    )
    proveedor.respuestas = [VALIDA, json.dumps({**ESTIMACION, "rendered": partidas})]

    with stack() as api:
        _estimate_rendered(API, _payload(), api)
        estimacion, *_ = _estimate_rendered(API, _payload("line_items"), api)

    assert len(proveedor.llamadas) == 2
    assert estimacion["rendered"] == partidas
