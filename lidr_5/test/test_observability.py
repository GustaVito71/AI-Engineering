"""Tests de `app/observability.py` (Paso 10): `request_id` y coste acumulado por turno.

QUÉ SE COMPRUEBA

- Cada petición recibe un `request_id` que aparece en TODOS los eventos que se emiten durante
  ella y en la cabecera `X-Request-ID` de la respuesta.
- Un identificador razonable enviado por el cliente se respeta; uno peligroso se descarta.
- Las llamadas al modelo suman tokens y coste a la hucha de la petición, y al terminar se emite
  un único evento `turno_costo` con el total. Sin llamadas no se emite nada.
- Un acierto de caché no cuenta como llamada ni como gasto.
- La hucha no se arrastra de una petición a la siguiente.
"""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from app.observability import (
    MAX_EXTERNAL_REQUEST_ID_CHARS,
    REQUEST_ID_HEADER,
    RequestContextMiddleware,
    current_request_id,
    current_turn_cost,
    record_llm_call,
    record_tool_call,
)
from app.services.llm_wrapper import LLMCallResult, emit_completion
from app.tracing import emit


@pytest.fixture
def cliente() -> TestClient:
    """Una app mínima con el middleware y tres rutas: una que emite, una que gasta y una vacía."""
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/emite")
    def emite() -> dict:
        """Emite dos eventos sueltos, como lo haría cualquier parte del servicio."""
        emit("prueba", "evento_uno")
        emit("prueba", "evento_dos")
        return {"request_id": current_request_id()}

    @app.get("/gasta")
    def gasta() -> dict:
        """Simula un turno con dos llamadas al modelo y una herramienta."""
        record_llm_call(prompt_tokens=100, completion_tokens=40, cost_usd=0.01)
        record_llm_call(prompt_tokens=50, completion_tokens=10, cost_usd=0.005)
        record_tool_call()
        return {}

    @app.get("/vacia")
    def vacia() -> dict:
        """No llama al modelo: no debe generar `turno_costo`."""
        return {}

    return TestClient(app)


def test_todos_los_eventos_de_la_peticion_llevan_el_mismo_request_id(cliente) -> None:
    """Dos eventos distintos dentro de la misma petición comparten el `request_id`."""
    with capture_logs() as logs:
        respuesta = cliente.get("/emite")

    ids = {e["request_id"] for e in logs if e["event"] in ("evento_uno", "evento_dos")}
    assert len(ids) == 1
    assert ids == {respuesta.json()["request_id"]}
    assert respuesta.headers[REQUEST_ID_HEADER] in ids


def test_dos_peticiones_tienen_request_id_distinto(cliente) -> None:
    """Cada petición genera su propio identificador."""
    a = cliente.get("/emite").headers[REQUEST_ID_HEADER]
    b = cliente.get("/emite").headers[REQUEST_ID_HEADER]
    assert a != b


def test_se_respeta_un_request_id_razonable_del_cliente(cliente) -> None:
    """Un identificador alfanumérico corto enviado por el cliente se conserva."""
    respuesta = cliente.get("/emite", headers={REQUEST_ID_HEADER: "mi-traza_42"})
    assert respuesta.headers[REQUEST_ID_HEADER] == "mi-traza_42"
    assert respuesta.json()["request_id"] == "mi-traza_42"


@pytest.mark.parametrize(
    "malo",
    ["con espacios", "a" * (MAX_EXTERNAL_REQUEST_ID_CHARS + 1), "x;rm -rf", "linea\tcon tab"],
)
def test_se_descarta_un_request_id_peligroso_o_demasiado_largo(cliente, malo) -> None:
    """Texto con caracteres raros o demasiado largo se reemplaza por uno generado."""
    respuesta = cliente.get("/emite", headers={REQUEST_ID_HEADER: malo})
    assert respuesta.headers[REQUEST_ID_HEADER] != malo
    assert len(respuesta.headers[REQUEST_ID_HEADER]) == 32


def test_el_turno_emite_un_unico_turno_costo_con_los_totales(cliente) -> None:
    """Dos llamadas y una herramienta suman en un solo evento con el `request_id`."""
    with capture_logs() as logs:
        respuesta = cliente.get("/gasta")

    eventos = [e for e in logs if e["event"] == "turno_costo"]
    assert len(eventos) == 1
    evento = eventos[0]
    assert evento["llamadas_llm"] == 2
    assert evento["llamadas_herramienta"] == 1
    assert evento["tokens_prompt"] == 150
    assert evento["tokens_completion"] == 50
    assert evento["coste_usd"] == pytest.approx(0.015)
    assert evento["ruta"] == "/gasta"
    assert evento["estado_http"] == 200
    assert evento["request_id"] == respuesta.headers[REQUEST_ID_HEADER]


def test_sin_llamadas_no_se_emite_turno_costo(cliente) -> None:
    """Una petición que no gasta nada (como /health) no llena el log."""
    with capture_logs() as logs:
        cliente.get("/vacia")
    assert not [e for e in logs if e["event"] == "turno_costo"]


def test_la_hucha_no_se_arrastra_entre_peticiones(cliente) -> None:
    """Tras una petición que gastó, la siguiente empieza de cero y no deja hucha abierta."""
    cliente.get("/gasta")
    with capture_logs() as logs:
        cliente.get("/gasta")
    assert next(e for e in logs if e["event"] == "turno_costo")["llamadas_llm"] == 2
    assert current_turn_cost() is None


def test_fuera_de_una_peticion_registrar_un_coste_no_hace_nada() -> None:
    """Sin hucha abierta (scripts, tests sueltos), `record_llm_call` no falla."""
    record_llm_call(prompt_tokens=1, completion_tokens=1, cost_usd=1.0)
    assert current_turn_cost() is None


def _resultado(*, cached: bool) -> LLMCallResult:
    """Un resultado de llamada al modelo de mentira, con o sin acierto de caché."""
    return LLMCallResult(
        content="{}",
        model="openai/gpt-4o-mini",
        provider="openai",
        prompt_tokens=10,
        completion_tokens=5,
        cost_usd=0.002,
        prompt_version="v5",
        cached=cached,
    )


def test_emit_completion_suma_la_llamada_a_la_hucha() -> None:
    """Todas las llamadas del wrapper pasan por `emit_completion`: ahí se contabilizan."""
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/x")
    def x() -> dict:
        """Dos llamadas reales y un acierto de caché dentro del mismo turno."""
        inicio = time.perf_counter()
        emit_completion(_resultado(cached=False), inicio, primary_model="openai/gpt-4o-mini")
        emit_completion(_resultado(cached=False), inicio, primary_model="openai/gpt-4o-mini")
        emit_completion(_resultado(cached=True), inicio, primary_model="openai/gpt-4o-mini")
        return {}

    with capture_logs() as logs:
        TestClient(app).get("/x")

    evento = next(e for e in logs if e["event"] == "turno_costo")
    assert evento["llamadas_llm"] == 2  # el acierto de caché no cuenta
    assert evento["coste_usd"] == pytest.approx(0.004)
    # Y los eventos de completado llevan el mismo request_id.
    ids = {e["request_id"] for e in logs if e["event"] == "estimacion_completada"}
    assert ids == {evento["request_id"]}
