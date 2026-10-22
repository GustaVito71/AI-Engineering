"""Tests del bucle agéntico dentro de los endpoints de sesiones (Paso 10).

Se levanta la aplicación real y solo se reemplazan el wrapper del actor (que además sabe hacer
rondas con herramientas), el crítico y el extractor. Sin red.

QUÉ SE COMPRUEBA

- Con el bucle apagado (por defecto) el comportamiento es el de antes: el actor no recibe
  bloque de herramientas y el modelo no recibe ninguna ronda con herramientas.
- Con el bucle encendido y una herramienta registrada, el actor recibe los resultados como un
  bloque `<tool_results>` junto a los adjuntos.
- Con el ACB, el bucle corre UNA sola vez aunque el actor haga varias vueltas, y las vueltas
  reutilizan lo recogido; el crítico recibe el mismo contexto.
- Todos los eventos del turno llevan el mismo `request_id`, y `turno_costo` suma la ronda
  de herramientas.
- Un contexto que supera el presupuesto se recorta y el usuario recibe el aviso.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs
from test_acb_critic_boss import indep, problema
from test_agent_loop import pide, responde
from test_estimate_structured_endpoint import _WrapperFalso
from test_sessions_acb import _CriticoFalso, estimar_acb
from test_sessions_memory_endpoint import FORM, TURNO_1, _ExtractorFalso

from app.agent.tools import Tool, ToolRegistry
from app.config import get_settings
from app.dependencies import (
    get_critic,
    get_llm_wrapper,
    get_metadata_extractor,
    get_tool_registry,
)
from app.main import create_app


class _WrapperConHerramientas(_WrapperFalso):
    """Actor de mentira que, además, responde rondas con herramientas según un guion."""

    def __init__(self, *rondas) -> None:
        super().__init__()
        self.rondas = list(rondas)
        self.rondas_herramientas: list[list[dict]] = []

    async def complete_with_tools(self, *, messages, tools, tool_choice="auto", max_tokens=None):
        """Guarda los mensajes de la ronda y devuelve la siguiente del guion."""
        self.rondas_herramientas.append(messages)
        return self.rondas[min(len(self.rondas_herramientas), len(self.rondas)) - 1]


def _registro() -> ToolRegistry:
    """Un registro con una herramienta de precios que devuelve un dato reconocible."""
    registro = ToolRegistry()
    registro.register(
        Tool(
            name="precios",
            description="Consulta tarifas de mercado.",
            parameters={"type": "object", "properties": {}},
            run=lambda args: "TARIFA-DE-MERCADO 55 EUR/h",
        )
    )
    return registro


@pytest.fixture
def montar(monkeypatch):
    """Fábrica: arma la app con el bucle encendido o apagado y crea una sesión."""

    def _montar(*, bucle: bool, wrapper=None, critico=None, **entorno):
        """Devuelve `(cliente, wrapper, session_id)`."""
        monkeypatch.setenv("REDIS_URL", "")
        monkeypatch.setenv("AGENT_LOOP_ENABLED", "true" if bucle else "false")
        for variable, valor in entorno.items():
            monkeypatch.setenv(variable, str(valor))
        get_settings.cache_clear()
        cliente = TestClient(create_app())
        wrapper = wrapper or _WrapperConHerramientas(pide(("precios", {})), responde("listo"))
        cliente.app.dependency_overrides[get_llm_wrapper] = lambda: wrapper
        cliente.app.dependency_overrides[get_metadata_extractor] = lambda: _ExtractorFalso()
        cliente.app.dependency_overrides[get_tool_registry] = _registro
        if critico is not None:
            cliente.app.dependency_overrides[get_critic] = lambda: critico
        return cliente, wrapper, cliente.post("/sessions").json()["session_id"]

    yield _montar
    get_settings.cache_clear()


def _estimar(cliente, session_id):
    """Envía un turno a `/estimate` como formulario."""
    return cliente.post(f"/sessions/{session_id}/estimate", data={"transcript": TURNO_1, **FORM})


def test_con_el_bucle_apagado_el_actor_no_recibe_herramientas(montar) -> None:
    """Por defecto no hay rondas con herramientas ni bloque `<tool_results>`."""
    cliente, wrapper, sid = montar(bucle=False)

    respuesta = _estimar(cliente, sid)

    assert respuesta.status_code == 200
    assert wrapper.rondas_herramientas == []
    assert "<tool_results>" not in wrapper.llamadas[0]["user_prompt"]


def test_con_el_bucle_encendido_el_actor_recibe_los_resultados(montar) -> None:
    """Una ronda pide `precios`, otra cierra; el actor ve el resultado como datos."""
    cliente, wrapper, sid = montar(bucle=True)

    respuesta = _estimar(cliente, sid)

    assert respuesta.status_code == 200
    assert len(wrapper.rondas_herramientas) == 2
    prompt_actor = wrapper.llamadas[0]["user_prompt"]
    assert "<tool_results>" in prompt_actor
    assert "TARIFA-DE-MERCADO 55 EUR/h" in prompt_actor
    assert '<tool_result name="precios">' in prompt_actor


def test_encendido_pero_sin_herramientas_no_gasta_rondas(montar) -> None:
    """El bucle está activo pero el registro llega vacío (estado real del Paso 10)."""
    cliente, wrapper, sid = montar(bucle=True)
    cliente.app.dependency_overrides[get_tool_registry] = lambda: ToolRegistry()

    respuesta = _estimar(cliente, sid)

    assert respuesta.status_code == 200
    assert wrapper.rondas_herramientas == []
    assert "<tool_results>" not in wrapper.llamadas[0]["user_prompt"]


def test_con_el_acb_el_bucle_corre_una_sola_vez_y_las_vueltas_lo_reutilizan(montar) -> None:
    """Dos vueltas del actor, una sola recogida de herramientas; el crítico ve lo mismo."""
    critico = _CriticoFalso(indep("needs_iteration", problema()), indep("accept"))
    cliente, wrapper, sid = montar(bucle=True, critico=critico)

    respuesta = estimar_acb(cliente, sid)

    assert respuesta.status_code == 200
    assert respuesta.json()["acb"]["iterations_run"] == 2
    assert len(wrapper.rondas_herramientas) == 2  # una recogida: pide + cierra, no ×2
    assert len(wrapper.llamadas) == 2
    for llamada in wrapper.llamadas:
        assert "TARIFA-DE-MERCADO 55 EUR/h" in llamada["user_prompt"]
    for revision in critico.llamadas:
        assert "TARIFA-DE-MERCADO 55 EUR/h" in revision["attachments_block"]


def test_los_eventos_del_turno_comparten_request_id_y_el_costo_suma_la_ronda(montar) -> None:
    """`herramienta_ejecutada`, `agente_terminado` y `turno_costo` llevan el mismo id."""
    cliente, _wrapper, sid = montar(bucle=True)

    with capture_logs() as logs:
        respuesta = _estimar(cliente, sid)

    ids = {
        e["request_id"]
        for e in logs
        if e["event"] in ("herramienta_ejecutada", "agente_terminado", "estimacion_conversacional")
    }
    assert ids == {respuesta.headers["X-Request-ID"]}
    turno = next(e for e in logs if e["event"] == "turno_costo")
    assert turno["llamadas_herramienta"] == 1
    assert turno["request_id"] in ids


def test_un_contexto_que_supera_el_presupuesto_se_recorta_y_avisa(montar) -> None:
    """Con un presupuesto mínimo y una herramienta que devuelve mucho, el usuario recibe el aviso."""
    registro = ToolRegistry()
    registro.register(
        Tool(
            name="precios",
            description="Consulta tarifas.",
            parameters={"type": "object", "properties": {}},
            run=lambda args: "dato largo " * 3000,
        )
    )
    cliente, wrapper, sid = montar(bucle=True, TURN_TOKEN_BUDGET=7_000)
    cliente.app.dependency_overrides[get_tool_registry] = lambda: registro

    respuesta = _estimar(cliente, sid)

    assert respuesta.status_code == 200
    assert any("presupuesto de 7000 tokens" in w for w in respuesta.json()["warnings"])
    assert "recortado por el presupuesto" in wrapper.llamadas[0]["user_prompt"]


def test_la_app_real_devuelve_el_request_id_en_todas_las_respuestas() -> None:
    """El middleware está montado en la aplicación real: `/health` también lleva la cabecera."""
    respuesta = TestClient(create_app()).get("/health")
    assert len(respuesta.headers["X-Request-ID"]) == 32


def test_el_registro_de_herramientas_llega_vacio_en_el_paso_10() -> None:
    """Las herramientas las traen los Pasos 11 y 12: hoy el registro real está vacío."""
    cliente = TestClient(create_app())
    cliente.app.state.tool_registry = None
    assert len(cliente.app.dependency_overrides) == 0

    class _Peticion:
        """Una petición mínima: solo expone `app.state`, que es lo que lee la dependencia."""

        app = cliente.app

    registro = get_tool_registry(_Peticion())
    assert len(registro) == 0
    assert get_tool_registry(_Peticion()) is registro  # se crea una vez y se comparte
