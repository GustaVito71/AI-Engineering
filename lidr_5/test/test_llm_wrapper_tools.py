"""Tests de `LLMWrapper.complete_with_tools` (Paso 10): una ronda del bucle agéntico.

Se usa el Router real de LiteLLM con respuestas simuladas (`mock_response`), sin red.

QUÉ SE COMPRUEBA

- Las herramientas y el historial llegan al Router tal cual, y el modelo que responde se informa.
- Una respuesta de texto devuelve contenido y ninguna petición de herramientas; una respuesta con
  `mock_tool_calls` devuelve las peticiones.
- Cada ronda se suma a la hucha de coste del turno y emite `agente_llamada`.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from structlog.testing import capture_logs
from test_llm_wrapper import PRIMARIO, _con_mock, wrapper  # noqa: F401 - `wrapper` es un fixture

from app.observability import RequestContextMiddleware

HERRAMIENTAS = [
    {
        "type": "function",
        "function": {
            "name": "buscar",
            "description": "Busca un dato.",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
]
MENSAJES = [
    {"role": "system", "content": "eres un asistente"},
    {"role": "user", "content": "busca algo"},
]


async def test_la_respuesta_de_texto_no_trae_peticiones_de_herramientas(wrapper) -> None:  # noqa: F811
    """Si el modelo responde con texto, `tool_calls` queda vacío y `content` trae el texto."""
    _con_mock(wrapper, mock_response="Sin consultas necesarias")
    ronda = await wrapper.complete_with_tools(messages=MENSAJES, tools=HERRAMIENTAS)
    assert ronda.tool_calls == []
    assert ronda.content == "Sin consultas necesarias"
    assert ronda.model == PRIMARIO
    assert ronda.provider == "openai"
    assert ronda.cost_usd > 0


async def test_las_herramientas_y_el_historial_llegan_al_router(wrapper) -> None:  # noqa: F811
    """El Router recibe los mensajes, las definiciones y la política de elección."""
    vistos = _con_mock(wrapper, mock_response="ok")
    await wrapper.complete_with_tools(messages=MENSAJES, tools=HERRAMIENTAS, tool_choice="none")
    assert vistos[0]["messages"] == MENSAJES
    assert vistos[0]["tools"] == HERRAMIENTAS
    assert vistos[0]["tool_choice"] == "none"
    assert vistos[0]["max_tokens"] == wrapper._max_tokens


async def test_una_respuesta_con_peticion_de_herramienta_la_devuelve(wrapper) -> None:  # noqa: F811
    """Con una respuesta simulada que pide `buscar`, la ronda trae la petición con su id."""
    _con_mock(
        wrapper,
        mock_tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "buscar", "arguments": '{"q": "django"}'},
            }
        ],
    )
    ronda = await wrapper.complete_with_tools(messages=MENSAJES, tools=HERRAMIENTAS)
    assert [c.function.name for c in ronda.tool_calls] == ["buscar"]
    assert ronda.tool_calls[0].id == "call_1"
    assert ronda.tool_calls[0].function.arguments == '{"q": "django"}'


async def test_la_ronda_emite_agente_llamada_y_suma_a_la_hucha(wrapper) -> None:  # noqa: F811
    """El evento `agente_llamada` lleva modelo, tokens y coste; el turno suma la llamada."""
    _con_mock(wrapper, mock_response="ok")
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/ronda")
    async def ronda() -> dict:
        """Una ronda del bucle dentro de una petición, como en producción."""
        await wrapper.complete_with_tools(messages=MENSAJES, tools=HERRAMIENTAS)
        return {}

    with capture_logs() as logs:
        TestClient(app).get("/ronda")

    llamada = next(e for e in logs if e["event"] == "agente_llamada")
    assert llamada["modelo"] == PRIMARIO
    assert llamada["uso_respaldo"] is False
    assert llamada["tokens_prompt"] > 0
    turno = next(e for e in logs if e["event"] == "turno_costo")
    assert turno["llamadas_llm"] == 1
    # `turno_costo` redondea a 6 decimales.
    assert turno["coste_usd"] == pytest.approx(llamada["coste_usd"], abs=1e-6)
    assert turno["request_id"] == llamada["request_id"]
