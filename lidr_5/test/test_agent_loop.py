"""Tests de `app/agent/loop.py` (Paso 10): el bucle agéntico, con un modelo de mentira.

El «modelo» es un guion: en cada ronda devuelve lo que el test preparó (texto o peticiones de
herramientas). Sin red ni claves.

QUÉ SE COMPRUEBA

- Sin herramientas registradas no se llama al modelo.
- Si el modelo responde con texto, el bucle termina en la primera ronda.
- Pide herramienta → se ejecuta → su resultado vuelve al modelo como dato → responde.
- Respeta el tope de rondas y el presupuesto de tokens.
- Un nombre desconocido, argumentos mal formados, un fallo o un tiempo agotado no tumban el
  bucle: el modelo recibe el error como texto.
- Los resultados se recortan al presupuesto y cada ejecución emite `herramienta_ejecutada`.
- El historial de herramientas que se reenvía es portable (diccionarios simples), para que el
  respaldo pueda continuar a mitad del bucle.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from structlog.testing import capture_logs

from app.agent import loop as loop_module
from app.agent.budget import count_tokens
from app.agent.loop import (
    STOP_BUDGET,
    STOP_FINAL,
    STOP_MAX_ITERATIONS,
    STOP_NO_TOOLS,
    run_agent_loop,
)
from app.agent.tools import Tool, ToolRegistry

MODELO = "openai/gpt-4o-mini"


def pide(*llamadas: tuple[str, dict | str]) -> SimpleNamespace:
    """Una ronda del modelo que pide herramientas: `pide(("buscar", {"q": "x"}))`."""
    calls = []
    for i, (nombre, args) in enumerate(llamadas):
        texto = args if isinstance(args, str) else json.dumps(args)
        calls.append(
            SimpleNamespace(id=f"call_{i}", function=SimpleNamespace(name=nombre, arguments=texto))
        )
    return SimpleNamespace(message=None, tool_calls=calls, content="")


def responde(texto: str = "listo") -> SimpleNamespace:
    """Una ronda del modelo que responde con texto y termina."""
    return SimpleNamespace(message=None, tool_calls=[], content=texto)


class ModeloGuionado:
    """Modelo de mentira: devuelve las rondas preparadas, en orden, y guarda lo que recibió."""

    def __init__(self, *rondas) -> None:
        self.rondas = list(rondas)
        self.llamadas: list[dict] = []

    async def complete_with_tools(self, *, messages, tools, tool_choice="auto", max_tokens=None):
        """Guarda una copia de los mensajes y devuelve la siguiente ronda del guion."""
        self.llamadas.append({"messages": json.loads(json.dumps(messages)), "tools": tools})
        return self.rondas[min(len(self.llamadas), len(self.rondas)) - 1]


def registro(**funciones) -> ToolRegistry:
    """Un registro con una herramienta por cada función indicada (`nombre=funcion`)."""
    reg = ToolRegistry()
    for nombre, funcion in funciones.items():
        reg.register(
            Tool(
                name=nombre,
                description=f"Herramienta {nombre}.",
                parameters={"type": "object", "properties": {}},
                run=funcion,
                timeout_seconds=0.2,
            )
        )
    return reg


async def correr(modelo, reg, *, max_rondas=4, presupuesto=60_000):
    """Atajo para correr el bucle con una descripción de proyecto cualquiera."""
    return await run_agent_loop(
        llm=modelo,
        registry=reg,
        transcript="Quiero una app de reservas",
        max_iterations=max_rondas,
        token_budget=presupuesto,
        model=MODELO,
    )


async def test_sin_herramientas_no_llama_al_modelo() -> None:
    """Un registro vacío devuelve un resultado vacío sin gastar nada."""
    modelo = ModeloGuionado(responde())
    resultado = await correr(modelo, ToolRegistry())
    assert resultado.stopped_by == STOP_NO_TOOLS
    assert resultado.outputs == []
    assert modelo.llamadas == []


async def test_si_el_modelo_responde_con_texto_termina_en_la_primera_ronda() -> None:
    """Sin pedir herramientas, una sola llamada y la nota final."""
    modelo = ModeloGuionado(responde("Sin consultas necesarias"))
    resultado = await correr(modelo, registro(buscar=lambda a: "x"))
    assert resultado.stopped_by == STOP_FINAL
    assert resultado.iterations == 1
    assert resultado.final_text == "Sin consultas necesarias"
    assert resultado.runs == []
    assert len(modelo.llamadas) == 1
    # El modelo recibió el prompt del bucle y la definición de la herramienta.
    assert modelo.llamadas[0]["messages"][0]["role"] == "system"
    assert "Quiero una app de reservas" in modelo.llamadas[0]["messages"][1]["content"]
    assert modelo.llamadas[0]["tools"][0]["function"]["name"] == "buscar"


async def test_pide_una_herramienta_la_ejecuta_y_el_resultado_vuelve_como_dato() -> None:
    """Ronda 1 pide `buscar`; ronda 2 recibe el resultado envuelto y responde."""
    modelo = ModeloGuionado(pide(("buscar", {"q": "django"})), responde("encontré datos"))
    resultado = await correr(modelo, registro(buscar=lambda a: f"info de {a['q']}"))

    assert resultado.stopped_by == STOP_FINAL
    assert resultado.iterations == 2
    assert len(resultado.outputs) == 1
    assert '<tool_result name="buscar">' in resultado.outputs[0]
    assert "info de django" in resultado.outputs[0]
    # La segunda ronda llevó el resultado como mensaje `tool` ligado a la petición.
    mensajes = modelo.llamadas[1]["messages"]
    assert mensajes[-1]["role"] == "tool"
    assert mensajes[-1]["tool_call_id"] == "call_0"
    assert "info de django" in mensajes[-1]["content"]


async def test_el_historial_de_herramientas_es_portable() -> None:
    """El mensaje del asistente se reenvía como diccionario simple con sus `tool_calls`:
    así el respaldo (otro proveedor) puede continuar el bucle con el mismo historial.
    """
    modelo = ModeloGuionado(pide(("buscar", {"q": "x"})), responde())
    await correr(modelo, registro(buscar=lambda a: "ok"))
    asistente = modelo.llamadas[1]["messages"][2]
    assert asistente["role"] == "assistant"
    llamada = asistente["tool_calls"][0]
    assert llamada["id"] == "call_0"
    assert llamada["type"] == "function"
    assert llamada["function"]["name"] == "buscar"
    assert json.loads(llamada["function"]["arguments"]) == {"q": "x"}


async def test_respeta_el_tope_de_rondas() -> None:
    """Un modelo que siempre pide herramientas se corta en `max_iterations` rondas."""
    modelo = ModeloGuionado(pide(("buscar", {})))
    resultado = await correr(modelo, registro(buscar=lambda a: "ok"), max_rondas=3)
    assert resultado.stopped_by == STOP_MAX_ITERATIONS
    assert resultado.iterations == 3
    assert len(modelo.llamadas) == 3
    assert len(resultado.outputs) == 3


async def test_respeta_el_presupuesto_de_tokens() -> None:
    """Si el contexto ya llegó al presupuesto, no se pide otra ronda."""
    modelo = ModeloGuionado(pide(("buscar", {})))
    resultado = await correr(
        modelo, registro(buscar=lambda a: "dato " * 2000), max_rondas=4, presupuesto=1_500
    )
    assert resultado.stopped_by == STOP_BUDGET
    assert len(modelo.llamadas) < 4


async def test_un_resultado_enorme_se_recorta_y_se_marca() -> None:
    """El resultado no puede pasar del presupuesto restante; el registro lo marca recortado."""
    modelo = ModeloGuionado(pide(("buscar", {})), responde())
    resultado = await correr(
        modelo, registro(buscar=lambda a: "palabra " * 20_000), presupuesto=5_000
    )
    assert resultado.runs[0].truncated
    assert "recortado por el presupuesto" in resultado.outputs[0]
    # Cabe en el presupuesto (medido en tokens) y es mucho más corto que el original.
    assert count_tokens(resultado.outputs[0], MODELO) <= 5_000
    assert len(resultado.outputs[0]) < len("palabra ") * 20_000 // 2


async def test_una_herramienta_desconocida_no_tumba_el_bucle() -> None:
    """El modelo recibe «no existe» como resultado y puede seguir."""
    modelo = ModeloGuionado(pide(("fantasma", {})), responde())
    resultado = await correr(modelo, registro(buscar=lambda a: "x"))
    assert resultado.stopped_by == STOP_FINAL
    assert "no existe" in resultado.outputs[0]
    assert "LookupError" in resultado.runs[0].error


@pytest.mark.parametrize("argumentos", ["esto no es json", "[1, 2, 3]"])
async def test_argumentos_mal_formados_se_devuelven_como_error(argumentos) -> None:
    """JSON inválido, o un JSON que no es un objeto, no ejecuta la herramienta."""
    llamadas = []
    modelo = ModeloGuionado(pide(("buscar", argumentos)), responde())
    resultado = await correr(modelo, registro(buscar=lambda a: llamadas.append(a) or "x"))
    assert llamadas == []
    assert resultado.runs[0].error is not None
    assert "Error al ejecutar" in resultado.outputs[0]


async def test_una_herramienta_que_falla_devuelve_el_error_al_modelo() -> None:
    """Una excepción dentro de la herramienta se convierte en texto de error."""

    def rota(_args):
        """Una herramienta que siempre falla."""
        raise RuntimeError("la base de datos no responde")

    modelo = ModeloGuionado(pide(("rota", {})), responde())
    resultado = await correr(modelo, registro(rota=rota))
    assert "la base de datos no responde" in resultado.outputs[0]
    assert resultado.runs[0].error.startswith("RuntimeError")


async def test_una_herramienta_lenta_se_cancela_por_tiempo() -> None:
    """Pasado `timeout_seconds` la herramienta se corta y el modelo recibe el aviso."""

    async def lenta(_args):
        """Tarda mucho más que el tiempo máximo del registro de prueba (0,2 s)."""
        await asyncio.sleep(5)
        return "tarde"

    modelo = ModeloGuionado(pide(("lenta", {})), responde())
    resultado = await correr(modelo, registro(lenta=lenta))
    assert resultado.runs[0].error == "tiempo agotado"
    assert "tardó demasiado" in resultado.outputs[0]


async def test_acepta_herramientas_asincronas() -> None:
    """Una función `async` se espera y su resultado se usa igual que el de una síncrona."""

    async def asincrona(args):
        """Devuelve un texto tras ceder el control una vez."""
        await asyncio.sleep(0)
        return "desde async"

    modelo = ModeloGuionado(pide(("a", {})), responde())
    resultado = await correr(modelo, registro(a=asincrona))
    assert "desde async" in resultado.outputs[0]


async def test_varias_herramientas_en_una_ronda_se_ejecutan_todas() -> None:
    """Si el modelo pide dos a la vez, ambas se ejecutan y vuelven con su id."""
    modelo = ModeloGuionado(pide(("a", {}), ("b", {})), responde())
    resultado = await correr(modelo, registro(a=lambda x: "A", b=lambda x: "B"))
    assert [r.name for r in resultado.runs] == ["a", "b"]
    mensajes = modelo.llamadas[1]["messages"]
    assert [m["tool_call_id"] for m in mensajes if m["role"] == "tool"] == ["call_0", "call_1"]


async def test_cada_ejecucion_emite_un_evento_con_nombre_duracion_y_tamano() -> None:
    """`herramienta_ejecutada` registra nombre, duración, tamaño y si se recortó; el bucle
    cierra con `agente_terminado`.
    """
    modelo = ModeloGuionado(pide(("buscar", {})), responde())
    with capture_logs() as logs:
        await correr(modelo, registro(buscar=lambda a: "datos"))

    ejecutada = [e for e in logs if e["event"] == "herramienta_ejecutada"]
    assert len(ejecutada) == 1
    assert ejecutada[0]["herramienta"] == "buscar"
    assert ejecutada[0]["caracteres_devueltos"] == len("datos")
    assert ejecutada[0]["recortado"] is False
    assert ejecutada[0]["duracion_ms"] >= 0
    terminado = next(e for e in logs if e["event"] == "agente_terminado")
    assert terminado["terminado_por"] == STOP_FINAL
    assert terminado["herramientas"] == ["buscar"]


async def test_cuenta_las_herramientas_en_la_hucha_del_turno(monkeypatch) -> None:
    """Cada ejecución se suma al contador de herramientas del turno."""
    llamadas = []
    monkeypatch.setattr(loop_module, "record_tool_call", lambda: llamadas.append(1))
    modelo = ModeloGuionado(pide(("a", {}), ("a", {})), responde())
    await correr(modelo, registro(a=lambda x: "ok"))
    assert len(llamadas) == 2
