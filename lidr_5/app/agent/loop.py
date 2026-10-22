"""El bucle agéntico: el modelo pide herramientas, el servicio las ejecuta y se repite (Paso 10).

CÓMO FUNCIONA (en lenguaje llano)

1. Se le da al modelo la descripción del proyecto y la lista de herramientas.
2. El modelo responde con texto («ya tengo lo que necesito») o pidiendo una o más herramientas.
3. Si pidió herramientas, el servicio las ejecuta, mete los resultados en la conversación
   (como DATOS entre etiquetas `<tool_result>`) y vuelve al paso 2.
4. Se para cuando el modelo responde con texto, cuando se agotan las rondas
   (`AGENT_MAX_ITERATIONS`) o cuando el contexto del bucle llega al presupuesto de tokens.

La salida del bucle NO es la estimación: es el conjunto de resultados de herramientas que
se entregará al actor como contexto. Por eso, con el ACB, el bucle corre una sola vez antes
de la primera vuelta del actor y las vueltas siguientes reutilizan lo recogido.

Cada ejecución de herramienta emite el evento `herramienta_ejecutada` con su nombre,
duración, tamaño devuelto y si se recortó; el límite de tamaño sale del presupuesto de tokens.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.agent.budget import count_tokens, truncate_to_tokens
from app.agent.tools import Tool, ToolRegistry, wrap_tool_result
from app.observability import record_tool_call
from app.prompts.loader import render_agent_prompt
from app.tracing import emit

# Por qué terminó el bucle (campo `stopped_by` del resultado y del evento `agente_terminado`).
STOP_FINAL = "final"  # el modelo respondió con texto
STOP_MAX_ITERATIONS = "max_iterations"  # se agotaron las rondas
STOP_BUDGET = "budget"  # el contexto llegó al presupuesto de tokens
STOP_NO_TOOLS = "no_tools"  # no había herramientas registradas

# Tokens máximos que puede ocupar UN resultado de herramienta, aunque quede presupuesto de
# sobra: evita que una sola página web se coma todo el contexto.
MAX_TOKENS_PER_TOOL_RESULT = 8_000


class ToolCompletionLike(Protocol):
    """Lo que el bucle necesita de una ronda del modelo (lo cumple `ToolCompletion`)."""

    message: Any
    tool_calls: list[Any]
    content: str


class ToolCallingLLM(Protocol):
    """Lo que el bucle necesita del wrapper: hacer una ronda con herramientas."""

    async def complete_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        max_tokens: int | None = None,
    ) -> ToolCompletionLike:
        """Una ronda: texto o petición de herramientas."""
        ...


@dataclass
class ToolRun:
    """Registro de una ejecución de herramienta (para la traza y los tests)."""

    name: str
    arguments: dict[str, Any]
    duration_ms: float
    result_chars: int
    truncated: bool
    error: str | None = None


@dataclass
class AgentLoopResult:
    """Lo que devuelve el bucle: lo recogido, cómo terminó y el detalle de cada ejecución."""

    # Resultados ya envueltos como datos (`<tool_result ...>`), en orden de ejecución.
    outputs: list[str] = field(default_factory=list)
    runs: list[ToolRun] = field(default_factory=list)
    iterations: int = 0
    stopped_by: str = STOP_FINAL
    final_text: str = ""


def _messages_tokens(messages: list[dict[str, Any]], model: str | None) -> int:
    """Tokens aproximados de toda la conversación del bucle (solo el texto)."""
    return sum(
        count_tokens(json.dumps(m, ensure_ascii=False, default=str), model) for m in messages
    )


def _assistant_message(completion: ToolCompletionLike) -> dict[str, Any]:
    """El mensaje `assistant` con sus peticiones, en un formato simple y portable.

    Se reconstruye como diccionario (en vez de reenviar el objeto del proveedor) para que
    el mismo historial lo entienda cualquier proveedor si el Router cambia al respaldo a
    mitad del bucle.
    """
    return {
        "role": "assistant",
        "content": completion.content or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in completion.tool_calls
        ],
    }


async def _run_tool(tool: Tool, arguments: dict[str, Any]) -> str:
    """Ejecuta la función de la herramienta (síncrona o asíncrona) con su tiempo máximo."""

    async def call() -> str:
        outcome = tool.run(arguments)
        if asyncio.iscoroutine(outcome) or isinstance(outcome, asyncio.Future):
            outcome = await outcome
        return str(outcome)

    return await asyncio.wait_for(call(), timeout=tool.timeout_seconds)


async def _execute_call(
    call: Any,
    registry: ToolRegistry,
    *,
    tokens_left: int,
    model: str | None,
) -> tuple[str, ToolRun]:
    """Ejecuta UNA petición de herramienta del modelo y devuelve (texto del resultado, registro).

    Nunca lanza: un nombre desconocido, argumentos mal formados, un fallo o un tiempo
    agotado se devuelven como texto de error para que el modelo pueda reaccionar.
    """
    name = call.function.name
    started = time.perf_counter()
    arguments: dict[str, Any] = {}
    error: str | None = None
    text = ""
    tool = registry.get(name)
    try:
        if tool is None:
            raise LookupError(f"La herramienta '{name}' no existe.")
        raw = call.function.arguments or "{}"
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(parsed, dict):
            raise TypeError("Los argumentos deben ser un objeto JSON.")
        arguments = parsed
        text = await _run_tool(tool, arguments)
    except TimeoutError:
        error = "tiempo agotado"
        text = f"Error: la herramienta '{name}' tardó demasiado y se canceló."
    except Exception as exc:  # noqa: BLE001 - un fallo de herramienta no debe tumbar el turno
        error = f"{type(exc).__name__}: {exc}"
        text = f"Error al ejecutar '{name}': {exc}"

    limit = max(min(tokens_left, MAX_TOKENS_PER_TOOL_RESULT), 0)
    text, truncated = truncate_to_tokens(text, limit, model)
    run = ToolRun(
        name=name,
        arguments=arguments,
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
        result_chars=len(text),
        truncated=truncated,
        error=error,
    )
    record_tool_call()
    emit(
        __name__,
        "herramienta_ejecutada",
        herramienta=name,
        duracion_ms=run.duration_ms,
        caracteres_devueltos=run.result_chars,
        recortado=truncated,
        error=error,
    )
    return text, run


async def run_agent_loop(
    *,
    llm: ToolCallingLLM,
    registry: ToolRegistry,
    transcript: str,
    attachments_block: str = "",
    max_iterations: int,
    token_budget: int,
    model: str | None = None,
) -> AgentLoopResult:
    """Corre el bucle agéntico y devuelve lo recogido.

    `max_iterations` es el tope de rondas modelo→herramientas; `token_budget` es el techo de
    tokens de la conversación del bucle (se comprueba antes de cada ronda nueva y limita el
    tamaño de cada resultado). `model` solo sirve para contar tokens. Sin herramientas
    registradas no se llama al modelo: se devuelve de inmediato un resultado vacío.
    """
    result = AgentLoopResult()
    if len(registry) == 0:
        result.stopped_by = STOP_NO_TOOLS
        return result

    system_prompt, user_prompt = render_agent_prompt(
        transcript=transcript, attachments_block=attachments_block
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    specs = registry.specs()

    for iteration in range(1, max_iterations + 1):
        # El presupuesto se mira antes de pedir otra ronda: una ronda más enviaría todo el
        # historial de nuevo, y el contexto solo crece.
        if _messages_tokens(messages, model) >= token_budget:
            result.stopped_by = STOP_BUDGET
            break
        completion = await llm.complete_with_tools(messages=messages, tools=specs)
        result.iterations = iteration
        if not completion.tool_calls:
            result.final_text = completion.content or ""
            result.stopped_by = STOP_FINAL
            break

        messages.append(_assistant_message(completion))
        for call in completion.tool_calls:
            tokens_left = token_budget - _messages_tokens(messages, model)
            text, run = await _execute_call(call, registry, tokens_left=tokens_left, model=model)
            wrapped = wrap_tool_result(run.name, text)
            result.outputs.append(wrapped)
            result.runs.append(run)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": wrapped})
    else:
        result.stopped_by = STOP_MAX_ITERATIONS

    emit(
        __name__,
        "agente_terminado",
        rondas=result.iterations,
        herramientas=[run.name for run in result.runs],
        terminado_por=result.stopped_by,
    )
    return result
