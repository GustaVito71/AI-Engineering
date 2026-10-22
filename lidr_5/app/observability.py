"""Trazabilidad por petición: `request_id` y coste acumulado del turno (Paso 10).

QUÉ HACE ESTE MÓDULO (en lenguaje llano)

Una estimación conversacional puede disparar muchas llamadas al modelo: el actor, el
crítico, el extractor de memoria, el resumidor, los reintentos... Cada una deja su propio
evento en el log, y sin una etiqueta común es imposible saber cuáles pertenecen a la misma
petición del usuario. Este módulo resuelve dos cosas:

1. **`request_id`**: un identificador único por petición HTTP. Se guarda en una variable de
   contexto (`structlog.contextvars`), así que TODOS los eventos que se emiten mientras se
   atiende esa petición lo llevan, sin que cada punto del código tenga que pasarlo. Se
   devuelve además en la cabecera `X-Request-ID` para poder buscarlo en el log.
2. **Acumulador de coste (`TurnCost`)**: una "hucha" por petición. Cada llamada al modelo
   suma ahí sus tokens y su coste; al terminar la petición se emite un único evento
   `turno_costo` con el total (bucle de herramientas, reintentos, ACB, extractor y
   resumidor incluidos).

Se implementa como un middleware ASGI puro (no `BaseHTTPMiddleware`) para que las variables
de contexto lleguen intactas a los endpoints, sean `async` o se ejecuten en un hilo.
"""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import structlog

from app.tracing import emit

# Cabecera por la que el cliente puede mandar su propio identificador (se respeta si es
# razonable) y por la que se devuelve el de la petición.
REQUEST_ID_HEADER = "X-Request-ID"
# Largo máximo aceptado para un identificador que venga de fuera: evita que alguien meta un
# texto enorme en cada línea del log.
MAX_EXTERNAL_REQUEST_ID_CHARS = 64


@dataclass
class TurnCost:
    """Lo que lleva gastado la petición en curso: llamadas, tokens y dinero."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    # Llamadas a herramientas del bucle agéntico (no son llamadas al modelo).
    tool_calls: int = 0
    # Búsquedas web hechas en la petición (Paso 11): sirve para aplicar el tope por petición.
    web_searches: int = 0
    # Consultas al backend de negocio hechas en la petición (Paso 12): mismo propósito.
    business_queries: int = 0


# La "hucha" de la petición actual. Es None fuera de una petición (scripts, tests sueltos):
# en ese caso registrar un coste simplemente no hace nada.
_current_cost: ContextVar[TurnCost | None] = ContextVar("turn_cost", default=None)


def current_request_id() -> str | None:
    """El `request_id` de la petición en curso, o None si no hay ninguna."""
    value = structlog.contextvars.get_contextvars().get("request_id")
    return str(value) if value else None


def start_turn_cost() -> TurnCost:
    """Abre una hucha nueva para la petición en curso y la devuelve."""
    cost = TurnCost()
    _current_cost.set(cost)
    return cost


def current_turn_cost() -> TurnCost | None:
    """La hucha de la petición en curso, o None si no hay ninguna."""
    return _current_cost.get()


def record_llm_call(*, prompt_tokens: int, completion_tokens: int, cost_usd: float) -> None:
    """Suma una llamada al modelo a la hucha de la petición (si existe)."""
    cost = _current_cost.get()
    if cost is None:
        return
    cost.calls += 1
    cost.prompt_tokens += prompt_tokens
    cost.completion_tokens += completion_tokens
    cost.cost_usd += cost_usd


def record_tool_call() -> None:
    """Cuenta una ejecución de herramienta del bucle agéntico (si hay hucha)."""
    cost = _current_cost.get()
    if cost is not None:
        cost.tool_calls += 1


def _new_request_id(headers: list[tuple[bytes, bytes]]) -> str:
    """Identificador de la petición: el que manda el cliente si es razonable, o uno nuevo."""
    wanted = REQUEST_ID_HEADER.lower().encode("latin-1")
    for name, value in headers:
        if name.lower() == wanted:
            candidate = value.decode("latin-1").strip()
            # Solo caracteres seguros para un log: letras, números, guion y guion bajo.
            if (
                candidate
                and len(candidate) <= MAX_EXTERNAL_REQUEST_ID_CHARS
                and all(ch.isalnum() or ch in "-_" for ch in candidate)
            ):
                return candidate
    return uuid.uuid4().hex


class RequestContextMiddleware:
    """Middleware ASGI: fija el `request_id`, abre la hucha de coste y cierra el turno.

    Por cada petición HTTP: genera (o respeta) el `request_id`, lo ata al contexto de
    structlog, abre un `TurnCost`, deja pasar la petición, añade la cabecera
    `X-Request-ID` a la respuesta y, si hubo llamadas al modelo, emite `turno_costo`.
    """

    def __init__(self, app: Any) -> None:
        """Guarda la aplicación ASGI que hay "debajo" de este middleware."""
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        """Atiende una petición (solo HTTP; websockets y lifespan pasan sin tocarse)."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _new_request_id(scope.get("headers", []))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        cost = start_turn_cost()
        started = time.perf_counter()
        status_holder = {"status": 500}

        async def send_with_header(message: dict) -> None:
            """Intercepta el inicio de la respuesta para añadir la cabecera y anotar el estado."""
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER.lower().encode("latin-1"), request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_header)
        finally:
            # Solo vale la pena un evento si la petición gastó algo: /health y similares
            # no llaman al modelo y no deben llenar el log.
            if cost.calls or cost.tool_calls:
                emit(
                    __name__,
                    "turno_costo",
                    ruta=scope.get("path", ""),
                    estado_http=status_holder["status"],
                    llamadas_llm=cost.calls,
                    llamadas_herramienta=cost.tool_calls,
                    tokens_prompt=cost.prompt_tokens,
                    tokens_completion=cost.completion_tokens,
                    coste_usd=round(cost.cost_usd, 6),
                    duracion_ms=round((time.perf_counter() - started) * 1000, 1),
                )
            _current_cost.set(None)
            structlog.contextvars.clear_contextvars()
