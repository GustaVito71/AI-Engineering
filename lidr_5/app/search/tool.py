"""La herramienta `web_search` que el modelo puede pedir durante el bucle (Paso 11).

QUÉ HACE ESTE MÓDULO (en lenguaje llano)

Convierte el cliente de búsqueda en una `Tool` del Paso 10: define qué le dice al modelo
sobre cuándo usarla y qué parámetros lleva, aplica el tope de búsquedas por petición y
devuelve los resultados como TEXTO (el bucle los envuelve en `<tool_result>` para que el
modelo los trate como datos y no como instrucciones). También deja el evento de traza
`busqueda_web` con la duración.

Si la búsqueda no está disponible, la herramienta NO lanza: devuelve un texto que lo explica
y el modelo sigue sin ella. Una búsqueda caída nunca debe romper una estimación.
"""

from __future__ import annotations

from typing import Any

from app.agent.tools import Tool
from app.config import Settings
from app.observability import current_turn_cost
from app.search.client import SearchOutcome, SearchUnavailableError, WebSearchClient
from app.tracing import emit

TOOL_NAME = "web_search"
# Largo máximo del texto de la consulta que se registra cuando SEARCH_LOG_QUERIES=true.
MAX_LOGGED_QUERY_CHARS = 80
# Largo máximo de la consulta que se acepta del modelo (evita consultas gigantes).
MAX_QUERY_CHARS = 300

TOOL_DESCRIPTION = (
    "Busca en internet información reciente o que no conoces. Úsala SOLO si la estimación "
    "depende de algo actual o posterior a tu entrenamiento (versiones, precios, tecnologías "
    "nuevas, normativa). No la uses para lo que ya está en la conversación o en los adjuntos. "
    "Devuelve hasta 3 resultados con título, dirección y un fragmento."
)

TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Consulta breve y concreta, como la escribirías en un buscador.",
        }
    },
    "required": ["query"],
}


def format_outcome(outcome: SearchOutcome) -> str:
    """Los resultados como texto numerado, listo para que el modelo los lea."""
    lines = [f"Resultados de búsqueda ({outcome.provider}):"]
    for position, result in enumerate(outcome.results, start=1):
        lines.append(f"{position}. {result.title}\n   {result.url}\n   {result.snippet}")
    return "\n".join(lines)


def _query_log_fields(query: str, log_queries: bool) -> dict[str, Any]:
    """Qué se registra de la consulta: su largo y, solo si está permitido, el texto truncado."""
    fields: dict[str, Any] = {"largo_consulta": len(query)}
    if log_queries:
        fields["consulta"] = query[:MAX_LOGGED_QUERY_CHARS]
    return fields


def build_web_search_tool(
    client: WebSearchClient,
    *,
    max_per_turn: int,
    log_queries: bool,
    timeout_seconds: float,
) -> Tool:
    """Arma la herramienta `web_search` sobre un cliente de búsqueda.

    `timeout_seconds` es el de cada servicio; la herramienta da margen para el principal y el
    respaldo seguidos.
    """

    async def run(arguments: dict[str, Any]) -> str:
        """Ejecuta una búsqueda pedida por el modelo y devuelve el texto del resultado."""
        query = str(arguments.get("query", "")).strip()[:MAX_QUERY_CHARS]
        if not query:
            return "Búsqueda no realizada: falta el parámetro 'query'."

        # El tope por petición vive en la "hucha" del turno (objeto compartido).
        cost = current_turn_cost()
        if cost is not None:
            if cost.web_searches >= max_per_turn:
                emit(
                    __name__,
                    "busqueda_web_tope",
                    level="warning",
                    tope=max_per_turn,
                    **_query_log_fields(query, log_queries),
                )
                return (
                    f"Búsqueda no realizada: se alcanzó el límite de {max_per_turn} "
                    "búsquedas por petición. Continúa con lo que ya tienes."
                )
            cost.web_searches += 1

        try:
            outcome = await client.search(query)
        except SearchUnavailableError:
            emit(
                __name__,
                "busqueda_web",
                level="warning",
                exito=False,
                **_query_log_fields(query, log_queries),
            )
            return "Búsqueda no disponible en este momento. Continúa sin ella."

        emit(
            __name__,
            "busqueda_web",
            exito=True,
            proveedor=outcome.provider,
            uso_respaldo=outcome.used_fallback,
            resultados=len(outcome.results),
            duracion_ms=outcome.duration_ms,
            **_query_log_fields(query, log_queries),
        )
        return format_outcome(outcome)

    return Tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        parameters=TOOL_PARAMETERS,
        run=run,
        # Principal + respaldo seguidos, más un pequeño margen.
        timeout_seconds=timeout_seconds * 2 + 5,
    )


def build_client_from_settings(settings: Settings) -> WebSearchClient:
    """El cliente de búsqueda configurado según `Settings` (claves, orden, límites)."""
    keys = {
        "tavily": settings.web_tavily_key.get_secret_value() if settings.web_tavily_key else "",
        "firecrawl": (
            settings.web_firecrawl_key.get_secret_value() if settings.web_firecrawl_key else ""
        ),
    }
    return WebSearchClient(
        keys=keys,
        primary=settings.search_default,
        fallback=settings.search_fallback,
        timeout_seconds=settings.search_timeout_seconds,
        max_snippet_chars=settings.search_max_snippet_chars,
    )


def build_tool_from_settings(settings: Settings) -> Tool | None:
    """La herramienta `web_search`, o None si no hay ninguna clave de búsqueda."""
    client = build_client_from_settings(settings)
    if not client.available:
        return None
    return build_web_search_tool(
        client,
        max_per_turn=settings.search_max_per_turn,
        log_queries=settings.search_log_queries,
        timeout_seconds=settings.search_timeout_seconds,
    )
