"""La herramienta `find_similar_projects` del bucle agéntico (Paso 12).

QUÉ HACE ESTE MÓDULO (en lenguaje llano)

Convierte el cliente del backend de negocio en una `Tool` del Paso 10: describe al modelo
cuándo conviene usarla, aplica el tope de consultas por petición y devuelve el resultado como
TEXTO (el bucle lo envuelve en `<tool_result>`, como dato y no como instrucción).

Igual que la búsqueda web (Paso 11): si el backend no responde, la herramienta NO lanza,
devuelve un texto que lo explica y el modelo sigue sin esa información.
"""

from __future__ import annotations

from typing import Any

from app.agent.tools import Tool
from app.business.client import (
    BusinessBackendClient,
    BusinessBackendUnavailableError,
    SimilarProject,
)
from app.config import Settings
from app.observability import current_turn_cost
from app.tracing import emit

TOOL_NAME = "find_similar_projects"
MAX_QUERY_CHARS = 200

TOOL_DESCRIPTION = (
    "Busca en el histórico de proyectos de la empresa proyectos parecidos al que se está "
    "estimando (horas reales, desviación sobre lo estimado y riesgos que se materializaron). "
    "Úsala cuando esos datos propios de la empresa puedan afinar la estimación. No la uses "
    "si el usuario no mencionó nada que amerite compararlo con proyectos anteriores."
)

TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "project_type": {
            "type": "string",
            "description": "Tipo de proyecto a buscar, p. ej. 'migración de API' o 'e-commerce'.",
        }
    },
    "required": ["project_type"],
}


def format_projects(projects: list[SimilarProject]) -> str:
    """Los proyectos históricos como texto numerado, listo para que el modelo los lea."""
    lines = ["Proyectos históricos parecidos:"]
    for position, project in enumerate(projects, start=1):
        riesgos = "; ".join(project.risks_materialized) or "sin riesgos registrados"
        lines.append(
            f"{position}. {project.project_type}: estimado {project.hours_estimated:.0f} h, "
            f"real {project.hours_actual:.0f} h (desviación {project.deviation_pct:+.1f}%). "
            f"Riesgos: {riesgos}."
        )
    return "\n".join(lines)


def build_find_similar_projects_tool(client: BusinessBackendClient, *, max_per_turn: int) -> Tool:
    """Arma la herramienta `find_similar_projects` sobre un cliente del backend de negocio."""

    async def run(arguments: dict[str, Any]) -> str:
        """Ejecuta una consulta pedida por el modelo y devuelve el texto del resultado."""
        project_type = str(arguments.get("project_type", "")).strip()[:MAX_QUERY_CHARS]
        if not project_type:
            return "Consulta no realizada: falta el parámetro 'project_type'."

        cost = current_turn_cost()
        if cost is not None:
            if cost.business_queries >= max_per_turn:
                emit(
                    __name__,
                    "consulta_negocio_tope",
                    level="warning",
                    tope=max_per_turn,
                )
                return (
                    f"Consulta no realizada: se alcanzó el límite de {max_per_turn} "
                    "consultas al histórico por petición. Continúa con lo que ya tienes."
                )
            cost.business_queries += 1

        try:
            projects = await client.find_similar(project_type)
        except BusinessBackendUnavailableError:
            emit(__name__, "consulta_negocio", level="warning", exito=False)
            return "Histórico de proyectos no disponible en este momento. Continúa sin él."

        emit(__name__, "consulta_negocio", exito=True, resultados=len(projects))
        if not projects:
            return "No se encontraron proyectos históricos parecidos."
        return format_projects(projects)

    return Tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        parameters=TOOL_PARAMETERS,
        run=run,
    )


def build_client_from_settings(settings: Settings) -> BusinessBackendClient | None:
    """El cliente del backend de negocio, o None si falta la URL o la clave."""
    if not settings.business_api_url or not settings.business_api_key:
        return None
    return BusinessBackendClient(
        base_url=settings.business_api_url,
        api_key=settings.business_api_key.get_secret_value(),
        timeout_seconds=settings.business_timeout_seconds,
    )


def build_tool_from_settings(settings: Settings) -> Tool | None:
    """La herramienta `find_similar_projects`, o None si no hay backend de negocio configurado."""
    client = build_client_from_settings(settings)
    if client is None:
        return None
    return build_find_similar_projects_tool(client, max_per_turn=settings.business_max_per_turn)
