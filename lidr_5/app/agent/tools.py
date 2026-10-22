"""Herramientas que el modelo puede pedir durante el bucle agéntico (Paso 10).

QUÉ ES UNA HERRAMIENTA (en lenguaje llano)

Una herramienta es una función de nuestro servicio (buscar en la web, consultar la base de
datos del negocio...) que el modelo puede "pedir" en medio de una conversación. El modelo no
la ejecuta: devuelve "quiero usar la herramienta X con estos parámetros", el servicio la
ejecuta y le devuelve el resultado. Este módulo define cómo se describe una herramienta
(`Tool`), dónde se registran (`ToolRegistry`) y cómo se presenta su resultado al modelo
(`wrap_tool_result`).

Los Pasos 11 (búsqueda web) y 12 (base de datos del negocio) registrarán herramientas aquí.
En el Paso 10 el registro llega vacío: el bucle existe, pero no tiene nada que ejecutar.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

# Una herramienta recibe los argumentos que pidió el modelo (un diccionario ya parseado) y
# devuelve su resultado como TEXTO, de forma síncrona o asíncrona.
ToolFunction = Callable[[dict[str, Any]], str | Awaitable[str]]

# Segundos máximos que puede tardar una herramienta antes de darla por fallida.
DEFAULT_TOOL_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class Tool:
    """Una herramienta: su definición para el modelo y la función que la ejecuta.

    `description` es lo que lee el modelo para decidir CUÁNDO usarla: debe decirlo con
    claridad («Úsala solo si el usuario menciona una tecnología que no conoces»).
    `parameters` es un JSON Schema de los argumentos (`{"type": "object", ...}`).
    """

    name: str
    description: str
    parameters: dict[str, Any]
    run: ToolFunction
    timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS

    def spec(self) -> dict[str, Any]:
        """La definición en el formato de herramientas de OpenAI, que LiteLLM traduce a cada
        proveedor (Anthropic incluido).
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolRegistry:
    """El conjunto de herramientas disponibles en el servicio, por nombre."""

    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        """Añade una herramienta. Un nombre repetido es un error de programación."""
        if tool.name in self._tools:
            raise ValueError(f"La herramienta '{tool.name}' ya está registrada.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """La herramienta con ese nombre, o None si el modelo pidió una que no existe."""
        return self._tools.get(name)

    def specs(self) -> list[dict[str, Any]]:
        """Las definiciones de todas las herramientas, listas para enviar al modelo."""
        return [tool.spec() for tool in self._tools.values()]

    @property
    def names(self) -> list[str]:
        """Los nombres de las herramientas registradas."""
        return list(self._tools)

    def __len__(self) -> int:
        """Cuántas herramientas hay registradas (0 = el bucle no tiene nada que hacer)."""
        return len(self._tools)


def wrap_tool_result(name: str, text: str) -> str:
    """Presenta el resultado de una herramienta como DATO, nunca como instrucción.

    El texto va entre etiquetas `<tool_result>` y se neutraliza cualquier etiqueta de cierre
    que traiga dentro (una página web maliciosa podría incluir `</tool_result>` y luego
    órdenes que parezcan venir del sistema). El prompt del bucle indica al modelo que lo que
    hay entre esas etiquetas son datos externos.
    """
    safe = text.replace("</tool_result", "&lt;/tool_result").replace(
        "<tool_result", "&lt;tool_result"
    )
    return f'<tool_result name="{name}">\n{safe}\n</tool_result>'
