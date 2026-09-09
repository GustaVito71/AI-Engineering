"""Interfaz común (el "target" del patrón Adapter).

Aquí se define la forma en la que TODO el resto del programa
habla con cualquier LLM, sin importar el proveedor real.
"""

from abc import ABC, abstractmethod  # Para crear clases "esqueleto" que obligan a implementar métodos
from dataclasses import dataclass  # Genera __init__, __repr__, etc. automáticamente
from typing import Literal

# Un rol solo puede ser uno de estos tres valores (ayuda de tipos)
Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    """Un turno de conversación, igual que en los chats."""
    role: Role      # quién habla (sistema, usuario o asistente)
    content: str    # el texto del mensaje


@dataclass
class LLMResponse:
    """Respuesta normalizada que devuelven todos los adaptadores."""
    content: str                 # el texto que respondió el LLM
    model: str                   # qué modelo respondió
    usage: dict | None = None    # consumo de tokens (opcional)


class BaseProvider(ABC):
    """Contrato que toda implementación (adaptador) debe cumplir."""

    name: str = "base"  # nombre corto con el que se registra en el factory

    @abstractmethod
    def get_available_models(self) -> list[str]:
        """Devuelve la lista de modelos disponibles (consulta en vivo)."""
        pass

    @abstractmethod
    def chat(self, messages: list[Message], model: str, **kwargs) -> LLMResponse:
        """Envía una conversación y devuelve la respuesta del LLM."""
        pass