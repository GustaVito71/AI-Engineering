"""Interfaz común de los proveedores LLM (target del patrón Adapter).

Quien usa un proveedor (el servicio) solo conoce estas clases; nunca el SDK.
Así el switch openai <-> anthropic no toca ni el router ni el servicio."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    """Un turno de conversación, igual que en los chats."""
    role: Role
    content: str


@dataclass
class LLMResponse:
    """Respuesta normalizada, sin vocabulario de ningún SDK.

    truncated es la señal más importante: el SDK decía por qué paró de escribir
    (finish_reason / stop_reason). Normalizarla a un bool evita que el servicio
    dependa de covención de cada proveedor."""
    content: str
    model: str
    truncated: bool
    usage: dict[str, int | None] | None


class BaseProvider(ABC):
    """Contrato que todo adaptador debe cumplir."""

    name: str = "base"
    default_model: str = ""

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Envía la conversación y devuelve la respuesta normalizada."""