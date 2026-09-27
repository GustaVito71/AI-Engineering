"""Interfaz común de los proveedores LLM (target del patrón Adapter).

Quien usa un proveedor (el servicio) solo conoce estas clases; nunca el SDK.
Así el switch openai <-> anthropic no toca ni el router ni el servicio."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
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


@dataclass(frozen=True)
class StreamChunk:
    """Un fragmento de texto del streaming.

    El consumidor (servicio) lo reenvía tal cual: el renderizado a eventos SSE
    es responsabilidad del router, este objeto es de dominio."""

    delta: str


@dataclass(frozen=True)
class StreamDone:
    """Cierre normalizado de un stream, mismos campos que LLMResponse."""

    model: str
    truncated: bool
    usage: dict[str, int | None] | None


class BaseProvider(ABC):
    """Contrato que todo adaptador debe cumplir."""

    name: str = "base"
    default_model: str = ""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Envía la conversación y devuelve la respuesta normalizada.

        Contrato async de punta a punta: los adaptadores usan los clientes
        async del SDK (AsyncOpenAI / AsyncAnthropic) para que una llamada
        lenta no bloquee el event loop."""

    @abstractmethod
    def chat_stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk | StreamDone]:
        """Transmite la conversación en fragmentos (streaming).

        Las implementaciones son generadores asíncronos: el llamador los
        recorre con `async for`. Deben ceder `StreamChunk` por cada fragmento
        de texto y UN `StreamDone` al cierre con los datos normalizados. Los
        errores de la API del proveedor se lanzan como LLMProviderError
        (mismo contrato que `chat`), tanto al abrir como a mitad del stream.
        Quien consume el stream decide qué hacer con un fallo a mitad según
        su política (el servicio solo hace fallback ANTES del primer
        fragmento: texto que arrancó = comprometido)."""
        yield  # pragma: no cover — los adaptadores implementan el generador
