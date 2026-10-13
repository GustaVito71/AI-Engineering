"""Adaptador Anthropic: traduce la Messages API a BaseProvider.

Misma robustez que el de OpenAI: timeout/max_retries en el cliente y lectura
de stop_reason (Anthropic llama "max_tokens" al corte por límite de salida).

Diferencia de contrato con OpenAI: el system prompt se envía por separado
(`system=`), no como un rol dentro de `messages`. El adaptador parte la lista
normalizada y rearma el payload que el SDK de Anthropic espera.

Usa AsyncAnthropic: `chat` es una corrutina y no bloquea el event loop."""

from __future__ import annotations

from collections.abc import AsyncIterator

from anthropic import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)
from anthropic import (
    AsyncAnthropic as SDKAsyncAnthropic,
)

from ..tracing import emitir
from .base import BaseProvider, LLMResponse, Message, StreamChunk, StreamDone
from .errors import LLMProviderError

_SDK_ERRORS = (AuthenticationError, RateLimitError, APIConnectionError, APIStatusError)


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    default_model = "claude-haiku-4-5"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self.client = SDKAsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.model = model

    async def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        if not max_tokens:
            raise ValueError("Anthropic exige max_tokens explícito")

        system_prompt = "\n".join(m.content for m in messages if m.role == "system")
        user_messages = [
            {"role": "user", "content": m.content} for m in messages if m.role == "user"
        ]

        try:
            response = await self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=user_messages,
                max_tokens=max_tokens,
                # La versión instalada del SDK (1.8.0) eliminó el parámetro
                # `temperature` de messages.create(), verificado contra la
                # firma del método. No se envía: por eso este adaptador ignora
                # el `temperature` del contrato BaseProvider. Si algún día hace
                # falta controlar esa palanca, hay que mirar la API de
                # Anthropic de ese momento, no asumir que sigue existiendo.
            )
        except _SDK_ERRORS as exc:
            # Dimensión 3 (camino) de la traza: fallo sanitizado del
            # proveedor (tipo + status), nunca el cuerpo crudo de la
            # respuesta. Los reintentos internos del SDK no se cuentan uno
            # a uno; el evento marca el fallo final de su cadena.
            emitir(
                __name__,
                "provider_error",
                nivel="warning",
                provider=self.name,
                tipo_error=type(exc).__name__,
                status_code=getattr(exc, "status_code", None),
                max_retries=self.client.max_retries,
            )
            raise LLMProviderError(
                provider=self.name,
                detail="Error de la API de Anthropic",
                status_code=getattr(exc, "status_code", None),
            ) from exc

        usage = response.usage
        return LLMResponse(
            content="".join(bloque.text for bloque in response.content if bloque.type == "text"),
            model=response.model,
            truncated=response.stop_reason == "max_tokens",
            usage=(
                {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                }
                if usage
                else None
            ),
        )

    async def chat_stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk | StreamDone]:
        """Streaming con la Messages API.

        El manager `client.messages.stream(...)` no es iterable directamente:
        el SDK expone `until_done()` para recorrer los eventos raw. Los que
        normalizamos:
        - message_start: trae message.model y el usage con input_tokens
          (los output_tokens llegan recién en message_delta).
        - content_block_delta: con delta.type == "text_delta" → texto nuevo.
        - message_delta: cierre con delta.stop_reason ("max_tokens" =
          truncado) y usage.output_tokens.

        Nota: igual que en `chat`, `temperature` del contrato BaseProvider
        se ignora porque el SDK 1.8.0 lo eliminó de esta API."""
        if not max_tokens:
            raise ValueError("Anthropic exige max_tokens explícito")

        system_prompt = "\n".join(m.content for m in messages if m.role == "system")
        user_messages = [
            {"role": "user", "content": m.content} for m in messages if m.role == "user"
        ]

        model = self.model
        input_tokens: int | None = None
        try:
            async with self.client.messages.stream(
                model=self.model,
                system=system_prompt,
                messages=user_messages,
                max_tokens=max_tokens,
            ) as stream:
                # El stream se itera DIRECTAMENTE: `AsyncMessageStream` es
                # asíncrono iterable. `until_done()` no sirve aquí — está
                # anotado `-> None`, consume el stream hasta el final y no
                # devuelve el iterador de eventos. Usarlo de las dos formas
                # rompe el streaming de Anthropic entero, que es justo el
                # camino del fallback.
                async for chunk in stream:
                    if chunk.type == "message_start":
                        model = chunk.message.model
                        usage = chunk.message.usage
                        if usage:
                            input_tokens = usage.input_tokens
                    elif chunk.type == "content_block_delta":
                        delta = chunk.delta
                        if delta.type == "text_delta":
                            yield StreamChunk(delta=delta.text)
                    elif chunk.type == "message_delta":
                        done_usage = chunk.usage
                        yield StreamDone(
                            model=model,
                            truncated=chunk.delta.stop_reason == "max_tokens",
                            usage=(
                                {
                                    "input_tokens": input_tokens,
                                    "output_tokens": done_usage.output_tokens,
                                }
                                if done_usage
                                else None
                            ),
                        )
        except _SDK_ERRORS as exc:
            # Mismo sanitizado que chat(): solo tipo y status, nunca el body.
            emitir(
                __name__,
                "provider_error",
                nivel="warning",
                provider=self.name,
                tipo_error=type(exc).__name__,
                status_code=getattr(exc, "status_code", None),
                max_retries=self.client.max_retries,
            )
            raise LLMProviderError(
                provider=self.name,
                detail="Error de la API de Anthropic",
                status_code=getattr(exc, "status_code", None),
            ) from exc
