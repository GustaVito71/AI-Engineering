"""Adaptador OpenAI: traduce la Responses API a BaseProvider.

Decisiones de robustez (ver revisión de la sesión):
- timeout y max_retries se pasan en la construcción del cliente. El default
  del SDK es 600 s con 2 reintentos: diez minutos por llamada, media hora
  hasta que muere. El número se define en Settings y se puede subir sin
  tocar código.
- Se leen los campos de la respuesta que importan: usage e incomplete_details.
  La Responses API no tiene finish_reason=="length" como Chat Completions:
  el truncado por límite de tokens se detecta con
  response.incomplete_details.reason == "max_output_tokens".
- Cliente async (AsyncOpenAI): `chat` es una corrutina, la llamada lenta al
  proveedor cede el event loop en lugar de congelarlo.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)
from openai import (
    AsyncOpenAI as SDKAsyncOpenAI,
)

from ..tracing import emitir
from .base import BaseProvider, LLMResponse, Message, StreamChunk, StreamDone
from .errors import LLMProviderError

# Errores del SDK que el adaptador sabe interpretar como fallos del proveedor.
# Los demás (bugs internos del adaptador) NO se traducen: deben salir como 500.
_SDK_ERRORS = (AuthenticationError, RateLimitError, APIConnectionError, APIStatusError)


class OpenAIProvider(BaseProvider):
    name = "openai"
    default_model = "gpt-4o-mini"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self.client = SDKAsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.model = model

    async def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        try:
            response = await self.client.responses.create(
                model=self.model,
                input=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature if temperature is not None else 0.3,
                max_output_tokens=max_tokens,
            )
        except _SDK_ERRORS as exc:
            # Dimensión 3 (camino) de la traza: cada fallo de la API del
            # proveedor queda registrado con datos SANITIZADOS (tipo y
            # status), nunca el cuerpo crudo de la respuesta — puede
            # contener datos del request. Los reintentos internos del SDK
            # (backoff) no se cuentan uno a uno; el evento marca el fallo
            # que la API devolvió al final de su cadena de reintentos.
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
                detail="Error de la API de OpenAI",
                status_code=getattr(exc, "status_code", None),
            ) from exc

        usage = response.usage
        return LLMResponse(
            content=response.output_text,
            model=response.model,
            truncated=(
                response.incomplete_details is not None
                and response.incomplete_details.reason == "max_output_tokens"
            ),
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
        """Streaming con la Responses API.

        El manager `client.responses.stream(...)` es un context manager
        asíncrono que ya deja el stream abierto. Cada evento de la iteración
        trae un type propio; los que nos interesan:
        - response.output_text.delta: un fragmento de texto (evento.delta).
        - response.completed: cierre normal con `response` ya completa
          (model, usage e incomplete_details para el flag de truncado).
        - response.failed: la API avisó que el request falló. En la práctica
          los errores HTTP/red llegan como excepciones del SDK durante la
          iteración; el try/except _SDK_ERRORS las captura igual que en
          `chat`, y response.failed sirve de red de seguridad si algún día
          el SDK decide emitirlo en vez de lanzar.
        """
        try:
            async with self.client.responses.stream(
                model=self.model,
                input=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature if temperature is not None else 0.3,
                max_output_tokens=max_tokens,
            ) as stream:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        yield StreamChunk(delta=event.delta)
                    elif event.type == "response.completed":
                        response = event.response
                        usage = response.usage
                        yield StreamDone(
                            model=response.model,
                            truncated=(
                                response.incomplete_details is not None
                                and response.incomplete_details.reason == "max_output_tokens"
                            ),
                            usage=(
                                {
                                    "input_tokens": usage.input_tokens,
                                    "output_tokens": usage.output_tokens,
                                }
                                if usage
                                else None
                            ),
                        )
                    elif event.type == "response.failed":
                        # La API avisó el fallo sin lanzar excepción (caso raro):
                        # lo traducimos igual que un error lanzado.
                        emitir(
                            __name__,
                            "provider_error",
                            nivel="warning",
                            provider=self.name,
                            tipo_error="ResponseFailedEvent",
                            max_retries=self.client.max_retries,
                        )
                        raise LLMProviderError(
                            provider=self.name,
                            detail="Error de la API de OpenAI",
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
                detail="Error de la API de OpenAI",
                status_code=getattr(exc, "status_code", None),
            ) from exc
