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

import structlog
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)
from openai import (
    AsyncOpenAI as SDKAsyncOpenAI,
)
from openai.types.responses import Response

from ..tracing import emitir
from .base import BaseProvider, LLMResponse, Message, StreamChunk, StreamDone
from .errors import LLMProviderError

# Errores del SDK que el adaptador sabe interpretar como fallos del proveedor.
# Los demás (bugs internos del adaptador) NO se traducen: deben salir como 500.
_SDK_ERRORS = (AuthenticationError, RateLimitError, APIConnectionError, APIStatusError)

# Los DOS eventos con los que una respuesta llega cerrada. Un stream cortado
# cierra con `response.incomplete`, NO con `response.completed`: escuchar solo
# el cierre feliz deja al service sin `StreamDone` y estalla con
# "terminó sin cierre". El flag de truncado NO se deduce del nombre del evento
# sino de `incomplete_details.reason` sobre la respuesta (ver _marcar_truncado).
_EVENTOS_TERMINALES = ("response.completed", "response.incomplete")


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

    def _marcar_truncado(self, respuesta: Response, *, techo: int | None) -> bool:
        """Si la respuesta llegó cortada, lo asienta en el log y devuelve True.

        Un solo lugar decide qué cuenta como truncado, para que el flag que
        viaja en la respuesta y la evidencia del log no puedan divergir.

        El log lleva el detalle justo para responder "¿hace falta subir
        `llm_max_tokens`?": si `tokens_generados` alcanzó `techo_tokens`, el
        techo fue lo que cortó y corresponde aumentarlo. Si se cortó muy por
        debajo del techo, el problema es otro y subir el número no lo arregla.
        Sin este registro esas dos preguntas se responden a ciegas.
        """
        details = respuesta.incomplete_details
        if details is None or details.reason != "max_output_tokens":
            return False

        usage = respuesta.usage
        # Proxy fresco por evento, como `emitir`: liga el log a la config
        # vigente en ese instante en vez de a un logger cacheado al importar.
        structlog.get_logger(__name__).warning(
            "response_truncated",
            provider=self.name,
            model=respuesta.model,
            motivo=details.reason,
            tokens_generados=usage.output_tokens if usage else None,
            techo_tokens=techo,
        )
        return True

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
                # effort "none" es lo que hace legal mandar `temperature` con modelos >5.0:
                # con cualquier otro effort la API lo rechaza.
                # reasoning={"effort": "none"},
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
            truncated=self._marcar_truncado(response, techo=max_tokens),
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
                # effort "none" es lo que hace legal mandar `temperature` con modelos >5.0:
                # con cualquier otro effort la API lo rechaza.
                # reasoning={"effort": "none"},
                temperature=temperature if temperature is not None else 0.3,
                max_output_tokens=max_tokens,
            ) as stream:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        yield StreamChunk(delta=event.delta)
                    elif event.type in _EVENTOS_TERMINALES:
                        response = event.response
                        usage = response.usage
                        yield StreamDone(
                            model=response.model,
                            truncated=self._marcar_truncado(response, techo=max_tokens),
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
