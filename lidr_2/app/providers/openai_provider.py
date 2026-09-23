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
"""
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)
from openai import (
    OpenAI as SDKOpenAI,
)

from .base import BaseProvider, LLMResponse, Message
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
        self.client = SDKOpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.model = model

    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        try:
            response = self.client.responses.create(
                model=self.model,
                input=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature if temperature is not None else 0.3,
                max_output_tokens=max_tokens,
            )
        except _SDK_ERRORS as exc:
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