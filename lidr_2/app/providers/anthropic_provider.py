"""Adaptador Anthropic: traduce la Messages API a BaseProvider.

Misma robustez que el de OpenAI: timeout/max_retries en el cliente y lectura
de stop_reason (Anthropic llama "max_tokens" al corte por límite de salida).

Diferencia de contrato con OpenAI: el system prompt se envía por separado
(`system=`), no como un rol dentro de `messages`. El adaptador parte la lista
normalizada y rearma el payload que el SDK de Anthropic espera."""

from anthropic import (
    Anthropic as SDKAnthropic,
)
from anthropic import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    RateLimitError,
)

from .base import BaseProvider, LLMResponse, Message
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
        self.client = SDKAnthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.model = model

    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        if not max_tokens:
            raise ValueError("Anthropic exige max_tokens explícito")

        system_prompt = "\n".join(
            m.content for m in messages if m.role == "system"
        )
        user_messages = [
            {"role": "user", "content": m.content} for m in messages if m.role == "user"
        ]

        try:
            response = self.client.messages.create(
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