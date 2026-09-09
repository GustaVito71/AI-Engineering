"""Patrón Adapter: traduce la API de DeepSeek a la interfaz común BaseProvider.

Dato de aprendizaje CLAVE: DeepSeek usa una API compatible con OpenAI.
El adaptador NO necesita un SDK propio: reutiliza el SDK de OpenAI
y solo cambia la URL base a la que apunta.
"""

import os

from openai import OpenAI, AuthenticationError, RateLimitError, APIStatusError

from .base import Message, LLMResponse, BaseProvider
from .errors import LLMError, RateLimitError as AppRateLimitError, AuthenticationError as AppAuthError


class DeepSeekProvider(BaseProvider):
    """Adaptador para DeepSeek. Solo cambia base_url del cliente de OpenAI."""

    name = "deepseek"  # con este nombre se registra en el factory

    BASE_URL = "https://api.deepseek.com"  # la única diferencia real con OpenAI

    def __init__(self, api_key: str | None = None, model: str = "deepseek-v4-flash"):
        # Mismo patrón de seguridad que los demás: key del entorno, nunca en código
        if not api_key:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise LLMError(self.name, "Falta DEEPSEEK_API_KEY")
        # Cliente de OpenAI apuntando al servidor de DeepSeek
        self.client = OpenAI(api_key=api_key, base_url=self.BASE_URL)
        self.model = model

    def get_available_models(self) -> list[str]:
        """Consulta en vivo /models vía el endpoint compatible con OpenAI."""
        try:
            return sorted(m.id for m in self.client.models.list())
        except APIStatusError as e:
            raise LLMError(self.name, str(e), getattr(e, "status_code", None)) from e

    def chat(self, messages: list[Message], model: str | None = None, temperature: float = 0.7) -> LLMResponse:
        """Estructura de llamada: IDÉNTICA a OpenAI (sirve este mismo SDK)."""
        model = model or self.model
        try:
            # Misma forma que en openai_provider: chat.completions.create
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
            )
            # Normalizamos la respuesta del SDK a nuestro formato común (LLMResponse)
            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage={  # mismo esquema de tokens que OpenAI
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if response.usage else None,
                },
            )
        # Mapeamos los errores del SDK a nuestros errores normalizados
        except AuthenticationError as e:
            raise AppAuthError(self.name, "API key inválida o sin permisos", getattr(e, "status_code", 401)) from e
        except RateLimitError as e:
            raise AppRateLimitError(self.name, "Rate limit superado", getattr(e, "status_code", 429)) from e
        except APIStatusError as e:
            raise LLMError(self.name, str(e), getattr(e, "status_code", None)) from e