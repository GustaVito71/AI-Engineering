"""Patrón Adapter: traduce la API de Anthropic (Claude) a la interfaz común BaseProvider."""

import os  # Para leer la API key de las variables de entorno

import anthropic  # SDK oficial de Anthropic
# Importamos los errores propios del SDK
from anthropic import AuthenticationError, RateLimitError, APIStatusError

# Nuestra interfaz común (base) y nuestros tipos de error normalizados
from .base import Message, LLMResponse, BaseProvider
from .errors import LLMError, RateLimitError as AppRateLimitError, AuthenticationError as AppAuthError


class AnthropicProvider(BaseProvider):
    """Adaptador para Anthropic (Claude): cumple el contrato de BaseProvider usando el SDK de Anthropic."""

    name = "anthropic"  # con este nombre se registra en el factory

    def __init__(self, api_key: str | None = None, model: str = "claude-haiku-4-5-20251001"):
        # Mismo patrón de seguridad que OpenAI: key del entorno, nunca en el código
        if not api_key:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError(self.name, "Falta ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)  # cliente autenticado del SDK
        self.model = model  # modelo por defecto de este adaptador

    def get_available_models(self) -> list[str]:
        """Consulta en vivo el endpoint /models de Anthropic."""
        try:
            # .data: la lista de modelos viene dentro de una estructura paginada
            return [m.id for m in self.client.models.list(limit=50).data]
        except anthropic.APIStatusError as e:
            raise LLMError(self.name, str(e), getattr(e, "status_code", None)) from e

    def chat(self, messages: list[Message], model: str | None = None, max_tokens: int = 1024) -> LLMResponse:
        """Estructura típica de una llamada al API de Anthropic (SDK).

        Dato de aprendizaje: la API de Anthropic es DISTINTA a la de OpenAI.
        La conversación se envía en messages, pero el "system prompt" viaja
        aparte, en el parámetro system. Por eso este adaptador lo separa.
        """
        model = model or self.model
        try:
            # Anthropic separa el system prompt del resto de mensajes
            system_text = "\n".join(m.content for m in messages if m.role == "system")
            user_msgs = [  # aquí solo van user/assistant, como dicts
                {"role": m.role, "content": m.content}
                for m in messages
                if m.role in ("user", "assistant")
            ]
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,  # Anthropic exige fijar el tope de tokens a generar
                system=[{"type": "text", "text": system_text}] if system_text else [],
                messages=user_msgs,
            )
            # La respuesta trae bloques; juntamos solo los de texto
            text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            # Normalizamos la respuesta del SDK a nuestro formato común (LLMResponse)
            return LLMResponse(
                content=text,
                model=response.model,
                usage={  # cuidado: Anthropic lo llama input/output, no prompt/completion
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            )
        # Mapeamos los errores del SDK a nuestros errores normalizados
        except AuthenticationError as e:
            raise AppAuthError(self.name, "API key inválida o sin permisos", getattr(e, "status_code", 401)) from e
        except RateLimitError as e:
            raise AppRateLimitError(self.name, "Rate limit superado", getattr(e, "status_code", 429)) from e
        except APIStatusError as e:
            raise LLMError(self.name, str(e), getattr(e, "status_code", None)) from e