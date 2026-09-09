"""Patrón Adapter: traduce la API de OpenAI a la interfaz común BaseProvider."""

import os  # Para leer la API key de las variables de entorno

# Importamos el SDK oficial de OpenAI y sus errores propios
from openai import OpenAI, AuthenticationError, RateLimitError, APIStatusError

# Nuestra interfaz común (base) y nuestros tipos de error normalizados
from .base import Message, LLMResponse, BaseProvider
from .errors import LLMError, RateLimitError as AppRateLimitError, AuthenticationError as AppAuthError


class OpenAIProvider(BaseProvider):
    """Adaptador para OpenAI: cumple el contrato de BaseProvider usando el SDK de OpenAI."""

    name = "openai"  # con este nombre se registra en el factory

    def __init__(self, api_key: str | None = None, model: str = "gpt-4.1-mini"):
        # Si no nos pasan key, la buscamos en el entorno (.env). Es un patrón de seguridad:
        # los secretos NUNCA van en el código fuente.
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            # "fail fast": mejor detenerse aquí que fallar raro a mitad de una llamada
            raise LLMError(self.name, "Falta OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)  # cliente autenticado del SDK
        self.model = model  # modelo por defecto de este adaptador

    def get_available_models(self) -> list[str]:
        """Consulta en vivo el endpoint /models de OpenAI."""
        try:
            ids = [m.id for m in self.client.models.list()]  # petición HTTP real al servidor
        except APIStatusError as e:
            # Cualquier fallo de la API se envuelve en nuestro error normalizado
            raise LLMError(self.name, str(e), getattr(e, "status_code", None)) from e
        # La API devuelve todo: embeddings, TTS, imágenes, etc.
        # Se filtran solo los modelos de chat/texto razonado.
        excluded = ("image", "audio", "realtime", "transcribe", "tts",
                    "whisper", "sora", "embedding", "moderation", "search",
                    "codex", "davinci", "babbage", "instruct")
        return sorted(
            m for m in ids
            if not any(flag in m for flag in excluded)  # se queda si no es excluido
        )

    def chat(self, messages: list[Message], model: str | None = None, temperature: float = 0.7) -> LLMResponse:
        """Estructura típica de una llamada a OpenAI (SDK): crear una "completion"."""
        model = model or self.model  # si no pasan modelo, usamos el por defecto
        try:
            # La petición clave: enviamos los mensajes con roles "system"/"user"/"assistant"
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,  # 0 = determinista, 1 = creativo
            )
            # Normalizamos la respuesta del SDK a nuestro formato común (LLMResponse)
            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage={  # el SDK expone el consumo de tokens en "usage"
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if response.usage else None,
                },
            )
        # Mapeamos los errores específicos del SDK a nuestros errores normalizados
        # para que quien llama al adaptador no dependa del SDK concreto.
        except AuthenticationError as e:
            raise AppAuthError(self.name, "API key inválida o sin permisos", getattr(e, "status_code", 401)) from e
        except RateLimitError as e:
            raise AppRateLimitError(self.name, "Rate limit superado", getattr(e, "status_code", 429)) from e
        except APIStatusError as e:
            # Cualquier otro fallo HTTP (404 modelo inexistente, 500 del servidor...)
            raise LLMError(self.name, str(e), getattr(e, "status_code", None)) from e