"""Patrón Adapter: traduce la API de Google Gemini a la interfaz común BaseProvider."""

import logging
import os

from google import genai  # SDK oficial de Google Gemini


# El SDK emite un logger.warning sobre AFC (automatic function calling).
# Solo aplica cuando se pasan herramientas; para texto simple es ruido,
# así que subimos el nivel de logging para silenciarlo.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
from google.genai import types  # tipos del SDK (Content, Part, configs de generación)
from google.genai.errors import APIError, ServerError  # errores propios del SDK

from .base import Message, LLMResponse, BaseProvider
from .errors import LLMError, RateLimitError as AppRateLimitError, AuthenticationError as AppAuthError


class GeminiProvider(BaseProvider):
    """Adaptador para Google Gemini: cumple el contrato de BaseProvider usando el SDK de Gemini."""

    name = "gemini"  # con este nombre se registra en el factory

    def __init__(self, api_key: str | None = None, model: str = "gemini-3.6-flash"):
        # Mismo patrón de seguridad que los demás adaptadores: key del entorno
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise LLMError(self.name, "Falta GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)  # cliente autenticado del SDK
        self.model = model  # modelo por defecto de este adaptador

    def get_available_models(self) -> list[str]:
        """Consulta en vivo el endpoint /models de Gemini.

        La API devuelve TODOS los modelos (imagen, audio, vídeo, TTS,
        robótica...). Se filtra a texto/chat: solo los 'gemini-*' y 'gemma-*'
        sin sufijos de tareas no textuales.
        """
        try:
            # Quitamos el prefijo "models/" que la API añade a cada nombre
            names = [m.name.removeprefix("models/") for m in self.client.models.list()]
        except APIError as e:
            raise LLMError(self.name, str(e), getattr(e, "code", None)) from e
        # Lista de subcadenas que delatan modelos NO textuales
        excluded = ("image", "audio", "live", "tts", "transcribe", "embedding",
                    "veo", "lyria", "robotics", "aqa", "nano-banana",
                    "computer-use", "deep-research", "antigravity")
        return sorted(
            n for n in names
            if (n.startswith("gemini-") or n.startswith("gemma-"))  # solo chat
            and not any(flag in n for flag in excluded)
        )

    def chat(self, messages: list[Message], model: str | None = None, temperature: float = 0.7) -> LLMResponse:
        """Estructura típica de una llamada a Google Gemini (SDK).

        Dato de aprendizaje: Gemini modela la conversación con objetos
        types.Content (rol + partes). El system prompt NO va en la lista
        de mensajes: viaja aparte, como system_instruction en el config.
        """
        model = model or self.model
        # Gemini usa el modelo de "transacción" para system prompts
        # Convierte nuestros Message a types.Content (roles "user"/"model")
        contents = [
            types.Content(role="model" if m.role == "assistant" else "user", parts=[types.Part(text=m.content)])
            for m in messages
            if m.role != "system"
        ]
        # El system prompt se separa y se pasa como instrucción de sistema
        system_parts = [types.Part(text=m.content) for m in messages if m.role == "system"]
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(  # los parámetros van en un "config"
                    system_instruction=system_parts if system_parts else None,
                    temperature=temperature,
                ),
            )
            # Normalizamos la respuesta del SDK a nuestro formato común (LLMResponse)
            return LLMResponse(
                content=response.text or "",
                model=model,
                usage={  # el SDK expone el gasto en usage_metadata
                    "prompt_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else None,
                    "completion_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else None,
                },
            )
        except ServerError as e:
            # Gemini incluye el motivo en el texto del error ("PERMISSION_DENIED"...)
            code = e.code if hasattr(e, "code") else None
            message = str(e)
            if "PERMISSION_DENIED" in message or (code and code == 403):
                raise AppAuthError(self.name, "API key inválida o sin permisos", code or 403) from e
            if "RESOURCE_EXHAUSTED" in message or (code and code == 429):
                raise AppRateLimitError(self.name, "Rate limit superado (RESOURCE_EXHAUSTED)", code or 429) from e
            raise LLMError(self.name, message, code) from e
        except APIError as e:
            raise LLMError(self.name, str(e), getattr(e, "code", None)) from e
