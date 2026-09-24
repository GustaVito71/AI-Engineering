"""providers: adaptadores de LLM con interfaz común BaseProvider.

El resto de la app solo conoce esta interfaz. Cambiar de proveedor no toca
ni routers ni servicios."""
from .base import BaseProvider, LLMResponse, Message
from .errors import LLMProviderError, UnknownProviderError
from .factory import create_provider

__all__ = [
    "BaseProvider",
    "LLMProviderError",
    "LLMResponse",
    "Message",
    "UnknownProviderError",
    "create_provider",
]