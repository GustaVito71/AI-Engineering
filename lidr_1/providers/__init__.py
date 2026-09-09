"""Paquete 'providers'.

Al importar este paquete (from providers import ...) ocurren dos cosas:
1. Se carga el .env con las API keys (seguridad: nunca en código).
2. Se exponen las clases públicas que el resto del programa usa.

El import de abajo y este archivo se recargan al hacer 'from providers import ...'
"""
from pathlib import Path

from dotenv import load_dotenv  # lee el archivo .env y lo mete en os.environ


_LOADED = False  # evita cargar el .env varias veces (importación múltiple)


def load_env_file() -> None:
    """Carga las API keys del .env en las variables de entorno (una sola vez)."""
    global _LOADED  # aviso: modificamos la variable del módulo
    if _LOADED:
        return
    # El .env está una carpeta por encima del paquete (en la raíz del proyecto)
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        # override=False -> las variables ya exportadas en el shell ganan al .env
        load_dotenv(env_path, override=False)
        print(f"  -> .env cargado desde {env_path}")
    _LOADED = True


load_env_file()  # se ejecuta al importar el paquete, antes que cualquier adaptador

# Exportaciones públicas: basta con 'from providers import Message, ProviderFactory'...
# Los import de código se ponen DESPUÉS de load_env_file para que los adaptadores
# ya encuentren las keys en el entorno al construirse.
from .base import BaseProvider, Message, LLMResponse
from .errors import LLMError, RateLimitError, AuthenticationError, retry_with_backoff
from .factory import ProviderFactory

__all__ = [  # los nombres "oficiales" del paquete (from providers import *)
    "BaseProvider",
    "Message",
    "LLMResponse",
    "LLMError",
    "RateLimitError",
    "AuthenticationError",
    "retry_with_backoff",
    "ProviderFactory",
]