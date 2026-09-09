"""Manejo de errores y reintentos con backoff exponencial."""

import functools  # Conserva el nombre/metadatos de la función al decorarla
import time
from typing import Callable


class LLMError(Exception):
    """Error base: cualquier fallo de un proveedor se envuelve en este tipo.

    Normalizar los errores permite tratarlos igual en los 4 adaptadores.
    """

    def __init__(self, provider: str, message: str, status_code: int | None = None):
        self.provider = provider      # qué proveedor falló (openai, gemini...)
        self.message = message        # descripción del fallo
        self.status_code = status_code  # código HTTP si la API lo devolvió
        # Mensaje legible: "[openai] API key inválida (HTTP 401)"
        super().__init__(f"[{provider}] {message} (HTTP {status_code})" if status_code else f"[{provider}] {message}")


class RetryableError(LLMError):
    """Marca los errores que SÍ merecen reintentarse (problemas temporales)."""
    pass


class RateLimitError(RetryableError):
    """El proveedor nos pide esperar: nos limitaron por cuota de peticiones."""
    pass


class AuthenticationError(LLMError):
    """Key inválida/prohibida: retintentar no sirve, hay que arreglar la credencial."""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> Callable:
    """Decorator: reintenta una función que llama a un LLM.

    - max_retries: cuántas veces reintentar antes de rendirse.
    - base_delay: segundos a esperar antes del primer reintento.
    - backoff_factor: multiplicador exponencial (1s, 2s, 4s...).

    Un decorador "envuelve" una función: se usa como @retry_with_backoff(...)
    sobre cualquier función que trate con la red.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)  # la llamada original funcionó
                except RateLimitError as e:
                    # Error temporal: esperamos y volvemos a intentar
                    attempt += 1
                    if attempt > max_retries:
                        raise LLMError(e.provider, f"Rate limit persistente tras {max_retries} reintentos") from e
                    print(f"  -> Rate limit, reintento {attempt}/{max_retries} en {delay:.1f}s")
                    time.sleep(delay)
                    delay *= backoff_factor  # cada intento espera más (backoff)
                except (AuthenticationError, LLMError) as e:
                    # Errores no recuperables: no tiene sentido reintentar
                    raise e

        return wrapper

    return decorator