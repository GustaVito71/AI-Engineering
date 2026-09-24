"""Fábrica de adaptadores LLM (patrón Factory).

Recibe argumentos ya resueltos (incluida la key, extraída en el punto de uso
por el servicio); aquí solo se mapea el nombre a la clase. Ninguna decisión
de configuración vive en esta capa."""
from .anthropic_provider import AnthropicProvider
from .base import BaseProvider
from .errors import UnknownProviderError
from .openai_provider import OpenAIProvider

_PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def create_provider(
    name: str,
    api_key: str,
    model: str,
    *,
    timeout: float = 30.0,
    max_retries: int = 2,
) -> BaseProvider:
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise UnknownProviderError(
            f"Proveedor '{name}' no implementado. Disponibles: {list(_PROVIDERS)}"
        )
    return cls(api_key=api_key, model=model, timeout=timeout, max_retries=max_retries)