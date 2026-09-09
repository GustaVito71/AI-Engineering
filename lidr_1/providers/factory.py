"""Patrón Factory: un único punto que crea el adaptador correcto.

Sin esto, el código cliente tendría que hacer un if/elif por proveedor
y conocer cada SDK. Con el factory solo pide un nombre.
"""

# Se importan las clases concretas para poder registrarlas abajo
from .base import BaseProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider
from .deepseek_provider import DeepSeekProvider


class ProviderFactory:
    """Patrón Factory: crea el adaptador correcto según el nombre.

    Así el código cliente no necesita saber qué SDK usar:
    solo pide un proveedor por nombre y recibe la instancia.
    """
    # Registro de proveedores: nombre -> clase del adaptador.
    # Crecer con más proveedores NO implica tocar este diccionario.
    _registry: dict[str, type[BaseProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_class: type[BaseProvider]) -> None:
        """Añade un proveedor nuevo al catálogo del factory (diccionario)."""
        cls._registry[name] = provider_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseProvider:
        """Devuelve una instancia ya construida del adaptador pedido.

        - name: "openai", "anthropic", "gemini" o "deepseek"
        - **kwargs: opcional, p.ej. pasar la api_key manualmente.
        """
        name = name.lower()  # toleramos mayúsculas: "OpenAI" -> "openai"
        if name not in cls._registry:
            # Fallo claro y con pista, mejor que un KeyError críptico
            available = ", ".join(sorted(cls._registry))
            raise ValueError(f"Proveedor desconocido '{name}'. Disponibles: {available}")
        return cls._registry[name](**kwargs)  # llama a la clase -> construye el adaptador

    @classmethod
    def list_available(cls) -> list[str]:
        """Devuelve los nombres de proveedores registrados, ordenados."""
        return sorted(cls._registry)


# Alta de los 4 proveedores al arrancar el módulo (es la "configuración")
ProviderFactory.register(OpenAIProvider.name, OpenAIProvider)
ProviderFactory.register(AnthropicProvider.name, AnthropicProvider)
ProviderFactory.register(GeminiProvider.name, GeminiProvider)
ProviderFactory.register(DeepSeekProvider.name, DeepSeekProvider)