"""Errores normalizados de los proveedores.

Quien llama a un adaptador captura SOLO estas clases, no `Exception`.
Un error propio del SDK se traduce aquí; un bug del adaptador (AttributeError,
TypeError...) no se traduce: tiene que salir como 500 para no confundirse
con un fallo del proveedor en el dashboard del proveedor."""
from dataclasses import dataclass


@dataclass
class LLMProviderError(Exception):
    """Cualquier fallo del proveedor envuelto en un tipo propio.

    El `detail` interno NO viaja al cliente: se loguea en el servidor."""
    provider: str
    detail: str
    status_code: int | None = None

    def __str__(self) -> str:  # legible en logs ("[openai] ... (HTTP 429)")
        sufijo = f" (HTTP {self.status_code})" if self.status_code else ""
        return f"[{self.provider}] {self.detail}{sufijo}"


class UnknownProviderError(Exception):
    """El nombre del proveedor no está registrado en la fábrica."""