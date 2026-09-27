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

    @property
    def can_fallback(self) -> bool:
        """¿Otro proveedor tiene chance de responder? Decidido por HTTP status.

        Regla ÚNICA y neutral: NO inspecciona clases del SDK. `openai.XError`
        y `anthropic.XError` son tipos distintos aunque se llamen igual; los
        adaptadores ya normalizaron todo a `status_code`, y esta propiedad
        solo lee ese int. Un tercer proveedor recibe la regla gratis: con
        traducir bien su error alcanza, no reimplementa nada.

        - None  -> no hubo respuesta HTTP (red/timeout): otro proveedor puede
        - 401   -> key del primario inválida/vencida: el fallback tiene OTRA key
        - 429   -> límite/cuota del primario: otro proveedor procesa
        - >=500 -> el servicio del primario está caído
        - 4xx   -> el MISMO request va a fallar igual: no gastar el fallback
        """
        return self.status_code is None or self.status_code in (401, 429) or self.status_code >= 500

    def __str__(self) -> str:  # legible en logs ("[openai] ... (HTTP 429)")
        sufijo = f" (HTTP {self.status_code})" if self.status_code else ""
        return f"[{self.provider}] {self.detail}{sufijo}"


class UnknownProviderError(Exception):
    """El nombre del proveedor no está registrado en la fábrica."""
