"""Servicio de estimación (dominio, agnóstico de HTTP).

El servicio no sabe qué es un 502. Dice «esto falló» con sus propias
excepciones; el router traduce. Por eso este módulo es reutilizable desde
un worker, un CLI o una cola sin arrastrar FastAPI detrás.

Flujo de una estimación:
1. Extraer la key en el punto de uso (falta -> LLMConfigurationError).
2. Construir el prompt: system (instrucciones + cache CAG) y user (la
   transcripción dentro de un delimitador impredecible por petición).
3. Llamar al proveedor y traducir sus errores normalizados a LLMServiceError.
4. Señalar si la respuesta se truncó: en CAG un 200 con la respuesta a medio
   escribir es la peor clase de fallo, la que no se ve.

El servicio es async de punta a punta: `generate_estimation` es corrutina y
hace `await provider.chat(...)`. Quien la use desde un contexto síncrono
(CLI, worker) debe envolver la llamada con `asyncio.run(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import LLMConfigurationError as _ConfigError
from ..config import Settings
from ..context.examples import build_system_prompt, transcription_delimiter
from ..providers import LLMProviderError, Message, UnknownProviderError, create_provider
from .pricing import estimate_cost


class LLMServiceError(Exception):
    """Error de dominio: falló la generación de la estimación (-> 502).

    Pensado para ser traducido por el router. Nunca contiene detalles del
    SDK: esos se quedan en el log del servidor."""


class LLMConfigurationError(LLMServiceError):
    """Falta configuración local para llamar al LLM (-> 503).

    Hereda del ServiceError pero merece un status distinto: el cliente no
    debería reintentar, es un problema del que opera el servicio."""


@dataclass(frozen=True)
class EstimationResult:
    estimation: str
    truncated: bool
    model: str
    provider: str
    usage: dict[str, int | None] | None
    cost_usd: float | None
    cost_note: str | None


def _build_messages(transcription: str) -> list[Message]:
    """arma los mensajes con roles y fronteras explícitas.

    system: instrucciones + cache CAG (la transcripción NO va aquí).
    user: la transcripción como DATOS, entre un par de etiquetas cuyo nombre
    solo se conoce en esta llamada. El usuario no puede cerrarlas porque no
    conoce el sufijo aleatorio."""
    apertura, cierre = transcription_delimiter()  # Delimitador aleatorio por llamada
    return [
        Message(role="system", content=build_system_prompt()),  # rol system + cache CAG
        Message(role="user", content=f"{apertura}\n{transcription}\n{cierre}"),  # rol user
    ]


async def generate_estimation(
    transcription: str,
    settings: Settings,
) -> EstimationResult:
    # Si el provider falla o la key falta, los mensajes jamás se construyen,
    # no hay trabajo desperdiciado en armar un prompt que no se va a enviar.
    try:
        # La key se valida en el punto de uso, no al importar.
        key = settings.require_api_key()
    except _ConfigError as exc:
        # config.py y este servicio define cada uno su error de config:
        # el de abajo es capa de infraestructura, este es de dominio.
        # Traducción explícita para no atrapar config en el router.
        raise LLMConfigurationError(str(exc)) from exc

    try:
        # Se construye el cliente del proveedor con la config ya validada.
        provider = create_provider(
            name=settings.llm_provider,
            api_key=key,
            model=settings.llm_model,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
    except UnknownProviderError as exc:
        raise LLMConfigurationError(str(exc)) from exc

    try:
        # Los mensajes (system + user) se construyen recién acá, como
        # argumento de la llamada: después de validar config y crear el
        # provider, antes de que la corrutina toque la red.
        response = await provider.chat(
            _build_messages(transcription),
            max_tokens=settings.llm_max_tokens,
        )
    except LLMProviderError as exc:
        raise LLMServiceError(f"Fallo del proveedor '{settings.llm_provider}'") from exc
    # NO se captura Exception: los bugs internos deben salir como 500,
    # no disfrazados de fallo del proveedor.

    # Costo del uso real
    cost_usd, cost_note = estimate_cost(response.model, response.usage)
    return EstimationResult(
        estimation=response.content,
        truncated=response.truncated,
        model=response.model,
        provider=settings.llm_provider,
        usage=response.usage,
        cost_usd=cost_usd,
        cost_note=cost_note,
    )
