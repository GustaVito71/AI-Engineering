"""Crítico del ACB (Paso 8): auditoría independiente de una estimación.

QUÉ HACE ESTE ARCHIVO

El crítico recibe una estimación que escribió el actor y devuelve una revisión
estructurada (`CriticFeedback`): un veredicto y la lista de problemas. No escribe una
estimación nueva y no decide qué hacer con su propia revisión: eso es del jefe
(`app/services/boss.py`). Es una función sin estado: no toca la sesión ni guarda nada
entre llamadas.

QUÉ VE EL CRÍTICO

Lo mismo que vio el actor: la transcripción, los adjuntos (el mismo bloque
`<attachments>`), la memoria del proyecto y la audiencia. Con eso puede detectar
componentes que la transcripción o los adjuntos mencionan y la estimación omite, y
contradicciones con la memoria.

INDEPENDENCIA: QUÉ PROVEEDOR REVISA

Para que la estimación no la revise el mismo modelo que la escribió, el crítico
intenta, en este orden:

1. `independent`: el proveedor distinto del que produjo la estimación (si el actor
   respondió con LLM_DEFAULT, revisa LLM_FALLBACK, y al revés).
2. `same_model`: si el anterior falla o no tiene clave, el mismo proveedor del actor.
   Se registra un aviso en el log.
3. `unavailable`: si tampoco eso funciona, no hay revisión.

El estado se devuelve junto con la revisión para que el jefe y la traza lo reflejen: en
vez de inventar un «aceptado con confianza 0» como la solución de referencia, el jefe
sabe que no hubo revisión. Cada proveedor usa su propio `LLMWrapper` SIN respaldo, para
saber con certeza quién respondió.

QUÉ SE REGISTRA

Solo tipos de error y cantidades, nunca el contenido de la conversación ni de los
adjuntos.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.config import Settings
from app.prompts.loader import CRITIC_PROMPT_VERSION, render_critic_prompt
from app.schemas.acb import ReviewStatus
from app.schemas.critic import CriticFeedback
from app.schemas.structured_estimation import StructuredResult
from app.services.llm_wrapper import IncompleteJSONError, LLMWrapper
from app.sessions.models import ProjectMetadata
from app.tracing import emit

# Tope de tokens de la revisión. 2.000 cortaba el JSON con el proveedor Anthropic (el
# crítico de OpenAI usa unos 300 tokens); se iguala al tope de las demás salidas
# estructuradas (`llm_max_tokens`, 4.000). Pagar de más es imposible: se cobra lo generado.
CRITIC_MAX_TOKENS = 4_000


@dataclass
class CriticReview:
    """Resultado de revisar una estimación: la revisión (si la hubo) y cómo se obtuvo."""

    # None solo cuando `status` es `unavailable`.
    feedback: CriticFeedback | None
    status: ReviewStatus
    # Proveedor que respondió; None si no hubo revisión.
    provider: str | None = None
    # Lo gastado en esta revisión (incluye los intentos que no cumplieron el schema).
    cost_usd: float = 0.0


# Fábrica de clientes del LLM: dado un proveedor, devuelve un wrapper que llama solo a ese
# proveedor. Se puede reemplazar en las pruebas.
WrapperFactory = Callable[[str], LLMWrapper]


class Critic:
    """Revisa estimaciones con un proveedor distinto del que las escribió."""

    def __init__(self, *, settings: Settings, wrapper_factory: WrapperFactory | None = None):
        self._settings = settings
        self._factory = wrapper_factory or self._default_factory
        # Un wrapper por proveedor, creado la primera vez que se usa.
        self._wrappers: dict[str, LLMWrapper] = {}

    def _default_factory(self, provider: str) -> LLMWrapper:
        """Un `LLMWrapper` que llama solo a `provider`, sin respaldo."""
        return LLMWrapper(settings=self._settings.for_provider(provider), use_fallback=False)

    def independent_provider(self, actor_provider: str) -> str:
        """El proveedor del par (LLM_DEFAULT, LLM_FALLBACK) que NO escribió la estimación."""
        if actor_provider == self._settings.llm_fallback:
            return self._settings.llm_default
        return self._settings.llm_fallback

    async def review(
        self,
        *,
        transcript: str,
        attachments_block: str,
        project_metadata: ProjectMetadata,
        tier: str,
        detail_level: str,
        result: StructuredResult,
        actor_version: str,
        actor_provider: str,
    ) -> CriticReview:
        """Revisa `result`, probando primero un proveedor distinto del `actor_provider`."""
        system_prompt, user_prompt = render_critic_prompt(
            transcript=transcript,
            attachments_block=attachments_block,
            project_metadata=project_metadata,
            tier=tier,
            detail_level=detail_level,
            result=result,
            actor_version=actor_version,
        )
        attempts: list[tuple[str, ReviewStatus]] = [
            (self.independent_provider(actor_provider), "independent"),
            (actor_provider, "same_model"),
        ]
        for provider, status in attempts:
            try:
                call = await self._wrapper(provider).estimate_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    prompt_version=f"critic-{CRITIC_PROMPT_VERSION}",
                    response_model=CriticFeedback,
                    max_tokens=CRITIC_MAX_TOKENS,
                )
            except Exception as exc:  # noqa: BLE001 - cualquier fallo pasa al siguiente intento
                emit(
                    __name__,
                    "critico_fallo",
                    level="warning",
                    proveedor=provider,
                    estado=status,
                    error=type(exc).__name__,
                    # Solo para la respuesta cortada: el mensaje trae el tope y el gasto,
                    # y no contiene datos de la conversación ni claves.
                    detalle=str(exc) if isinstance(exc, IncompleteJSONError) else "",
                )
                continue
            if status == "same_model":
                emit(
                    __name__,
                    "critico_mismo_modelo",
                    level="warning",
                    proveedor=provider,
                    detalle="La revisión no es independiente: revisó el mismo proveedor del actor.",
                )
            return CriticReview(
                feedback=call.estimation, status=status, provider=provider, cost_usd=call.cost_usd
            )

        emit(__name__, "critico_no_disponible", level="warning", proveedor_actor=actor_provider)
        return CriticReview(feedback=None, status="unavailable")

    def _wrapper(self, provider: str) -> LLMWrapper:
        """El wrapper de `provider`; si falta su clave lanza `LLMConfigurationError` (lo trata `review`)."""
        if provider not in self._wrappers:
            self._wrappers[provider] = self._factory(provider)
        return self._wrappers[provider]
