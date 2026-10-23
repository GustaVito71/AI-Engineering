"""Jefe del ACB (Paso 8): orquesta al actor y al crítico y decide qué se entrega.

QUÉ HACE ESTE ARCHIVO

El jefe es una máquina de estados pequeña, sin llamadas propias al LLM: solo elige el
paso siguiente. Por eso cada decisión se puede reproducir con las mismas entradas y
queda anotada en la traza (`BossTrace`).

    actor → crítico → decidir
                       ├── aceptar     → se entrega el borrador tal cual
                       ├── iterar      → se vuelve a llamar al actor con el feedback
                       └── sintetizar  → se entrega el último borrador con salvedades

El número de vueltas está acotado por `max_iterations`: producción antes que
perfección. Cuando se agota, o cuando el crítico rechaza, el jefe NO tira el borrador a
la basura (eso escondería la mejor respuesta disponible detrás de un «fuera de alcance»):
lo entrega con las salvedades abiertas al inicio del resumen y la confianza reducida.

REGLAS SEGÚN QUIÉN REVISÓ (`review_status`)

- `independent`: se respeta el veredicto del crítico (`accept`, `needs_iteration` o
  `reject`).
- `same_model`: el veredicto es real, pero un `accept` baja a `needs_iteration`, porque un
  modelo que se revisa a sí mismo tiende a aprobar lo suyo. `needs_iteration` y `reject`
  se mantienen. Como el crítico seguirá siendo del mismo modelo, lo normal es que se
  agoten las vueltas y el resultado sea una síntesis.
- `unavailable`: no hubo revisión. El jefe sintetiza de inmediato, sin iterar (otra
  vuelta no cambiaría nada), con una nota que dice que la estimación no pudo revisarse.
- `skipped`: el actor rechazó la estimación por falta de información; no hay nada que
  revisar y se entrega tal cual.

El actor y el crítico llegan como funciones (`ActorCallable`, `CriticCallable`): el jefe
no sabe cómo se llama al LLM, solo cuándo. El armado real está en el endpoint.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.schemas.acb import (
    REVIEW_NOTE_MARK,
    REVIEW_NOTE_SEPARATOR,
    ACBIteration,
    BossDecision,
    BossTrace,
    ReviewStatus,
    TraceVerdict,
)
from app.schemas.critic import MAX_ISSUES, CriticFeedback
from app.schemas.structured_estimation import MIN_CONFIDENCE_PCT, StructuredResult
from app.services.critic import CriticReview
from app.tracing import emit

# Confianza mínima de una síntesis: por debajo de MIN_CONFIDENCE_PCT el schema exigiría el
# prefijo «Fuera de alcance:», lo que contradice entregar un borrador utilizable.
SYNTHESIS_CONFIDENCE_FLOOR = MIN_CONFIDENCE_PCT

# Largo máximo del resumen de una estimación (el del schema) y de la nota de salvedades
# que se le antepone: la nota no puede comerse el resumen.
MAX_SUMMARY_CHARS = 1_200
MAX_NOTE_CHARS = 600


@dataclass
class Draft:
    """Un borrador del actor: la estimación y quién la escribió.

    `payload` es opaco para el jefe: el llamador guarda ahí lo que necesite recuperar
    después (por ejemplo los avisos de la llamada) sin que el jefe tenga que conocerlo.
    """

    result: StructuredResult
    provider: str
    payload: Any = None


@dataclass
class BossOutcome:
    """Lo que devuelve el jefe: la estimación final, el borrador del que sale y la traza."""

    result: StructuredResult
    draft: Draft
    trace: BossTrace


# El actor recibe el feedback de la vuelta anterior (None en la primera) y devuelve un borrador.
ActorCallable = Callable[[CriticFeedback | None], Awaitable[Draft]]
# El crítico recibe un borrador y devuelve su revisión.
CriticCallable = Callable[[Draft], Awaitable[CriticReview]]


class Boss:
    """Orquestador sin estado: una instancia puede atender muchas estimaciones."""

    def __init__(self, *, max_iterations: int = 3) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations tiene que ser al menos 1.")
        self.max_iterations = max_iterations

    async def run(self, *, actor: ActorCallable, critic: CriticCallable) -> BossOutcome:
        """Ejecuta el bucle actor → crítico → decisión, con como máximo `max_iterations` vueltas."""
        trace = BossTrace(final_decision="accept", iterations_run=0)
        feedback: CriticFeedback | None = None
        draft: Draft | None = None
        review: CriticReview | None = None

        for iteration in range(self.max_iterations):
            try:
                new_draft = await actor(feedback)
            except Exception as exc:
                if draft is None or review is None:
                    # Sin ningún borrador no hay nada que entregar: el error sube al endpoint.
                    raise
                # Ya hay un borrador de una vuelta anterior: se entrega ese con salvedades
                # en vez de perder todo por un fallo del proveedor en una vuelta de mejora.
                emit(__name__, "jefe_actor_fallo", level="warning", error=type(exc).__name__)
                return self._finish(draft, review, trace, replace_last_decision=True)
            draft = new_draft

            if draft.result.out_of_scope:
                # El actor rechazó la estimación por falta de información: no hay nada que auditar.
                trace.iterations.append(self._iteration(iteration, None, "skipped", "accept"))
                trace.iterations_run = iteration + 1
                trace.final_decision = "accept"
                return BossOutcome(result=draft.result, draft=draft, trace=trace)

            review = await critic(draft)
            decision = self.decide(review, iterations_left=self.max_iterations - iteration - 1)
            trace.iterations.append(self._iteration(iteration, review, review.status, decision))
            trace.iterations_run = iteration + 1
            emit(
                __name__,
                "jefe_vuelta",
                vuelta=iteration,
                decision=decision,
                estado_revision=review.status,
                veredicto=review.feedback.verdict if review.feedback else "unavailable",
                problemas=len(review.feedback.issues) if review.feedback else 0,
                # Qué problemas vio el crítico («major:math_error@totals.hours»), sin su
                # texto: sirve para saber por qué el bucle no converge (por ejemplo, si lo
                # frena siempre la misma categoría) sin volcar contenido al log.
                categorias=[
                    f"{issue.severity}:{issue.category}@{issue.field_path}"
                    for issue in (review.feedback.issues if review.feedback else [])
                ],
            )

            if decision == "accept":
                trace.final_decision = "accept"
                return BossOutcome(result=draft.result, draft=draft, trace=trace)
            if decision == "synthesize":
                return self._finish(draft, review, trace)
            # decision == "iterate": el feedback de esta revisión viaja al actor.
            feedback = review.feedback

        # Se agotaron las vueltas con el crítico todavía señalando problemas.
        emit(__name__, "jefe_presupuesto_agotado", level="info", vueltas=self.max_iterations)
        assert draft is not None and review is not None  # max_iterations >= 1
        return self._finish(draft, review, trace)

    # -- decisión -----------------------------------------------------------------------

    @staticmethod
    def decide(review: CriticReview, *, iterations_left: int) -> BossDecision:
        """Qué hacer tras una revisión, según quién revisó y qué concluyó (ver el docstring del módulo)."""
        if review.status == "unavailable" or review.feedback is None:
            return "synthesize"
        verdict = review.feedback.verdict
        if verdict == "reject":
            return "synthesize"
        if verdict == "accept" and review.status == "independent":
            return "accept"
        # needs_iteration, o un accept de un crítico no independiente (se baja a iterar).
        return "iterate" if iterations_left > 0 else "synthesize"

    # -- salidas --------------------------------------------------------------------------

    def _finish(
        self,
        draft: Draft,
        review: CriticReview,
        trace: BossTrace,
        *,
        replace_last_decision: bool = False,
    ) -> BossOutcome:
        """Cierra el bucle entregando el último borrador con salvedades."""
        if replace_last_decision and trace.iterations:
            # La vuelta anterior había decidido iterar, pero la iteración no pudo ocurrir.
            trace.iterations[-1] = trace.iterations[-1].model_copy(
                update={"decision_after": "synthesize"}
            )
        trace.final_decision = "synthesize"
        return BossOutcome(result=self._synthesize(draft.result, review), draft=draft, trace=trace)

    @staticmethod
    def _synthesize(result: StructuredResult, review: CriticReview) -> StructuredResult:
        """El borrador con una nota de salvedades al inicio del resumen y la confianza reducida.

        `model_copy` no vuelve a validar. La nota cita cifras del crítico que pueden no estar
        en la estimación, por eso la respuesta usa `ReviewedResult`, que no las somete a la
        comprobación de cifras del resumen (ver `app/schemas/acb.py`). El resto de las reglas
        se cumple: el borrador no es un rechazo y la confianza queda en el piso (30) o más.
        """
        feedback = review.feedback
        if review.status == "unavailable" or feedback is None:
            note = f"{REVIEW_NOTE_MARK} Esta estimación no pudo ser revisada por el crítico.{REVIEW_NOTE_SEPARATOR}"
            confidence = min(result.confidence_pct, SYNTHESIS_CONFIDENCE_FLOOR)
        else:
            who = "revisión independiente" if review.status == "independent" else "revisión"
            lines = (
                [
                    f"- [{issue.severity}] {issue.category} ({issue.field_path}): "
                    f"{' '.join(issue.description.split())}"  # en una línea: la nota termina en una línea en blanco
                    for issue in feedback.issues
                ]
                or ["- (sin detalle)"]
            )
            header = f"{REVIEW_NOTE_MARK} Salvedades abiertas de la {who} (el proceso no convergió del todo)"
            if review.status == "same_model":
                header += "; el crítico fue del mismo proveedor que el actor"
            note = f"{header}:\n" + "\n".join(lines)
            if len(note) > MAX_NOTE_CHARS:
                note = note[: MAX_NOTE_CHARS - 1].rstrip() + "…"
            note += REVIEW_NOTE_SEPARATOR
            confidence = max(SYNTHESIS_CONFIDENCE_FLOOR, result.confidence_pct // 2)
        summary = (note + result.summary)[:MAX_SUMMARY_CHARS]
        return result.model_copy(update={"summary": summary, "confidence_pct": confidence})

    @staticmethod
    def _iteration(
        iteration: int,
        review: CriticReview | None,
        status: ReviewStatus,
        decision: BossDecision,
    ) -> ACBIteration:
        """La fila de la traza para una vuelta."""
        feedback = review.feedback if review else None
        verdict: TraceVerdict = (
            feedback.verdict if feedback else ("skipped" if status == "skipped" else "unavailable")
        )
        return ACBIteration(
            iteration=iteration,
            review_status=status,
            critic_provider=review.provider if review else None,
            critic_verdict=verdict,
            critic_confidence=feedback.confidence_in_review if feedback else 0,
            decision_after=decision,
            issue_summary=[
                f"[{issue.severity}] {issue.category} @ {issue.field_path}"
                for issue in (feedback.issues if feedback else [])[:MAX_ISSUES]
            ],
        )
