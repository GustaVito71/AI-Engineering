"""Tests del crítico y del jefe del ACB (Paso 8): esquemas, servicio del crítico y orquestación.

QUÉ SE COMPRUEBA

- Esquema `CriticFeedback`: `needs_iteration` exige un problema critical/major y `reject`
  exige algún problema.
- Crítico (`Critic`): prueba primero un proveedor distinto del actor (`independent`), cae al
  mismo proveedor si el otro falla (`same_model`) y, si todo falla, informa `unavailable`.
  Recibe los adjuntos, la memoria del proyecto y la transcripción.
- Jefe (`Boss`): reglas según quién revisó, tope de vueltas, síntesis con salvedades y
  confianza mínima 30, rechazo del actor sin revisar y fallo del actor en una vuelta tardía.

Todo con falsos: sin red ni claves.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from structlog.testing import capture_logs
from test_estimate_structured_endpoint import ESTIMACION

from app.schemas.critic import CriticFeedback, CriticIssue
from app.schemas.structured_estimation import (
    OUT_OF_SCOPE_PREFIX,
    UNESTIMATED_PHASE,
    StructuredResult,
)
from app.services.boss import Boss, Draft
from app.services.critic import Critic, CriticReview
from app.services.llm_wrapper import LLMWrapper
from app.sessions.models import ProjectMetadata

# --- Ayudas ------------------------------------------------------------------------------


def problema(severity: str = "major", category: str = "math_error") -> CriticIssue:
    """Un problema de ejemplo del crítico."""
    return CriticIssue(
        category=category,
        severity=severity,
        field_path="totals.hours",
        description="Las horas del total no cuadran con las fases.",
        suggested_fix="",
    )


def revision(verdict: str, *issues: CriticIssue) -> CriticFeedback:
    """Una revisión del crítico con el veredicto y los problemas dados."""
    return CriticFeedback(verdict=verdict, issues=list(issues), confidence_in_review=80)


def estimacion(confidence: int = 70) -> StructuredResult:
    """La estimación de prueba, con la confianza global indicada."""
    return StructuredResult.model_validate({**ESTIMACION, "confidence_pct": confidence})


def rechazo() -> StructuredResult:
    """Una estimación rechazada por falta de información (fuera de alcance)."""
    return StructuredResult.model_validate(
        {
            "phases": [
                {
                    "name": UNESTIMATED_PHASE,
                    "summary": "No hay información suficiente para estimar.",
                    "duration_weeks": 0,
                    "hours": 0,
                    "cost_eur": 0,
                    "confidence_pct": 0,
                    "assumptions": [],
                    "risks": [],
                }
            ],
            "team": [],
            "totals": {"hours": 0, "cost_eur": 0, "duration_weeks": 0},
            "summary": f"{OUT_OF_SCOPE_PREFIX} falta saber qué producto se quiere construir.",
            "confidence_pct": 10,
        }
    )


# --- Esquema -----------------------------------------------------------------------------


def test_needs_iteration_exige_un_problema_bloqueante() -> None:
    """Con solo problemas minor, el veredicto no puede ser needs_iteration."""
    with pytest.raises(ValidationError, match="critical o major"):
        revision("needs_iteration", problema("minor"))
    assert revision("needs_iteration", problema("critical")).verdict == "needs_iteration"


def test_reject_exige_algun_problema() -> None:
    """Un rechazo sin explicación no es válido."""
    with pytest.raises(ValidationError, match="reject"):
        revision("reject")


def test_las_categorias_incluyen_las_del_enunciado() -> None:
    """Las dos categorías que pide el enunciado del ACB existen."""
    for categoria in ("missing_component", "inconsistency_with_metadata"):
        assert problema(category=categoria).category == categoria


# --- Crítico -----------------------------------------------------------------------------


class _WrapperCritico:
    """Wrapper falso del crítico: devuelve la revisión preparada o lanza el error preparado."""

    def __init__(self, resultado: CriticFeedback | Exception) -> None:
        self.resultado = resultado
        self.llamadas: list[dict] = []

    async def estimate_structured(self, **kwargs):
        """Guarda los argumentos y devuelve la revisión (o lanza el error preparado)."""
        self.llamadas.append(kwargs)
        if isinstance(self.resultado, Exception):
            raise self.resultado
        return SimpleNamespace(estimation=self.resultado, cost_usd=0.002)


def construir_critico(construir_settings, wrappers: dict[str, _WrapperCritico]) -> Critic:
    """Un crítico cuya fábrica entrega los wrappers falsos por proveedor."""
    settings = construir_settings(llm_default="openai", llm_fallback="anthropic")

    def fabrica(proveedor: str):
        """Devuelve el wrapper falso del proveedor, o falla como si faltara su clave."""
        if proveedor not in wrappers:
            raise RuntimeError("sin clave")
        return wrappers[proveedor]

    return Critic(settings=settings, wrapper_factory=fabrica)


async def revisar(critico: Critic, *, actor: str = "openai", adjuntos: str = "") -> CriticReview:
    """Pide una revisión de la estimación de prueba escrita por el proveedor `actor`."""
    return await critico.review(
        transcript="Cliente: necesitamos un SaaS de préstamos de equipos.",
        attachments_block=adjuntos,
        project_metadata=ProjectMetadata(project_name="Bookflow"),
        tier="default",
        detail_level="detailed",
        result=estimacion(),
        actor_version="v5",
        actor_provider=actor,
    )


@pytest.mark.parametrize(("actor", "esperado"), [("openai", "anthropic"), ("anthropic", "openai")])
async def test_el_critico_independiente_es_el_otro_proveedor(
    construir_settings, actor, esperado
) -> None:
    """Si el actor respondió con un proveedor, revisa el otro del par."""
    wrappers = {
        "openai": _WrapperCritico(revision("accept")),
        "anthropic": _WrapperCritico(revision("accept")),
    }
    resultado = await revisar(construir_critico(construir_settings, wrappers), actor=actor)

    assert (resultado.status, resultado.provider) == ("independent", esperado)
    assert len(wrappers[esperado].llamadas) == 1
    assert len(wrappers[actor].llamadas) == 0


async def test_si_el_otro_proveedor_falla_revisa_el_mismo_del_actor(construir_settings) -> None:
    """El estado pasa a same_model y la revisión es la del proveedor del actor."""
    wrappers = {
        "anthropic": _WrapperCritico(RuntimeError("caído")),
        "openai": _WrapperCritico(revision("needs_iteration", problema())),
    }
    resultado = await revisar(construir_critico(construir_settings, wrappers))

    assert (resultado.status, resultado.provider) == ("same_model", "openai")
    assert resultado.feedback.verdict == "needs_iteration"


async def test_sin_ninguna_revision_el_estado_es_unavailable(construir_settings) -> None:
    """Si los dos intentos fallan (o falta una clave), no hay revisión y no se inventa un accept."""
    wrappers = {"anthropic": _WrapperCritico(RuntimeError("caído"))}  # openai: sin clave
    resultado = await revisar(construir_critico(construir_settings, wrappers))

    assert resultado.status == "unavailable"
    assert resultado.feedback is None
    assert resultado.provider is None


async def test_el_critico_recibe_adjuntos_memoria_y_transcripcion(construir_settings) -> None:
    """El prompt del crítico lleva los mismos datos que vio el actor."""
    wrappers = {"anthropic": _WrapperCritico(revision("accept"))}
    await revisar(
        construir_critico(construir_settings, wrappers),
        adjuntos="<attachments>El ERP es LEGACYSOFT.</attachments>",
    )

    llamada = wrappers["anthropic"].llamadas[0]
    assert "LEGACYSOFT" in llamada["user_prompt"]
    assert "Bookflow" in llamada["user_prompt"]
    assert "SaaS de préstamos" in llamada["user_prompt"]
    assert "<estimation_under_review>" in llamada["user_prompt"]
    assert llamada["prompt_version"] == "critic-v1"


# --- Jefe --------------------------------------------------------------------------------


class _Actor:
    """Actor falso: devuelve un borrador por llamada y anota el feedback que recibió."""

    def __init__(self, borradores: list[StructuredResult | Exception] | None = None) -> None:
        self.borradores = borradores or [estimacion()]
        self.feedbacks: list[CriticFeedback | None] = []

    async def __call__(self, feedback: CriticFeedback | None) -> Draft:
        """Entrega el borrador de esta vuelta (el último se repite)."""
        self.feedbacks.append(feedback)
        indice = min(len(self.feedbacks), len(self.borradores)) - 1
        borrador = self.borradores[indice]
        if isinstance(borrador, Exception):
            raise borrador
        return Draft(result=borrador, provider="openai")


class _Critico:
    """Crítico falso: devuelve las revisiones preparadas, en orden (la última se repite)."""

    def __init__(self, *revisiones: CriticReview) -> None:
        self.revisiones = list(revisiones)
        self.llamadas = 0

    async def __call__(self, draft: Draft) -> CriticReview:
        """Entrega la revisión de esta vuelta."""
        self.llamadas += 1
        return self.revisiones[min(self.llamadas, len(self.revisiones)) - 1]


def indep(verdict: str, *issues: CriticIssue) -> CriticReview:
    """Una revisión de un crítico independiente."""
    return CriticReview(feedback=revision(verdict, *issues), status="independent", provider="x")


def mismo(verdict: str, *issues: CriticIssue) -> CriticReview:
    """Una revisión de un crítico del mismo proveedor que el actor."""
    return CriticReview(feedback=revision(verdict, *issues), status="same_model", provider="y")


SIN_REVISION = CriticReview(feedback=None, status="unavailable")


async def test_independiente_accept_entrega_el_borrador_tal_cual() -> None:
    """Un accept independiente cierra en la primera vuelta, sin tocar la estimación."""
    actor, critico = _Actor(), _Critico(indep("accept"))
    salida = await Boss(max_iterations=3).run(actor=actor, critic=critico)

    assert salida.result == estimacion()
    assert (salida.trace.final_decision, salida.trace.iterations_run) == ("accept", 1)
    assert salida.trace.iterations[0].review_status == "independent"


async def test_needs_iteration_vuelve_al_actor_con_el_feedback() -> None:
    """La segunda llamada del actor recibe la revisión de la primera vuelta."""
    actor = _Actor()
    critico = _Critico(indep("needs_iteration", problema()), indep("accept"))
    salida = await Boss(max_iterations=3).run(actor=actor, critic=critico)

    assert actor.feedbacks[0] is None
    assert actor.feedbacks[1].verdict == "needs_iteration"
    assert (salida.trace.final_decision, salida.trace.iterations_run) == ("accept", 2)
    assert [i.decision_after for i in salida.trace.iterations] == ["iterate", "accept"]


async def test_nunca_supera_el_maximo_de_vueltas_y_sintetiza() -> None:
    """Un crítico que siempre pide otra vuelta agota el presupuesto: tres vueltas y síntesis."""
    actor = _Actor()
    critico = _Critico(indep("needs_iteration", problema()))
    salida = await Boss(max_iterations=3).run(actor=actor, critic=critico)

    assert len(actor.feedbacks) == critico.llamadas == 3
    assert (salida.trace.final_decision, salida.trace.iterations_run) == ("synthesize", 3)
    assert salida.trace.iterations[-1].decision_after == "synthesize"
    assert "Salvedades abiertas de la revisión independiente" in salida.result.summary


async def test_reject_sintetiza_de_inmediato_con_confianza_reducida() -> None:
    """Un rechazo no itera: entrega el borrador con la nota y la confianza a la mitad (mín. 30)."""
    actor, critico = _Actor(), _Critico(indep("reject", problema("critical")))
    salida = await Boss(max_iterations=3).run(actor=actor, critic=critico)

    assert salida.trace.iterations_run == 1
    assert salida.result.summary.startswith("⚠ Salvedades abiertas")
    assert salida.result.summary.endswith(estimacion().summary)
    assert salida.result.confidence_pct == 35  # 70 // 2


async def test_la_confianza_de_la_sintesis_no_baja_de_30() -> None:
    """Con confianza 40 la mitad sería 20: el piso es 30."""
    actor = _Actor([estimacion(confidence=40)])
    salida = await Boss(max_iterations=2).run(
        actor=actor, critic=_Critico(indep("reject", problema("critical")))
    )
    assert salida.result.confidence_pct == 30


async def test_same_model_baja_un_accept_a_needs_iteration() -> None:
    """Un accept de un crítico no independiente no cierra: el jefe itera."""
    actor = _Actor()
    critico = _Critico(mismo("accept"), mismo("accept"))
    salida = await Boss(max_iterations=2).run(actor=actor, critic=critico)

    assert [i.decision_after for i in salida.trace.iterations] == ["iterate", "synthesize"]
    assert salida.trace.final_decision == "synthesize"
    # La síntesis dice «revisión» a secas y avisa que no fue independiente.
    resumen = salida.result.summary
    assert "Salvedades abiertas de la revisión (" in resumen
    assert "independiente" not in resumen.split("\n")[0].replace("no independiente", "")
    assert "mismo proveedor" in resumen


async def test_same_model_mantiene_needs_iteration_y_reject() -> None:
    """Los veredictos negativos de un crítico del mismo modelo se respetan."""
    r1 = await Boss(max_iterations=3).run(
        actor=_Actor(), critic=_Critico(mismo("reject", problema("critical")))
    )
    assert r1.trace.iterations_run == 1 and r1.trace.final_decision == "synthesize"

    actor = _Actor()
    await Boss(max_iterations=3).run(
        actor=actor, critic=_Critico(mismo("needs_iteration", problema()), mismo("accept"))
    )
    assert actor.feedbacks[1] is not None  # iteró con el feedback real


async def test_unavailable_sintetiza_de_inmediato_con_nota_y_confianza_30() -> None:
    """Sin revisión no se itera: se entrega el borrador con la nota y confianza 30."""
    actor = _Actor()
    salida = await Boss(max_iterations=3).run(actor=actor, critic=_Critico(SIN_REVISION))

    assert len(actor.feedbacks) == 1
    assert salida.result.summary.startswith("⚠ Esta estimación no pudo ser revisada")
    assert salida.result.confidence_pct == 30
    iteracion = salida.trace.iterations[0]
    assert (iteracion.review_status, iteracion.critic_verdict, iteracion.critic_confidence) == (
        "unavailable",
        "unavailable",
        0,
    )
    assert iteracion.critic_provider is None


async def test_unavailable_no_sube_la_confianza_si_ya_era_menor() -> None:
    """El piso es 30 solo hacia abajo: una confianza de 30 o menos no se modifica."""
    actor = _Actor([estimacion(confidence=30)])
    salida = await Boss(max_iterations=3).run(actor=actor, critic=_Critico(SIN_REVISION))
    assert salida.result.confidence_pct == 30


async def test_el_log_de_cada_vuelta_dice_que_problemas_vio_el_critico() -> None:
    """`jefe_vuelta` lista severidad, categoría y campo de cada problema (sin su texto), para
    diagnosticar por qué el bucle no converge.
    """
    critico = _Critico(indep("needs_iteration", problema("major", "math_error")))
    with capture_logs() as eventos:
        await Boss(max_iterations=1).run(actor=_Actor(), critic=critico)

    vuelta = next(e for e in eventos if e["event"] == "jefe_vuelta")
    assert vuelta["categorias"] == ["major:math_error@totals.hours"]
    assert "Las horas del total" not in str(vuelta)


async def test_un_rechazo_del_actor_no_se_revisa() -> None:
    """Si el actor rechazó por falta de información, no hay nada que auditar."""
    critico = _Critico(indep("reject", problema("critical")))
    salida = await Boss(max_iterations=3).run(actor=_Actor([rechazo()]), critic=critico)

    assert critico.llamadas == 0
    assert salida.result == rechazo()
    assert salida.trace.iterations[0].review_status == "skipped"
    assert salida.trace.final_decision == "accept"


async def test_el_resumen_nunca_supera_el_maximo_ni_pierde_su_final() -> None:
    """Muchos problemas largos no se comen el resumen: la nota se acota y el total cabe."""
    muchos = [
        CriticIssue(
            category="hallucination",
            severity="major",
            field_path=f"phases[{i}].summary",
            description="d" * 480,
            suggested_fix="",
        )
        for i in range(12)
    ]
    salida = await Boss(max_iterations=1).run(
        actor=_Actor(), critic=_Critico(indep("reject", *muchos))
    )
    assert len(salida.result.summary) <= 1200
    assert "…" in salida.result.summary
    assert salida.result.summary.endswith(estimacion().summary)


async def test_un_fallo_del_actor_en_la_primera_vuelta_sube() -> None:
    """Sin ningún borrador no hay qué entregar: el error llega al endpoint."""
    with pytest.raises(RuntimeError, match="proveedor"):
        await Boss(max_iterations=3).run(
            actor=_Actor([RuntimeError("proveedor caído")]), critic=_Critico(indep("accept"))
        )


async def test_un_fallo_del_actor_en_una_vuelta_tardia_entrega_el_borrador_anterior() -> None:
    """Producción antes que perfección: se entrega la vuelta anterior con salvedades."""
    actor = _Actor([estimacion(), RuntimeError("proveedor caído")])
    critico = _Critico(indep("needs_iteration", problema()))
    salida = await Boss(max_iterations=3).run(actor=actor, critic=critico)

    assert salida.trace.final_decision == "synthesize"
    assert salida.trace.iterations[-1].decision_after == "synthesize"
    assert "Salvedades abiertas" in salida.result.summary


def test_el_jefe_exige_al_menos_una_vuelta() -> None:
    """Un presupuesto de cero vueltas no tiene sentido."""
    with pytest.raises(ValueError):
        Boss(max_iterations=0)


# --- Configuración y wrapper del crítico ---------------------------------------------------


def test_boss_max_iterations_vale_tres_por_defecto_y_vacio_es_el_defecto(
    construir_settings,
) -> None:
    """Sin configurar, o con la variable vacía, el tope de vueltas es 3."""
    assert construir_settings().boss_max_iterations == 3
    assert construir_settings(boss_max_iterations="").boss_max_iterations == 3
    assert construir_settings(boss_max_iterations="2").boss_max_iterations == 2


def test_boss_max_iterations_menor_que_uno_rompe_al_arrancar(construir_settings) -> None:
    """Cero vueltas no tendría sentido: el error nombra la variable."""
    with pytest.raises(ValidationError, match="BOSS_MAX_ITERATIONS"):
        construir_settings(boss_max_iterations=0)


def test_for_provider_cambia_solo_el_primario(construir_settings) -> None:
    """La copia para el crítico llama al proveedor pedido y deja el resto igual."""
    base = construir_settings(llm_default="openai", llm_fallback="anthropic")
    copia = base.for_provider("anthropic")

    assert copia.primary_provider == "anthropic"
    assert copia.primary_model == base.fallback_model
    assert base.primary_provider == "openai"  # el original no cambia


def test_el_wrapper_sin_respaldo_no_arma_el_deployment_de_respaldo(construir_settings) -> None:
    """Con `use_fallback=False` el Router solo tiene el primario y no hay aviso de respaldo."""
    settings = construir_settings(
        llm_default="anthropic",
        llm_fallback="openai",
        anthropic_api_key="sk-a",
        openai_api_key="sk-o",
    )
    con = LLMWrapper(settings=settings)
    sin = LLMWrapper(settings=settings, use_fallback=False)

    assert len(con._router.model_list) == 2
    assert len(sin._router.model_list) == 1
    assert sin._router.model_list[0]["litellm_params"]["model"] == settings.primary_model
    assert sin.warnings == ()
