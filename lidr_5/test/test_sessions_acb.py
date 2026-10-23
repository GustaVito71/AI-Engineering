"""Tests de `POST /sessions/{id}/estimate-acb` (Paso 8): el endpoint Actor-Crítico-Jefe.

Se levanta la aplicación real (rutas, router, almacén de sesiones, servicio de estimación,
prompts y jefe) y solo se reemplazan los puntos que llaman a un proveedor: el wrapper del
actor, el crítico y el extractor de metadata. Sin red ni Redis.

QUÉ SE COMPRUEBA

- Los tres estados de revisión de punta a punta: crítico independiente (se acepta o se
  itera con el feedback), crítico del mismo modelo (un accept baja a iterar y el bucle
  termina en síntesis) y crítico no disponible (síntesis inmediata con nota).
- El tope `BOSS_MAX_ITERATIONS`: nunca hay más vueltas que las configuradas.
- El crítico recibe los adjuntos y la memoria; el actor recibe el feedback en la vuelta 2.
- El turno se anota una sola vez en el historial, con la estimación final.
- Las comprobaciones de entrada son las de `/estimate` (404, 422) y el tier se valida.
- Un fallo del actor en la primera vuelta es un 502, como en `/estimate`.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from test_acb_critic_boss import SIN_REVISION, indep, mismo, problema
from test_attachments import hacer_pdf
from test_estimate_structured_endpoint import ESTIMACION, _WrapperFalso
from test_sessions_memory_endpoint import FORM, TURNO_1, _ExtractorFalso

from app.config import get_settings
from app.dependencies import get_critic, get_llm_wrapper, get_metadata_extractor
from app.main import create_app
from app.schemas.critic import CriticIssue
from app.services.llm_wrapper import StructuredOutputError


class _CriticoFalso:
    """Crítico de prueba: devuelve las revisiones preparadas, en orden (la última se repite)."""

    def __init__(self, *revisiones) -> None:
        self.revisiones = list(revisiones)
        self.llamadas: list[dict] = []

    async def review(self, **kwargs):
        """Guarda lo recibido y devuelve la revisión de esta vuelta."""
        self.llamadas.append(kwargs)
        return self.revisiones[min(len(self.llamadas), len(self.revisiones)) - 1]


@pytest.fixture
def montar(monkeypatch):
    """Fábrica: arma la app con actor, crítico y extractor falsos y crea una sesión.

    Devuelve `(cliente, wrapper, extractor, session_id)`.
    """

    def _montar(critico, *, wrapper=None, max_iteraciones: int | None = None):
        """Crea la app de prueba con el crítico indicado y una sesión vacía."""
        monkeypatch.setenv("REDIS_URL", "")
        if max_iteraciones is not None:
            monkeypatch.setenv("BOSS_MAX_ITERATIONS", str(max_iteraciones))
        get_settings.cache_clear()
        cliente = TestClient(create_app())
        wrapper = wrapper or _WrapperFalso()
        extractor = _ExtractorFalso()
        cliente.app.dependency_overrides[get_llm_wrapper] = lambda: wrapper
        cliente.app.dependency_overrides[get_critic] = lambda: critico
        cliente.app.dependency_overrides[get_metadata_extractor] = lambda: extractor
        session_id = cliente.post("/sessions").json()["session_id"]
        return cliente, wrapper, extractor, session_id

    yield _montar
    get_settings.cache_clear()


def estimar_acb(cliente, session_id, transcript=TURNO_1, **extra):
    """Envía un turno al endpoint ACB como formulario multipart."""
    datos = {"transcript": transcript, **FORM, **extra.pop("data", {})}
    return cliente.post(f"/sessions/{session_id}/estimate-acb", data=datos, **extra)


# --- Crítico independiente ---------------------------------------------------------------


def test_independiente_accept_devuelve_la_estimacion_y_la_traza(montar) -> None:
    """Una sola vuelta: una llamada del actor, una del crítico y traza `accept`."""
    cliente, wrapper, _extractor, sid = montar(_CriticoFalso(indep("accept")))

    respuesta = estimar_acb(cliente, sid)

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["estimation"]["summary"] == ESTIMACION["summary"]
    assert cuerpo["prompt_version"] == "v5"
    assert cuerpo["acb"]["final_decision"] == "accept"
    assert cuerpo["acb"]["iterations_run"] == 1
    assert cuerpo["acb"]["iterations"][0]["review_status"] == "independent"
    assert len(wrapper.llamadas) == 1


def test_needs_iteration_le_pasa_el_feedback_al_actor_en_la_segunda_vuelta(montar) -> None:
    """La vuelta 1 no lleva `<critic_feedback>`; la vuelta 2, sí, con el problema señalado."""
    critico = _CriticoFalso(indep("needs_iteration", problema()), indep("accept"))
    cliente, wrapper, _extractor, sid = montar(critico)

    respuesta = estimar_acb(cliente, sid)

    assert respuesta.status_code == 200
    assert respuesta.json()["acb"]["iterations_run"] == 2
    assert "<critic_feedback>" not in wrapper.llamadas[0]["user_prompt"]
    assert "[major] math_error en totals.hours" in wrapper.llamadas[1]["user_prompt"]
    assert [i["decision_after"] for i in respuesta.json()["acb"]["iterations"]] == [
        "iterate",
        "accept",
    ]


def test_nunca_supera_boss_max_iterations(montar) -> None:
    """Un crítico que siempre pide otra vuelta se corta en el tope configurado."""
    critico = _CriticoFalso(indep("needs_iteration", problema()))
    cliente, wrapper, _extractor, sid = montar(critico, max_iteraciones=2)

    cuerpo = estimar_acb(cliente, sid).json()

    assert len(wrapper.llamadas) == len(critico.llamadas) == 2
    assert cuerpo["acb"]["iterations_run"] == 2
    assert cuerpo["acb"]["final_decision"] == "synthesize"
    assert "Salvedades abiertas de la revisión independiente" in cuerpo["estimation"]["summary"]


def test_la_sintesis_con_cifras_en_las_salvedades_no_es_un_500(montar) -> None:
    """Las salvedades del crítico citan cifras («250 EUR») que no están en la estimación.

    Esa nota va al inicio del resumen y, al armar la respuesta, el schema revisa las cifras
    del resumen: antes eso daba un 500 («Un resumen cita cifras que no están en la
    estimación»). La nota la escribe el código con lo que dijo el crítico, no el modelo,
    así que su contenido no se somete a esa comprobación.
    """
    con_cifras = CriticIssue(
        category="unrealistic_estimate",
        severity="major",
        field_path="phases[0].cost_eur",
        description="El coste de 250 EUR y las 62,5 horas son poco realistas para esa fase.",
        suggested_fix="",
    )
    critico = _CriticoFalso(indep("needs_iteration", con_cifras))
    cliente, _wrapper, _extractor, sid = montar(critico, max_iteraciones=2)

    respuesta = estimar_acb(cliente, sid)

    assert respuesta.status_code == 200, respuesta.text
    cuerpo = respuesta.json()
    assert cuerpo["acb"]["final_decision"] == "synthesize"
    assert "250 EUR" in cuerpo["estimation"]["summary"]  # la salvedad no se recorta


def test_por_defecto_son_tres_vueltas(montar) -> None:
    """Sin configurar nada, el tope es 3 (el de la solución de referencia)."""
    critico = _CriticoFalso(indep("needs_iteration", problema()))
    cliente, wrapper, _extractor, sid = montar(critico)

    estimar_acb(cliente, sid)

    assert len(wrapper.llamadas) == 3


# --- Crítico del mismo modelo -------------------------------------------------------------


def test_same_model_baja_el_accept_a_iterar_y_termina_en_sintesis(montar) -> None:
    """Un accept del mismo proveedor no cierra: itera y, agotadas las vueltas, sintetiza."""
    critico = _CriticoFalso(mismo("accept"))
    cliente, wrapper, _extractor, sid = montar(critico, max_iteraciones=2)

    cuerpo = estimar_acb(cliente, sid).json()

    assert len(wrapper.llamadas) == 2
    assert [i["review_status"] for i in cuerpo["acb"]["iterations"]] == ["same_model"] * 2
    assert cuerpo["acb"]["final_decision"] == "synthesize"
    assert "el crítico fue del mismo proveedor que el actor" in cuerpo["estimation"]["summary"]


# --- Crítico no disponible ----------------------------------------------------------------


def test_unavailable_entrega_el_borrador_con_nota_y_confianza_30(montar) -> None:
    """Sin revisión: una sola llamada del actor, nota al inicio y confianza 30."""
    cliente, wrapper, _extractor, sid = montar(_CriticoFalso(SIN_REVISION))

    cuerpo = estimar_acb(cliente, sid).json()

    assert len(wrapper.llamadas) == 1
    assert cuerpo["estimation"]["summary"].startswith("⚠ Esta estimación no pudo ser revisada")
    assert cuerpo["estimation"]["confidence_pct"] == 30
    iteracion = cuerpo["acb"]["iterations"][0]
    assert (iteracion["review_status"], iteracion["critic_verdict"]) == (
        "unavailable",
        "unavailable",
    )
    assert cuerpo["acb"]["final_decision"] == "synthesize"


# --- Lo que ve el crítico y lo que queda en la sesión -------------------------------------


def test_el_critico_recibe_los_adjuntos_y_la_memoria(montar) -> None:
    """El crítico ve el mismo bloque `<attachments>` que el actor y la memoria de la sesión."""
    critico = _CriticoFalso(indep("accept"))
    cliente, wrapper, _extractor, sid = montar(critico)
    pdf = hacer_pdf(["El sistema debe integrarse con el ERP LEGACYSOFT."])

    respuesta = estimar_acb(
        cliente, sid, files={"attachments": ("requisitos.pdf", pdf, "application/pdf")}
    )

    assert respuesta.status_code == 200
    llamada = critico.llamadas[0]
    assert "LEGACYSOFT" in llamada["attachments_block"]
    assert llamada["attachments_block"] == _bloque_del_actor(wrapper)
    assert llamada["actor_provider"] == "openai"
    assert llamada["actor_version"] == "v5"


def _bloque_del_actor(wrapper: _WrapperFalso) -> str:
    """El bloque `<attachments>` que viajó en el mensaje de usuario del actor."""
    prompt = wrapper.llamadas[0]["user_prompt"]
    return prompt[prompt.index("<attachments>") :]


def test_el_turno_se_anota_una_sola_vez_y_actualiza_la_memoria(montar) -> None:
    """Aunque haya dos vueltas, el historial recibe un solo turno y el extractor se llama una vez."""
    critico = _CriticoFalso(indep("needs_iteration", problema()), indep("accept"))
    cliente, _wrapper, extractor, sid = montar(critico)

    estimar_acb(cliente, sid)

    estado = cliente.get(f"/sessions/{sid}").json()
    assert estado["turns"] == 1
    assert len(extractor.llamadas) == 1


def test_el_segundo_turno_lleva_la_ventana_en_todas_las_vueltas(montar) -> None:
    """Las vueltas de un mismo turno ven el mismo historial (el de los turnos anteriores)."""
    # Turno 1: una vuelta (accept). Turno 2: dos vueltas (needs_iteration y accept).
    critico = _CriticoFalso(indep("accept"), indep("needs_iteration", problema()), indep("accept"))
    cliente, wrapper, _extractor, sid = montar(critico)
    estimar_acb(cliente, sid)

    wrapper.llamadas.clear()
    estimar_acb(cliente, sid, "Cliente: sumemos además un panel de métricas para gerencia.")

    assert [len(c["history"]) for c in wrapper.llamadas] == [2, 2]


# --- Tier ------------------------------------------------------------------------------------


def test_el_tier_del_formulario_llega_al_actor_y_al_critico(montar) -> None:
    """Con `tier=executive` el system prompt del actor lleva la audiencia y el crítico la recibe."""
    critico = _CriticoFalso(indep("accept"))
    cliente, wrapper, _extractor, sid = montar(critico)

    respuesta = estimar_acb(cliente, sid, data={"tier": "executive"})

    assert respuesta.status_code == 200
    assert "<audience>" in wrapper.llamadas[0]["system_prompt"]
    assert critico.llamadas[0]["tier"] == "executive"


def test_sin_tier_se_usa_el_configurado(montar) -> None:
    """Sin `tier`, `resolve_tier` devuelve el tier por defecto: no hay bloque de audiencia."""
    critico = _CriticoFalso(indep("accept"))
    cliente, wrapper, _extractor, sid = montar(critico)

    estimar_acb(cliente, sid)

    assert "<audience>" not in wrapper.llamadas[0]["system_prompt"]
    assert critico.llamadas[0]["tier"] == "default"


def test_un_tier_invalido_es_un_422(montar) -> None:
    """Un tier fuera de la lista se rechaza antes de llamar a ningún modelo."""
    critico = _CriticoFalso(indep("accept"))
    cliente, wrapper, _extractor, sid = montar(critico)

    respuesta = estimar_acb(cliente, sid, data={"tier": "ceo"})

    assert respuesta.status_code == 422
    assert wrapper.llamadas == []


def test_tier_view_viaja_en_la_respuesta_segun_el_tier(montar) -> None:
    """`tier_view` trae la vista propia del tier pedido, además de `estimation` de siempre."""
    critico = _CriticoFalso(indep("accept"))
    cliente, _wrapper, _extractor, sid = montar(critico)

    respuesta = estimar_acb(cliente, sid, data={"tier": "executive"})

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["estimation"]["phases"]  # la forma de siempre no desaparece
    assert cuerpo["tier_view"].keys() == {
        "headline_cost_eur",
        "headline_duration_weeks",
        "confidence_level",
        "top_risks",
        "recommendation",
    }
    assert cuerpo["tier_view"]["headline_cost_eur"] == ESTIMACION["totals"]["cost_eur"]


def test_sin_tier_tier_view_es_null(montar) -> None:
    """Con tier `default` (sin pedir ninguno) no hay vista propia: `tier_view` es `null`."""
    critico = _CriticoFalso(indep("accept"))
    cliente, _wrapper, _extractor, sid = montar(critico)

    respuesta = estimar_acb(cliente, sid)

    assert respuesta.json()["tier_view"] is None


# --- Entrada y errores -----------------------------------------------------------------------


def test_una_sesion_que_no_existe_es_un_404(montar) -> None:
    """Misma respuesta que `/estimate` para una sesión desconocida."""
    cliente, wrapper, _extractor, _sid = montar(_CriticoFalso(indep("accept")))

    respuesta = estimar_acb(cliente, "no-existe")

    assert respuesta.status_code == 404
    assert wrapper.llamadas == []


def test_una_transcripcion_corta_es_un_422(montar) -> None:
    """Las reglas de largo son las de `/estimate`."""
    cliente, wrapper, _extractor, sid = montar(_CriticoFalso(indep("accept")))

    respuesta = estimar_acb(cliente, sid, "corta")

    assert respuesta.status_code == 422
    assert wrapper.llamadas == []


def test_un_fallo_del_actor_en_la_primera_vuelta_es_un_502(montar) -> None:
    """Sin ningún borrador, el error del proveedor se traduce igual que en `/estimate`."""
    error = StructuredOutputError(attempts=3, cost_usd=0.01, last_error="no cumple el schema")
    critico = _CriticoFalso(indep("accept"))
    cliente, _wrapper, extractor, sid = montar(critico, wrapper=_WrapperFalso(error=error))

    respuesta = estimar_acb(cliente, sid)

    assert respuesta.status_code == 502
    assert critico.llamadas == []
    assert extractor.llamadas == []
    assert cliente.get(f"/sessions/{sid}").json()["turns"] == 0
