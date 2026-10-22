"""Tests de `LLMWrapper.estimate_structured`: Instructor sobre el Router de LiteLLM.

Sin red: cada llamada al Router recibe un `mock_response`, que puede ser un JSON
(válido o no contra el schema) o una excepción de LiteLLM. Se verifica:

- que una respuesta que no cumple el schema se vuelve a pedir, y que el coste y
  los tokens suman todos los intentos;
- que agotar los intentos da `StructuredOutputError`, con intentos y coste;
- que los fallos del proveedor llegan como las excepciones de LiteLLM, no
  envueltos por Instructor (el endpoint distingue 504 de 502 por el tipo);
- respaldo y trazabilidad, como en `estimate` (la caché no es del wrapper).
"""

from __future__ import annotations

import json

import litellm
import pytest
from litellm import completion_cost
from structlog.testing import capture_logs

from app.config import get_settings
from app.services.llm_wrapper import IncompleteJSONError, LLMWrapper, StructuredOutputError

PRIMARIO = "openai/gpt-4o-mini"
RESPALDO = "anthropic/claude-haiku-4-5"


def _modelos(primario: str, respaldo: str) -> dict[str, str]:
    """'proveedor/modelo' de cada rol -> variables de entorno que los fijan."""
    prov_p, modelo_p = primario.split("/", 1)
    prov_r, modelo_r = respaldo.split("/", 1)
    return {
        "LLM_DEFAULT": prov_p,
        f"{prov_p.upper()}_MODEL": modelo_p,
        "LLM_FALLBACK": prov_r,
        f"{prov_r.upper()}_MODEL": modelo_r,
    }


ESTIMACION = {
    "phases": [
        {
            "name": "Implementación",
            "summary": "Trabajo de la fase de Implementación.",
            "duration_weeks": 5,
            "hours": 320,
            "cost_eur": 20000,
            "confidence_pct": 70,
            "assumptions": [],
            "risks": [],
        }
    ],
    "team": [{"role": "Desarrollador", "headcount": 2}],
    "totals": {"hours": 320, "cost_eur": 20000, "duration_weeks": 5},
    "summary": "Estimación de prueba con las fases y el equipo indicados.",
    "confidence_pct": 70,
}
VALIDA = json.dumps(ESTIMACION)
INVALIDA = json.dumps({**ESTIMACION, "team": []})  # sin equipo: no cumple el schema


def _construir(monkeypatch, **entorno: str) -> LLMWrapper:
    """Crea el cliente del modelo de prueba con claves ficticias; los parámetros añaden o
    cambian variables de entorno.
    """
    base = {
        "OPENAI_API_KEY": "sk-openai",
        "ANTHROPIC_API_KEY": "sk-anthropic",
        **_modelos(PRIMARIO, RESPALDO),
        "LLM_MAX_RETRIES": "0",
    }
    for variable, valor in {**base, **entorno}.items():
        monkeypatch.setenv(variable, valor)
    get_settings.cache_clear()
    return LLMWrapper(settings=get_settings())


def _respuestas(wrapper: LLMWrapper, *respuestas, **extra) -> list[dict]:
    """Cada llamada al Router recibe la respuesta siguiente; devuelve los argumentos vistos."""
    original = wrapper._router.acompletion
    vistos: list[dict] = []

    async def espia(**kwargs):
        """Anota cada llamada y devuelve la siguiente respuesta preparada, guardando
        también la respuesta completa.
        """
        vistos.append(kwargs)
        respuesta = await original(**kwargs, mock_response=respuestas[len(vistos) - 1], **extra)
        devueltas.append(respuesta)
        return respuesta

    wrapper._router.acompletion = espia
    devueltas: list = []
    wrapper.devueltas = devueltas  # lo que devolvió el Router, para los tests de coste
    return vistos


async def _estimate(wrapper: LLMWrapper):
    """Atajo: pide una estimación estructurada con prompts de prueba."""
    return await wrapper.estimate_structured(
        system_prompt="sistema", user_prompt="usuario", prompt_version="v3"
    )


# --- Camino normal --------------------------------------------------------------------


async def test_el_historial_va_entre_el_system_prompt_y_el_mensaje_actual(monkeypatch) -> None:
    """Los turnos anteriores se envían en orden, después del system y antes del usuario."""
    w = _construir(monkeypatch)
    vistos = _respuestas(w, VALIDA)
    historial = [
        {"role": "user", "content": "turno anterior"},
        {"role": "assistant", "content": "respuesta anterior"},
    ]

    await w.estimate_structured(
        system_prompt="sistema", user_prompt="usuario", prompt_version="v3", history=historial
    )

    [llamada] = vistos
    assert [(m["role"], m["content"]) for m in llamada["messages"]] == [
        ("system", "sistema"),
        ("user", "turno anterior"),
        ("assistant", "respuesta anterior"),
        ("user", "usuario"),
    ]


async def test_una_respuesta_valida_se_acepta_en_el_primer_intento(monkeypatch) -> None:
    """Una respuesta válida se acepta al primer intento."""
    w = _construir(monkeypatch)
    vistos = _respuestas(w, VALIDA)

    result = await _estimate(w)

    assert result.estimation.model_dump() == ESTIMACION
    assert json.loads(result.content) == ESTIMACION
    assert result.attempts == 1
    assert result.model == PRIMARIO
    assert result.provider == "openai"
    assert result.cost_usd > 0
    assert result.prompt_version == "v3"
    assert not result.cached

    [llamada] = vistos
    assert llamada["model"] == PRIMARIO
    assert [m["role"] for m in llamada["messages"]] == ["system", "user"]
    # El schema viaja como response_format, con las fases primero.
    formato = llamada["response_format"]
    assert formato["type"] == "json_schema"
    assert list(formato["json_schema"]["schema"]["properties"]) == [
        "phases",
        "team",
        "totals",
        "summary",
        "confidence_pct",
    ]
    # La temperatura va en el deployment, no en la llamada (ver `_deployment`).
    assert "temperature" not in llamada
    assert [d["litellm_params"]["temperature"] for d in w._router.model_list] == [0.1, 0.1]


async def test_una_respuesta_invalida_se_vuelve_a_pedir_con_el_error(monkeypatch) -> None:
    """Una respuesta inválida se vuelve a pedir pasando al modelo su respuesta anterior y
    el error de validación.
    """
    w = _construir(monkeypatch)
    vistos = _respuestas(w, INVALIDA, VALIDA)

    result = await _estimate(w)

    assert result.attempts == 2
    assert result.estimation.team[0].role == "Desarrollador"
    # El segundo intento lleva la respuesta mala y el error de validación.
    segundo = vistos[1]["messages"]
    assert len(segundo) > 2
    assert "team" in segundo[-1]["content"]


async def test_coste_y_tokens_suman_todos_los_intentos(monkeypatch) -> None:
    """Cada intento se paga: no solo el que se aceptó."""
    w = _construir(monkeypatch)
    _respuestas(w, INVALIDA, VALIDA)

    result = await _estimate(w)

    primera, segunda = w.devueltas
    assert result.cost_usd == pytest.approx(completion_cost(primera) + completion_cost(segunda))
    assert result.cost_usd > completion_cost(segunda)
    assert result.prompt_tokens == primera.usage.prompt_tokens + segunda.usage.prompt_tokens
    assert result.completion_tokens == (
        primera.usage.completion_tokens + segunda.usage.completion_tokens
    )


async def test_el_coste_de_los_intentos_fallidos_llega_al_error(monkeypatch) -> None:
    """El error final incluye el coste sumado de todos los intentos fallidos."""
    w = _construir(monkeypatch, STRUCTURED_MAX_RETRIES="1")
    _respuestas(w, INVALIDA, INVALIDA)

    with pytest.raises(StructuredOutputError) as error:
        await _estimate(w)

    assert error.value.cost_usd == pytest.approx(sum(completion_cost(r) for r in w.devueltas))


async def test_agotar_los_intentos_da_structured_output_error(monkeypatch) -> None:
    """Al agotar los reintentos se lanza `StructuredOutputError` con el número de intentos."""
    w = _construir(monkeypatch, STRUCTURED_MAX_RETRIES="2")
    vistos = _respuestas(w, INVALIDA, INVALIDA, INVALIDA)

    with pytest.raises(StructuredOutputError) as error:
        await _estimate(w)

    assert len(vistos) == 3  # el primero más dos reintentos
    assert error.value.attempts == 3
    assert error.value.cost_usd > 0
    assert "team" in error.value.last_error


async def test_sin_reintentos_se_pide_una_sola_vez(monkeypatch) -> None:
    """Con cero reintentos se pide una sola vez."""
    w = _construir(monkeypatch, STRUCTURED_MAX_RETRIES="0")
    vistos = _respuestas(w, INVALIDA, VALIDA)

    with pytest.raises(StructuredOutputError):
        await _estimate(w)
    assert len(vistos) == 1


async def test_un_json_roto_tambien_se_vuelve_a_pedir(monkeypatch) -> None:
    """Un JSON roto también se vuelve a pedir."""
    w = _construir(monkeypatch)
    vistos = _respuestas(w, '{"phases": [', VALIDA)

    result = await _estimate(w)

    assert len(vistos) == 2
    assert result.attempts == 2


# --- Fallos del proveedor -----------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        litellm.Timeout(message="sin respuesta", model="gpt-4o-mini", llm_provider="openai"),
        litellm.APIConnectionError(message="caído", model="gpt-4o-mini", llm_provider="openai"),
    ],
    ids=["timeout", "conexion"],
)
async def test_un_fallo_del_proveedor_llega_como_excepcion_de_litellm(monkeypatch, error) -> None:
    """Sin respaldo, para que el fallo llegue: Instructor lo envuelve y el wrapper lo saca."""
    w = _construir(monkeypatch, ANTHROPIC_API_KEY="")
    vistos = _respuestas(w, error, VALIDA)

    with pytest.raises(type(error)):
        await _estimate(w)
    assert len(vistos) == 1  # un fallo del proveedor no es un error de validación


async def test_el_respaldo_responde_si_cae_el_primario(monkeypatch) -> None:
    """Si falla el modelo principal, responde el de respaldo y el resultado lo indica."""
    w = _construir(monkeypatch)
    _respuestas(w, VALIDA, mock_testing_fallbacks=True)

    result = await _estimate(w)

    assert result.model == RESPALDO
    assert result.provider == "anthropic"


# --- Sin caché -----------------------------------------------------------------------------


async def test_el_wrapper_no_guarda_cache_siempre_llama_al_proveedor(monkeypatch) -> None:
    """La caché es del servicio (test_estimation_service.py): dos llamadas, dos respuestas."""
    w = _construir(monkeypatch)
    vistos = _respuestas(w, VALIDA, VALIDA)

    primera = await _estimate(w)
    segunda = await _estimate(w)

    assert len(vistos) == 2
    assert not primera.cached
    assert not segunda.cached
    assert segunda.attempts == 1


# --- Trazabilidad ----------------------------------------------------------------------------


async def test_el_evento_registra_salida_e_intentos(monkeypatch) -> None:
    """El evento del log indica que la salida fue estructurada y cuántos intentos hicieron
    falta.
    """
    w = _construir(monkeypatch)
    _respuestas(w, INVALIDA, VALIDA)

    with capture_logs() as logs:
        result = await _estimate(w)

    [evento] = [e for e in logs if e["event"] == "estimacion_completada"]
    assert evento["salida"] == "estructurada"
    assert evento["intentos"] == 2
    assert evento["coste_usd"] == result.cost_usd
    assert evento["prompt_version"] == "v3"


async def test_estimate_de_texto_no_cambia_su_evento(monkeypatch) -> None:
    """Los campos nuevos son solo de la salida estructurada."""
    w = _construir(monkeypatch)
    _respuestas(w, "texto libre")

    with capture_logs() as logs:
        await w.estimate(system_prompt="s", user_prompt="u", prompt_version="v2")

    [evento] = [e for e in logs if e["event"] == "estimacion_completada"]
    assert "salida" not in evento
    assert "intentos" not in evento


async def test_un_rechazo_mal_formado_se_vuelve_a_pedir(monkeypatch) -> None:
    """Confianza baja sin «Fuera de alcance:»: el validador lo rechaza y el modelo
    recibe el motivo en el reintento."""
    w = _construir(monkeypatch)
    sin_prefijo = json.dumps({**ESTIMACION, "confidence_pct": 10})
    vistos = _respuestas(w, sin_prefijo, VALIDA)

    result = await _estimate(w)

    assert result.attempts == 2
    assert "Fuera de alcance:" in vistos[1]["messages"][-1]["content"]


# --- Guardrail de salida y respuesta cortada (WU9 B) -----------------------------------

# 500 h en 5 semanas con 2 personas: el tope es 400 h. Cumple todo lo demás.
HORAS_IMPOSIBLES = json.dumps(
    {
        **ESTIMACION,
        "phases": [{**ESTIMACION["phases"][0], "hours": 500}],
        "totals": {**ESTIMACION["totals"], "hours": 500},
    }
)


async def test_unas_horas_que_no_caben_en_el_equipo_se_vuelven_a_pedir(monkeypatch) -> None:
    """Si las horas no caben en el equipo y la duración, se vuelve a pedir pasando al
    modelo el error.
    """
    w = _construir(monkeypatch)
    vistos = _respuestas(w, HORAS_IMPOSIBLES, VALIDA)

    result = await _estimate(w)

    assert result.attempts == 2
    assert result.estimation.phases[0].hours == 320
    # El segundo intento lleva el error de la regla, para que el modelo lo corrija.
    segundo = vistos[1]["messages"]
    assert "no caben en el equipo" in json.dumps(segundo, ensure_ascii=False)


async def test_unas_horas_imposibles_que_se_repiten_agotan_los_intentos(monkeypatch) -> None:
    """Si las horas imposibles se repiten, se agotan los intentos y se lanza error."""
    w = _construir(monkeypatch, STRUCTURED_MAX_RETRIES="1")
    _respuestas(w, HORAS_IMPOSIBLES, HORAS_IMPOSIBLES)

    with pytest.raises(StructuredOutputError) as error:
        await _estimate(w)

    assert error.value.attempts == 2
    assert "no caben en el equipo" in error.value.last_error


def _cortada(contenido: str = '{"phases": [{"name": "Impl'):
    """Una respuesta que el proveedor cortó por max_tokens (finish_reason=length)."""
    return litellm.ModelResponse(
        model=PRIMARIO,
        choices=[
            {
                "index": 0,
                "finish_reason": "length",
                "message": {"role": "assistant", "content": contenido},
            }
        ],
        usage={"prompt_tokens": 100, "completion_tokens": 4000, "total_tokens": 4100},
    )


async def test_una_respuesta_cortada_no_se_reintenta_y_dice_cuanto_costo(monkeypatch) -> None:
    """Una respuesta cortada por límite de tokens no se reintenta y el error informa lo que
    costó.
    """
    w = _construir(monkeypatch, LLM_MAX_TOKENS="4000", STRUCTURED_MAX_RETRIES="3")
    llamadas: list[dict] = []

    async def acompletion(**kwargs):
        """Registra la llamada y devuelve siempre una respuesta cortada."""
        llamadas.append(kwargs)
        return _cortada()

    w._router.acompletion = acompletion

    with pytest.raises(IncompleteJSONError) as error:
        await _estimate(w)

    assert len(llamadas) == 1  # con el mismo presupuesto se cortaría igual
    assert error.value.attempts == 1
    assert error.value.max_tokens == 4000
    assert error.value.cost_usd > 0
    assert "max_tokens=4000" in str(error.value)


async def test_max_tokens_de_la_llamada_manda_en_el_error(monkeypatch) -> None:
    """El máximo de tokens que se usó en la llamada es el que aparece en el error."""
    w = _construir(monkeypatch, LLM_MAX_TOKENS="4000")

    async def acompletion(**_kwargs):
        """Devuelve siempre una respuesta cortada."""
        return _cortada()

    w._router.acompletion = acompletion

    with pytest.raises(IncompleteJSONError) as error:
        await w.estimate_structured(
            system_prompt="s", user_prompt="u", prompt_version="v3", max_tokens=123
        )
    assert error.value.max_tokens == 123


async def test_cortada_tras_un_reintento_suma_los_dos_intentos(monkeypatch) -> None:
    """Si la respuesta se corta tras un reintento, el coste suma los dos intentos."""
    w = _construir(monkeypatch, STRUCTURED_MAX_RETRIES="3")
    original = w._router.acompletion
    pedidas: list[dict] = []

    async def acompletion(**kwargs):
        """Devuelve primero una respuesta inválida y después una cortada."""
        pedidas.append(kwargs)
        if len(pedidas) == 1:  # primera respuesta: JSON completo que no cumple el schema
            return await original(**kwargs, mock_response=INVALIDA)
        return _cortada()

    w._router.acompletion = acompletion

    with pytest.raises(IncompleteJSONError) as error:
        await _estimate(w)

    assert len(pedidas) == 2
    assert error.value.attempts == 2
    una = completion_cost(await original(**pedidas[0], mock_response=INVALIDA))
    assert error.value.cost_usd > una
