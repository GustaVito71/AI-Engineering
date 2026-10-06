"""Tests de `LLMWrapper.estimate_structured`: Instructor sobre el Router de LiteLLM.

Sin red: cada llamada al Router recibe un `mock_response`, que puede ser un JSON
(válido o no contra el schema) o una excepción de LiteLLM. Se verifica:

- que una respuesta que no cumple el schema se vuelve a pedir, y que el coste y
  los tokens suman todos los intentos;
- que agotar los intentos da `StructuredOutputError`, con intentos y coste;
- que los fallos del proveedor llegan como las excepciones de LiteLLM, no
  envueltos por Instructor (el endpoint distingue 504 de 502 por el tipo);
- caché, respaldo y trazabilidad, como en `estimate`.
"""

from __future__ import annotations

import json

import litellm
import pytest
from litellm import completion_cost
from structlog.testing import capture_logs

from app.config import get_settings
from app.services.llm_wrapper import LLMWrapper, StructuredOutputError

PRIMARIO = "openai/gpt-4o-mini"
RESPALDO = "anthropic/claude-haiku-4-5"

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


class _CacheMemoria:
    def __init__(self) -> None:
        self.datos: dict[str, dict] = {}

    def make_key(self, system_prompt, user_prompt):
        return f"{system_prompt}|{user_prompt}"

    async def get(self, key):
        return self.datos.get(key)

    async def set(self, key, value):
        self.datos[key] = value


def _construir(monkeypatch, cache=None, **entorno: str) -> LLMWrapper:
    base = {
        "OPENAI_API_KEY": "sk-openai",
        "ANTHROPIC_API_KEY": "sk-anthropic",
        "PRIMARY_MODEL": PRIMARIO,
        "FALLBACK_MODEL": RESPALDO,
        "LLM_MAX_RETRIES": "0",
    }
    for variable, valor in {**base, **entorno}.items():
        monkeypatch.setenv(variable, valor)
    get_settings.cache_clear()
    return LLMWrapper(settings=get_settings(), cache=cache or _CacheMemoria())


def _respuestas(wrapper: LLMWrapper, *respuestas, **extra) -> list[dict]:
    """Cada llamada al Router recibe la respuesta siguiente; devuelve los argumentos vistos."""
    original = wrapper._router.acompletion
    vistos: list[dict] = []

    async def espia(**kwargs):
        vistos.append(kwargs)
        respuesta = await original(**kwargs, mock_response=respuestas[len(vistos) - 1], **extra)
        devueltas.append(respuesta)
        return respuesta

    wrapper._router.acompletion = espia
    devueltas: list = []
    wrapper.devueltas = devueltas  # lo que devolvió el Router, para los tests de coste
    return vistos


async def _estimate(wrapper: LLMWrapper):
    return await wrapper.estimate_structured(
        system_prompt="sistema", user_prompt="usuario", prompt_version="v3"
    )


# --- Camino normal --------------------------------------------------------------------


async def test_una_respuesta_valida_se_acepta_en_el_primer_intento(monkeypatch) -> None:
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
    assert llamada["temperature"] == 0.1


async def test_una_respuesta_invalida_se_vuelve_a_pedir_con_el_error(monkeypatch) -> None:
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
    w = _construir(monkeypatch, STRUCTURED_MAX_RETRIES="1")
    _respuestas(w, INVALIDA, INVALIDA)

    with pytest.raises(StructuredOutputError) as error:
        await _estimate(w)

    assert error.value.cost_usd == pytest.approx(sum(completion_cost(r) for r in w.devueltas))


async def test_agotar_los_intentos_da_structured_output_error(monkeypatch) -> None:
    w = _construir(monkeypatch, STRUCTURED_MAX_RETRIES="2")
    vistos = _respuestas(w, INVALIDA, INVALIDA, INVALIDA)

    with pytest.raises(StructuredOutputError) as error:
        await _estimate(w)

    assert len(vistos) == 3  # el primero más dos reintentos
    assert error.value.attempts == 3
    assert error.value.cost_usd > 0
    assert "team" in error.value.last_error


async def test_sin_reintentos_se_pide_una_sola_vez(monkeypatch) -> None:
    w = _construir(monkeypatch, STRUCTURED_MAX_RETRIES="0")
    vistos = _respuestas(w, INVALIDA, VALIDA)

    with pytest.raises(StructuredOutputError):
        await _estimate(w)
    assert len(vistos) == 1


async def test_un_json_roto_tambien_se_vuelve_a_pedir(monkeypatch) -> None:
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
    w = _construir(monkeypatch)
    _respuestas(w, VALIDA, mock_testing_fallbacks=True)

    result = await _estimate(w)

    assert result.model == RESPALDO
    assert result.provider == "anthropic"


# --- Caché ---------------------------------------------------------------------------------


async def test_la_segunda_estimacion_sale_de_la_cache(monkeypatch) -> None:
    w = _construir(monkeypatch)
    vistos = _respuestas(w, VALIDA, VALIDA)

    primera = await _estimate(w)
    segunda = await _estimate(w)

    assert len(vistos) == 1
    assert segunda.cached
    assert segunda.attempts == 0
    assert segunda.estimation == primera.estimation
    assert segunda.cost_usd == primera.cost_usd


async def test_una_entrada_de_cache_que_no_cumple_el_schema_se_ignora(monkeypatch) -> None:
    cache = _CacheMemoria()
    cache.datos["sistema|usuario"] = {
        "content": "| Fase | Semanas |",  # una estimación de texto, no JSON
        "model": PRIMARIO,
        "provider": "openai",
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "cost_usd": 0.1,
    }
    w = _construir(monkeypatch, cache=cache)
    vistos = _respuestas(w, VALIDA)

    result = await _estimate(w)

    assert len(vistos) == 1
    assert not result.cached
    assert json.loads(cache.datos["sistema|usuario"]["content"]) == ESTIMACION


# --- Trazabilidad ----------------------------------------------------------------------------


async def test_el_evento_registra_salida_e_intentos(monkeypatch) -> None:
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
