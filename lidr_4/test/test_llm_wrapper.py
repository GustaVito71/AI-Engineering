"""Tests del LLMWrapper: claves, Router, respaldo, resultado, configuración y coste.

Sin red: el Router de LiteLLM acepta `mock_response` (respuesta simulada) y
`mock_testing_fallbacks` (simula la caída del primario para probar el respaldo).
Las claves y la configuración se fijan con variables de entorno y se reconstruye
Settings, igual que haría el servicio al arrancar.
"""

from __future__ import annotations

import pytest

from app.config import LLMConfigurationError, get_settings
from app.services import llm_wrapper as W
from app.services.llm_wrapper import LLMWrapper

PRIMARIO = "openai/gpt-4o-mini"
RESPALDO = "anthropic/claude-haiku-4-5"


class _CacheVacia:
    """Caché que nunca acierta: obliga a pasar siempre por el Router."""

    def make_key(self, *_):
        return "k"

    async def get(self, _):
        return None

    async def set(self, *_):
        pass


def _construir(monkeypatch, **entorno: str) -> LLMWrapper:
    """Fija el entorno, reconstruye Settings y crea el wrapper como en dependencies.py."""
    for variable in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    for variable, valor in entorno.items():
        monkeypatch.setenv(variable, valor)
    get_settings.cache_clear()
    s = get_settings()
    return LLMWrapper(
        openai_api_key=s.openai_api_key,
        anthropic_api_key=s.anthropic_api_key,
        primary_model=s.primary_model,
        fallback_model=s.fallback_model,
        timeout=s.llm_timeout,
        num_retries=s.llm_max_retries,
        model_group=s.llm_model_group,
        cache=_CacheVacia(),
    )


@pytest.fixture
def wrapper(monkeypatch) -> LLMWrapper:
    return _construir(
        monkeypatch,
        OPENAI_API_KEY="sk-openai",
        ANTHROPIC_API_KEY="sk-anthropic",
        PRIMARY_MODEL=PRIMARIO,
        FALLBACK_MODEL=RESPALDO,
        LLM_MAX_RETRIES="5",
        LLM_TIMEOUT="12",
        LLM_MAX_TOKENS="1500",
    )


def _con_mock(wrapper: LLMWrapper, **extra) -> list[dict]:
    """Hace que el Router responda sin red y registra los argumentos que recibe."""
    original = wrapper._router.acompletion
    vistos: list[dict] = []

    async def espia(**kwargs):
        vistos.append(kwargs)
        return await original(**kwargs, **extra)

    wrapper._router.acompletion = espia
    return vistos


# --- Claves ------------------------------------------------------------------


def test_missing_primary_key_raises_config_error(monkeypatch) -> None:
    with pytest.raises(LLMConfigurationError, match="primary model"):
        _construir(monkeypatch, ANTHROPIC_API_KEY="sk-anthropic")


def test_missing_fallback_key_raises_config_error(monkeypatch) -> None:
    with pytest.raises(LLMConfigurationError, match="fallback model"):
        _construir(monkeypatch, OPENAI_API_KEY="sk-openai")


def test_each_deployment_gets_its_provider_key(wrapper) -> None:
    claves = {
        d["litellm_params"]["model"]: d["litellm_params"]["api_key"]
        for d in wrapper._router.model_list
    }
    assert claves.get(PRIMARIO) == "sk-openai"
    assert claves.get(RESPALDO) == "sk-anthropic"


# --- Router y respaldo ---------------------------------------------------------


def test_router_contains_both_models(wrapper) -> None:
    modelos = {d["litellm_params"]["model"] for d in wrapper._router.model_list}
    assert modelos == {PRIMARIO, RESPALDO}


async def test_fallback_triggers_on_primary_failure(wrapper) -> None:
    """`mock_testing_fallbacks` hace fallar al primario; `mock_response` evita que
    el respaldo llame a la API real (sin él, el test saldría a la red)."""
    vistos = _con_mock(wrapper, mock_testing_fallbacks=True, mock_response="ok")
    res = await wrapper.estimate(system_prompt="s", user_prompt="u", prompt_version="v1")
    assert vistos[0]["model"] == PRIMARIO  # se pidió el primario...
    assert res.model == RESPALDO  # ...y respondió el respaldo
    assert res.provider == "anthropic"


# --- Resultado -----------------------------------------------------------------


async def test_result_model_is_name_and_provider_matches(wrapper) -> None:
    _con_mock(wrapper, mock_response="ok")
    res = await wrapper.estimate(system_prompt="s", user_prompt="u", prompt_version="v1")
    assert res.model == PRIMARIO
    assert res.provider == "openai"
    assert res.cost_usd > 0  # el coste se calcula en el camino normal


async def test_prompt_version_respected(wrapper) -> None:
    _con_mock(wrapper, mock_response="ok")
    res = await wrapper.estimate(system_prompt="s", user_prompt="u", prompt_version="v7")
    assert res.prompt_version == "v7"


# --- Configuración -------------------------------------------------------------


def test_router_uses_settings_retries_and_timeout(wrapper) -> None:
    assert wrapper._router.num_retries == 5
    assert wrapper._router.timeout == 12


async def test_max_tokens_comes_from_settings(wrapper) -> None:
    vistos = _con_mock(wrapper, mock_response="ok")
    await wrapper.estimate(system_prompt="s", user_prompt="u", prompt_version="v1")
    assert vistos[0]["max_tokens"] == 1500


# --- Coste ---------------------------------------------------------------------


async def test_cost_zero_on_exception(wrapper, monkeypatch) -> None:
    _con_mock(wrapper, mock_response="ok")

    def explota(*_a, **_k):
        raise ValueError("modelo sin precio")

    monkeypatch.setattr(W, "completion_cost", explota)
    res = await wrapper.estimate(system_prompt="s", user_prompt="u", prompt_version="v1")
    assert res.cost_usd == 0.0
