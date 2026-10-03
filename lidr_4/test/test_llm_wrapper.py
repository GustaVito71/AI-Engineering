"""Tests del LLMWrapper: claves, Router, respaldo, resultado, configuración, coste
y trazabilidad.

Sin red: el Router de LiteLLM acepta `mock_response` (respuesta simulada) y
`mock_testing_fallbacks` (simula la caída del primario para probar el respaldo).
Las claves y la configuración se fijan con variables de entorno y se reconstruye
Settings, igual que haría el servicio al arrancar. El `.env` de quien corre los
tests no interviene (ver `entorno_aislado` en conftest.py).

Ningún test da por hecho qué proveedor es el primario: los que dependen de eso
lo fijan explícitamente, y los de claves y respaldo corren con cada proveedor
como primario.
"""

from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from app.config import LLMConfigurationError, Settings, get_settings
from app.services import llm_wrapper as W
from app.services.llm_wrapper import LLMWrapper

# Combinación fija para los tests que no tratan de proveedores. Los que sí
# usan la fixture `modelos` de conftest.py, que prueba los dos órdenes.
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


class _CacheMemoria:
    """Caché en memoria que sí guarda: la segunda llamada igual es un acierto."""

    def __init__(self) -> None:
        self.datos: dict[str, dict] = {}

    def make_key(self, system_prompt, user_prompt):
        return f"{system_prompt}|{user_prompt}"

    async def get(self, key):
        return self.datos.get(key)

    async def set(self, key, value):
        self.datos[key] = value


def _construir(monkeypatch, **entorno: str) -> LLMWrapper:
    """Fija el entorno, reconstruye Settings y crea el wrapper como en dependencies.py.

    Parte de un entorno sin claves ni `.env` (conftest.py): solo cuenta lo que
    se pasa acá."""
    for variable, valor in entorno.items():
        monkeypatch.setenv(variable, valor)
    get_settings.cache_clear()
    return LLMWrapper(settings=get_settings(), cache=_CacheVacia())


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


def test_missing_primary_key_raises_config_error(monkeypatch, modelos) -> None:
    (primario, var_primario, _), (respaldo, var_respaldo, _) = modelos
    with pytest.raises(LLMConfigurationError, match=f"{var_primario}.*modelo primario"):
        _construir(
            monkeypatch,
            PRIMARY_MODEL=primario,
            FALLBACK_MODEL=respaldo,
            **{var_respaldo: "sk-respaldo"},
        )


def test_missing_fallback_key_starts_without_fallback(monkeypatch, modelos) -> None:
    """Sin la clave del respaldo el wrapper arranca solo con el primario, lo deja
    en el log y guarda el aviso para el usuario."""
    (primario, var_primario, _), (respaldo, var_respaldo, _) = modelos
    with capture_logs() as logs:
        w = _construir(
            monkeypatch,
            PRIMARY_MODEL=primario,
            FALLBACK_MODEL=respaldo,
            **{var_primario: "sk-primario"},
        )

    desplegados = [d["litellm_params"]["model"] for d in w._router.model_list]
    assert desplegados == [primario]
    assert not w._router.fallbacks
    [aviso] = w.avisos
    assert var_respaldo in aviso
    assert respaldo in aviso
    [evento] = [e for e in logs if e["event"] == "respaldo_no_disponible"]
    assert evento["log_level"] == "warning"
    assert evento["modelo_respaldo"] == respaldo


async def test_estimates_work_without_fallback_key(monkeypatch, modelos) -> None:
    (primario, var_primario, proveedor), (respaldo, _, _) = modelos
    w = _construir(
        monkeypatch,
        PRIMARY_MODEL=primario,
        FALLBACK_MODEL=respaldo,
        **{var_primario: "sk-primario"},
    )
    _con_mock(w, mock_response="ok")
    res = await w.estimate(system_prompt="s", user_prompt="u", prompt_version="v1")
    assert res.content == "ok"
    assert res.model == primario
    assert res.provider == proveedor


def test_uses_the_settings_it_receives() -> None:
    """El wrapper se arma con los Settings que recibe, no relee el entorno: acá
    el entorno no tiene claves (conftest.py) y los Settings pasados sí."""
    s = Settings(
        openai_api_key="sk-p",
        anthropic_api_key="sk-r",
        primary_model=PRIMARIO,
        fallback_model=RESPALDO,
        llm_max_retries=4,
    )
    w = LLMWrapper(settings=s, cache=_CacheVacia())
    assert w.avisos == ()
    assert w._router.num_retries == 4


def test_no_notices_when_both_keys_are_set(wrapper) -> None:
    assert wrapper.avisos == ()


def test_each_deployment_gets_its_provider_key(monkeypatch, modelos) -> None:
    """Cada deployment recibe la clave de su proveedor, sea primario o respaldo."""
    (primario, var_primario, _), (respaldo, var_respaldo, _) = modelos
    w = _construir(
        monkeypatch,
        PRIMARY_MODEL=primario,
        FALLBACK_MODEL=respaldo,
        **{var_primario: "sk-primario", var_respaldo: "sk-respaldo"},
    )
    claves = {
        d["litellm_params"]["model"]: d["litellm_params"]["api_key"] for d in w._router.model_list
    }
    assert claves == {primario: "sk-primario", respaldo: "sk-respaldo"}


# --- Router y respaldo ---------------------------------------------------------


def test_router_contains_both_models(wrapper) -> None:
    modelos = {d["litellm_params"]["model"] for d in wrapper._router.model_list}
    assert modelos == {PRIMARIO, RESPALDO}


async def test_fallback_triggers_on_primary_failure(monkeypatch, modelos) -> None:
    """`mock_testing_fallbacks` hace fallar al primario; `mock_response` evita que
    el respaldo llame a la API real (sin él, el test saldría a la red)."""
    (primario, var_primario, _), (respaldo, var_respaldo, proveedor_respaldo) = modelos
    w = _construir(
        monkeypatch,
        PRIMARY_MODEL=primario,
        FALLBACK_MODEL=respaldo,
        **{var_primario: "sk-primario", var_respaldo: "sk-respaldo"},
    )
    vistos = _con_mock(w, mock_testing_fallbacks=True, mock_response="ok")
    res = await w.estimate(system_prompt="s", user_prompt="u", prompt_version="v1")
    assert vistos[0]["model"] == primario  # se pidió el primario...
    assert res.model == respaldo  # ...y respondió el respaldo
    assert res.provider == proveedor_respaldo


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


# --- Trazabilidad ---------------------------------------------------------------


def _eventos(logs: list[dict]) -> list[dict]:
    return [e for e in logs if e["event"] == "estimacion_completada"]


async def test_tracing_records_a_normal_call(wrapper) -> None:
    _con_mock(wrapper, mock_response="ok")
    with capture_logs() as logs:
        res = await wrapper.estimate(system_prompt="s", user_prompt="u", prompt_version="v1")

    [evento] = _eventos(logs)
    assert evento["log_level"] == "info"
    assert evento["modelo"] == PRIMARIO
    assert evento["proveedor"] == "openai"
    assert evento["uso_respaldo"] is False
    assert evento["desde_cache"] is False
    assert evento["tokens_prompt"] == res.prompt_tokens > 0
    assert evento["tokens_completion"] == res.completion_tokens > 0
    assert evento["coste_usd"] == res.cost_usd > 0
    assert evento["coste_evitado_usd"] == 0.0
    assert evento["latencia_ms"] >= 0
    assert evento["prompt_version"] == "v1"


async def test_tracing_flags_the_fallback(wrapper) -> None:
    _con_mock(wrapper, mock_testing_fallbacks=True, mock_response="ok")
    with capture_logs() as logs:
        await wrapper.estimate(system_prompt="s", user_prompt="u", prompt_version="v1")

    [evento] = _eventos(logs)
    assert evento["modelo"] == RESPALDO
    assert evento["proveedor"] == "anthropic"
    assert evento["uso_respaldo"] is True


async def test_tracing_on_cache_hit_reports_avoided_cost(wrapper) -> None:
    wrapper._cache = _CacheMemoria()
    vistos = _con_mock(wrapper, mock_response="ok")

    primera = await wrapper.estimate(system_prompt="s", user_prompt="u", prompt_version="v1")
    with capture_logs() as logs:
        segunda = await wrapper.estimate(system_prompt="s", user_prompt="u", prompt_version="v1")

    assert len(vistos) == 1  # la segunda no llamó al proveedor
    assert primera.cached is False
    assert segunda.cached is True
    [evento] = _eventos(logs)
    assert evento["desde_cache"] is True
    assert evento["coste_usd"] == 0.0
    assert evento["coste_evitado_usd"] == primera.cost_usd > 0


async def test_tracing_never_logs_the_prompt(wrapper) -> None:
    """El evento lleva métricas, no contenido: la descripción del cliente no va al log."""
    _con_mock(wrapper, mock_response="ok")
    with capture_logs() as logs:
        await wrapper.estimate(
            system_prompt="SYSTEM-SECRETO",
            user_prompt="DESCRIPCION-DEL-CLIENTE",
            prompt_version="v1",
        )

    [evento] = _eventos(logs)
    volcado = repr(evento)
    assert "SYSTEM-SECRETO" not in volcado
    assert "DESCRIPCION-DEL-CLIENTE" not in volcado
