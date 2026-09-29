"""Tests de Settings para las variables de la Sesión 3 (fallback y cache).

El punto que cubre este archivo: el campo `llm_fallback` NO es `Literal` por un
motivo — pydantic rechaza un `""` como valor de `Literal` antes de que el
validator de Settings pueda normalizarlo a `None`. Por eso la validación vive
en `aplicar_defaults` y acá verificamos que se comporte como un .env copiado
de .env.example: vacío = fallback desactivado, typo o igual al primario = fail fast.
"""

from __future__ import annotations

import pytest
from conftest import make_settings

from app.config import LLMConfigurationError


def test_fallback_vacio_se_normaliza_a_none():
    """Un `LLM_FALLBACK=` vacío (como en .env.example) no puede romper el arranque."""
    settings = make_settings(llm_fallback="")
    assert settings.llm_fallback is None


def test_fallback_valido_se_mantiene():
    """Un fallback distinto al primario queda configurado tal cual."""
    settings = make_settings(llm_provider="openai", llm_fallback="anthropic")
    assert settings.llm_provider == "openai"
    assert settings.llm_fallback == "anthropic"


def test_fallback_sin_configurar_es_none():
    """Sin LLM_FALLBACK el campo queda None (fallback desactivado)."""
    settings = make_settings()
    assert settings.llm_fallback is None


def test_fallback_con_typo_falla_al_arrancar():
    """Un valor fuera del dominio es un error de config, no un fallback raro."""
    with pytest.raises(ValueError):
        make_settings(llm_fallback="openia")


def test_fallback_igual_al_primario_falla_al_arrancar():
    """Fallback al mismo proveedor sería un no-op con costo extra: fail fast."""
    with pytest.raises(ValueError):
        make_settings(llm_provider="openai", llm_fallback="openai")


def test_defaults_de_cache_redis():
    """Los defaults de cache quedan documentados: Redis local y 24h de TTL."""
    settings = make_settings()
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.cache_ttl == 86400


def test_active_api_key_sin_provider_usa_el_activo():
    """Sin argumento, la key es la del proveedor activo (contrato previo intacto)."""
    settings = make_settings(openai_api_key="key-oa")
    assert settings.active_api_key() == "key-oa"


def test_active_api_key_con_provider_pide_key_de_ese_proveedor():
    """Con provider explícito, devuelve la key de ESE proveedor (no el activo)."""
    settings = make_settings(
        openai_api_key="key-oa",
        anthropic_api_key="key-ant",
        llm_provider="openai",
    )
    assert settings.active_api_key("openai") == "key-oa"
    assert settings.active_api_key("anthropic") == "key-ant"


def test_active_api_key_devuelve_none_si_falta_la_key_del_proveedor():
    """Una key faltante devuelve None sin lanzar (lo que /health necesita contar)."""
    settings = make_settings(openai_api_key="key-oa")
    assert settings.active_api_key("anthropic") is None


def test_require_api_key_con_provider_errado_falla_en_el_punto_de_uso():
    """El fallback sin key falla recién cuando se pide, no al arrancar (Opción 1)."""
    settings = make_settings(openai_api_key="key-oa", llm_fallback="anthropic")
    # El arranque no lanza ni aunque el fallback esté configurado sin key
    assert settings.llm_fallback == "anthropic"
    # El punto de uso sí exige la key del proveedor pedido
    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        settings.require_api_key("anthropic")


def test_require_api_key_sin_provider_sigue_usando_el_activo():
    """El comportamiento previo de require_api_key() no cambia sin argumento."""
    settings = make_settings(openai_api_key="key-oa")
    assert settings.require_api_key() == "key-oa"


def test_resolve_model_activo_usa_llm_model_resuelto():
    """El modelo del proveedor activo respeta el override LLM_MODEL ya resuelto."""
    settings = make_settings(llm_provider="openai", llm_model="")
    assert settings.resolve_model() == "gpt-4o-mini"


def test_resolve_model_alterno_usa_su_default_no_el_override():
    """Un proveedor alterno usa su default, aunque LLM_MODEL tenga valor de otro."""
    settings = make_settings(
        llm_provider="openai",
        llm_model="my-custom-model",
        llm_fallback="anthropic",
    )
    assert settings.resolve_model() == "my-custom-model"
    assert settings.resolve_model("anthropic") == "claude-haiku-4-5"


# --- Piso de LLM_MAX_TOKENS -------------------------------------------------
#
# La API de OpenAI responde 400 "integer below minimum value" a cualquier
# max_output_tokens < 16, sin nombrar la variable. Estos tests fijan que un
# valor así muera al arrancar con un mensaje que sí dice qué corregir.


def test_max_tokens_en_el_piso_se_acepta():
    """16 es el mínimo válido: no se rechaza el borde."""
    assert make_settings(llm_max_tokens=16).llm_max_tokens == 16


@pytest.mark.parametrize("invalido", [1, 0, -5, 15])
def test_max_tokens_debajo_del_piso_falla_al_arrancar(invalido):
    with pytest.raises(ValueError, match="LLM_MAX_TOKENS"):
        make_settings(llm_max_tokens=invalido)


def test_max_tokens_invalido_dice_cual_variable_y_cual_minimo():
    """El mensaje tiene que ser accionable, no un 'value error' a secas.

    El 400 crudo de la API no nombra la variable; si el error de arranque
    tampoco lo hace, se vuelve a caer en la misma búsqueda.
    """
    with pytest.raises(ValueError) as exc:
        make_settings(llm_max_tokens=1)
    mensaje = str(exc.value)
    assert "LLM_MAX_TOKENS=1" in mensaje
    assert "16" in mensaje
