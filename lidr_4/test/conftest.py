"""Fixtures compartidas.

`get_settings` está cacheado con `lru_cache`, igual que en `lidr_3`. Eso es lo
correcto en producción (Settings se lee una vez) pero en los tests rompe el
aislamiento: un `monkeypatch.setenv` no se ve si la caché ya se llenó, y el
orden de ejecución pasa a importar.

Por eso los tests que dependen del entorno construction Settings explícitamente
y vacían la caché, en vez de confiar en que el `.env` de la máquina coincida con
lo que el test asume. Sin esto, `test_el_techo_estrechado_rechaza_lo_que_el
_contrato_aceptaba` pasaría en tu máquina y fallaría en CI, o al revés.

`entorno_aislado` completa ese aislamiento para todos los tests: ninguno lee el
`.env` ni las variables de configuración de quien los corre. Cada test parte de
los defaults de Settings y fija con `monkeypatch.setenv` lo que necesita.
"""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings


@pytest.fixture(autouse=True)
def entorno_aislado(monkeypatch) -> None:
    """Settings no lee el `.env` ni las variables de configuración del entorno.

    Sin esto, los tests heredan la configuración de quien los corre: con otro
    PRIMARY_MODEL, otras claves o otro DESCRIPCION_MAX_CHARS en su `.env`,
    fallan tests que en otra máquina pasan.

    - `env_file=None` deja de leer el `.env` del proyecto (se restaura al
      terminar el test).
    - Se quitan del entorno las variables que corresponden a campos de
      Settings, por si el shell o el CI exportan alguna.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for campo in Settings.model_fields:
        monkeypatch.delenv(campo.upper(), raising=False)
        monkeypatch.delenv(campo, raising=False)


# (modelo, variable de su API key, proveedor)
_OPENAI = ("openai/gpt-4o-mini", "OPENAI_API_KEY", "openai")
_ANTHROPIC = ("anthropic/claude-haiku-4-5", "ANTHROPIC_API_KEY", "anthropic")


@pytest.fixture(
    params=[(_OPENAI, _ANTHROPIC), (_ANTHROPIC, _OPENAI)],
    ids=["openai-primario", "anthropic-primario"],
)
def modelos(request) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    """(primario, respaldo), cada uno como (modelo, variable de la clave, proveedor).

    Los tests que dependen de qué proveedor es el primario la usan y corren
    con los dos órdenes: así ninguno da por hecho uno en particular.
    """
    return request.param


@pytest.fixture(autouse=True)
def limpiar_cache_settings() -> None:
    """Vacía la caché de get_settings antes y después de cada test.

    autouse porque el aislamiento es una propiedad de todos los tests, no de
    algunos: basta con que uno lea Settings cacheado para que el resto mienta.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def construir_settings():
    """Construye Settings sin leer `.env`, con los overrides que se le pasen.

    `_env_file=None` es lo que aísla el test del archivo: sin eso, una variable
    que sobre en el `.env` del desarrollador pisa el default del test y el rojo
    aparece en la máquina de otro, no en la tuya.
    """

    def _construir(**overrides) -> Settings:
        return Settings(_env_file=None, **overrides)

    return _construir


@pytest.fixture
def settings_predeterminada(construir_settings) -> Settings:
    """Settings con los defaults de la clase, sin tocar el entorno."""
    return construir_settings()
