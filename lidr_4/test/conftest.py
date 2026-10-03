"""Fixtures compartidas.

`get_settings` está cacheado con `lru_cache`, igual que en `lidr_3`. Eso es lo
correcto en producción (Settings se lee una vez) pero en los tests rompe el
aislamiento: un `monkeypatch.setenv` no se ve si la caché ya se llenó, y el
orden de ejecución pasa a importar.

Por eso los tests que dependen del entorno construction Settings explícitamente
y vacían la caché, en vez de confiar en que el `.env` de la máquina coincida con
lo que el test asume. Sin esto, `test_el_techo_estrechado_rechaza_lo_que_el
_contrato_aceptaba` pasaría en tu máquina y fallaría en CI, o al revés.
"""

from __future__ import annotations

import pytest

from app.config import Settings, get_settings


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
