"""Fixtures compartidos.

El detalle que importa: `make_settings` pasa `_env_file=None` y los valores
explícitamente. Así un test no puede leer el .env local ni una API key real
del entorno. Esa es la causa más común de suites que pasan en tu máquina y
fallan en CI: una variable que tenés seteada en el shell."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app


def make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


@pytest.fixture
def settings() -> Settings:
    return make_settings(openai_api_key="test-key-openai")


@pytest.fixture
def client(settings: Settings):
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    # TestClient usado como context manager: solo así Starlette ejecuta el
    # lifespan. Fuera del `with`, la app no llega a arrancar de verdad y un
    # fallo de arranque pasaría desapercibido. Es un test que parece correr.
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _limpiar_cache_de_settings():
    """Sin esto, un lru_cache arrastra configuración entre tests y el orden
    de ejecución empieza a importar. De los bugs que más tiempo cuestan."""
    yield
    get_settings.cache_clear()