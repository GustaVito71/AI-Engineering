"""Vigila que la versión que anuncia la API coincida con la del proyecto.

La versión vive en dos sitios: `pyproject.toml` (la del paquete) y `app/main.py` (la que
muestra la documentación automática de FastAPI en `/docs` y `/openapi.json`). Ya se
desalinearon una vez (0.4.0 frente a 0.5.0); este test avisa si vuelve a pasar.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from app.main import create_app

RAIZ = Path(__file__).resolve().parent.parent


def test_la_version_de_la_api_coincide_con_la_de_pyproject() -> None:
    """La versión de FastAPI (la que ve quien abre `/docs`) es la de `pyproject.toml`."""
    with (RAIZ / "pyproject.toml").open("rb") as archivo:
        version_del_proyecto = tomllib.load(archivo)["project"]["version"]

    assert create_app().version == version_del_proyecto
