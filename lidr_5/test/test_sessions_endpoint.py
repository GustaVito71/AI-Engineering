"""Tests de `POST /sessions`: crea una sesión vacía y devuelve su identificador.

QUÉ SE COMPRUEBA

- La respuesta: código 201 y un cuerpo con un único campo, `session_id`, que es un UUID v4.
- Que cada llamada crea una sesión distinta.
- Que la sesión queda guardada en el almacén compartido de la app (si no, la
  petición siguiente no la encontraría).
- Que nace vacía y con la ventana de turnos que fija `MAX_TURNS`.
- Que se deja un evento de trazabilidad sin contenido de la conversación.
- Que no existe una forma de listar las sesiones ajenas.

Sin red ni Redis: `REDIS_URL` vacío desactiva la caché, y este endpoint no llama a
ningún proveedor de LLM, así que tampoco hacen falta claves. El `.env` de quien corre
los tests no interviene (ver `entorno_aislado` en conftest.py).
"""

from __future__ import annotations

from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from app.config import get_settings
from app.main import create_app


@pytest.fixture
def app_y_cliente(monkeypatch):
    """Devuelve una función que arma la app real y un cliente HTTP para probarla.

    Se devuelven los dos porque algunos tests miran dentro de `app.state` (donde vive
    el almacén de sesiones) además de hacer peticiones. Cada test puede fijar
    variables de entorno (por ejemplo `MAX_TURNS`) al llamarla.
    """

    def _crear(**entorno: str) -> tuple:
        """Arma la app y su cliente HTTP; los parámetros fijan variables de entorno antes
        de crearla.
        """
        monkeypatch.setenv("REDIS_URL", "")  # sin Redis: caché desactivada
        for variable, valor in entorno.items():
            monkeypatch.setenv(variable, valor)
        get_settings.cache_clear()  # que Settings relea el entorno recién fijado
        app = create_app()
        return app, TestClient(app)

    yield _crear
    get_settings.cache_clear()


def test_crea_una_sesion_y_devuelve_su_identificador(app_y_cliente) -> None:
    """`POST /sessions` responde 201 con un único campo: el identificador (UUID versión 4)."""
    _app, cliente = app_y_cliente()

    respuesta = cliente.post("/sessions")

    assert respuesta.status_code == 201
    cuerpo = respuesta.json()
    # El cuerpo tiene un solo campo: no se filtra nada más sobre la sesión.
    assert set(cuerpo) == {"session_id"}
    # Es un UUID versión 4 válido (si no lo fuera, `UUID(...)` lanzaría un error).
    assert UUID(cuerpo["session_id"]).version == 4


def test_cada_llamada_crea_una_sesion_distinta(app_y_cliente) -> None:
    """Cada llamada crea una sesión con un identificador distinto."""
    _app, cliente = app_y_cliente()

    ids = {cliente.post("/sessions").json()["session_id"] for _ in range(5)}

    assert len(ids) == 5


def test_la_sesion_queda_guardada_en_el_almacen_compartido(app_y_cliente) -> None:
    """Las sesiones creadas quedan en el mismo almacén compartido por todas las peticiones."""
    app, cliente = app_y_cliente()

    id_1 = cliente.post("/sessions").json()["session_id"]
    id_2 = cliente.post("/sessions").json()["session_id"]

    # Las dos peticiones usaron el MISMO almacén: ambas sesiones están en él.
    almacen = app.state.session_store
    assert len(almacen) == 2
    assert almacen.get(id_1) is not None
    assert almacen.get(id_2) is not None


def test_la_sesion_nace_vacia(app_y_cliente) -> None:
    """Una sesión nueva no tiene mensajes ni datos del proyecto."""
    app, cliente = app_y_cliente()

    session_id = cliente.post("/sessions").json()["session_id"]

    sesion = app.state.session_store.get(session_id)
    assert sesion.history.messages == []  # sin mensajes
    assert sesion.metadata.is_empty()  # sin hechos del proyecto


def test_la_ventana_de_la_sesion_sale_de_max_turns(app_y_cliente) -> None:
    """La ventana de la sesión (turnos que recuerda) sale de `MAX_TURNS`."""
    app, cliente = app_y_cliente(MAX_TURNS="3")

    session_id = cliente.post("/sessions").json()["session_id"]

    assert app.state.session_store.get(session_id).history.max_turns == 3


def test_la_ventana_por_defecto_es_de_seis_turnos(app_y_cliente) -> None:
    """Sin `MAX_TURNS`, la ventana es de 6 turnos."""
    app, cliente = app_y_cliente()

    session_id = cliente.post("/sessions").json()["session_id"]

    assert app.state.session_store.get(session_id).history.max_turns == 6


def test_con_el_arranque_de_la_app_tambien_funciona(app_y_cliente) -> None:
    """Con `with`, el cliente ejecuta el arranque real (`lifespan`) de la app.

    Es la forma en que corre en producción: el almacén lo crea el arranque y no la
    primera petición. El comportamiento tiene que ser el mismo.
    """
    app, cliente = app_y_cliente(MAX_TURNS="2")

    with cliente:
        session_id = cliente.post("/sessions").json()["session_id"]
        sesion = app.state.session_store.get(session_id)
        assert sesion is not None
        assert sesion.history.max_turns == 2


def test_deja_un_evento_de_traza_sin_contenido(app_y_cliente) -> None:
    """Crear una sesión deja un evento en el log con el identificador y el número de
    sesiones, sin contenido de la conversación.
    """
    _app, cliente = app_y_cliente()

    # `capture_logs` recoge los eventos de structlog que se emiten dentro del bloque.
    with capture_logs() as eventos:
        session_id = cliente.post("/sessions").json()["session_id"]

    creados = [e for e in eventos if e["event"] == "sesion_creada"]
    assert len(creados) == 1
    assert creados[0]["session_id"] == session_id
    assert creados[0]["sesiones_activas"] == 1


def test_no_hay_forma_de_listar_las_sesiones(app_y_cliente) -> None:
    """Conocer el identificador es lo único que da acceso a una sesión.

    Si existiera un `GET /sessions` que las enumerara, cualquiera podría leer o
    continuar las conversaciones de otros. Se comprueba que no existe.
    """
    _app, cliente = app_y_cliente()
    cliente.post("/sessions")

    respuesta = cliente.get("/sessions")

    assert respuesta.status_code in (404, 405)
