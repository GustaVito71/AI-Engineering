"""Tests de la pestaña «Estimación Dinámica» de `streamlit_app.py` (Paso 6).

QUÉ SE COMPRUEBA

Funciones puras del cliente (con un servidor HTTP falso, `httpx.MockTransport`):
- crear sesión, leer su estado y enviar un turno con adjuntos: la forma de la petición
  (formulario multipart, nombres de campo, `prompt_version`) coincide con el contrato de
  la API, que es lo que se rompe en silencio;
- errores de la API (404, 422) como `_ApiError`;
- validaciones locales del mensaje y de los adjuntos;
- el panel de memoria muestra solo los hechos conocidos;
- «Nueva conversación» deja el estado limpio.

Pantalla completa (`streamlit.testing.v1.AppTest`, sin navegador ni red; la API se reemplaza
por funciones falsas): al cargar se crea la sesión; un turno aparece en pantalla y llena el
panel lateral; «Nueva conversación» pide una sesión nueva; una sesión caducada (404) se
reemplaza con un aviso; un mensaje corto no llega a la API.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from streamlit_app import (
    DETAIL_LEVELS,
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENTS,
    TRANSCRIPT_MAX_CHARS,
    TRANSCRIPT_MIN_CHARS,
    _acb_trace_lines,
    _ApiError,
    _attachments_error,
    _compression_lines,
    _create_session,
    _estimate_in_session,
    _estimate_in_session_acb,
    _get_session_state,
    _metadata_rows,
    _reset_conversation,
    _transcript_error,
)

# Carpeta del proyecto, donde vive `streamlit_app.py`.
RAIZ = Path(__file__).resolve().parent.parent

ESTIMACION = {
    "summary": "Un sistema de reservas.",
    "phases": [],
    "team": [{"role": "backend", "headcount": 2}],
    "totals": {"hours": 480, "cost_eur": 36000, "duration_weeks": 8},
    "confidence_pct": 80,
}
ESTADO = {
    "session_id": "abc",
    "turns": 1,
    "max_turns": 6,
    "project_metadata": {
        "project_name": "Bookflow",
        "assumed_team_size": 3,
        "mentioned_technologies": ["Node", "Postgres"],
        "agreed_scope": None,
        "explicit_constraints": [],
        "rejected_options": ["Rails"],
        "other_facts": ["Cliente: cadena de gimnasios"],
    },
}


# Trazas del ACB como las devuelve la API, una por estado de revisión.
TRAZA_INDEPENDIENTE = {
    "final_decision": "accept",
    "iterations_run": 2,
    "iterations": [
        {
            "iteration": 0,
            "review_status": "independent",
            "critic_provider": "anthropic",
            "critic_verdict": "needs_iteration",
            "critic_confidence": 80,
            "decision_after": "iterate",
            "issue_summary": ["[major] math_error @ totals.hours"],
        },
        {
            "iteration": 1,
            "review_status": "independent",
            "critic_provider": "anthropic",
            "critic_verdict": "accept",
            "critic_confidence": 90,
            "decision_after": "accept",
            "issue_summary": [],
        },
    ],
}
TRAZA_MISMO_MODELO = {
    "final_decision": "synthesize",
    "iterations_run": 1,
    "iterations": [
        {
            "iteration": 0,
            "review_status": "same_model",
            "critic_provider": "openai",
            "critic_verdict": "accept",
            "critic_confidence": 70,
            "decision_after": "synthesize",
            "issue_summary": [],
        }
    ],
}
TRAZA_SIN_REVISION = {
    "final_decision": "synthesize",
    "iterations_run": 1,
    "iterations": [
        {
            "iteration": 0,
            "review_status": "unavailable",
            "critic_provider": None,
            "critic_verdict": "unavailable",
            "critic_confidence": 0,
            "decision_after": "synthesize",
            "issue_summary": [],
        }
    ],
}


def _cliente(handler) -> httpx.Client:
    """Un cliente HTTP cuyo servidor es la función `handler`."""
    return httpx.Client(transport=httpx.MockTransport(handler))


# --- Funciones puras ------------------------------------------------------------------


def test_crear_sesion_devuelve_el_identificador() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert (request.method, request.url.path) == ("POST", "/sessions")
        return httpx.Response(201, json={"session_id": "abc"})

    assert _create_session("http://api", _cliente(handler)) == "abc"


def test_crear_sesion_con_error_lanza_api_error() -> None:
    cliente = _cliente(lambda _r: httpx.Response(500, json={"detail": "caído"}))
    with pytest.raises(_ApiError) as info:
        _create_session("http://api", cliente)
    assert info.value.status == 500


def test_leer_el_estado_de_la_sesion() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/sessions/abc"
        return httpx.Response(200, json=ESTADO)

    assert _get_session_state("http://api", "abc", _cliente(handler)) == ESTADO


def test_el_turno_se_envia_como_formulario_multipart_con_adjuntos() -> None:
    vistos: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        vistos.append(request)
        return httpx.Response(
            200, json={"estimation": ESTIMACION, "prompt_version": "v5", "warnings": ["aviso"]}
        )

    resultado = _estimate_in_session(
        "http://api",
        "abc",
        "Queremos un sistema de reservas",
        "web_saas",
        "medium",
        "phases_table",
        [("notas.txt", b"contenido del adjunto"), ("otro.md", b"# otro")],
        _cliente(handler),
        prompt_version="v3",
    )

    assert resultado == (ESTIMACION, "v5", ["aviso"])
    [peticion] = vistos
    assert peticion.url.path == "/sessions/abc/estimate"
    assert peticion.url.params["prompt_version"] == "v3"
    assert peticion.headers["content-type"].startswith("multipart/form-data")
    cuerpo = peticion.content.decode()
    for esperado in (
        'name="transcript"',
        "Queremos un sistema de reservas",
        'name="project_type"',
        'name="detail_level"',
        'name="output_format"',
        'name="attachments"; filename="notas.txt"',
        'filename="otro.md"',
        "contenido del adjunto",
    ):
        assert esperado in cuerpo


def test_un_turno_sin_adjuntos_ni_version_no_manda_parametros_extra() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "prompt_version" not in request.url.params
        assert b'name="attachments"' not in request.content
        return httpx.Response(200, json={"estimation": ESTIMACION, "prompt_version": "v5"})

    estimation, version, avisos = _estimate_in_session(
        "http://api", "abc", "Un mensaje suficientemente largo", "web_saas", "medium",
        "narrative", None, _cliente(handler),
    )  # fmt: skip
    assert (estimation, version, avisos) == (ESTIMACION, "v5", [])


@pytest.mark.parametrize(
    ("status", "detalle"),
    [(404, "La sesión no existe."), (422, [{"msg": "mensaje inválido"}])],
)
def test_los_errores_de_la_api_llegan_como_api_error(status, detalle) -> None:
    cliente = _cliente(lambda _r: httpx.Response(status, json={"detail": detalle}))
    with pytest.raises(_ApiError) as info:
        _estimate_in_session(
            "http://api", "abc", "Un mensaje suficientemente largo", "web_saas", "medium",
            "narrative", None, cliente,
        )  # fmt: skip
    assert info.value.status == status
    assert info.value.detail == detalle


def test_validacion_del_mensaje() -> None:
    assert _transcript_error("x" * TRANSCRIPT_MIN_CHARS) is None
    assert _transcript_error("x" * TRANSCRIPT_MAX_CHARS) is None
    assert "al menos" in _transcript_error("x" * (TRANSCRIPT_MIN_CHARS - 1))
    assert "hasta" in _transcript_error("x" * (TRANSCRIPT_MAX_CHARS + 1))


def test_validacion_de_los_adjuntos() -> None:
    assert _attachments_error([("a.pdf", 10)]) is None
    assert _attachments_error([("a.pdf", MAX_ATTACHMENT_BYTES)]) is None
    assert "como máximo" in _attachments_error([("a.pdf", 1)] * (MAX_ATTACHMENTS + 1))
    assert "a.pdf" in _attachments_error([("a.pdf", MAX_ATTACHMENT_BYTES + 1)])


def test_el_panel_de_memoria_muestra_solo_lo_conocido() -> None:
    filas = dict(_metadata_rows(ESTADO))
    assert filas["Proyecto"] == "Bookflow"
    assert filas["Equipo asumido"] == "3"
    assert filas["Tecnologías"] == "Node, Postgres"
    assert filas["Opciones descartadas"] == "Rails"
    assert filas["Otros hechos"] == "Cliente: cadena de gimnasios"
    assert "Alcance acordado" not in filas
    assert "Restricciones" not in filas
    assert _metadata_rows(None) == []


def test_el_panel_muestra_audiencia_anclas_y_resumen_solo_si_existen() -> None:
    assert _compression_lines(None) == []
    assert _compression_lines(ESTADO) == []
    estado = {**ESTADO, "tier": "executive", "tier_rule": "nda_detected", "anchors": 2}
    assert _compression_lines(estado) == [
        "Audiencia: executive (nda_detected).",
        "Compromisos conservados textuales: 2.",
    ]
    assert _compression_lines({**ESTADO, "has_summary": True}) == ["Lo más antiguo está resumido."]


def test_nueva_conversacion_deja_el_estado_limpio() -> None:
    estado = {"conv_turnos": [1, 2], "conv_estado": {"turns": 2}, "conv_aviso": "viejo"}
    _reset_conversation(estado, "nueva")
    assert estado == {"conv_session_id": "nueva", "conv_turnos": [], "conv_estado": None}
    _reset_conversation(estado, "otra", "caducó")
    assert estado["conv_aviso"] == "caducó"


# --- Pantalla completa con AppTest -----------------------------------------------------


class _ApiFalsa:
    """Servidor falso a nivel de `httpx.Client`: responde a las rutas que usa la pantalla.

    La pantalla corre como script propio (AppTest), así que sustituir funciones del módulo
    no la afecta; lo que sí comparte con el test es la librería `httpx`, que se sustituye
    acá. Registra lo que se pidió.
    """

    def __init__(self) -> None:
        self.sesiones_creadas = 0
        self.turnos: list[dict] = []
        # Última parte de la URL de cada turno: `estimate` o `estimate-acb`.
        self.rutas: list[str] = []
        # `?prompt_version=` de cada turno (None si no se mandó).
        self.versiones: list[str | None] = []
        # Traza que devuelve `estimate-acb` (cada test puede cambiarla).
        self.traza = TRAZA_INDEPENDIENTE
        self.error_al_estimar: tuple[int, str] | None = None

    def post(self, url, **kwargs) -> httpx.Response:
        """`POST /sessions` crea una sesión; `POST /sessions/{id}/estimate` registra el turno."""
        if url.endswith("/sessions"):
            self.sesiones_creadas += 1
            return httpx.Response(201, json={"session_id": f"sesion-{self.sesiones_creadas}"})
        if self.error_al_estimar:
            status, detalle = self.error_al_estimar
            return httpx.Response(status, json={"detail": detalle})
        self.turnos.append(
            {"session_id": url.split("/")[-2], "transcript": kwargs["data"]["transcript"]}
        )
        self.rutas.append(url.split("/")[-1])
        self.versiones.append((kwargs.get("params") or {}).get("prompt_version"))
        cuerpo = {"estimation": ESTIMACION, "prompt_version": "v5", "warnings": []}
        if url.endswith("/estimate-acb"):
            cuerpo["acb"] = self.traza
        return httpx.Response(200, json=cuerpo)

    def get(self, url, **_kwargs) -> httpx.Response:
        """`GET /health` y `GET /sessions/{id}` (estado fijo)."""
        if url.endswith("/health"):
            return httpx.Response(200, json={"structured_prompt_versions": ["v5"]})
        return httpx.Response(200, json={**ESTADO, "session_id": url.split("/")[-1]})


@pytest.fixture
def api(monkeypatch) -> _ApiFalsa:
    """Sustituye `httpx.Client.post/get` por el servidor falso, sin red."""
    falsa = _ApiFalsa()
    monkeypatch.setattr(httpx.Client, "post", falsa.post)
    monkeypatch.setattr(httpx.Client, "get", falsa.get)
    return falsa


def _app() -> AppTest:
    """La pantalla completa, ejecutada sin navegador."""
    return AppTest.from_file(str(RAIZ / "streamlit_app.py"), default_timeout=30).run()


def _textos(app: AppTest) -> str:
    """Todo el texto visible (markdown, subtítulos y avisos) junto, para buscar en él."""
    elementos = [*app.markdown, *app.caption, *app.warning, *app.error, *app.subheader]
    return "\n".join(e.value for e in elementos)


def _enviar(app: AppTest, mensaje: str) -> AppTest:
    """Escribe un mensaje en la pestaña dinámica y lo envía."""
    app.text_area(key="conv_transcript").set_value(mensaje)
    app.button[[b.label for b in app.button].index("Enviar")].click()
    return app.run()


def test_al_cargar_se_crea_la_sesion_y_hay_dos_pestanas(api) -> None:
    app = _app()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Estimación", "Estimación Dinámica"]
    assert api.sesiones_creadas == 1
    assert app.session_state["conv_session_id"] == "sesion-1"
    assert "Todavía no hay hechos anotados" in _textos(app)


def test_un_turno_se_muestra_y_llena_el_panel_de_memoria(api) -> None:
    app = _enviar(_app(), "Queremos un sistema de reservas con Node")
    assert not app.exception
    assert api.turnos == [
        {"session_id": "sesion-1", "transcript": "Queremos un sistema de reservas con Node"}
    ]
    texto = _textos(app)
    assert "Un sistema de reservas." in texto
    assert "Bookflow" in texto
    assert "Turnos recordados: 1 de 6" in texto


def test_nueva_conversacion_pide_otra_sesion_y_vacia_la_pantalla(api) -> None:
    app = _enviar(_app(), "Queremos un sistema de reservas con Node")
    etiquetas = [b.label for b in app.button]
    app.button[etiquetas.index("🧹 Nueva conversación")].click()
    app = app.run()
    assert api.sesiones_creadas == 2
    assert app.session_state["conv_session_id"] == "sesion-2"
    assert app.session_state["conv_turnos"] == []
    assert "Un sistema de reservas." not in _textos(app)


def test_una_sesion_caducada_se_reemplaza_con_un_aviso(api) -> None:
    app = _app()
    api.error_al_estimar = (404, "La sesión no existe.")
    app = _enviar(app, "Queremos un sistema de reservas con Node")
    assert api.sesiones_creadas == 2
    assert app.session_state["conv_session_id"] == "sesion-2"
    assert "caducó" in _textos(app)


def test_un_mensaje_corto_no_llega_a_la_api(api) -> None:
    app = _enviar(_app(), "corto")
    assert api.turnos == []
    assert "al menos" in _textos(app)


def test_el_selector_de_salida_esta_en_la_pestana_estimacion_y_no_en_la_barra_lateral(api) -> None:
    """El selector de salida y versión de lidr_4 aplica solo a la pestaña «Estimación»: no
    debe verse en la barra lateral ni mezclarse con la pestaña dinámica, que tiene el suyo.
    """
    app = _app()
    assert not app.exception
    assert len(app.sidebar.selectbox) == 0
    assert app.tabs[0].selectbox[0].label == "Salida y versión del prompt"
    # La pestaña dinámica no ofrece elegir versión: es siempre v5.
    assert [s.label for s in app.tabs[1].selectbox] == [
        "Tipo de proyecto",
        "Nivel de detalle",
        "Formato de salida",
    ]


def test_la_pestana_dinamica_pide_siempre_el_prompt_v5(api) -> None:
    """No hay selector de versión: los dos endpoints reciben `?prompt_version=v5`."""
    app = _enviar(_app(), "Queremos un sistema de reservas con Node")
    app.toggle(key="conv_acb").set_value(True)
    _enviar(app, "Ahora con revisión del crítico, por favor, sobre el mismo proyecto")
    assert api.rutas == ["estimate", "estimate-acb"]
    assert api.versiones == ["v5", "v5"]


def test_tras_un_error_del_servicio_se_muestra_y_los_botones_vuelven_a_estar_activos(api) -> None:
    """Un error del servicio (por ejemplo un 500) no deja la pantalla bloqueada: se muestra el
    detalle y «Probar conexión», «Nueva conversación» y «Enviar» quedan activos de nuevo.
    """
    app = _app()
    api.error_al_estimar = (500, "Detalle de prueba.")
    app = _enviar(app, "Queremos un sistema de reservas con Node")
    assert "Detalle de prueba." in _textos(app)
    botones = {b.label: b for b in [*app.sidebar.button, *app.button]}
    assert not botones["Probar conexión"].disabled
    assert not botones["🧹 Nueva conversación"].disabled
    assert "conv_pendiente" not in app.session_state


def test_mientras_se_estima_los_botones_que_cortarian_el_turno_estan_desactivados(
    api, monkeypatch
) -> None:
    """Tocar «Probar conexión» o «Nueva conversación» durante una estimación la corta (Streamlit
    interrumpe el script). Para ver la pantalla tal como está mientras la API trabaja, la
    llamada falsa detiene el script con `st.stop()`: la pasada queda a medias y su pantalla
    muestra los botones desactivados.
    """
    app = _app()

    def api_que_se_cuelga(*_args, **_kwargs):
        st.stop()

    monkeypatch.setattr(httpx.Client, "post", api_que_se_cuelga)
    app = _enviar(app, "Queremos un sistema de reservas con Node")
    botones = {b.label: b for b in [*app.sidebar.button, *app.button]}
    assert botones["Probar conexión"].disabled
    assert botones["🧹 Nueva conversación"].disabled


def test_un_error_que_no_es_de_la_api_no_deja_los_botones_bloqueados(api, monkeypatch) -> None:
    """Si algo falla por un motivo ajeno a la API (acá, una respuesta con otra forma), no hay
    traceback ni bloqueo: se avisa, el mensaje se conserva y los botones quedan activos.
    """
    app = _app()

    def respuesta_rara(url, **_kwargs):
        if url.endswith("/sessions"):
            return httpx.Response(201, json={"session_id": "sesion-1"})
        # Sin las claves `estimation` y `prompt_version` que la pantalla espera.
        return httpx.Response(200, json={"algo": "inesperado"})

    monkeypatch.setattr(httpx.Client, "post", respuesta_rara)
    app = _enviar(app, "Queremos un sistema de reservas con Node")

    assert not app.exception
    assert "Ocurrió un error inesperado" in _textos(app)
    botones = {b.label: b for b in [*app.sidebar.button, *app.button]}
    assert not botones["Probar conexión"].disabled
    assert not botones["🧹 Nueva conversación"].disabled
    assert not any(b.disabled for b in app.button if b.label == "Enviar")
    assert app.text_area(key="conv_transcript").value == "Queremos un sistema de reservas con Node"
    assert "conv_pendiente" not in app.session_state


def test_los_ajustes_del_formulario_persisten_entre_turnos_y_el_mensaje_se_vacia(api) -> None:
    """Tipo de proyecto, detalle, formato, versión e interruptor ACB se conservan de un turno
    al siguiente; solo se vacía el mensaje. Si el turno falla, el mensaje no se pierde.
    """
    app = _app()
    segundo = list(DETAIL_LEVELS)[1]
    app.selectbox(key="conv_detail_level").set_value(segundo)
    app.toggle(key="conv_acb").set_value(True)
    app = _enviar(app, "Queremos un sistema de reservas con Node")
    assert app.selectbox(key="conv_detail_level").value == segundo
    assert app.toggle(key="conv_acb").value is True
    assert app.text_area(key="conv_transcript").value == ""
    api.error_al_estimar = (500, "Detalle de prueba.")
    app = _enviar(app, "Segundo mensaje que falla")
    assert app.text_area(key="conv_transcript").value == "Segundo mensaje que falla"
    assert app.selectbox(key="conv_detail_level").value == segundo


def test_terminada_la_estimacion_los_botones_vuelven_a_estar_activos(api) -> None:
    """Con el turno ya resuelto, «Probar conexión» y «Nueva conversación» se pueden usar."""
    app = _enviar(_app(), "Queremos un sistema de reservas con Node")
    botones = {b.label: b for b in [*app.sidebar.button, *app.button]}
    assert not botones["Probar conexión"].disabled
    assert not botones["🧹 Nueva conversación"].disabled
    assert "conv_pendiente" not in app.session_state


# --- Interruptor del ACB ------------------------------------------------------------------


def test_el_turno_acb_va_a_estimate_acb_y_devuelve_la_traza() -> None:
    """La petición ACB usa la misma forma multipart que la normal, en otra ruta, y trae `acb`."""
    vistos: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        vistos.append(request)
        cuerpo = {
            "estimation": ESTIMACION,
            "prompt_version": "v5",
            "warnings": ["aviso"],
            "acb": TRAZA_INDEPENDIENTE,
        }
        return httpx.Response(200, json=cuerpo)

    resultado = _estimate_in_session_acb(
        "http://api",
        "abc",
        "Queremos un sistema de reservas",
        "web_saas",
        "medium",
        "phases_table",
        [("notas.txt", b"contenido")],
        _cliente(handler),
    )

    assert resultado == (ESTIMACION, "v5", ["aviso"], TRAZA_INDEPENDIENTE)
    [peticion] = vistos
    assert peticion.url.path == "/sessions/abc/estimate-acb"
    assert 'name="attachments"; filename="notas.txt"' in peticion.content.decode()


def test_la_traza_rotula_quien_reviso_cada_vuelta() -> None:
    """Cada estado de revisión tiene su rótulo; los problemas se listan bajo su vuelta."""
    independiente = _acb_trace_lines(TRAZA_INDEPENDIENTE)
    assert "Revisión independiente (anthropic)" in independiente[0]
    assert "se pide otra vuelta" in independiente[0]
    assert independiente[1] == "- [major] math_error @ totals.hours"

    assert "Revisión no independiente (openai)" in _acb_trace_lines(TRAZA_MISMO_MODELO)[0]
    sin_revision = _acb_trace_lines(TRAZA_SIN_REVISION)[0]
    assert "Sin revisión del crítico" in sin_revision
    assert "(" not in sin_revision.split("·")[1]  # sin proveedor entre paréntesis


def _enviar_con_acb(app: AppTest, mensaje: str, *, acb: bool) -> AppTest:
    """Escribe un mensaje, fija el interruptor del ACB y lo envía."""
    app.toggle(key="conv_acb").set_value(acb)
    return _enviar(app, mensaje)


def test_el_interruptor_apagado_usa_estimate_y_no_muestra_traza(api) -> None:
    """Por defecto el ACB está apagado: ruta normal y ningún recuadro de revisión."""
    app = _app()
    assert app.toggle(key="conv_acb").label == "Revisar con crítico (ACB)"
    assert app.toggle(key="conv_acb").value is False

    app = _enviar_con_acb(app, "Queremos un sistema de reservas con Node", acb=False)

    assert not app.exception
    assert api.rutas == ["estimate"]
    assert "Revisión del crítico" not in _textos(app)


def test_el_interruptor_encendido_usa_estimate_acb_y_muestra_la_traza(api) -> None:
    """Con el ACB encendido el turno va al endpoint ACB y la traza se ve en el turno."""
    app = _enviar_con_acb(_app(), "Queremos un sistema de reservas con Node", acb=True)

    assert not app.exception
    assert api.rutas == ["estimate-acb"]
    texto = _textos(app)
    assert "aceptada en 2 vuelta(s)" in texto
    assert "Revisión independiente (anthropic)" in texto


def test_una_estimacion_rechazada_no_se_muestra_como_aceptada(api) -> None:
    """Si el crítico no revisó (estado `skipped`) el encabezado no dice «aceptada en N vuelta(s)»."""
    api.traza = {
        "final_decision": "accept",
        "iterations_run": 1,
        "iterations": [
            {
                "iteration": 0,
                "review_status": "skipped",
                "critic_provider": None,
                "critic_verdict": "skipped",
                "critic_confidence": 0,
                "decision_after": "accept",
                "issue_summary": [],
            }
        ],
    }
    app = _enviar_con_acb(_app(), "Por favor, revisa la estimación", acb=True)

    assert not app.exception
    texto = _textos(app)
    assert "sin revisión (estimación rechazada)" in texto
    assert "aceptada en" not in texto


@pytest.mark.parametrize(
    ("traza", "rotulo"),
    [
        (TRAZA_MISMO_MODELO, "Revisión no independiente"),
        (TRAZA_SIN_REVISION, "Sin revisión del crítico"),
    ],
)
def test_la_pantalla_muestra_los_rotulos_de_revision_no_independiente(api, traza, rotulo) -> None:
    """Las revisiones no independientes o ausentes se distinguen en la pantalla."""
    api.traza = traza
    app = _enviar_con_acb(_app(), "Queremos un sistema de reservas con Node", acb=True)

    assert not app.exception
    assert rotulo in _textos(app)
    assert "entregada con salvedades" in _textos(app)
