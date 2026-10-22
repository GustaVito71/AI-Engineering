"""Frontend Streamlit del servicio Estimador (lidr_4 + pestaña conversacional de lidr_5).

La pantalla tiene dos pestañas:

- «Estimación»: la interfaz de lidr_4, sin cambios (descripción corta, una sola
  petición independiente por envío).
- «Estimación Dinámica» (lidr_5): una conversación de varios turnos con memoria. Al
  cargar se crea una sesión en el servicio (`POST /sessions`); cada envío manda el
  mensaje y los adjuntos opcionales a `POST /sessions/{id}/estimate`, y el panel lateral
  muestra qué recuerda el servicio del proyecto (`GET /sessions/{id}`).

Lo que sigue describe la pestaña «Estimación» (lidr_4):


La interfaz es un cliente liviano de una sola petición: se completa un
formulario tipado y se recibe una estimación en texto libre o, con «Salida
estructurada», como datos (fases, equipo y totales). Con «Salida renderizada»
llegan los mismos datos más la presentación que pidió el formato de salida,
escrita por el modelo. Nunca toca una API
key de proveedor; el backend lee sus propias claves de Settings (.env). El
frontend solo necesita la URL base de la API en ejecución
(`ESTIMATOR_API_BASE_URL`, por defecto http://localhost:8001).

Por qué una sola petición: no hay una conversación que recordar. Cada envío es
un `POST /api/v1/estimate` independiente, así que mantener en pantalla un
historial de varios turnos sugeriría una continuidad que no existe.

Por qué un formulario y no un chat: la petición es un contrato tipado
(`description` + tres enums), no prosa libre. El formulario hace visible ese
contrato al usuario y permite que el navegador valide las longitudes antes de
gastar un token.

Por qué la llamada HTTP es una función pura: `_estimate` se puede probar sin un
runtime de Streamlit en marcha. El test de contrato de test/test_frontend.py le
pasa un transporte HTTP simulado. Las llamadas a Streamlit viven solo dentro de
`main()`, que Streamlit ejecuta como __main__; importar el módulo desde pytest
define las funciones puras y nada más.
"""

from __future__ import annotations

import logging
import os

import httpx
import streamlit as st

# Registro del frontend: los fallos inesperados quedan en `docker compose logs frontend`.
logger = logging.getLogger(__name__)

DEFAULT_API_BASE = os.environ.get("ESTIMATOR_API_BASE_URL", "http://localhost:8001")

# Límites de longitud de la descripción, copiados del contrato de la API
# (`EstimationRequest`: 20 a 2000 caracteres). El formulario los usa para avisar
# si el texto no llega al mínimo o pasa del máximo, sin esperar a que la API
# responda 422. Están copiados y no importados porque el frontend es un cliente
# aparte que no depende del paquete `app`.
#
# El máximo NO se pasa como `max_chars` al campo de texto: con ese tope, Streamlit
# descarta entero un pegado que lo supere, sin ningún aviso, y el campo queda
# vacío. Sin tope se pega todo y `_length_error` explica qué sobra.
DESCRIPTION_MIN_CHARS = 20
DESCRIPTION_MAX_CHARS = 2000

# Prefijo con el que el servicio marca una estimación rechazada en la salida
# estructurada. Copiado del schema (`OUT_OF_SCOPE_PREFIX`) por la misma
# razón que los límites de arriba: el frontend no depende del paquete `app`.
OUT_OF_SCOPE_PREFIX = "Fuera de alcance:"

PROJECT_TYPES = {
    "mobile_app": "App móvil",
    "web_saas": "SaaS web",
    "internal_tool": "Herramienta interna",
    "data_pipeline": "Pipeline de datos",
}

DETAIL_LEVELS = {
    "summary": "Resumen",
    "medium": "Medio",
    "detailed": "Detallado",
}

OUTPUT_FORMATS = {
    "phases_table": "Tabla de fases",
    "line_items": "Partidas detalladas",
    "narrative": "Narrativa",
}


class _ApiError(Exception):
    """El backend respondió con un código de estado distinto de 200."""

    def __init__(self, status: int, detail: object) -> None:
        super().__init__(str(detail))
        self.status = status
        self.detail = detail


def _detail_text(detail: object) -> str:
    """Convierte el `detail` de una respuesta de error en texto legible.

    En los errores de validación (422), FastAPI manda `detail` como una lista de
    objetos `{msg, ...}`: se toma el `msg` de cada uno, en una línea por error.
    En el resto de los errores `detail` ya es un texto y se devuelve tal cual.
    """
    if isinstance(detail, list):
        return "\n".join(str(item.get("msg", item)) for item in detail)
    return str(detail)


def _length_error(description: str) -> str | None:
    """Mensaje de error si la descripción está fuera de los límites, o None si es válida.

    Recibe el texto ya sin espacios en los extremos, que es lo que se envía a la API.
    """
    length = len(description)
    if length < DESCRIPTION_MIN_CHARS:
        return (
            f"La descripción necesita al menos {DESCRIPTION_MIN_CHARS} caracteres (tiene {length})."
        )
    if length > DESCRIPTION_MAX_CHARS:
        return (
            f"La descripción admite hasta {DESCRIPTION_MAX_CHARS} caracteres (tiene {length}). "
            "Acortala antes de estimar."
        )
    return None


def _build_payload(
    description: str, project_type: str, detail_level: str, output_format: str
) -> dict:
    """El cuerpo de `POST /api/v1/estimate` a partir de los campos del formulario.

    Fuera de `main()` para que los tests usen el mismo armado que la interfaz: una
    clave mal escrita acá rompe el contrato con la API sin ningún error local.
    """
    return {
        "description": description,
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
    }


def _read_health(api_base: str, client: httpx.Client | None = None) -> dict | None:
    """El JSON de `/health`, o None si la API no responde o no devuelve JSON."""
    http_client = client if client is not None else httpx.Client()
    try:
        return http_client.get(f"{api_base}/health", timeout=3.0).json()
    except (httpx.HTTPError, ValueError):
        return None
    finally:
        if client is None:
            http_client.close()


def _version_options(health: dict | None) -> list[tuple[str, str | None]]:
    """Opciones del selector de versión: `(etiqueta, valor de ?prompt_version=)`.

    La primera es siempre la predeterminada, con valor None: no se manda el
    parámetro y decide `PROMPT_VERSION` en el servicio. Las demás salen de
    `/health` (`prompt_versions`); si la API no respondió, queda solo la primera
    y el formulario funciona igual.
    """
    if not health:
        return [("Predeterminada", None)]
    configured = health.get("prompt_version")
    label = f"Predeterminada ({configured})" if configured else "Predeterminada"
    return [(label, None)] + [(v, v) for v in health.get("prompt_versions") or []]


def _estimate(
    api_base: str,
    payload: dict,
    client: httpx.Client | None = None,
    prompt_version: str | None = None,
) -> tuple[str, str, list[str]]:
    """Envía el formulario a `/api/v1/estimate` y devuelve `(text, prompt_version, avisos)`.

    El contrato de respuesta es `{text, prompt_version, avisos}`: texto libre,
    sin estructura que parsear. `warnings` puede faltar si la API es anterior a
    ese campo, y entonces se toma como lista vacía. Una respuesta distinta de
    200 lanza `_ApiError` con el código HTTP real y el `detail` parseado (el
    422 de FastAPI llega como lista de `{msg, ...}`, que `_detail_text`
    aplana). `client` lo inyectan los tests (httpx.MockTransport); la UI deja
    que la función cree el suyo. Con `prompt_version` se pide esa versión del
    prompt (`?prompt_version=`); sin él, la API usa la configurada.
    """
    owns_client = client is None
    http_client = client if client is not None else httpx.Client()
    try:
        response = http_client.post(
            f"{api_base}/api/v1/estimate",
            json=payload,
            params={"prompt_version": prompt_version} if prompt_version else None,
            timeout=None,
        )
        if response.status_code != 200:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise _ApiError(response.status_code, detail)
        body = response.json()
        return body["text"], body["prompt_version"], list(body.get("warnings") or [])
    finally:
        if owns_client:
            http_client.close()


# --- Structured response ----------------------------------------------------
# Con «Salida estructurada» el formulario llama a `/api/v1/estimate/structured`,
# que devuelve la estimación como JSON (fases, equipo y totales). El modelo
# devuelve siempre la misma estructura: `output_format` decide acá cómo se
# muestra, no qué se pide. Las funciones que arman lo que se muestra son puras,
# como `_estimate`, para poder probarlas sin runtime de Streamlit.


def _structured_version_options(health: dict | None) -> list[tuple[str, str | None]]:
    """Como `_version_options`, con las versiones de salida estructurada de `/health`."""
    if not health:
        return [("Predeterminada", None)]
    configured = health.get("structured_prompt_version")
    label = f"Predeterminada ({configured})" if configured else "Predeterminada"
    return [(label, None)] + [(v, v) for v in health.get("structured_prompt_versions") or []]


def _estimate_structured(
    api_base: str,
    payload: dict,
    client: httpx.Client | None = None,
    prompt_version: str | None = None,
    path: str = "/api/v1/estimate/structured",
) -> tuple[dict, str, list[str], bool]:
    """Envía el formulario a `/api/v1/estimate/structured`.

    Devuelve `(estimation, prompt_version, avisos, cached)`, con `estimation`
    como el dict de fases, equipo, totales, resumen y confianza, y `cached` en
    True si la estimación salió de la caché del servicio. Errores y `client`
    como en `_estimate`. `path` lo cambia `_estimate_rendered`, que tiene la
    misma respuesta.
    """
    owns_client = client is None
    http_client = client if client is not None else httpx.Client()
    try:
        response = http_client.post(
            f"{api_base}{path}",
            json=payload,
            params={"prompt_version": prompt_version} if prompt_version else None,
            timeout=None,
        )
        if response.status_code != 200:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise _ApiError(response.status_code, detail)
        body = response.json()
        return (
            body["estimation"],
            body["prompt_version"],
            list(body.get("warnings") or []),
            bool(body.get("cached")),
        )
    finally:
        if owns_client:
            http_client.close()


# --- Rendered response ------------------------------------------------------
# Con «Salida renderizada» el formulario llama a `/api/v1/estimate/rendered`. La
# respuesta es la de la salida estructurada más `estimation.rendered`: la
# presentación que pidió `output_format`, escrita por el modelo en Markdown y
# validada por el servicio contra las cifras. Acá se muestra tal cual.

# Salida y versión del prompt: un solo selector con las tres versiones de lidr_4 (v2 texto
# libre, v3 estructurada y v4 renderizada). Cada una decide el modo, es decir, a qué
# endpoint se envía el formulario. v1 (texto, histórica) y v5 (la de las sesiones) no se
# ofrecen en esta pestaña: la API sigue aceptándolas con `?prompt_version=`.
# Cada elemento es `(versión, modo, etiqueta)`.
OUTPUT_CHOICES = (
    ("v2", "text", "Texto libre (v2)"),
    ("v3", "structured", "Salida estructurada (v3)"),
    ("v4", "rendered", "Salida renderizada (v4)"),
)


def _output_choices(health: dict | None) -> list[tuple[str, str, str]]:
    """Las opciones del selector de salida que la API ofrece según `/health`.

    Cada opción se muestra solo si su versión figura entre las que informa el servicio para
    su modo (por ejemplo, v3 debe estar en `structured_prompt_versions`). Si la API no
    respondió, o no informa ninguna de las tres, se ofrecen todas: el formulario funciona
    igual y un error de versión lo explicaría la propia API.
    """
    if not health:
        return list(OUTPUT_CHOICES)
    offered = {
        "text": {value for _label, value in _version_options(health)},
        "structured": {value for _label, value in _structured_version_options(health)},
        "rendered": {value for _label, value in _rendered_version_options(health)},
    }
    choices = [choice for choice in OUTPUT_CHOICES if choice[0] in offered[choice[1]]]
    return choices or list(OUTPUT_CHOICES)


def _rendered_version_options(health: dict | None) -> list[tuple[str, str | None]]:
    """Como `_version_options`, con las versiones de salida renderizada de `/health`."""
    if not health:
        return [("Predeterminada", None)]
    configured = health.get("rendered_prompt_version")
    label = f"Predeterminada ({configured})" if configured else "Predeterminada"
    return [(label, None)] + [(v, v) for v in health.get("rendered_prompt_versions") or []]


def _estimate_rendered(
    api_base: str,
    payload: dict,
    client: httpx.Client | None = None,
    prompt_version: str | None = None,
) -> tuple[dict, str, list[str], bool]:
    """Envía el formulario a `/api/v1/estimate/rendered`. Devuelve lo mismo que
    `_estimate_structured`; `estimation` trae además `rendered`."""
    return _estimate_structured(
        api_base,
        payload,
        client,
        prompt_version=prompt_version,
        path="/api/v1/estimate/rendered",
    )


# --- Structured response ----------------------------------------------------


def _format_number(n: float) -> str:
    """29850 -> '29.850'; 2.5 -> '2,5'. Números en formato castellano."""
    if float(n).is_integer():
        return f"{int(n):,}".replace(",", ".")
    return f"{n:,.2f}".rstrip("0").replace(",", "X").replace(".", ",").replace("X", ".")


def _format_weeks(n: float) -> str:
    """Escribe un número de semanas con la palabra en singular o plural ("1 semana", "3
    semanas").
    """
    return f"{_format_number(n)} {'semana' if n == 1 else 'semanas'}"


def _phase_rows(estimation: dict) -> list[dict]:
    """Filas de la tabla de fases, con encabezados en castellano."""
    return [
        {
            "Fase": p["name"],
            "Semanas": _format_number(p["duration_weeks"]),
            "Horas": _format_number(p["hours"]),
            "Coste (EUR)": _format_number(p["cost_eur"]),
            "Confianza (%)": p["confidence_pct"],
            # Al final: es la columna larga, y las cifras tienen que verse primero.
            "Descripción": p["summary"],
        }
        for p in estimation["phases"]
    ]


def _line_items(estimation: dict) -> list[str]:
    """Una línea por fase para el formato «Partidas detalladas», numeradas desde 1."""
    return [
        f"{i}. {p['name']} — {_format_weeks(p['duration_weeks'])} — {_format_number(p['hours'])} horas — "
        f"{_format_number(p['cost_eur'])} EUR (confianza: {p['confidence_pct']} %). {p['summary']}"
        for i, p in enumerate(estimation["phases"], start=1)
    ]


def _narrative(estimation: dict) -> list[str]:
    """Un párrafo por fase para el formato «Narrativa»."""
    return [
        f"**{p['name']}.** {p['summary']} Dura {_format_weeks(p['duration_weeks'])}, con "
        f"{_format_number(p['hours'])} horas de equipo y un coste de {_format_number(p['cost_eur'])} EUR "
        f"(confianza: {p['confidence_pct']} %)."
        for p in estimation["phases"]
    ]


def _team_summary(estimation: dict) -> str:
    """'Desarrollador × 2, Diseñador × 1'."""
    return ", ".join(f"{member['role']} × {member['headcount']}" for member in estimation["team"])


def _is_out_of_scope(estimation: dict) -> bool:
    """True si el modelo rechazó estimar. Mismo criterio que el schema del servicio."""
    return estimation["summary"].startswith(OUT_OF_SCOPE_PREFIX)


def _show_structured(estimation: dict, output_format: str) -> None:
    """Muestra la estimación estructurada en el formato elegido en el formulario."""
    # Un rechazo no tiene cifras que mostrar: solo qué información falta.
    if _is_out_of_scope(estimation):
        st.info(estimation["summary"])
        return

    st.markdown(estimation["summary"])
    _show_metrics(estimation)

    if output_format == "phases_table":
        st.dataframe(_phase_rows(estimation), hide_index=True, use_container_width=True)
    elif output_format == "line_items":
        st.markdown("\n".join(_line_items(estimation)))
    else:
        for paragraph in _narrative(estimation):
            st.markdown(paragraph)

    _show_team_and_risks(estimation)


def _show_metrics(estimation: dict) -> None:
    """Totales y confianza global, como métricas en una fila."""
    totals = estimation["totals"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Horas", _format_number(totals["hours"]))
    c2.metric("Coste (EUR)", _format_number(totals["cost_eur"]))
    c3.metric("Duración", _format_weeks(totals["duration_weeks"]))
    c4.metric("Confianza", f"{estimation['confidence_pct']} %")


def _show_team_and_risks(estimation: dict) -> None:
    """Equipo, y supuestos y riesgos por fase cuando el nivel de detalle los trae."""
    st.markdown(f"**Equipo:** {_team_summary(estimation)}")

    # Supuestos y riesgos solo existen con nivel de detalle medio o detallado.
    for phase in estimation["phases"]:
        if not phase["assumptions"] and not phase["risks"]:
            continue
        with st.expander(f"Supuestos y riesgos: {phase['name']}"):
            for assumption in phase["assumptions"]:
                st.markdown(f"- {assumption}")
            for risk in phase["risks"]:
                st.markdown(f"- **Riesgo:** {risk['risk']} — **Mitigación:** {risk['mitigation']}")


# --- Rendered response ------------------------------------------------------
def _show_rendered(estimation: dict) -> None:
    """Muestra la estimación de la salida renderizada: la presentación del modelo.

    `rendered` ya tiene el formato elegido en el formulario (el servicio lo
    validó), así que no se arma nada acá: se muestra tal cual, entre las
    métricas y el equipo. Un rechazo se muestra como aviso, igual que en la
    salida estructurada.
    """
    if _is_out_of_scope(estimation):
        st.info(estimation["rendered"])
        return

    st.markdown(estimation["summary"])
    _show_metrics(estimation)
    st.markdown(estimation["rendered"])
    _show_team_and_risks(estimation)


# --- Pestaña «Estimación Dinámica» (conversación con memoria) ----------------------
# Reglas de entrada copiadas del contrato de `POST /sessions/{id}/estimate`, por la
# misma razón que los límites de la descripción: el frontend no importa el paquete `app`.
TRANSCRIPT_MIN_CHARS = 20
TRANSCRIPT_MAX_CHARS = 80_000
ATTACHMENT_EXTENSIONS = ["pdf", "docx", "txt", "md"]
MAX_ATTACHMENTS = 5
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024

# Nombres legibles de los hechos que guarda la sesión (`project_metadata`), en el orden
# en que se muestran en el panel lateral.
METADATA_LABELS = {
    "project_name": "Proyecto",
    "assumed_team_size": "Equipo asumido",
    "mentioned_technologies": "Tecnologías",
    "agreed_scope": "Alcance acordado",
    "explicit_constraints": "Restricciones",
    "rejected_options": "Opciones descartadas",
    "other_facts": "Otros hechos",
}

# Claves de `st.session_state` de la conversación.
CONV_SESSION_ID = "conv_session_id"
CONV_TURNS = "conv_turnos"
CONV_STATE = "conv_estado"
CONV_NOTICE = "conv_aviso"
# Pedido en curso: se guarda al pulsar «Enviar» y se atiende en la pasada siguiente, con los
# botones que cortarían la estimación desactivados (ver `_render_dynamic_tab`).
CONV_PENDING = "conv_pendiente"
# Error del último pedido: se guarda y la pantalla se redibuja, para que los botones
# vuelvan a estar activos (si no, quedarían desactivados hasta el próximo clic).
CONV_ERROR = "conv_error"
# Orden de vaciar el mensaje y los adjuntos en la próxima pasada, y número del cargador de
# archivos (cambiar su clave es la forma de vaciarlo). Los demás campos del formulario
# (tipo de proyecto, detalle, formato, versión, ACB) se conservan entre turnos.
CONV_CLEAR = "conv_vaciar"
CONV_UPLOAD_N = "conv_cargador"


def _api_error(response: httpx.Response) -> _ApiError:
    """El error de la API a partir de una respuesta que no fue 200/201.

    Toma el `detail` del JSON si lo hay; si la respuesta no es JSON, usa el texto.
    """
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return _ApiError(response.status_code, detail)


def _transcript_error(transcript: str) -> str | None:
    """Mensaje de error si el mensaje está fuera de los límites, o None si es válido.

    Recibe el texto ya sin espacios en los extremos, que es lo que se envía.
    """
    length = len(transcript)
    if length < TRANSCRIPT_MIN_CHARS:
        return f"El mensaje necesita al menos {TRANSCRIPT_MIN_CHARS} caracteres (tiene {length})."
    if length > TRANSCRIPT_MAX_CHARS:
        return (
            f"El mensaje admite hasta {TRANSCRIPT_MAX_CHARS:,} caracteres (tiene {length:,}). "
            "Acortalo antes de enviar."
        ).replace(",", ".")
    return None


def _attachments_error(files: list[tuple[str, int]]) -> str | None:
    """Mensaje de error si los adjuntos superan los límites, o None si están bien.

    Recibe pares `(nombre, tamaño en bytes)`. Avisa antes de subir nada: así no se
    espera una respuesta 413/422 del servicio por un archivo que nunca iba a aceptar.
    """
    if len(files) > MAX_ATTACHMENTS:
        return f"Se admiten como máximo {MAX_ATTACHMENTS} adjuntos por mensaje."
    for name, size in files:
        if size > MAX_ATTACHMENT_BYTES:
            return f"«{name}» pesa más de {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB."
    return None


def _create_session(api_base: str, client: httpx.Client | None = None) -> str:
    """Crea una sesión en el servicio (`POST /sessions`) y devuelve su `session_id`."""
    owns_client = client is None
    http_client = client if client is not None else httpx.Client()
    try:
        response = http_client.post(f"{api_base}/sessions", timeout=30)
        if response.status_code != 201:
            raise _api_error(response)
        return response.json()["session_id"]
    finally:
        if owns_client:
            http_client.close()


def _get_session_state(api_base: str, session_id: str, client: httpx.Client | None = None) -> dict:
    """El estado visible de la sesión (`GET /sessions/{id}`): turnos y `project_metadata`."""
    owns_client = client is None
    http_client = client if client is not None else httpx.Client()
    try:
        response = http_client.get(f"{api_base}/sessions/{session_id}", timeout=30)
        if response.status_code != 200:
            raise _api_error(response)
        return response.json()
    finally:
        if owns_client:
            http_client.close()


def _post_session_turn(
    api_base: str,
    session_id: str,
    route: str,
    transcript: str,
    project_type: str,
    detail_level: str,
    output_format: str,
    attachments: list[tuple[str, bytes]] | None,
    client: httpx.Client | None,
    prompt_version: str | None,
) -> dict:
    """Envía un turno (formulario multipart) a `route` y devuelve el cuerpo JSON de la respuesta.

    Es lo común de `_estimate_in_session` y `_estimate_in_session_acb`: la única diferencia
    entre los dos endpoints es la ruta y que el ACB agrega la traza a la respuesta.
    """
    owns_client = client is None
    http_client = client if client is not None else httpx.Client()
    try:
        response = http_client.post(
            f"{api_base}/sessions/{session_id}/{route}",
            data={
                "transcript": transcript,
                "project_type": project_type,
                "detail_level": detail_level,
                "output_format": output_format,
            },
            files=[("attachments", (name, content)) for name, content in attachments or []],
            params={"prompt_version": prompt_version} if prompt_version else None,
            timeout=None,
        )
        if response.status_code != 200:
            raise _api_error(response)
        return response.json()
    finally:
        if owns_client:
            http_client.close()


def _estimate_in_session(
    api_base: str,
    session_id: str,
    transcript: str,
    project_type: str,
    detail_level: str,
    output_format: str,
    attachments: list[tuple[str, bytes]] | None = None,
    client: httpx.Client | None = None,
    prompt_version: str | None = None,
) -> tuple[dict, str, list[str]]:
    """Envía un turno a `POST /sessions/{id}/estimate` (formulario multipart).

    Devuelve `(estimation, prompt_version, avisos)`. `attachments` son pares
    `(nombre, contenido)`. Errores y `client` como en `_estimate`.
    """
    body = _post_session_turn(
        api_base,
        session_id,
        "estimate",
        transcript,
        project_type,
        detail_level,
        output_format,
        attachments,
        client,
        prompt_version,
    )
    return body["estimation"], body["prompt_version"], list(body.get("warnings") or [])


def _estimate_in_session_acb(
    api_base: str,
    session_id: str,
    transcript: str,
    project_type: str,
    detail_level: str,
    output_format: str,
    attachments: list[tuple[str, bytes]] | None = None,
    client: httpx.Client | None = None,
    prompt_version: str | None = None,
) -> tuple[dict, str, list[str], dict]:
    """Envía un turno a `POST /sessions/{id}/estimate-acb` (actor, crítico y jefe).

    Devuelve `(estimation, prompt_version, avisos, traza)`: lo mismo que
    `_estimate_in_session` más la traza del ACB (`acb` en la respuesta).
    """
    body = _post_session_turn(
        api_base,
        session_id,
        "estimate-acb",
        transcript,
        project_type,
        detail_level,
        output_format,
        attachments,
        client,
        prompt_version,
    )
    return (
        body["estimation"],
        body["prompt_version"],
        list(body.get("warnings") or []),
        body["acb"],
    )


# Cómo se rotula en la traza quién revisó cada vuelta (`review_status` de la API).
ACB_REVIEW_LABELS = {
    "independent": "Revisión independiente",
    "same_model": "Revisión no independiente",
    "unavailable": "Sin revisión del crítico",
    "skipped": "Sin revisión (estimación rechazada)",
}
# Qué decidió el jefe tras cada vuelta (`decision_after` de la API).
ACB_DECISION_LABELS = {
    "accept": "aceptada",
    "iterate": "se pide otra vuelta",
    "synthesize": "se entrega con salvedades",
}


def _acb_trace_lines(trace: dict) -> list[str]:
    """La traza del ACB como líneas de texto para mostrar, una por vuelta y problema.

    Cada vuelta dice quién revisó (con el rótulo de `ACB_REVIEW_LABELS`), qué veredicto dio y
    qué decidió el jefe; debajo, los problemas que el crítico encontró.
    """
    lines = []
    for iteration in trace.get("iterations") or []:
        review = ACB_REVIEW_LABELS.get(iteration["review_status"], iteration["review_status"])
        provider = f" ({iteration['critic_provider']})" if iteration.get("critic_provider") else ""
        decision = ACB_DECISION_LABELS.get(iteration["decision_after"], iteration["decision_after"])
        lines.append(
            f"**Vuelta {iteration['iteration'] + 1}** · {review}{provider} · "
            f"veredicto: {iteration['critic_verdict']} · {decision}"
        )
        lines.extend(f"- {issue}" for issue in iteration.get("issue_summary") or [])
    return lines


def _show_acb_trace(trace: dict) -> None:
    """Dibuja la traza del ACB de un turno, en un recuadro dentro del desplegable del turno.

    No usa otro desplegable: Streamlit no permite un `expander` dentro de otro y cada turno
    ya se muestra en uno.
    """
    rounds = trace.get("iterations_run", 0)
    iterations = trace.get("iterations") or []
    # Si el actor rechazó la estimación (p. ej. «Fuera de alcance:») el crítico no revisó nada:
    # decir «aceptada en 1 vuelta» sería engañoso, así que se avisa de que no hubo revisión.
    not_reviewed = bool(iterations) and all(
        it.get("review_status") == "skipped" for it in iterations
    )
    with st.container(border=True):
        if not_reviewed:
            st.markdown("**Revisión del crítico (ACB):** sin revisión (estimación rechazada)")
        else:
            ending = (
                "aceptada"
                if trace.get("final_decision") == "accept"
                else "entregada con salvedades"
            )
            st.markdown(f"**Revisión del crítico (ACB):** {ending} en {rounds} vuelta(s)")
        for line in _acb_trace_lines(trace):
            st.markdown(line)


def _metadata_rows(state: dict | None) -> list[tuple[str, str]]:
    """Los hechos conocidos del proyecto como pares `(etiqueta, texto)` para el panel.

    Se omiten los campos vacíos: el panel muestra solo lo que el servicio ya recuerda.
    """
    if not state:
        return []
    metadata = state.get("project_metadata") or {}
    rows = []
    for key, label in METADATA_LABELS.items():
        value = metadata.get(key)
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        if value not in (None, ""):
            rows.append((label, str(value)))
    return rows


# Versión del prompt de la pestaña «Estimación Dinámica»: siempre v5, la única que inyecta la
# memoria del proyecto y la audiencia; las anteriores las ignoran, así que no se ofrece
# otra. Se envía siempre de forma explícita en `?prompt_version=`.
CONVERSATION_PROMPT_VERSION = "v5"


def _reset_conversation(state, session_id: str, notice: str | None = None) -> None:
    """Deja el estado de la pantalla como el de una conversación nueva.

    Recibe el estado de la sesión de Streamlit como parámetro para poder probarlo con
    un dict. `notice` es un aviso que se muestra una vez en la pestaña.
    """
    state[CONV_SESSION_ID] = session_id
    state[CONV_TURNS] = []
    state[CONV_STATE] = None
    if notice:
        state[CONV_NOTICE] = notice
    else:
        state.pop(CONV_NOTICE, None)


def _connection_error(api_base: str) -> str:
    """El mensaje estándar cuando no se puede conectar con la API."""
    return f"No se pudo conectar a {api_base}. Levantá los servicios con:\n\n`docker compose up -d`"


def _compression_lines(state: dict | None) -> list[str]:
    """Líneas del panel sobre la audiencia y la compresión del historial (Paso 9).

    Solo muestra lo que existe: la audiencia con la regla que la decidió («executive
    (nda_detected)»), cuántos turnos con compromisos se conservan textuales y si lo que
    salió de la ventana ya se resumió. Un estado sin esos campos (un servicio anterior)
    no produce líneas.
    """
    if not state:
        return []
    lines = []
    if state.get("tier"):
        lines.append(f"Audiencia: {state['tier']} ({state.get('tier_rule') or 'default'}).")
    if state.get("anchors"):
        lines.append(f"Compromisos conservados textuales: {state['anchors']}.")
    if state.get("has_summary"):
        lines.append("Lo más antiguo está resumido.")
    return lines


def _show_memory_panel(state: dict | None) -> None:
    """Panel lateral con lo que el servicio recuerda del proyecto."""
    st.sidebar.divider()
    st.sidebar.subheader("🧠 Memoria del proyecto")
    rows = _metadata_rows(state)
    if not rows:
        st.sidebar.caption("Todavía no hay hechos anotados. Aparecerán tras el primer turno.")
    for label, value in rows:
        st.sidebar.markdown(f"**{label}:** {value}")
    if state:
        st.sidebar.caption(
            f"Turnos recordados: {state['turns']} de {state['max_turns']} (ventana deslizante)."
        )
        for line in _compression_lines(state):
            st.sidebar.caption(line)


def _render_dynamic_tab(api_base: str) -> None:
    """Dibuja la pestaña «Estimación Dinámica»: la conversación con memoria.

    Orden: aviso de la sesión, turnos anteriores, formulario del turno nuevo y botón de
    «Nueva conversación». La sesión se crea al cargar y su identificador queda en el estado.
    """
    st.title("💬 Conversá el proyecto")
    st.caption(
        "Cada mensaje suma a la conversación: el servicio recuerda los últimos turnos y "
        "los hechos del proyecto (panel lateral). Podés adjuntar PDF, Word, .txt o .md."
    )

    # La sesión se crea la primera vez que se abre la pantalla.
    if CONV_SESSION_ID not in st.session_state:
        try:
            _reset_conversation(st.session_state, _create_session(api_base))
        except (_ApiError, httpx.HTTPError):
            st.error(_connection_error(api_base))
            return

    # El panel lateral se dibuja acá, con la sesión ya creada, para que aparezca desde la
    # primera carga (Streamlit permite escribir en la barra lateral desde cualquier lugar).
    _show_memory_panel(st.session_state.get(CONV_STATE))

    notice = st.session_state.pop(CONV_NOTICE, None)
    if notice:
        st.warning(notice)
    error_text = st.session_state.pop(CONV_ERROR, None)
    if error_text:
        st.error(error_text)

    turns = st.session_state[CONV_TURNS]
    for number, turn in enumerate(turns, start=1):
        with st.expander(f"Turno {number}", expanded=number == len(turns)):
            st.markdown(f"**Vos:** {turn['transcript']}")
            if turn["attachments"]:
                st.caption("Adjuntos: " + ", ".join(turn["attachments"]))
            for warning in turn["warnings"]:
                st.warning(warning)
            _show_structured(turn["estimation"], turn["output_format"])
            st.caption(f"Prompt: {turn['prompt_version']}")
            if turn.get("acb"):
                _show_acb_trace(turn["acb"])

    # Tras un turno resuelto (o «Nueva conversación») se vacían solo el mensaje y los adjuntos.
    # Se hace aquí, antes de dibujar los campos: después ya no se puede cambiar su valor.
    if st.session_state.pop(CONV_CLEAR, False):
        st.session_state["conv_transcript"] = ""
        st.session_state[CONV_UPLOAD_N] = st.session_state.get(CONV_UPLOAD_N, 0) + 1

    with st.form("conversacion", clear_on_submit=False):
        transcript = st.text_area(
            "Tu mensaje",
            height=160,
            placeholder="Contá el proyecto o aclará algo del turno anterior...",
            key="conv_transcript",
        )
        uploads = st.file_uploader(
            f"Adjuntos (hasta {MAX_ATTACHMENTS}, 10 MB cada uno)",
            type=ATTACHMENT_EXTENSIONS,
            accept_multiple_files=True,
            key=f"conv_attachments_{st.session_state.get(CONV_UPLOAD_N, 0)}",
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            project_type = st.selectbox(
                "Tipo de proyecto",
                options=list(PROJECT_TYPES),
                format_func=PROJECT_TYPES.get,
                key="conv_project_type",
            )
        with c2:
            detail_level = st.selectbox(
                "Nivel de detalle",
                options=list(DETAIL_LEVELS),
                format_func=DETAIL_LEVELS.get,
                key="conv_detail_level",
            )
        with c3:
            output_format = st.selectbox(
                "Formato de salida",
                options=list(OUTPUT_FORMATS),
                format_func=OUTPUT_FORMATS.get,
                key="conv_output_format",
            )
        use_acb = st.toggle(
            "Revisar con crítico (ACB)",
            key="conv_acb",
            help=(
                "Un segundo modelo revisa la estimación y, si encuentra problemas, se vuelve "
                "a estimar (hasta el tope del servicio). Tarda más y cuesta más."
            ),
        )
        submitted = st.form_submit_button(
            "Enviar", type="primary", disabled=CONV_PENDING in st.session_state
        )

    if st.button("🧹 Nueva conversación", disabled=CONV_PENDING in st.session_state):
        try:
            _reset_conversation(st.session_state, _create_session(api_base))
            st.session_state[CONV_CLEAR] = True
        except (_ApiError, httpx.HTTPError):
            st.error(_connection_error(api_base))
            return
        st.rerun()

    if submitted:
        cleaned = transcript.strip()
        error = _transcript_error(cleaned) or _attachments_error(
            [(upload.name, upload.size) for upload in uploads]
        )
        if error:
            st.error(error)
            return
        # El formulario se vacía al enviar. Se guarda el pedido y se vuelve a dibujar la
        # pantalla con «Probar conexión» y «Nueva conversación» desactivados: pulsarlos
        # mientras se estima cortaba la petición y el turno se perdía.
        st.session_state[CONV_PENDING] = {
            "transcript": cleaned,
            "project_type": project_type,
            "detail_level": detail_level,
            "output_format": output_format,
            "use_acb": use_acb,
            "attachments": [(upload.name, upload.getvalue()) for upload in uploads],
        }
        st.rerun()

    # Se retira ya: los botones se dibujaron desactivados en esta pasada y, pase lo que
    # pase con la petición (error, 404, corte), el pedido no se repite solo.
    pending = st.session_state.pop(CONV_PENDING, None)
    if pending is None:
        return

    cleaned = pending["transcript"]
    project_type = pending["project_type"]
    detail_level = pending["detail_level"]
    output_format = pending["output_format"]
    use_acb = pending["use_acb"]
    attachments = pending["attachments"]
    session_id = st.session_state[CONV_SESSION_ID]
    try:
        with st.spinner("Estimando y revisando..." if use_acb else "Estimando..."):
            send = _estimate_in_session_acb if use_acb else _estimate_in_session
            result = send(
                api_base,
                session_id,
                cleaned,
                project_type,
                detail_level,
                output_format,
                attachments,
                prompt_version=CONVERSATION_PROMPT_VERSION,
            )
            estimation, prompt_version, warnings = result[:3]
            # Solo el endpoint ACB devuelve la traza.
            trace = result[3] if use_acb else None
            state = _get_session_state(api_base, session_id)
    except _ApiError as exc:
        if exc.status == 404:
            # La sesión caducó o el servicio se reinició: la memoria anterior se perdió.
            try:
                _reset_conversation(
                    st.session_state,
                    _create_session(api_base),
                    "La conversación anterior caducó y se empezó una nueva. "
                    "Volvé a enviar tu mensaje; el servicio ya no recuerda lo anterior.",
                )
            except (_ApiError, httpx.HTTPError):
                st.session_state[CONV_ERROR] = _connection_error(api_base)
            st.rerun()
        st.session_state[CONV_ERROR] = _detail_text(exc.detail)
        st.rerun()
    except httpx.HTTPError:
        st.session_state[CONV_ERROR] = _connection_error(api_base)
        st.rerun()
    except Exception as exc:
        # Un error que no es de la API (por ejemplo una respuesta con otra forma de la esperada)
        # cortaría el script con los botones ya dibujados como desactivados, y no habría
        # forma de reactivarlos sin recargar la página (y perder la conversación). Se
        # registra el detalle en el log del frontend y se muestra un aviso: el mensaje no
        # se pierde y los botones vuelven a estar activos. `st.rerun()` no cae acá porque
        # Streamlit lo implementa con una excepción que no hereda de `Exception`.
        logger.exception("Fallo inesperado al procesar el turno de la conversación")
        st.session_state[CONV_ERROR] = (
            f"Ocurrió un error inesperado ({type(exc).__name__}). Tu mensaje se conservó: "
            "podés reintentar. Si se repite, revisá los logs del frontend."
        )
        st.rerun()

    turns.append(
        {
            "transcript": cleaned,
            "attachments": [name for name, _content in attachments],
            "estimation": estimation,
            "output_format": output_format,
            "prompt_version": prompt_version,
            "warnings": warnings,
            "acb": trace,
        }
    )
    st.session_state[CONV_STATE] = state
    st.session_state[CONV_CLEAR] = True
    st.rerun()


def main() -> None:
    """Dibuja la pantalla de Streamlit: barra lateral (URL de la API, prueba de conexión y
    memoria del proyecto) y las dos pestañas. Los selectores de salida y de versión de
    prompt viven dentro de la pestaña «Estimación», que es a la que aplican.
    """
    st.set_page_config(page_title="Estimador", page_icon="📋", layout="wide")

    st.sidebar.title("Estimador")
    st.sidebar.caption(
        "Cliente del servicio `estimador`. Las API keys viven "
        "en el backend (.env); acá no se piden."
    )
    api_base = st.sidebar.text_input("URL de la API", value=DEFAULT_API_BASE)

    # /health se lee una vez por URL y se guarda en la sesión: alimenta el
    # selector de versión sin consultar la API en cada interacción. "Probar
    # conexión" lo vuelve a leer.
    if st.session_state.get("health_de") != api_base:
        st.session_state["health"] = _read_health(api_base)
        st.session_state["health_de"] = api_base

    if st.sidebar.button(
        "Probar conexión",
        use_container_width=True,
        # Pulsarlo con una estimación en curso cortaba la petición (Streamlit interrumpe
        # el script al tocar un widget): mientras hay un pedido pendiente, desactivado.
        disabled=CONV_PENDING in st.session_state,
    ):
        st.session_state["health"] = _read_health(api_base)
        health = st.session_state["health"]
        if health is not None:
            st.sidebar.success(
                f"Conectado: {health.get('primary_model')} · {health.get('fallback_model')}"
                f"{' · key ok' if health.get('llm_configured') else ' · FALTA API KEY'}"
            )
            if not health.get("llm_configured"):
                st.sidebar.warning(
                    "Falta la API key del modelo primario en el .env del backend: "
                    "no se pueden generar estimaciones."
                )
            # Avisos que no impiden estimar (por ejemplo, respaldo no disponible).
            for warning in health.get("warnings") or []:
                st.sidebar.warning(warning)
        else:
            st.sidebar.error(
                f"No se pudo conectar a {api_base}. Levantá los servicios con:\n\n`docker compose up -d`"
            )

    tab_single, tab_dynamic = st.tabs(["Estimación", "Estimación Dinámica"])
    with tab_single:
        _render_single_tab(api_base)
    with tab_dynamic:
        _render_dynamic_tab(api_base)


def _output_controls() -> tuple[str, str]:
    """Dibuja el selector de salida de la pestaña «Estimación» y devuelve la elección.

    Un solo selector dentro de un recuadro: la versión del prompt (v2, v3 o v4) decide
    también el tipo de salida y, con él, el endpoint. Devuelve `(modo, versión)`, con modo
    `text`, `structured` o `rendered`.
    """
    choices = _output_choices(st.session_state.get("health"))
    with st.container(border=True):
        version, mode, _label = st.selectbox(
            "Salida y versión del prompt",
            options=choices,
            format_func=lambda choice: choice[2],
            help=(
                "Cada versión es de un tipo de salida: v2 es texto libre; v3 devuelve los "
                "datos (fases, equipo y totales); v4 suma la presentación del formato de "
                "salida escrita por el modelo y validada contra las cifras."
            ),
            key="output_choice",
        )
    return mode, version


def _render_single_tab(api_base: str) -> None:
    """Dibuja la pestaña «Estimación» (lidr_4): controles de salida, formulario de una sola
    petición y su resultado.
    """
    st.title("📋 Estimá un proyecto")
    st.caption(
        f"Describí el proyecto entre {DESCRIPTION_MIN_CHARS} y {DESCRIPTION_MAX_CHARS} caracteres."
    )
    mode, chosen_prompt_version = _output_controls()

    # Una sola estimación en pantalla: cada resultado nuevo reemplaza al anterior
    # en vez de sumarse debajo. No se guarda historial porque no hay conversación.
    result = st.session_state.get("resultado")

    if result:
        # Los avisos van antes del texto: condicionan cómo leer la estimación.
        for warning in result.get("warnings") or []:
            st.warning(warning)
        # --- Structured response ----------------------------------------------------
        if "estimation" in result and "rendered" in result["estimation"]:
            _show_rendered(result["estimation"])
        elif "estimation" in result:
            _show_structured(result["estimation"], result["output_format"])
        else:
            st.markdown(result["text"])
        cache_note = " · desde la caché" if result.get("cached") else ""
        st.caption(f"Prompt: {result['prompt_version']}{cache_note}")
        if st.button("🧹 Nueva estimación"):
            st.session_state.pop("resultado", None)
            st.rerun()

    with st.form("estimacion", clear_on_submit=False):
        description = st.text_area(
            "Descripción del proyecto",
            height=180,
            placeholder="¿Qué hay que construir, para quién y con qué requisitos?...",
            key="description_entrada",
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            project_type = st.selectbox(
                "Tipo de proyecto",
                options=list(PROJECT_TYPES),
                format_func=PROJECT_TYPES.get,
                key="project_type_entrada",
            )
        with c2:
            detail_level = st.selectbox(
                "Nivel de detalle",
                options=list(DETAIL_LEVELS),
                format_func=DETAIL_LEVELS.get,
                key="detail_level_entrada",
            )
        with c3:
            output_format = st.selectbox(
                "Formato de salida",
                options=list(OUTPUT_FORMATS),
                format_func=OUTPUT_FORMATS.get,
                key="output_format_entrada",
            )
        submitted = st.form_submit_button("Estimar", type="primary")

    if not submitted:
        return

    cleaned = description.strip()
    error = _length_error(cleaned)
    if error:
        st.error(error)
        return

    payload = _build_payload(cleaned, project_type, detail_level, output_format)
    try:
        # --- Structured response ----------------------------------------------------
        if mode != "text":
            estimate = _estimate_rendered if mode == "rendered" else _estimate_structured
            estimation, prompt_version, warnings, cached = estimate(
                api_base, payload, prompt_version=chosen_prompt_version
            )
            st.session_state["resultado"] = {
                "estimation": estimation,
                "cached": cached,
                "output_format": output_format,
                "prompt_version": prompt_version,
                "warnings": warnings,
            }
            st.rerun()
        text, prompt_version, warnings = _estimate(
            api_base, payload, prompt_version=chosen_prompt_version
        )
    except _ApiError as exc:
        st.error(_detail_text(exc.detail))
        return
    except httpx.HTTPError:
        st.error(
            f"No se pudo conectar a {api_base}. Levantá los servicios con:\n\n`docker compose up -d`"
        )
        return

    st.session_state["resultado"] = {
        "text": text,
        "prompt_version": prompt_version,
        "warnings": warnings,
    }
    st.rerun()


if __name__ == "__main__":
    main()
