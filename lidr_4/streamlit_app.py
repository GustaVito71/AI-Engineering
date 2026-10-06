"""Frontend Streamlit del servicio Estimador (entrega de lidr_4).

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

import os

import httpx
import streamlit as st

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

# Interruptores de tipo de salida (claves de `st.session_state`). Son
# excluyentes: encender uno apaga el otro, y con los dos apagados la salida es
# texto libre.
STRUCTURED_TOGGLE = "structured_output"
RENDERED_TOGGLE = "rendered_output"


def _switch_off_other(state, active: str, other: str) -> None:
    """Callback de un interruptor: si quedó encendido, apaga el otro.

    Recibe el estado de la sesión como parámetro para poder probarlo con un dict.
    """
    if state.get(active):
        state[other] = False


def _output_mode(state) -> str:
    """'rendered', 'structured' o 'text', según el interruptor encendido."""
    if state.get(RENDERED_TOGGLE):
        return "rendered"
    if state.get(STRUCTURED_TOGGLE):
        return "structured"
    return "text"


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


def main() -> None:
    st.set_page_config(page_title="Estimador", page_icon="📋", layout="centered")

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

    if st.sidebar.button("Probar conexión", use_container_width=True):
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

    # --- Structured response ----------------------------------------------------
    # El interruptor decide el endpoint y, con él, qué versiones se ofrecen: cada
    # endpoint acepta solo las de su tipo de salida.
    st.sidebar.toggle(
        "Salida estructurada",
        key=STRUCTURED_TOGGLE,
        help="Pide la estimación como datos (fases, equipo y totales) en vez de texto libre.",
        on_change=_switch_off_other,
        args=(st.session_state, STRUCTURED_TOGGLE, RENDERED_TOGGLE),
    )
    # --- Rendered response ------------------------------------------------------
    st.sidebar.toggle(
        "Salida renderizada",
        key=RENDERED_TOGGLE,
        help=(
            "Pide los mismos datos y, además, la presentación del formato de salida "
            "escrita por el modelo y validada contra las cifras."
        ),
        on_change=_switch_off_other,
        args=(st.session_state, RENDERED_TOGGLE, STRUCTURED_TOGGLE),
    )
    mode = _output_mode(st.session_state)
    if mode == "rendered":
        options = _rendered_version_options(st.session_state.get("health"))
        help_text = "Predeterminada usa RENDERED_PROMPT_VERSION del servicio."
    elif mode == "structured":
        options = _structured_version_options(st.session_state.get("health"))
        help_text = "Predeterminada usa STRUCTURED_PROMPT_VERSION del servicio."
    else:
        options = _version_options(st.session_state.get("health"))
        help_text = "Predeterminada usa la versión configurada en el servicio (PROMPT_VERSION)."
    _label, chosen_prompt_version = st.sidebar.selectbox(
        "Versión del prompt",
        options=options,
        format_func=lambda option: option[0],
        help=help_text,
        # Una clave por tipo: al cambiar el interruptor no queda elegida una
        # versión del otro tipo.
        key=f"version_{mode}",
    )

    st.title("📋 Estimá un proyecto")
    st.caption(
        f"Describí el proyecto entre {DESCRIPTION_MIN_CHARS} y {DESCRIPTION_MAX_CHARS} caracteres."
    )

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
