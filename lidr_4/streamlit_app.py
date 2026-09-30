"""Streamlit frontend for the Estimador service (lidr_4 deliverable).

The UI is a thin, single-request client: fill a typed form, get an estimate
back as free text. It never touches a provider API key; the backend reads its
own keys from Settings (.env). The frontend only needs the base URL of the
running API (`ESTIMATOR_API_BASE_URL`, default http://localhost:8001).

Why single-shot: there is no conversation to remember. Each submission is an
independent `POST /api/v1/estimate`, so keeping a multi-turn history on screen
would imply continuity that does not exist.

Why the form and not a chat: the request is a typed contract
(`description` + three enums), not prose. A form makes that contract visible
to the user and lets the browser validate lengths before spending a token.

Why the HTTP call is a pure function: `_estimar` is testable without a running
Streamlit runtime. The contract test in test/test_frontend.py feeds it a mocked
HTTP transport. The Streamlit calls live only inside `main()`, which Streamlit
runs as __main__; importing the module from pytest defines the pure functions
and does nothing else.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

DEFAULT_API_BASE = os.environ.get("ESTIMATOR_API_BASE_URL", "http://localhost:8001")

# The 422 errors raised by `EstimationRequest` before the prompt even runs.
DESCRIPTION_MIN_CHARS = 20
DESCRIPTION_MAX_CHARS = 2000

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
    """The backend answered with a non-200 status."""

    def __init__(self, status: int, detail: object) -> None:
        super().__init__(str(detail))
        self.status = status
        self.detail = detail


def _detail_texto(detail: object) -> str:
    """FastAPI validation errors arrive as a list of `{msg, ...}` items."""
    if isinstance(detail, list):
        return "\n".join(str(item.get("msg", item)) for item in detail)
    return str(detail)


def _estimar(
    api_base: str,
    payload: dict,
    client: httpx.Client | None = None,
) -> tuple[str, str]:
    """POST the typed payload to `/api/v1/estimate` and return `(text, prompt_version)`.

    The response contract is `{text: str, prompt_version: str}` — free text, no
    structure to parse, so nothing here walks the body beyond those two keys.
    A non-200 raises `_ApiError` carrying the real HTTP status and the parsed
    `detail` (FastAPI's 422 arrives as a list of `{msg, ...}` items, which
    `_detail_texto` flattens). `client` is injected by tests
    (httpx.MockTransport); the UI always lets the function create its own.
    """
    propio = client is None
    cliente = client if client is not None else httpx.Client()
    try:
        respuesta = cliente.post(
            f"{api_base}/api/v1/estimate",
            json=payload,
            timeout=None,
        )
        if respuesta.status_code != 200:
            try:
                detail = respuesta.json().get("detail", respuesta.text)
            except ValueError:
                detail = respuesta.text
            raise _ApiError(respuesta.status_code, detail)
        cuerpo = respuesta.json()
        return cuerpo["text"], cuerpo["prompt_version"]
    finally:
        if propio:
            cliente.close()


def main() -> None:
    st.set_page_config(page_title="Estimador", page_icon="📋", layout="centered")

    st.sidebar.title("Estimador")
    st.sidebar.caption(
        "Cliente del servicio `estimador`. Las API keys viven "
        "en el backend (.env); acá no se piden."
    )
    api_base = st.sidebar.text_input("URL de la API", value=DEFAULT_API_BASE)

    if st.sidebar.button("Probar conexión", use_container_width=True):
        try:
            health = httpx.get(f"{api_base}/health", timeout=3.0).json()
            st.sidebar.success(
                f"Conectado: {health.get('provider')} · {health.get('model')}"
                f"{' · key ok' if health.get('llm_configured') else ' · FALTA API KEY'}"
            )
            if not health.get("llm_configured"):
                st.sidebar.warning(
                    "Poné OPENAI_API_KEY o ANTHROPIC_API_KEY en el .env del backend."
                )
        except httpx.HTTPError:
            st.sidebar.error(
                f"No se pudo conectar a {api_base}. Levantá la API con:\n\n`uv run python -m app`"
            )

    st.title("📋 Estimá un proyecto")
    st.caption(
        f"Describí el proyecto entre {DESCRIPTION_MIN_CHARS} y {DESCRIPTION_MAX_CHARS} caracteres."
    )

    # Single-shot: the previous result is replaced, not appended to. There is no
    # conversation state because there is no conversation.
    resultado = st.session_state.get("resultado")

    if resultado:
        st.markdown(resultado["text"])
        st.caption(f"Prompt: {resultado['prompt_version']}")
        if st.button("🧹 Nueva estimación"):
            st.session_state.pop("resultado", None)
            st.rerun()

    with st.form("estimacion", clear_on_submit=False):
        description = st.text_area(
            "Descripción del proyecto",
            height=180,
            max_chars=DESCRIPTION_MAX_CHARS,
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
        enviar = st.form_submit_button("Estimar", type="primary")

    if not enviar:
        return

    limpio = description.strip()
    if len(limpio) < DESCRIPTION_MIN_CHARS:
        st.error(
            f"La descripción necesita al menos {DESCRIPTION_MIN_CHARS} caracteres "
            f"(tiene {len(limpio)})."
        )
        return

    payload = {
        "description": limpio,
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
    }
    try:
        texto, prompt_version = _estimar(api_base, payload)
    except _ApiError as exc:
        st.error(_detail_texto(exc.detail))
        return
    except httpx.HTTPError:
        st.error(f"No se pudo conectar a {api_base}. Levantá la API con:\n\n`uv run python -m app`")
        return

    st.session_state["resultado"] = {"text": texto, "prompt_version": prompt_version}
    st.rerun()


if __name__ == "__main__":
    main()
