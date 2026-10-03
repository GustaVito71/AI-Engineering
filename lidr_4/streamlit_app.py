"""Frontend Streamlit del servicio Estimador (entrega de lidr_4).

La interfaz es un cliente liviano de una sola petición: se completa un
formulario tipado y se recibe una estimación en texto libre. Nunca toca una API
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

Por qué la llamada HTTP es una función pura: `_estimar` se puede probar sin un
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
# (`EstimationRequest`: 20 a 2000 caracteres). El formulario los usa para cortar
# el texto en el máximo y avisar si no llega al mínimo, sin esperar a que la API
# responda 422. Están copiados y no importados porque el frontend es un cliente
# aparte que no depende del paquete `app`.
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
    """El backend respondió con un código de estado distinto de 200."""

    def __init__(self, status: int, detail: object) -> None:
        super().__init__(str(detail))
        self.status = status
        self.detail = detail


def _detail_texto(detail: object) -> str:
    """Convierte el `detail` de una respuesta de error en texto legible.

    En los errores de validación (422), FastAPI manda `detail` como una lista de
    objetos `{msg, ...}`: se toma el `msg` de cada uno, en una línea por error.
    En el resto de los errores `detail` ya es un texto y se devuelve tal cual.
    """
    if isinstance(detail, list):
        return "\n".join(str(item.get("msg", item)) for item in detail)
    return str(detail)


def _estimar(
    api_base: str,
    payload: dict,
    client: httpx.Client | None = None,
) -> tuple[str, str, list[str]]:
    """Envía el formulario a `/api/v1/estimate` y devuelve `(text, prompt_version, avisos)`.

    El contrato de respuesta es `{text, prompt_version, avisos}`: texto libre,
    sin estructura que parsear. `avisos` puede faltar si la API es anterior a
    ese campo, y entonces se toma como lista vacía. Una respuesta distinta de
    200 lanza `_ApiError` con el código HTTP real y el `detail` parseado (el
    422 de FastAPI llega como lista de `{msg, ...}`, que `_detail_texto`
    aplana). `client` lo inyectan los tests (httpx.MockTransport); la UI deja
    que la función cree el suyo.
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
        return cuerpo["text"], cuerpo["prompt_version"], list(cuerpo.get("avisos") or [])
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
                f"Conectado: {health.get('primary_model')} · {health.get('fallback_model')}"
                f"{' · key ok' if health.get('llm_configured') else ' · FALTA API KEY'}"
            )
            if not health.get("llm_configured"):
                st.sidebar.warning(
                    "Falta la API key del modelo primario en el .env del backend: "
                    "no se pueden generar estimaciones."
                )
            # Avisos que no impiden estimar (por ejemplo, respaldo no disponible).
            for aviso in health.get("avisos") or []:
                st.sidebar.warning(aviso)
        except httpx.HTTPError:
            st.sidebar.error(
                f"No se pudo conectar a {api_base}. Levantá la API con:\n\n`uv run python -m app`"
            )

    st.title("📋 Estimá un proyecto")
    st.caption(
        f"Describí el proyecto entre {DESCRIPTION_MIN_CHARS} y {DESCRIPTION_MAX_CHARS} caracteres."
    )

    # Una sola estimación en pantalla: cada resultado nuevo reemplaza al anterior
    # en vez de sumarse debajo. No se guarda historial porque no hay conversación.
    resultado = st.session_state.get("resultado")

    if resultado:
        # Los avisos van antes del texto: condicionan cómo leer la estimación.
        for aviso in resultado.get("avisos") or []:
            st.warning(aviso)
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
        texto, prompt_version, avisos = _estimar(api_base, payload)
    except _ApiError as exc:
        st.error(_detail_texto(exc.detail))
        return
    except httpx.HTTPError:
        st.error(f"No se pudo conectar a {api_base}. Levantá la API con:\n\n`uv run python -m app`")
        return

    st.session_state["resultado"] = {
        "text": texto,
        "prompt_version": prompt_version,
        "avisos": avisos,
    }
    st.rerun()


if __name__ == "__main__":
    main()
