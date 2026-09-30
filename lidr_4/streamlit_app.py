"""Streamlit frontend for the Estimador CAG service (lidr_3 deliverable).

The UI is a thin, single-turn client: paste a meeting transcription, get a
live streaming estimate. It never touches a provider API key; the backend
reads its own keys from Settings (.env). The frontend only needs the base URL
of the running API (`ESTIMATOR_API_BASE_URL`, default http://localhost:8001).

Why single-turn: the model is stateless per request (system prompt CAG + the
transcription as data). There is no conversation to remember, so keeping a
multi-turn history on screen would imply continuity that does not exist. Each
new transcription replaces the previous turn on purpose.

Why the streaming loop is a pure generator: `_parsear_sse` and
`_stream_deltas` are testable without a running Streamlit runtime, and the
contract test in test/test_frontend.py feeds them the EXACT lines the router
produces (`_evento_sse`) plus a mocked HTTP transport. The Streamlit calls
live only inside `main()`, which Streamlit runs as __main__; importing the
module from pytest defines the pure functions and does nothing else.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st
import json

from collections.abc import Iterable, Iterator




DEFAULT_API_BASE = os.environ.get("ESTIMATOR_API_BASE_URL", "http://localhost:8001")


class _ApiError(Exception):
    """The backend answered with a non-200 status or an SSE `error` event."""

    def __init__(self, status: int, detail: object) -> None:
        super().__init__(str(detail))
        self.status = status
        self.detail = detail


def _parsear_sse(lineas: Iterable[str]) -> Iterator[tuple[str, dict]]:
    """Parse `event:`/`data:` lines (the format `_evento_sse` emits) into
    (tipo, datos) pairs. Blank lines are separators and are ignored; the
    `event` line always precedes its `data` line, so one slot is enough."""
    evento: str | None = None
    for linea in lineas:
        if not linea:
            continue
        if linea.startswith("event: "):
            evento = linea[len("event: ") :]
        elif linea.startswith("data: "):
            if evento is None:
                raise _ApiError(502, "SSE malformado: data sin event previo")
            datos = json.loads(linea[len("data: ") :])
            yield evento, datos
            evento = None


def _detail_texto(detail: object) -> str:
    """FastAPI validation errors arrive as a list of `{msg, ...}` items."""
    if isinstance(detail, list):
        return "\n".join(str(item.get("msg", item)) for item in detail)
    return str(detail)


def _stream_deltas(
    api_base: str,
    transcription: str,
    captura: dict,
    client: httpx.Client | None = None,
) -> Iterator[str]:
    """Consume `POST /api/v1/estimate/stream` and yield each delta as text.

    Metadata captured for the final summary lives in the mutable `captura`
    dict (closed over by the caller): `meta` (provider/route, only on a real
    miss) and `final` (the complete EstimationResult). A failure before the
    first byte raises `_ApiError` with the real HTTP status; a mid-stream
    failure arrives as an SSE `error` event and is raised the same way after
    whatever deltas were already shown. `client` is injected by tests
    (httpx.MockTransport); the UI always lets the function create its own."""
    propio = client is None
    cliente = client if client is not None else httpx.Client()
    try:
        with cliente.stream(
            "POST",
            f"{api_base}/api/v1/estimate/stream",
            json={"transcription": transcription},
            timeout=None,  # SSE: the backend stream can pause between deltas.
        ) as respuesta:
            if respuesta.status_code != 200:
                # En modo stream el body NO se lee solo: acceder a .json()/.text
                # sin read() lanza httpx.ResponseNotRead (un RuntimeError que
                # ni siquiera hereda de HTTPError; lo destapó el repaso manual,
                # no MockTransport, porque el mock entrega el body ya leído).
                cuerpo = respuesta.read()
                try:
                    detail = respuesta.json().get("detail", cuerpo.decode("utf-8", "replace"))
                except ValueError:
                    detail = cuerpo.decode("utf-8", "replace")
                raise _ApiError(respuesta.status_code, detail)
            for tipo, datos in _parsear_sse(respuesta.iter_lines()):
                if tipo == "delta":
                    yield datos["texto"]
                elif tipo == "meta":
                    captura["meta"] = datos
                elif tipo == "estimation":
                    captura["final"] = datos
                elif tipo == "error":
                    raise _ApiError(502, datos.get("detail", "No se pudo completar la estimación."))
    finally:
        if propio:
            cliente.close()


def _resumen(captura: dict) -> str:
    """One caption line with the tracing metadata the backend exposes."""
    piezas: list[str] = []
    meta = captura.get("meta")
    if meta:
        piezas.append(f"Camino: {meta.get('camino', '—')}")
        piezas.append(f"Proveedor: {meta.get('proveedor', '—')}")
    final = captura.get("final")
    if final:
        piezas.append(f"Modelo: {final.get('model', '—')}")
        costo = final.get("cost_usd")
        piezas.append(f"Costo: ${costo:.6f}" if costo is not None else "Costo: —")
        piezas.append("Truncado: sí" if final.get("truncated") else "Truncado: no")
    return " · ".join(piezas) if piezas else ""


def main() -> None:
    st.set_page_config(page_title="Estimador CAG", page_icon="📋", layout="centered")

    st.sidebar.title("Estimador CAG")
    st.sidebar.caption(
        "Cliente del servicio `estimador-cag`. Las API keys viven "
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

    st.title("📋 Estimá la duración del proyecto")
    st.caption(
        "Pegá la transcripción de la reunión; cada nueva transcripción "
        "reemplaza la anterior (turno único)."
    )

    # Single-turn: la conversación siempre tiene exactamente UN turno
    # (user + assistant). Enviar una transcripción nueva lo reemplaza.
    conversacion = st.session_state.get("conversacion")

    if conversacion:
        with st.chat_message("user"):
            st.write(conversacion["transcripcion"])
        with st.chat_message("assistant"):
            st.write(conversacion["texto"])
            resumen = _resumen(conversacion["captura"])
            if resumen:
                st.caption(resumen)
        if st.button("🧹 Nueva transcripción"):
            st.session_state.pop("conversacion", None)
            st.rerun()

    with st.form("transcripcion", clear_on_submit=False):
        transcripcion = st.text_area(
            "Transcripción de la reunión",
            height=180,
            placeholder="Pegá acá la transcripción (mínimo 50 caracteres)...",
            key="transcripcion_entrada",
        )
        enviar = st.form_submit_button("Estimar duración", type="primary")

    if enviar and transcripcion.strip():
        # Single-turn: cualquier turno anterior desaparece ante la nueva entrada.
        st.session_state.pop("conversacion", None)
        captura: dict = {}
        with st.chat_message("user"):
            st.write(transcripcion.strip())
        with st.chat_message("assistant"):
            try:
                texto = st.write_stream(_stream_deltas(api_base, transcripcion.strip(), captura))
            except _ApiError as exc:
                st.error(_detail_texto(exc.detail))
                st.stop()
            except httpx.HTTPError:
                st.error(
                    f"No se pudo conectar a {api_base}. Levantá la API con:\n\n"
                    "`uv run python -m app`"
                )
                st.stop()
            resumen = _resumen(captura)
            if resumen:
                st.caption(resumen)
        st.session_state["conversacion"] = {
            "transcripcion": transcripcion.strip(),
            "texto": texto,
            "captura": captura,
        }
        st.rerun()
    elif enviar:
        st.info("La transcripción no puede estar vacía.")


if __name__ == "__main__":
    main()
