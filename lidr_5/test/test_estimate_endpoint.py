"""Tests del endpoint POST /api/v1/estimate y de /health: respuesta normal, avisos y errores.

Sin red ni Redis: REDIS_URL vacío desactiva la caché. Los casos de 503 usan el
wrapper real (sin la clave que falta ni siquiera llega a construirse); los de
200, 502 y 504 reemplazan el wrapper con `dependency_overrides`.

El `.env` de quien corre los tests no interviene (ver `entorno_aislado` en
conftest.py), y los tests de claves corren con cada proveedor como primario
(fixture `modelos`).
"""

from __future__ import annotations

import httpx
import litellm
import pytest
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from app.config import get_settings
from app.dependencies import get_llm_wrapper
from app.main import create_app
from app.routers.estimations import PROVIDER_FAILURE_MESSAGE, TIMEOUT_MESSAGE
from app.services.llm_wrapper import LLMCallResult

BODY = {
    "description": "A small B2B SaaS to manage employee equipment loans.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}

# Texto que un proveedor podría devolver y que nunca debe llegar al cliente.
DETALLE_INTERNO = "org-ACME-1234: upstream said no (request_id=abc)"


@pytest.fixture
def cliente(monkeypatch):
    """App real con el entorno que fije cada test; sin claves ni `.env` de partida."""

    def _crear(**entorno: str) -> TestClient:
        """Crea un cliente HTTP de prueba con la app completa, sin Redis; los parámetros
        añaden variables de entorno.
        """
        monkeypatch.setenv("REDIS_URL", "")
        for variable, valor in entorno.items():
            monkeypatch.setenv(variable, valor)
        get_settings.cache_clear()
        return TestClient(create_app())

    yield _crear
    get_settings.cache_clear()


class _WrapperFalso:
    """Reemplaza al LLMWrapper: devuelve un resultado fijo o lanza `error`."""

    primary_model = "openai/gpt-4o-mini"

    def __init__(
        self,
        error: Exception | None = None,
        warnings: tuple[str, ...] = (),
        contenido: str = "| phase | duration_weeks | cost_eur | confidence_pct |",
    ) -> None:
        self._error = error
        self.warnings = warnings
        self._contenido = contenido
        self.llamadas: list[dict] = []

    async def estimate(self, *, prompt_version: str, **kwargs) -> LLMCallResult:
        """Simula la llamada al modelo: guarda los argumentos recibidos y devuelve el texto
        preparado (o lanza el error preparado).
        """
        self.llamadas.append({"prompt_version": prompt_version, **kwargs})
        if self._error is not None:
            raise self._error
        return LLMCallResult(
            content=self._contenido,
            model="openai/gpt-4o-mini",
            provider="openai",
            prompt_tokens=10,
            completion_tokens=20,
            cost_usd=0.001,
            prompt_version=prompt_version,
        )


def _con_wrapper(cliente, wrapper: _WrapperFalso) -> TestClient:
    """Crea un cliente de prueba cuya app usa el modelo falso en vez del real."""
    c = cliente()
    c.app.dependency_overrides[get_llm_wrapper] = lambda: wrapper
    return c


def _error_de_conexion() -> litellm.APIConnectionError:
    """Crea un error de conexión del proveedor, con un detalle interno que no debe llegar
    al cliente.
    """
    return litellm.APIConnectionError(
        message=DETALLE_INTERNO, llm_provider="openai", model="gpt-4o-mini"
    )


def _timeout() -> litellm.Timeout:
    """Crea un error de tiempo de espera agotado del proveedor."""
    return litellm.Timeout(message=DETALLE_INTERNO, model="gpt-4o-mini", llm_provider="openai")


def _error_de_estado() -> litellm.RateLimitError:
    """Crea un error 429 (demasiadas peticiones) del proveedor."""
    return litellm.RateLimitError(
        message=DETALLE_INTERNO,
        llm_provider="openai",
        model="gpt-4o-mini",
        response=httpx.Response(429, request=httpx.Request("POST", "https://x")),
    )


# --- Camino normal --------------------------------------------------------------


def test_returns_200_with_text_and_prompt_version(cliente) -> None:
    """Una petición válida devuelve 200 con el texto, la versión de prompt y la lista de
    avisos.
    """
    with _con_wrapper(cliente, _WrapperFalso()) as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert r.status_code == 200
    assert r.json() == {
        "text": "| phase | duration_weeks | cost_eur | confidence_pct |",
        "prompt_version": "v2",
        "warnings": [],
    }


@pytest.mark.parametrize(
    "contenido",
    [
        "Totales: 500 horas.\nEquipo: 1 Desarrollador.\n</estimation>",
        "<estimation>\nTotales: 500 horas.\nEquipo: 1 Desarrollador.\n</estimation>",
        "<estimation>Totales: 500 horas.\nEquipo: 1 Desarrollador.</estimation>",
    ],
    ids=["solo-cierre", "apertura-y-cierre", "en-la-misma-linea"],
)
def test_quita_las_etiquetas_del_prompt_que_copia_el_modelo(cliente, contenido) -> None:
    """El modelo a veces copia el <estimation> que envuelve los ejemplos del prompt."""
    with _con_wrapper(cliente, _WrapperFalso(contenido=contenido)) as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert r.json()["text"] == "Totales: 500 horas.\nEquipo: 1 Desarrollador."


def test_texto_sin_etiquetas_llega_igual(cliente) -> None:
    """Si el modelo responde sin etiquetas de estructura, el texto llega al cliente tal
    cual.
    """
    texto = "| Fase | Semanas |\n|---|---|\n| QA | 1 |"
    with _con_wrapper(cliente, _WrapperFalso(contenido=texto)) as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert r.json()["text"] == texto


# --- Versión del prompt por query ---------------------------------------------------


def test_sin_query_usa_la_version_configurada(cliente) -> None:
    """Sin `prompt_version` en la petición se usa la versión configurada (v2)."""
    wrapper = _WrapperFalso()
    with _con_wrapper(cliente, wrapper) as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert r.json()["prompt_version"] == "v2"
    assert "Responde siempre en castellano" in wrapper.llamadas[0]["system_prompt"]


def test_la_query_elige_otra_version(cliente) -> None:
    """Con ?prompt_version=v1 se renderiza el prompt de v1 y la respuesta lo dice."""
    wrapper = _WrapperFalso()
    with _con_wrapper(cliente, wrapper) as c:
        r = c.post("/api/v1/estimate?prompt_version=v1", json=BODY)
    assert r.status_code == 200
    assert r.json()["prompt_version"] == "v1"
    [llamada] = wrapper.llamadas
    assert llamada["prompt_version"] == "v1"
    assert "Always answer in English" in llamada["system_prompt"]


@pytest.mark.parametrize(
    ("query", "esperada"),
    [("", "v1"), ("?prompt_version=v2", "v2")],
    ids=["sin-query-usa-PROMPT_VERSION", "la-query-gana"],
)
def test_query_y_configuracion(cliente, query, esperada) -> None:
    """Con PROMPT_VERSION=v1 configurado: sin query se usa v1; la query lo pisa."""
    c = cliente(PROMPT_VERSION="v1")
    c.app.dependency_overrides[get_llm_wrapper] = lambda: _WrapperFalso()
    with c:
        r = c.post(f"/api/v1/estimate{query}", json=BODY)
    assert r.json()["prompt_version"] == esperada


@pytest.mark.parametrize("pedida", ["v9", "../v1", "v1/../v2", "V1", ""])
def test_una_version_que_no_existe_es_422(cliente, pedida) -> None:
    """Pedir una versión de prompt que no existe devuelve 422 con las versiones
    disponibles, sin llamar al modelo.
    """
    wrapper = _WrapperFalso()
    with _con_wrapper(cliente, wrapper) as c:
        r = c.post("/api/v1/estimate", params={"prompt_version": pedida}, json=BODY)
    assert r.status_code == 422
    assert r.json() == {
        "detail": f"La versión de prompt '{pedida}' no existe. Versiones disponibles: v1, v2."
    }
    assert wrapper.llamadas == []


def test_health_lista_las_versiones_disponibles(cliente) -> None:
    """`/health` informa la versión de prompt por defecto y las disponibles."""
    with cliente() as c:
        h = c.get("/health").json()
    assert h["prompt_version"] == "v2"
    assert h["prompt_versions"] == ["v1", "v2"]


def test_versiones_disponibles_se_ordenan_por_numero(tmp_path, monkeypatch) -> None:
    """Las versiones se ordenan por número (v2 antes que v10) y solo cuentan las que tienen
    su `system.j2`.
    """
    from app.prompts import loader

    for nombre in ("v10", "v2", "v1", "borrador", "v3"):
        (tmp_path / "estimation" / nombre).mkdir(parents=True)
    for nombre in ("v10", "v2", "v1", "borrador"):  # v3 sin system.j2: no está publicada
        (tmp_path / "estimation" / nombre / "system.j2").write_text("x", encoding="utf-8")
    monkeypatch.setattr(loader, "_BASE_DIR", tmp_path)

    assert loader.available_versions() == ["v1", "v2", "v10"]


# --- Falta configuración: 503 ---------------------------------------------------


def _entorno(modelos, **claves_por_rol: str) -> dict[str, str]:
    """Modelos fijados más las claves pedidas por rol ("primario"/"respaldo")."""
    (primario, var_primario, _), (respaldo, var_respaldo, _) = modelos
    variables = {"primario": var_primario, "respaldo": var_respaldo}
    prov_p, modelo_p = primario.split("/", 1)
    prov_r, modelo_r = respaldo.split("/", 1)
    entorno = {
        "LLM_DEFAULT": prov_p,
        f"{prov_p.upper()}_MODEL": modelo_p,
        "LLM_FALLBACK": prov_r,
        f"{prov_r.upper()}_MODEL": modelo_r,
    }
    entorno.update({variables[rol]: valor for rol, valor in claves_por_rol.items()})
    return entorno


def test_missing_primary_key_returns_503_naming_the_variable(cliente, modelos) -> None:
    """Sin la clave del modelo principal, la API responde 503 y nombra la variable que
    falta.
    """
    (_, var_primario, _), _ = modelos
    with cliente(**_entorno(modelos, respaldo="sk-respaldo")) as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert r.status_code == 503
    assert f"Falta {var_primario}" in r.json()["detail"]


def test_503_never_leaks_the_configured_key(cliente, modelos) -> None:
    """El mensaje de error 503 nunca incluye el valor de una clave configurada."""
    with cliente(**_entorno(modelos, respaldo="sk-respaldo-secreta")) as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert r.status_code == 503
    assert "sk-respaldo-secreta" not in r.text


def test_missing_fallback_key_still_estimates_and_warns(cliente, monkeypatch, modelos) -> None:
    """Sin la clave del respaldo la estimación sale con el primario, y la
    respuesta trae el aviso para el usuario."""
    _, (_, var_respaldo, _) = modelos

    async def sin_red(self, **kwargs):
        """Sustituye la llamada al modelo por una respuesta simulada, sin usar la red."""
        return await litellm.acompletion(
            model=kwargs["model"], messages=kwargs["messages"], mock_response="estimación"
        )

    monkeypatch.setattr(litellm.Router, "acompletion", sin_red)
    with cliente(**_entorno(modelos, primario="sk-primario-secreta")) as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["text"] == "estimación"
    [aviso] = cuerpo["warnings"]
    assert var_respaldo in aviso
    assert "sk-primario-secreta" not in r.text


def test_avisos_reach_the_response(cliente) -> None:
    """Los avisos generados por el servicio llegan al cliente en la respuesta."""
    with _con_wrapper(cliente, _WrapperFalso(warnings=("aviso de prueba",))) as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert r.json()["warnings"] == ["aviso de prueba"]


def test_health_reports_missing_fallback(cliente, modelos) -> None:
    """`/health` avisa de que falta la clave del modelo de respaldo."""
    _, (_, var_respaldo, _) = modelos
    with cliente(**_entorno(modelos, primario="sk-primario")) as c:
        h = c.get("/health").json()
    assert h["llm_configured"] is True
    assert h["fallback_configured"] is False
    [aviso] = h["warnings"]
    assert var_respaldo in aviso


def test_health_reports_missing_primary(cliente, modelos) -> None:
    """`/health` indica que el modelo principal no está configurado cuando falta su clave."""
    with cliente(**_entorno(modelos, respaldo="sk-respaldo")) as c:
        h = c.get("/health").json()
    assert h["llm_configured"] is False


def test_health_without_notices_when_both_keys_are_set(cliente, modelos) -> None:
    """Con ambas claves configuradas, `/health` no trae avisos."""
    with cliente(**_entorno(modelos, primario="sk-p", respaldo="sk-r")) as c:
        h = c.get("/health").json()
    assert h["llm_configured"] is True
    assert h["fallback_configured"] is True
    assert h["warnings"] == []


def test_health_still_answers_without_keys(cliente) -> None:
    """El servicio arranca sin claves y /health lo informa (no hay crashloop)."""
    with cliente() as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["llm_configured"] is False


# --- Falla el proveedor: 502 y 504 con mensaje limpio ----------------------------


@pytest.mark.parametrize(
    ("crear_error", "codigo", "mensaje"),
    [
        (_error_de_conexion, 502, PROVIDER_FAILURE_MESSAGE),
        (_error_de_estado, 502, PROVIDER_FAILURE_MESSAGE),
        (lambda: RuntimeError(DETALLE_INTERNO), 502, PROVIDER_FAILURE_MESSAGE),
        (_timeout, 504, TIMEOUT_MESSAGE),
    ],
    ids=["conexion", "rate-limit", "inesperado", "timeout"],
)
def test_provider_failure_returns_clean_message(cliente, crear_error, codigo, mensaje) -> None:
    """Cada fallo del proveedor devuelve su código HTTP y un mensaje limpio, sin el detalle
    interno.
    """
    with _con_wrapper(cliente, _WrapperFalso(crear_error())) as c:
        r = c.post("/api/v1/estimate", json=BODY)
    assert r.status_code == codigo
    assert r.json() == {"detail": mensaje}
    assert DETALLE_INTERNO not in r.text


def test_provider_failure_detail_goes_to_the_log(cliente) -> None:
    """Lo que el cliente no ve tiene que quedar en el log para poder diagnosticar."""
    with _con_wrapper(cliente, _WrapperFalso(_error_de_conexion())) as c, capture_logs() as logs:
        c.post("/api/v1/estimate", json=BODY)
    fallos = [e for e in logs if e["event"] == "estimacion_fallida"]
    assert len(fallos) == 1
    assert fallos[0]["log_level"] == "error"
    assert fallos[0]["codigo_http"] == 502
    assert fallos[0]["tipo_error"] == "APIConnectionError"
    assert DETALLE_INTERNO in fallos[0]["detalle"]


# --- Request validation -----------------------------------------------------


def test_una_descripcion_en_blanco_es_422_sin_llamar_al_modelo(cliente) -> None:
    """Una descripción en blanco devuelve 422 y no llega al modelo."""
    wrapper = _WrapperFalso()
    with cliente() as c:
        c.app.dependency_overrides[get_llm_wrapper] = lambda: wrapper
        r = c.post("/api/v1/estimate", json={**BODY, "description": " \n\t" * 10})

    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "description"]
    assert wrapper.llamadas == []


def test_el_prompt_recibe_la_descripcion_recortada(cliente) -> None:
    """El prompt recibe la descripción sin los espacios y saltos de línea de los extremos."""
    wrapper = _WrapperFalso()
    with cliente() as c:
        c.app.dependency_overrides[get_llm_wrapper] = lambda: wrapper
        r = c.post(
            "/api/v1/estimate", json={**BODY, "description": f"\n  {BODY['description']}  \n"}
        )

    assert r.status_code == 200
    [llamada] = wrapper.llamadas
    assert (
        f"<project_description>\n{BODY['description']}\n</project_description>"
        in (llamada["user_prompt"])
    )


@pytest.mark.parametrize(
    "ruta", ["/api/v1/estimate", "/api/v1/estimate/structured", "/api/v1/estimate/rendered"]
)
def test_un_campo_fuera_del_contrato_es_422_en_los_tres_endpoints(cliente, ruta) -> None:
    """Un campo que no está en el contrato devuelve 422 en los tres endpoints de
    estimación.
    """
    wrapper = _WrapperFalso()
    with cliente() as c:
        c.app.dependency_overrides[get_llm_wrapper] = lambda: wrapper
        r = c.post(ruta, json={**BODY, "prompt_version": "v1"})

    assert r.status_code == 422
    [detalle] = r.json()["detail"]
    assert detalle["type"] == "extra_forbidden"
    assert detalle["loc"] == ["body", "prompt_version"]
    assert wrapper.llamadas == []
