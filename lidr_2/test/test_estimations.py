"""Tests del Estimador CAG.

El invariante de este ejercicio es uno solo: que los ejemplos de referencia
llegan dentro del system prompt. Eso es CAG. Si eso no ocurre, tenés un
wrapper de una API con pasos extra. Varios tests acá prueban exactamente eso;
el resto prueban que el servicio cuida el coste y no miente:
- no llama al LLM cuando la entrada no pasa la validación (no gastar tokens),
- un error del proveedor no llega entero al cliente (no filtrar secrets),
- una respuesta cortada se expone como `truncated: true` (no mentir con un 200).
"""
from __future__ import annotations

from conftest import make_settings
from fastapi.testclient import TestClient

from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES, build_system_prompt
from app.main import create_app
from app.providers import LLMProviderError, LLMResponse
from app.services import llm_service

TRANSCRIPCION = (
    "Reunión de planificación del sprint para el módulo de facturación. "
    "El cliente quiere alta de clientes, emisión de comprobantes y reporte "
    "de cobranzas. Se definieron los límites del MVP para esta iteración."
)
TRANSCRIPCION_CORTA = "reunión"


# ─── Coste estimado: tokens reales x LLMPrice, nunca se inventa ─────────

def test_coste_se_calcula_de_tokens_reales():
    from app.services.pricing import estimate_cost

    # 1M+1M de gpt-4o-mini: $0.15 + $0.60 (precios del snapshot de LLMPrice).
    cost, nota = estimate_cost(
        "gpt-4o-mini-2024-07-18",
        {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
    )
    assert cost == 0.75
    assert "LLMPrice" in nota
    # 1M+1M de claude-haiku-4-5: $1.00 + $5.00
    cost2, _ = estimate_cost(
        "claude-haiku-4-5-20251001",
        {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
    )
    assert cost2 == 6.0


def test_coste_es_none_si_no_se_puede_calcular_con_datos_reales():
    from app.services.pricing import estimate_cost

    # Modelo fuera del snapshot: coste ausente (mejor que inventar) y la nota
    # explica por qué.
    cost, nota = estimate_cost("fake-xyz", {"input_tokens": 10, "output_tokens": 5})
    assert cost is None
    assert "no tiene datos" in nota
    # Sin metadatos de tokens tampoco hay coste.
    cost2, _ = estimate_cost("gpt-4o-mini", None)
    assert cost2 is None


# ─── El invariante: CAG significa que el cache llega al system prompt ────

def test_los_ejemplos_llegan_al_system_prompt():
    prompt = build_system_prompt()
    for ejemplo in ESTIMATION_EXAMPLES:
        assert ejemplo.estimation in prompt


def test_cada_ejemplo_incluye_equipo_y_duracion_estimada():
    """El system prompt exige Equipo recomendado y Duración estimada; los
    ejemplos deben enseñar ese formato (en CAG los ejemplos SON el sistema)."""
    for ejemplo in ESTIMATION_EXAMPLES:
        assert "Equipo recomendado" in ejemplo.estimation
        assert "Duración estimada" in ejemplo.estimation


def test_los_ejemplos_son_aritmeticamente_coherentes():
    """En CAG los ejemplos son el sistema: un total que no suma su desglose
    le enseña al modelo a inventar totales."""
    import re

    for ejemplo in ESTIMATION_EXAMPLES:
        filas = re.findall(r"^\|\s+[^|]+\|\s+(\d+)\s*\|$", ejemplo.estimation, re.MULTILINE)
        total_match = re.search(r"\*\*(\d+) horas\*\*", ejemplo.estimation)
        assert filas, f"{ejemplo}: no hay tareas en el desglose"
        assert total_match, f"{ejemplo}: no hay total declarado"
        total = int(total_match.group(1))
        assert sum(int(h) for h in filas) == total, f"{ejemplo}: total no suma"


def test_los_roles_van_en_orden_y_la_transcripcion_es_dato(monkeypatch):
    capturado: dict = {}

    def fake_create(name, api_key, model, *, timeout, max_retries):
        class FakeProvider:
            def chat(self, messages, *, max_tokens=None, temperature=None):
                capturado["messages"] = messages
                return LLMResponse(content="", model="fake", truncated=False, usage=None)

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)
    llm_service.generate_estimation(
        TRANSCRIPCION, make_settings(openai_api_key="k")
    )

    mensajes = capturado["messages"]
    assert [m.role for m in mensajes] == ["system", "user"]
    # El cache de CAG vive en system; la transcripción, sin modificar, en user.
    assert ESTIMATION_EXAMPLES[0].estimation in mensajes[0].content
    assert TRANSCRIPCION in mensajes[1].content


def test_la_transcripcion_va_envuelta_en_delimitador_impredecible(monkeypatch):
    """La transcripción es DATOS: va entre etiquetas cuyo nombre solo se
    conoce en la llamada. Un usuario no puede cerrar una etiqueta que no
    conoce, así que un `</transcripcion-...>` malicioso no escapa del bloque."""
    capturado: dict = {}

    def fake_create(name, api_key, model, *, timeout, max_retries):
        class FakeProvider:
            def chat(self, messages, *, max_tokens=None, temperature=None):
                capturado["messages"] = messages
                return LLMResponse(content="", model="fake", truncated=False, usage=None)

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)
    llm_service.generate_estimation(
        TRANSCRIPCION, make_settings(openai_api_key="k")
    )

    user = capturado["messages"][1].content
    apertura = user.split("\n", 1)[0]
    cierre = user.rsplit("\n", 1)[1]
    contenido = user.split("\n", 1)[1].rsplit("\n", 1)[0]

    assert apertura.startswith("<transcripcion-") and apertura.endswith(">")
    assert cierre.startswith("</transcripcion-") and cierre.endswith(">")
    # El sufijo del cierre coincide con el de la apertura.
    assert apertura[1:-1].split("-", 1)[1] == cierre[2:-1].split("-", 1)[1]
    # Y ninguna etiqueta aparece dentro del bloque de datos.
    assert "transcripcion-" not in contenido


# ─── Robustez: no gastar plata ni mentir ────────────────────────────────

def test_entrada_corta_no_llama_al_llm(client, monkeypatch):
    def no_deberia_llamarse(*args, **kwargs):
        raise AssertionError("El LLM no debería invocarse con entrada inválida")

    monkeypatch.setattr(
        "app.routers.estimations.generate_estimation", no_deberia_llamarse
    )
    r = client.post("/api/v1/estimate", json={"transcription": TRANSCRIPCION_CORTA})
    assert r.status_code == 422
    assert "al menos" in r.text


def test_entrada_demasiado_larga_no_llama_al_llm(client, monkeypatch):
    def no_deberia_llamarse(*args, **kwargs):
        raise AssertionError("El LLM no debería invocarse con entrada inválida")

    monkeypatch.setattr(
        "app.routers.estimations.generate_estimation", no_deberia_llamarse
    )
    r = client.post(
        "/api/v1/estimate",
        json={"transcription": "a" * 60_000},
    )
    assert r.status_code == 422
    assert "no puede superar" in r.text


def test_truncado_se_expone_como_flag_no_como_200_completo(client, monkeypatch):
    """El SDK dice por qué paró. Si paró por max_tokens, la respuesta está
    incompleta y el ejercicio no miente: expone `truncated: true`."""

    def fake_create(name, api_key, model, *, timeout, max_retries):
        class FakeProvider:
            def chat(self, messages, *, max_tokens=None, temperature=None):
                return LLMResponse(
                    content="media tabla cortada",
                    model="fake",
                    truncated=True,
                    usage={"input_tokens": 100, "output_tokens": 10},
                )

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)
    r = client.post("/api/v1/estimate", json={"transcription": TRANSCRIPCION})
    assert r.status_code == 200
    assert r.json()["truncated"] is True


def test_error_del_proveedor_no_se_filtra_al_cliente(client, monkeypatch):
    """Un error del SDK arrastra fragmentos de key y URLs internas. El cliente
    solo debe ver un mensaje fijo; la traza se queda en el server."""

    def fake_create(name, api_key, model, *, timeout, max_retries):
        class FakeProvider:
            def chat(self, messages, *, max_tokens=None, temperature=None):
                raise LLMProviderError(
                    provider="openai",
                    detail="CLAVE_SECRETA_DE_PRUEBA_QUE_NO_DEBE_SALIR",
                    status_code=429,
                )

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)
    r = client.post("/api/v1/estimate", json={"transcription": TRANSCRIPCION})
    assert r.status_code == 502
    body = r.json()["detail"]
    assert "CLAVE_SECRETA_DE_PRUEBA" not in body
    assert "No se pudo generar la estimación" in body


# ─── El servicio arranca y /health sobrevive sin API key ────────────────

def test_health_responde_sin_key():
    settings = make_settings(openai_api_key=None, anthropic_api_key=None)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    # Sin el override, este test leería la OPENAI_API_KEY del shell real de la
    # máquina y fallaría en CI: la trampa exacta que describe la revisión.
    with TestClient(app) as c:
        r = c.get("/health")
    app.dependency_overrides.clear()
    assert r.status_code == 200
    assert r.json()["llm_configured"] is False


def test_el_modelo_se_resuelve_desde_el_default_del_proveedor():
    """LLM_MODEL vacío -> se usa OPENAI_MODEL|ANTHROPIC_MODEL, nunca el del
    proveedor contrario."""
    openai = make_settings(openai_api_key="k", openai_model="gpt-5-mini")
    assert openai.llm_model == "gpt-5-mini"

    anthropic = make_settings(
        llm_provider="anthropic",
        anthropic_api_key="k",
        anthropic_model="claude-4-5",
    )
    assert anthropic.llm_model == "claude-4-5"

    # El default sigue siendo el built-in si OPENAI_MODEL no viene.
    assert make_settings(openai_api_key="k").llm_model == "gpt-4o-mini"

    # Un `LLM_MODEL=` vacío en .env es "", no None (así lo parsea
    # pydantic-settings): también debe resolver al default, no quedarse vacío.
    assert make_settings(openai_api_key="k", llm_model="").llm_model == "gpt-4o-mini"

    # Y la incoherencia cruzada solo es posible eligiendo LLM_MODEL expreso.
    mezclado = make_settings(
        llm_provider="openai",
        openai_api_key="k",
        llm_model="claude-x",
    )
    assert mezclado.llm_model == "claude-x"


def test_campos_opcionales_vacios_se_normalizan_al_default():
    """Un .env copiado de .env.example trae APP_ENV= y LOG_LEVEL= vacíos;
    eso significa "no configurado", no debe romper ni quedar en vacío."""
    s = make_settings(app_env="", log_level="")
    assert s.app_env == "local"
    assert s.log_level == "INFO"


def test_el_puerto_default_es_8001_para_no_chocar_con_docker():
    """El 8000 lo suele tener tomado otro proceso (p. ej. Ganttly en Docker):
    el default del servicio es 8001, y se cambia con APP_PORT / --port."""
    assert make_settings().app_port == 8001
    assert make_settings(app_port=8010).app_port == 8010


def test_estimate_sin_key_devuelve_503():
    settings = make_settings(openai_api_key=None, anthropic_api_key=None)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as c:
        r = c.post("/api/v1/estimate", json={"transcription": TRANSCRIPCION})
    app.dependency_overrides.clear()
    assert r.status_code == 503
    assert "OPENAI_API_KEY" in r.json()["detail"]


def test_estimate_completo_con_fake(client, monkeypatch):
    def fake_create(name, api_key, model, *, timeout, max_retries):
        class FakeProvider:
            def chat(self, messages, *, max_tokens=None, temperature=None):
                return LLMResponse(
                    content="## Total\n**80 horas**\n",
                    model="fake",
                    truncated=False,
                    usage={"input_tokens": 50, "output_tokens": 20},
                )

        return FakeProvider()

    monkeypatch.setattr(llm_service, "create_provider", fake_create)
    r = client.post("/api/v1/estimate", json={"transcription": TRANSCRIPCION})
    assert r.status_code == 200
    data = r.json()
    assert data["estimation"].startswith("## Total")
    assert data["truncated"] is False
    assert data["model"] == "fake"
    # El fixture usa el proveedor por defecto: openai.
    assert data["provider"] == "openai"
    # El fake no está en el snapshot de LLMPrice: coste ausente y la nota lo
    # dice, en vez de inventar un número.
    assert data["cost_usd"] is None
    assert data["cost_note"].startswith("LLMPrice")