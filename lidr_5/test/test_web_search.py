"""Tests de la búsqueda web como herramienta (Paso 11), todos con falsos: sin red ni claves.

QUÉ SE COMPRUEBA

- Cliente: límite de 3 resultados, recorte del fragmento, `scrapeOptions={}` solo en Firecrawl,
  respaldo ante error / resultados vacíos / tiempo agotado, y que la clave nunca salga en un error.
- Herramienta: formato del texto, tope de búsquedas por petición, mensaje cuando no hay servicio,
  y qué se registra de la consulta (solo el largo, o el texto truncado si se permite).
- Settings y registro: validaciones y registro condicional de `web_search`.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from structlog.testing import capture_logs

from app.config import Settings
from app.observability import start_turn_cost
from app.search.client import (
    MAX_RESULTS,
    SNIPPET_TRUNCATION_MARK,
    SearchUnavailableError,
    WebSearchClient,
)
from app.search.tool import (
    MAX_LOGGED_QUERY_CHARS,
    TOOL_NAME,
    build_tool_from_settings,
    build_web_search_tool,
)

CLAVE_TAVILY = "tvly-SECRETO-1"
CLAVE_FIRECRAWL = "fc-SECRETO-2"


def _respuesta(n: int = 3, snippet: str = "texto") -> SimpleNamespace:
    """Una respuesta falsa del buscador con `n` resultados."""
    return SimpleNamespace(
        results=[
            SimpleNamespace(title=f"T{i}", url=f"https://x/{i}", snippet=snippet) for i in range(n)
        ]
    )


class _Buscador:
    """Buscador falso: guarda las llamadas y responde según un guion por servicio."""

    def __init__(self, guion: dict) -> None:
        """`guion` mapea servicio → respuesta o excepción a lanzar."""
        self.guion = guion
        self.llamadas: list[dict] = []

    async def __call__(self, **kwargs):
        """Registra la llamada y devuelve (o lanza) lo que el guion indique."""
        self.llamadas.append(kwargs)
        accion = self.guion[kwargs["search_provider"]]
        if isinstance(accion, Exception):
            raise accion
        return accion


def _cliente(buscador, *, max_chars: int = 1500, claves=None) -> WebSearchClient:
    """Cliente de prueba con Tavily principal y Firecrawl de respaldo."""
    return WebSearchClient(
        keys=claves
        if claves is not None
        else {"tavily": CLAVE_TAVILY, "firecrawl": CLAVE_FIRECRAWL},
        primary="tavily",
        fallback="firecrawl",
        timeout_seconds=5,
        max_snippet_chars=max_chars,
        search_fn=buscador,
    )


# ----------------------------------------------------------------------------- cliente
async def test_devuelve_como_maximo_tres_resultados() -> None:
    """Aunque el servicio devuelva 5, solo llegan 3."""
    b = _Buscador({"tavily": _respuesta(5)})
    salida = await _cliente(b).search("hola")
    assert len(salida.results) == MAX_RESULTS == 3
    assert salida.provider == "tavily" and not salida.used_fallback
    assert b.llamadas[0]["max_results"] == 3


async def test_el_fragmento_se_recorta_con_marca() -> None:
    """Un fragmento largo se recorta al máximo, marca incluida."""
    b = _Buscador({"tavily": _respuesta(1, snippet="a" * 5000)})
    salida = await _cliente(b, max_chars=200).search("q")
    snippet = salida.results[0].snippet
    assert len(snippet) <= 200 and snippet.endswith(SNIPPET_TRUNCATION_MARK)


async def test_scrape_options_vacio_solo_en_firecrawl() -> None:
    """Firecrawl lleva `scrapeOptions={}`; Tavily no lo recibe."""
    b = _Buscador({"tavily": RuntimeError("x"), "firecrawl": _respuesta(1)})
    await _cliente(b).search("q")
    tavily, firecrawl = b.llamadas
    assert "scrapeOptions" not in tavily
    assert firecrawl["scrapeOptions"] == {}


@pytest.mark.parametrize(
    "fallo",
    [RuntimeError("caído"), TimeoutError("lento"), SimpleNamespace(results=[])],
    ids=["error", "timeout", "vacío"],
)
async def test_el_respaldo_toma_el_relevo(fallo) -> None:
    """Si el principal falla, da error de tiempo o no trae resultados, responde Firecrawl."""
    b = _Buscador({"tavily": fallo, "firecrawl": _respuesta(2)})
    salida = await _cliente(b).search("q")
    assert salida.provider == "firecrawl" and salida.used_fallback


async def test_si_fallan_los_dos_lanza_y_no_filtra_claves() -> None:
    """Con los dos servicios caídos se lanza `SearchUnavailableError` sin claves en el texto."""
    b = _Buscador(
        {
            "tavily": RuntimeError(f"401 con {CLAVE_TAVILY}"),
            "firecrawl": RuntimeError(f"402 con {CLAVE_FIRECRAWL}"),
        }
    )
    with capture_logs() as logs, pytest.raises(SearchUnavailableError) as info:
        await _cliente(b).search("q")
    visible = str(info.value) + str(logs)
    assert CLAVE_TAVILY not in visible and CLAVE_FIRECRAWL not in visible
    assert "***" in visible


async def test_solo_usa_servicios_con_clave() -> None:
    """Sin clave de Tavily, el principal se salta y responde Firecrawl sin contarse como relevo."""
    b = _Buscador({"firecrawl": _respuesta(1)})
    cliente = _cliente(b, claves={"tavily": "", "firecrawl": CLAVE_FIRECRAWL})
    assert cliente.available
    salida = await cliente.search("q")
    assert salida.provider == "firecrawl"
    assert [c["search_provider"] for c in b.llamadas] == ["firecrawl"]


async def test_sin_ninguna_clave_no_esta_disponible() -> None:
    """Sin claves el cliente no está disponible y buscar lanza."""
    cliente = _cliente(_Buscador({}), claves={"tavily": "", "firecrawl": ""})
    assert not cliente.available
    with pytest.raises(SearchUnavailableError):
        await cliente.search("q")


# --------------------------------------------------------------------------- herramienta
def _herramienta(buscador, *, tope: int = 2, log_queries: bool = False):
    """La herramienta `web_search` sobre un buscador falso."""
    return build_web_search_tool(
        _cliente(buscador), max_per_turn=tope, log_queries=log_queries, timeout_seconds=5
    )


async def test_el_resultado_es_texto_numerado() -> None:
    """La herramienta devuelve título, dirección y fragmento numerados."""
    texto = await _herramienta(_Buscador({"tavily": _respuesta(2)})).run({"query": "qué hay"})
    assert "1. T0" in texto and "https://x/1" in texto and "tavily" in texto


async def test_consulta_vacia_no_busca() -> None:
    """Sin `query` no se llama al buscador."""
    b = _Buscador({"tavily": _respuesta()})
    texto = await _herramienta(b).run({})
    assert "falta" in texto and not b.llamadas


async def test_el_tope_por_peticion_se_aplica() -> None:
    """La tercera búsqueda de una petición con tope 2 no llega al buscador."""
    start_turn_cost()
    b = _Buscador({"tavily": _respuesta()})
    herramienta = _herramienta(b, tope=2)
    await herramienta.run({"query": "uno"})
    await herramienta.run({"query": "dos"})
    texto = await herramienta.run({"query": "tres"})
    assert "límite de 2" in texto and len(b.llamadas) == 2


async def test_el_tope_se_cuenta_aunque_la_herramienta_corra_en_otra_tarea() -> None:
    """El bucle ejecuta las herramientas dentro de `wait_for` (otra tarea): el conteo se mantiene."""
    start_turn_cost()
    b = _Buscador({"tavily": _respuesta()})
    herramienta = _herramienta(b, tope=1)
    await asyncio.wait_for(herramienta.run({"query": "uno"}), timeout=5)
    texto = await asyncio.wait_for(herramienta.run({"query": "dos"}), timeout=5)
    assert "límite" in texto and len(b.llamadas) == 1


async def test_sin_servicio_devuelve_texto_y_no_lanza() -> None:
    """Con los dos servicios caídos el modelo recibe un aviso, no una excepción."""
    b = _Buscador({"tavily": RuntimeError("x"), "firecrawl": RuntimeError("y")})
    texto = await _herramienta(b).run({"query": "q"})
    assert "no disponible" in texto


async def test_por_defecto_solo_se_registra_el_largo_de_la_consulta() -> None:
    """Con `log_queries` apagado el evento lleva el largo y no el texto."""
    b = _Buscador({"tavily": _respuesta()})
    with capture_logs() as logs:
        await _herramienta(b).run({"query": "cliente secreto ACME"})
    evento = next(e for e in logs if e["event"] == "busqueda_web")
    assert evento["largo_consulta"] == len("cliente secreto ACME")
    assert "consulta" not in evento and "ACME" not in str(logs)
    assert evento["exito"] and evento["proveedor"] == "tavily" and "duracion_ms" in evento


async def test_con_log_queries_se_registra_la_consulta_truncada() -> None:
    """Con `log_queries` encendido el texto se registra recortado a 80 caracteres."""
    b = _Buscador({"tavily": _respuesta()})
    with capture_logs() as logs:
        await _herramienta(b, log_queries=True).run({"query": "x" * 500})
    evento = next(e for e in logs if e["event"] == "busqueda_web")
    assert len(evento["consulta"]) == MAX_LOGGED_QUERY_CHARS == 80


# ------------------------------------------------------------------ Settings y registro
def _settings(**extra) -> Settings:
    """Settings de prueba sin leer el `.env` real."""
    return Settings(_env_file=None, **extra)


def test_la_herramienta_no_se_crea_sin_claves() -> None:
    """Sin claves de búsqueda no hay herramienta."""
    assert build_tool_from_settings(_settings()) is None


def test_la_herramienta_se_crea_con_una_clave() -> None:
    """Con la clave de Tavily la herramienta existe y se llama `web_search`."""
    herramienta = build_tool_from_settings(_settings(web_tavily_key=CLAVE_TAVILY))
    assert herramienta is not None and herramienta.name == TOOL_NAME == "web_search"


def test_principal_y_respaldo_distintos_y_conocidos() -> None:
    """Servicios repetidos o desconocidos hacen fallar el arranque."""
    with pytest.raises(ValueError, match="distintos"):
        _settings(search_default="tavily", search_fallback="tavily")
    with pytest.raises(ValueError, match="no es válido"):
        _settings(search_default="exa")


@pytest.mark.parametrize(
    "campo", ["search_max_per_turn", "search_max_snippet_chars", "search_timeout_seconds"]
)
def test_limites_invalidos_fallan_al_arrancar(campo) -> None:
    """Un límite en cero (o demasiado bajo) es un error de configuración."""
    with pytest.raises(ValueError):
        _settings(**{campo: 0})


def test_valores_por_defecto_aprobados() -> None:
    """Los valores acordados: 2 búsquedas, 1.500 caracteres, 15 s, consultas sin registrar."""
    s = _settings()
    assert (s.search_max_per_turn, s.search_max_snippet_chars) == (2, 1500)
    assert s.search_timeout_seconds == 15 and s.search_log_queries is False


def _registro_con(monkeypatch, **valores):
    """Llama a `get_tool_registry` con Settings falsos y devuelve el registro resultante."""
    from types import SimpleNamespace as NS

    from app import dependencies

    monkeypatch.setattr(dependencies, "get_settings", lambda: _settings(**valores))
    peticion = NS(app=NS(state=NS()))
    return dependencies.get_tool_registry(peticion)


def test_el_registro_incluye_web_search_con_bucle_y_clave(monkeypatch) -> None:
    """Bucle encendido + clave de búsqueda → `web_search` registrada."""
    registro = _registro_con(monkeypatch, agent_loop_enabled=True, web_tavily_key=CLAVE_TAVILY)
    assert registro.names == ["web_search"]


def test_el_registro_queda_vacio_sin_bucle_o_sin_clave(monkeypatch) -> None:
    """Sin el bucle encendido, o sin claves, el registro sigue vacío."""
    assert len(_registro_con(monkeypatch, web_tavily_key=CLAVE_TAVILY)) == 0
    assert len(_registro_con(monkeypatch, agent_loop_enabled=True)) == 0
