"""Comprueba con llamadas reales que las claves de búsqueda web funcionan.

Uso (desde la raíz del proyecto, con las claves en el .env):

    uv run python test/verificar_busqueda.py

Hace UNA búsqueda mínima (3 resultados) por servicio con clave, por la misma vía que
usaría el servicio (`litellm.search`). Cuesta unos pocos créditos por servicio. Las
claves se leen de WEB_TAVILY_KEY y WEB_FIRECRAWL_KEY (del entorno o del .env) y nunca
se imprimen: tampoco aparecen en los mensajes de error. Exa está descartada (su cuenta
tiene los créditos pendientes de verificación de pago).

Con `--tamano` mide además cuántos caracteres devuelve cada resultado (el campo que el
modelo recibiría como contexto) y cuánto tarda, con y sin límites, para Tavily y
Firecrawl. Hace 5 búsquedas más y los créditos de Firecrawl suben con el contenido de
cada página (la variante `summary` genera un resumen por página).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import litellm
from dotenv import dotenv_values

RAIZ = Path(__file__).resolve().parents[1]
CONSULTA = "FastAPI latest release version"

# (nombre mostrado, variable de la clave, proveedor de litellm.search)
SERVICIOS = [
    ("tavily", "WEB_TAVILY_KEY", "tavily"),
    ("firecrawl", "WEB_FIRECRAWL_KEY", "firecrawl"),
]


def leer_clave(variable: str, desde_archivo: dict[str, str | None]) -> str:
    """Busca la clave primero en el entorno del sistema y, si no está, en el `.env`.

    Devuelve texto vacío si no existe en ninguno de los dos sitios.
    """
    return (os.environ.get(variable) or desde_archivo.get(variable) or "").strip()


def probar(proveedor: str, clave: str) -> tuple[bool, str]:
    """Hace una búsqueda de prueba y devuelve `(funciona, detalle)`.

    `detalle` es un texto legible con el resultado y el tiempo que tardó. Nunca
    contiene la clave, ni siquiera dentro de un mensaje de error.
    """
    inicio = time.perf_counter()
    try:
        respuesta = litellm.search(
            query=CONSULTA,
            search_provider=proveedor,
            max_results=3,
            api_key=clave,
            timeout=30,
        )
    except Exception as e:  # noqa: BLE001 - se muestra el tipo y el motivo
        motivo = str(e).replace("\n", " ").replace(clave, "***")
        return False, f"RECHAZA ({type(e).__name__}): {motivo[:160]}"
    segundos = time.perf_counter() - inicio
    resultados = getattr(respuesta, "results", None) or []
    if not resultados:
        return (
            False,
            f"SIN RESULTADOS ({segundos:.1f} s): la clave respondió pero no hubo resultados",
        )
    titulo = (getattr(resultados[0], "title", "") or "").strip()[:70]
    return True, f"FUNCIONA ({segundos:.1f} s, {len(resultados)} resultados): «{titulo}»"


# (etiqueta, servicio, proveedor, parámetros extra de litellm.search)
VARIANTES_TAMANO = [
    ("tavily, por defecto", "WEB_TAVILY_KEY", "tavily", {}),
    ("firecrawl, por defecto", "WEB_FIRECRAWL_KEY", "firecrawl", {}),
    (
        "firecrawl + max_tokens_per_page=500",
        "WEB_FIRECRAWL_KEY",
        "firecrawl",
        {"max_tokens_per_page": 500},
    ),
    # LiteLLM solo agrega scrapeOptions (markdown de la página) si no lo pasamos.
    ("firecrawl + scrapeOptions vacío", "WEB_FIRECRAWL_KEY", "firecrawl", {"scrapeOptions": {}}),
    # No confirmado en la documentación de /search: puede rechazarlo o ignorarlo.
    (
        "firecrawl + formato summary",
        "WEB_FIRECRAWL_KEY",
        "firecrawl",
        {"scrapeOptions": {"formats": ["summary"], "onlyMainContent": True}},
    ),
]


def medir_tamano(archivo: dict[str, str | None]) -> None:
    """Imprime los caracteres del `snippet` de cada resultado, por variante."""
    print("\nTamaño del contenido por resultado (caracteres del snippet):")
    for etiqueta, variable, proveedor, extra in VARIANTES_TAMANO:
        clave = leer_clave(variable, archivo)
        if not clave:
            print(f"  {etiqueta:<38} SIN CLAVE: {variable}")
            continue
        inicio = time.perf_counter()
        try:
            respuesta = litellm.search(
                query=CONSULTA,
                search_provider=proveedor,
                max_results=3,
                api_key=clave,
                timeout=60,
                **extra,
            )
        except Exception as e:  # noqa: BLE001 - se muestra el tipo y el motivo
            motivo = str(e).replace("\n", " ").replace(clave, "***")
            print(f"  {etiqueta:<38} RECHAZA ({type(e).__name__}): {motivo[:120]}")
            continue
        segundos = time.perf_counter() - inicio
        largos = [len(getattr(r, "snippet", "") or "") for r in respuesta.results]
        print(f"  {etiqueta:<38} {largos} (suma {sum(largos)}, {segundos:.1f} s)")


def main() -> int:
    """Prueba cada servicio de búsqueda con clave y muestra si funciona.

    Con `--tamano` mide además el tamaño del contenido. Devuelve 1 si algún servicio
    falla, 0 si todo va bien.
    """
    archivo = dotenv_values(RAIZ / ".env")
    fallos = 0
    for nombre, variable, proveedor in SERVICIOS:
        clave = leer_clave(variable, archivo)
        if not clave:
            print(f"{nombre:<10} SIN CLAVE: {variable} está vacía o no existe")
            continue
        funciona, detalle = probar(proveedor, clave)
        fallos += not funciona
        print(f"{nombre:<10} {detalle}")
    default = leer_clave("SEARCH_DEFAULT", archivo)
    fallback = leer_clave("SEARCH_FALLBACK", archivo)
    if default and default == fallback:
        print("Aviso: SEARCH_DEFAULT y SEARCH_FALLBACK son iguales; tienen que ser distintos.")
    if "--tamano" in sys.argv[1:]:
        medir_tamano(archivo)
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
