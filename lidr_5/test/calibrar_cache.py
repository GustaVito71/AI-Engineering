"""Calibra el umbral de la caché semántica con el modelo de embeddings real.

NO es un test: llama a la API de embeddings (unos 50 textos cortos, céntimos de dólar),
no tiene aprobado ni fallo y pytest no lo recoge. Usa la clave y el modelo que lea
Settings del `.env` (`EMBEDDING_MODEL`, `OPENAI_API_KEY`).

Mide la similitud coseno de pares de descripciones escritos a mano:

- «debe acertar»: la misma idea con otras palabras, sin tildes o sin mayúsculas
  (devolver la estimación de una a la otra es lo que se busca);
- «NO debe acertar»: proyectos distintos, o el mismo proyecto con un matiz que cambia el
  alcance («sin pagos» / «con pagos»), o con cifras distintas.

Por cada umbral posible cuenta cuántos pares «debe acertar» se sirven y cuántos «NO debe
acertar» se servirían por error, aplicando también la guarda de las mismas cifras (la que
usa el servicio). Recomienda el umbral más bajo sin falsos aciertos y puede dejar un CSV.

Uso, desde la raíz de lidr_4:

    uv run python test/calibrar_cache.py
    uv run python test/calibrar_cache.py --csv similitudes.csv

Los pares son una muestra, no tráfico real: sirve para elegir un umbral de partida.
Después se afina con `SEMANTIC_CACHE_LOG_ONLY=true` y descripciones reales.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

# El script vive en test/: se agrega la raíz del proyecto para importar `app`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import LLMConfigurationError, get_settings, provider_from_model
from app.services.semantic_cache import LiteLLMEmbedder, extract_numbers

# (descripción A, descripción B, ¿debe acertar?, nota)
PAIRS: list[tuple[str, str, bool, str]] = [
    # --- Debe acertar: misma idea ---------------------------------------------------
    (
        "Plataforma web para que una empresa gestione el préstamo de equipos a sus empleados, con inventario y aprobaciones.",
        "Aplicación web donde una compañía administra los préstamos de equipos de sus empleados: inventario y flujo de aprobación.",
        True,
        "paráfrasis",
    ),
    (
        "App móvil de delivery de comida para un restaurante, con pedidos y pagos con tarjeta.",
        "Aplicación para celulares de pedidos de comida a domicilio de un restaurante, con pago con tarjeta.",
        True,
        "paráfrasis",
    ),
    (
        "Tienda online de ropa con catálogo, carrito y pasarela de pago.",
        "E-commerce de indumentaria con catálogo de productos, carrito de compras y pasarela de pagos.",
        True,
        "paráfrasis",
    ),
    (
        "Sistema de turnos para una clínica con recordatorios por correo.",
        "Sistema para reservar turnos en una clínica, con avisos por email.",
        True,
        "paráfrasis",
    ),
    (
        "API REST para gestionar usuarios, roles y permisos de una aplicación interna.",
        "API REST de gestión de usuarios, roles y permisos para una app interna.",
        True,
        "paráfrasis cercana",
    ),
    (
        "Panel de métricas para el equipo comercial con gráficos de ventas mensuales.",
        "Dashboard para el equipo de ventas con gráficos de las ventas por mes.",
        True,
        "paráfrasis",
    ),
    (
        "Sitio web institucional de 8 páginas para un estudio contable.",
        "Página web institucional de 8 secciones para un estudio contable.",
        True,
        "paráfrasis con cifra",
    ),
    (
        "Plataforma web SaaS B2B para gestionar el préstamo de equipos con aprobación y avisos por correo.",
        "plataforma web saas b2b para gestionar el prestamo de equipos con aprobacion y avisos por correo",
        True,
        "sin mayúsculas ni tildes",
    ),
    (
        "Aplicación de gestión de inventario para un almacén, con lectura de códigos de barras.",
        "Aplicación de gestion de inventario para un almacen con lector de codigos de barras.",
        True,
        "sin tildes",
    ),
    (
        "Portal de autogestión para clientes de una empresa de telefonía: facturas, consumos y reclamos.",
        "Portal de autogestión de clientes de una telefónica con facturas, consumos y reclamos.",
        True,
        "paráfrasis",
    ),
    # --- NO debe acertar: proyectos distintos ----------------------------------------
    (
        "Plataforma web para gestionar el préstamo de equipos a empleados.",
        "App móvil de delivery de comida para un restaurante.",
        False,
        "proyectos distintos",
    ),
    (
        "Tienda online de ropa con catálogo, carrito y pasarela de pago.",
        "Sistema de turnos para una clínica con recordatorios por correo.",
        False,
        "proyectos distintos",
    ),
    (
        "API REST para gestionar usuarios y roles.",
        "Panel de métricas con gráficos de ventas mensuales.",
        False,
        "proyectos distintos",
    ),
    (
        "Sitio web institucional para un estudio contable.",
        "Sitio web institucional para un club de fútbol.",
        False,
        "mismo tipo, otro cliente",
    ),
    # --- NO debe acertar: mismo proyecto, otro alcance --------------------------------
    (
        "Tienda online de ropa con catálogo y carrito de compras.",
        "Tienda online de ropa con catálogo, carrito de compras y pasarela de pago.",
        False,
        "con y sin pagos",
    ),
    (
        "App móvil de delivery para un restaurante, solo pedidos.",
        "App móvil de delivery para un restaurante, con pedidos, pagos, seguimiento en mapa y programa de puntos.",
        False,
        "alcance mucho mayor",
    ),
    (
        "Sistema de turnos para una clínica.",
        "Sistema de turnos para una clínica, con historia clínica electrónica e integración con obras sociales.",
        False,
        "alcance mucho mayor",
    ),
    (
        "Plataforma web para gestionar préstamos de equipos, solo para uso interno.",
        "Plataforma web para gestionar préstamos de equipos, multiempresa con facturación por suscripción.",
        False,
        "interno vs SaaS",
    ),
    (
        "API REST de usuarios con autenticación básica.",
        "API REST de usuarios con autenticación SSO, auditoría y MFA.",
        False,
        "seguridad distinta",
    ),
    (
        "Sitio web institucional estático.",
        "Sitio web institucional con CMS a medida y multiidioma.",
        False,
        "estático vs CMS",
    ),
    # --- NO debe acertar: mismas palabras, otras cifras --------------------------------
    (
        "Sitio web institucional de 8 páginas para un estudio contable.",
        "Sitio web institucional de 80 páginas para un estudio contable.",
        False,
        "cifra distinta",
    ),
    (
        "Plataforma para 20.000 usuarios simultáneos con inventario y aprobaciones.",
        "Plataforma para 200.000 usuarios simultáneos con inventario y aprobaciones.",
        False,
        "cifra distinta",
    ),
    (
        "App móvil con 5 pantallas para una veterinaria.",
        "App móvil con 25 pantallas para una veterinaria.",
        False,
        "cifra distinta",
    ),
]


@dataclass
class Result:
    """Resultado de comparar un par de descripciones: similitud, si tienen las mismas
    cifras, si debería acertar y el texto de ambas.
    """

    similarity: float
    same_numbers: bool
    should_hit: bool
    note: str
    a: str
    b: str


def cosine(a: list[float], b: list[float]) -> float:
    """Similitud coseno entre dos vectores: 1 si apuntan igual, 0 si no tienen relación."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def served(result: Result, threshold: float) -> bool:
    """Lo que haría el servicio: similitud suficiente Y las mismas cifras."""
    return result.similarity >= threshold and result.same_numbers


async def measure(model: str, api_key: str, timeout: float) -> list[Result]:
    """Pide al proveedor el vector de cada texto de los pares de prueba y calcula la
    similitud de cada par.
    """
    embedder = LiteLLMEmbedder(api_key=api_key, model=model, timeout=timeout)
    texts = sorted({t for a, b, _, _ in PAIRS for t in (a, b)})
    vectors = dict(
        zip(texts, await asyncio.gather(*(embedder.embed(t) for t in texts)), strict=True)
    )
    return [
        Result(
            similarity=cosine(vectors[a], vectors[b]),
            same_numbers=extract_numbers(a) == extract_numbers(b),
            should_hit=should_hit,
            note=note,
            a=a,
            b=b,
        )
        for a, b, should_hit, note in PAIRS
    ]


def print_pairs(results: list[Result]) -> None:
    """Muestra los pares ordenados de mayor a menor similitud, con si debían acertar o no."""
    print("\nSimilitud por par (de mayor a menor):\n")
    print(f"{'sim':>6}  {'cifras':<9} {'esperado':<13} nota")
    for r in sorted(results, key=lambda r: -r.similarity):
        cifras = "iguales" if r.same_numbers else "DISTINTAS"
        esperado = "debe acertar" if r.should_hit else "NO debe"
        print(f"{r.similarity:6.3f}  {cifras:<9} {esperado:<13} {r.note}")


def print_sweep(results: list[Result]) -> float | None:
    """Prueba umbrales de 0,70 a 0,99, muestra aciertos y falsos aciertos de cada uno y
    devuelve el más bajo sin falsos aciertos.
    """
    positives = [r for r in results if r.should_hit]
    negatives = [r for r in results if not r.should_hit]
    print(
        f"\nBarrido de umbrales ({len(positives)} pares «debe acertar», "
        f"{len(negatives)} «NO debe»):\n"
    )
    print(f"{'umbral':>7}  {'aciertos servidos':>18}  {'falsos aciertos':>16}")
    recommended: float | None = None
    for step in range(70, 100):
        threshold = step / 100
        hits = sum(served(r, threshold) for r in positives)
        false = sum(served(r, threshold) for r in negatives)
        print(f"{threshold:7.2f}  {hits:>10}/{len(positives):<7}  {false:>16}")
        if false == 0 and recommended is None:
            recommended = threshold
    return recommended


def write_csv(results: list[Result], path: Path) -> None:
    """Guarda en un archivo CSV la similitud de cada par."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["similitud", "mismas_cifras", "debe_acertar", "nota", "a", "b"])
        for r in sorted(results, key=lambda r: -r.similarity):
            writer.writerow([f"{r.similarity:.4f}", r.same_numbers, r.should_hit, r.note, r.a, r.b])


def main() -> int:
    """Ejecuta la calibración: mide los pares con el modelo de embeddings configurado,
    muestra el barrido de umbrales y recomienda uno.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", type=Path, help="Guarda las similitudes de cada par.")
    args = parser.parse_args()

    settings = get_settings()
    api_key = settings.active_api_key(provider_from_model(settings.embedding_model))
    if api_key is None:
        raise LLMConfigurationError(
            f"Falta la clave del proveedor de {settings.embedding_model} en el .env."
        )
    print(f"Modelo de embeddings: {settings.embedding_model}")
    print(f"Umbral actual (CACHE_SEMANTIC_THRESHOLD): {settings.cache_semantic_threshold}")

    results = asyncio.run(measure(settings.embedding_model, api_key, settings.embedding_timeout))
    print_pairs(results)
    recommended = print_sweep(results)
    if args.csv:
        write_csv(results, args.csv)
        print(f"\nSimilitudes en {args.csv}")

    positives = [r for r in results if r.should_hit]
    if recommended is None:
        print("\nNingún umbral entre 0.70 y 0.99 evita todos los falsos aciertos.")
    else:
        hits = sum(served(r, recommended) for r in positives)
        print(
            f"\nUmbral más bajo sin falsos aciertos: {recommended:.2f} "
            f"(sirve {hits} de {len(positives)} paráfrasis)."
        )
        print("Es el piso con esta muestra: elige unas centésimas por encima, por prudencia.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
