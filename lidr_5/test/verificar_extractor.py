"""Comprueba con llamadas reales qué guarda el extractor de `project_metadata` entre dos turnos.

Uso (desde la raíz del proyecto, con las claves del extractor en el .env):

    uv run python test/verificar_extractor.py                       # primario del .env
    uv run python test/verificar_extractor.py --proveedor deepseek  # fuerza otro proveedor

Prueba UN solo proveedor y sin respaldo a propósito: con el respaldo activo, si el
proveedor probado falla, el Router responde con el otro en silencio y la prueba parecería
buena sin haberlo probado (así pasó la primera vez con DeepSeek). Sin respaldo, un fallo
se ve con su motivo.

No levanta el servidor: llama directamente a `MetadataExtractor`, con la misma
configuración que usa el servicio. Simula dos turnos de una conversación:

1. el usuario presenta el proyecto (nombre, equipo, tecnología, presupuesto y un dato
   suelto: quién es el cliente, que debe ir a `other_facts`);
2. el usuario cambia de opinión («ya no Rails, vamos con Node»).

Después del segundo turno comprueba, hecho por hecho, que la memoria conservó lo que
no cambió y movió Rails a «descartadas». El log del servicio solo cuenta hechos; este
script muestra cuáles son, así que sirve para detectar si el extractor perdió alguno.

Hace 2 llamadas cortas por proveedor (menos de un centavo). Imprime la metadata de
la conversación de ejemplo, nunca las claves: lee el .env a través de Settings.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import EXTRACTOR_KEY_VARIABLES, LLMConfigurationError, get_settings
from app.services.metadata_extractor import MetadataExtractor
from app.sessions import ProjectMetadata

# Los dos turnos de la conversación de ejemplo: lo que dice el usuario y un resumen de
# lo que respondería el asistente (el extractor recibe ambos).
TURNO_1 = (
    "Queremos Bookflow, un sistema de reservas con Rails y un equipo de tres personas. "
    "El presupuesto máximo es de 40.000 EUR. El cliente es una cadena de gimnasios."
)
RESPUESTA_1 = "Estimación de prueba. Equipo: 3 Desarrollador. Totales: 390 h, 23950 EUR, 6 semanas."
TURNO_2 = "Cambiamos de idea: ya no usaremos Rails, vamos con Node en todo el backend."
RESPUESTA_2 = "Estimación de prueba con Node. Equipo: 3 Desarrollador. Totales: 400 h, 24500 EUR."


def _minusculas(items: list[str]) -> list[str]:
    """Los textos de una lista en minúsculas, para comparar sin distinguir mayúsculas."""
    return [item.lower() for item in items]


def comprobaciones(metadata: ProjectMetadata) -> list[tuple[str, bool]]:
    """Qué debe cumplir la memoria después del segundo turno, como pares (descripción, ok)."""
    tecnologias = _minusculas(metadata.mentioned_technologies)
    descartadas = _minusculas(metadata.rejected_options)
    restricciones = " ".join(metadata.explicit_constraints)
    otros = " ".join(metadata.other_facts).lower()
    return [
        ("conservó el nombre (Bookflow)", "bookflow" in (metadata.project_name or "").lower()),
        ("conservó el equipo (3 personas)", metadata.assumed_team_size == 3),
        ("conservó el presupuesto en las restricciones", "40" in restricciones),
        ("Node figura entre las tecnologías", "node" in tecnologias),
        ("Rails figura entre las descartadas", "rails" in descartadas),
        ("Rails ya no figura entre las tecnologías", "rails" not in tecnologias),
        ("conservó el dato suelto (cliente: gimnasios) en other_facts", "gimnasio" in otros),
    ]


def mostrar(titulo: str, metadata: ProjectMetadata, segundos: float, coste_usd: float) -> None:
    """Imprime la memoria que dejó un turno, con lo que tardó y costó la extracción."""
    print(f"\n--- {titulo} ({segundos:.1f} s, {coste_usd:.5f} USD) ---")
    print(metadata.model_dump_json(indent=2, exclude_defaults=True))


async def ejecutar(extractor: MetadataExtractor) -> bool:
    """Corre los dos turnos, imprime la memoria de cada uno y devuelve si pasó todo."""
    memoria = ProjectMetadata()
    for titulo, usuario, asistente in (
        ("Turno 1", TURNO_1, RESPUESTA_1),
        ("Turno 2", TURNO_2, RESPUESTA_2),
    ):
        inicio = time.perf_counter()
        resultado = await extractor.extract(memoria, user_turn=usuario, assistant_turn=asistente)
        mostrar(titulo, resultado.metadata, time.perf_counter() - inicio, resultado.cost_usd)
        print(f"modelo que respondió: {resultado.model}")
        memoria = resultado.metadata

    print("\nComprobaciones después del turno 2:")
    todo_ok = True
    for descripcion, ok in comprobaciones(memoria):
        print(f"  [{'OK' if ok else 'FALLA'}] {descripcion}")
        todo_ok = todo_ok and ok
    return todo_ok


def main() -> None:
    """Arma el extractor con la configuración del servicio y corre la conversación de ejemplo."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--proveedor",
        choices=sorted(EXTRACTOR_KEY_VARIABLES),
        help="proveedor a probar (por defecto, EXTRACTOR_DEFAULT del .env)",
    )
    args = parser.parse_args()

    settings = get_settings()
    probado = args.proveedor or settings.extractor_default
    # El «respaldo» es el otro proveedor y se le quita la clave: sin ella el Router no
    # arma ese deployment y, si el probado falla, el error sale en vez de taparse.
    otro = (
        settings.extractor_fallback
        if probado == settings.extractor_default
        else (settings.extractor_default)
    )
    settings = settings.model_copy(
        update={
            "extractor_default": probado,
            "extractor_fallback": otro,
            f"{otro}_api_key": None,
        }
    )
    print(f"Extractor: {settings.for_extractor().primary_model} (sin respaldo)")

    try:
        extractor = MetadataExtractor.from_settings(settings)
    except LLMConfigurationError as exc:
        raise SystemExit(str(exc)) from None
    try:
        todo_ok = asyncio.run(ejecutar(extractor))
    except Exception as exc:  # noqa: BLE001 - se muestra el tipo y el motivo del fallo
        motivo = str(exc).strip().splitlines()[-1][:300] if str(exc).strip() else ""
        raise SystemExit(f"\nLa llamada falló ({type(exc).__name__}): {motivo}") from None
    if not todo_ok:
        raise SystemExit("\nAlguna comprobación falló: la memoria perdió o no movió hechos.")
    print("\nTodo OK.")


if __name__ == "__main__":
    main()
