"""Métricas binarias del framework de evals (Bloque 1, LIDR_6).

QUÉ MIDE CADA MÉTRICA

Las tres son binarias (pasa / no pasa) y miran lo que de verdad hay que corregir si se
rompe: el contrato que ve el cliente (forma de la respuesta), la factura (coste por
turno y acumulado) y si el sistema de verdad usó lo que se le dijo (recuerda los
hechos del turno). Ninguna repite lo que ya validan los guardrails de `app/guardrails/`
o las reglas del propio schema (`app/schemas/structured_estimation.py`): esas corren
siempre, DENTRO del servicio, y ya garantizan que una respuesta 200 es internamente
coherente (fases con horas positivas, totales que cuadran con la suma, etc.). Estas
métricas comprueban algo distinto: que ese resultado coherente es además el que el
caso golden esperaba (el tier correcto, el coste razonable, los hechos recordados).

CÓMO SE USAN

`evals/run.py` arma un `CaseResult` por caso golden (las respuestas HTTP de cada turno,
en orden) y le pasa cada métrica de `METRICS`. Cada métrica devuelve `(aprobado,
motivo)`: un booleano para el CSV y una frase corta para que el reporte sea legible sin
abrir el JSON. Un caso donde algún turno no respondió 200 reprueba las tres sin mirar
el contenido: no hay nada fiable que medir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.schemas.structured_estimation import OUT_OF_SCOPE_PREFIX


@dataclass
class TurnResult:
    """Lo que dejó UN turno de un caso golden: la respuesta HTTP, lo que costó y cuánto
    tardó. `body` es `None` cuando el turno no respondió 200 (un 422, un 500...).
    """

    status_code: int
    body: dict[str, Any] | None
    cost_usd: float
    latency_s: float


@dataclass
class CaseResult:
    """Lo que dejó un caso golden completo: sus turnos, en orden, y la sesión usada.

    `session_metadata` es la memoria persistida de la sesión (`GET /sessions/{id}`),
    que `evals/run.py` solo consulta cuando el caso declara `expected_facts`: así
    `ContentRecallMetric` puede comprobar recuerdo real de la memoria, no solo lo que
    el modelo escribió en la última respuesta.
    """

    case_id: str
    session_id: str | None
    turns: list[TurnResult] = field(default_factory=list)
    session_metadata: dict[str, Any] | None = None

    @property
    def all_turns_ok(self) -> bool:
        """True si CADA turno del caso respondió 200. Si uno falló, las tres métricas
        reprueban sin mirar el contenido: no hay nada fiable que medir."""
        return bool(self.turns) and all(t.status_code == 200 for t in self.turns)

    @property
    def last_body(self) -> dict[str, Any] | None:
        """El cuerpo del ÚLTIMO turno exitoso: la estimación final del caso."""
        for turn in reversed(self.turns):
            if turn.status_code == 200 and turn.body is not None:
                return turn.body
        return None

    @property
    def total_cost_usd(self) -> float:
        """Coste sumado de todos los turnos del caso."""
        return sum(t.cost_usd for t in self.turns)


class Metric(Protocol):
    """Contrato de una métrica binaria: un nombre (para el CSV) y una función que
    evalúa un caso golden contra lo que de verdad respondió el servicio."""

    name: str

    def evaluate(self, case: dict[str, Any], result: CaseResult) -> tuple[bool, str]: ...


class SchemaAdherenceMetric:
    """¿La respuesta final tiene la forma que el caso golden esperaba?

    No repite lo que el schema ya valida solo (una respuesta 200 YA es internamente
    coherente, con re-pregunta si no lo era: ver `StructuredResult`). Comprueba lo que
    el schema no sabe, porque depende del caso: cuántas fases son razonables para su
    `detail_level` (`expected_phase_range`), si se esperaba un rechazo del modelo
    («Fuera de alcance», `expect_out_of_scope`) y si la vista por tier (`tier_view`)
    tiene las claves propias de ese tier (`expected_tier_view_keys`; `None` significa
    "este caso no lo comprueba", no "tiene que venir vacío").
    """

    name = "schema_adherence"

    def evaluate(self, case: dict[str, Any], result: CaseResult) -> tuple[bool, str]:
        if not result.all_turns_ok:
            return False, "algún turno no respondió 200"
        body = result.last_body
        if body is None:
            return False, "sin respuesta exitosa que evaluar"
        estimation = body["estimation"]

        if case.get("expect_out_of_scope"):
            ok = estimation["summary"].startswith(OUT_OF_SCOPE_PREFIX)
            return ok, ("rechazo esperado, llegó" if ok else "se esperaba «Fuera de alcance»")

        lo, hi = case["expected_phase_range"]
        n_phases = len(estimation["phases"])
        if not (lo <= n_phases <= hi):
            return False, f"{n_phases} fases, fuera de [{lo}, {hi}]"

        expected_keys = case.get("expected_tier_view_keys")
        if expected_keys is not None:
            tier_view = body.get("tier_view")
            got_keys = set(tier_view.keys()) if tier_view else set()
            if got_keys != set(expected_keys):
                return False, f"tier_view {sorted(got_keys)}, se esperaba {sorted(expected_keys)}"

        return True, f"{n_phases} fases, forma correcta"


class CostBoundsMetric:
    """¿El turno más caro y el total del caso caen dentro de lo presupuestado?

    Los topes (`max_cost_usd_per_turn`, `max_total_cost_usd`) son techos de
    REGRESIÓN, no un benchmark de precio: generosos a propósito, para no reprobar por
    la variación normal entre llamadas al proveedor, pero sí marcar un desbarranco real
    (un bucle de reintentos que no corta, un prompt que se disparó de tamaño).

    El coste de cada turno no viaja en el cuerpo de la respuesta HTTP (no es un dato
    del cliente): `evals/run.py` lo junta de los eventos de trazabilidad que emite el
    servicio en cada llamada al proveedor (`coste_usd` en `estimacion_completada`,
    `critico_completado`...), capturados con `structlog.testing.capture_logs`.
    """

    name = "cost_bounds"

    def evaluate(self, case: dict[str, Any], result: CaseResult) -> tuple[bool, str]:
        if not result.all_turns_ok:
            return False, "algún turno no respondió 200"

        max_per_turn = case["max_cost_usd_per_turn"]
        worst = max((t.cost_usd for t in result.turns), default=0.0)
        if worst > max_per_turn:
            return False, f"turno de ${worst:.5f}, tope ${max_per_turn:.5f}"

        max_total = case["max_total_cost_usd"]
        total = result.total_cost_usd
        if total > max_total:
            return False, f"total ${total:.5f}, tope ${max_total:.5f}"

        return True, f"${total:.5f} en {len(result.turns)} turno(s)"


class ContentRecallMetric:
    """¿Los hechos del caso (`expected_facts`) sobreviven en la respuesta final?

    Busca cada hecho (sin mayúsculas) como subcadena de un texto que junta la memoria
    persistida de la sesión (`session_metadata`, si `run.py` la consultó: lo que de
    verdad recuerda el servicio turno a turno) y la estimación del último turno (lo
    que el modelo escribió en esta respuesta). Una lista vacía de hechos (los casos
    fuera de alcance no tienen ninguno que recordar) aprueba siempre, sin mirar nada
    más.

    Es una búsqueda de texto, no de significado: un hecho ausente puede ser que el
    servicio lo olvidó, o que el modelo lo parafraseó distinto. Por eso los hechos del
    dataset son términos concretos (nombres, tecnologías, cifras), nunca frases largas
    que un modelo difícilmente repita palabra por palabra.
    """

    name = "content_recall"

    def evaluate(self, case: dict[str, Any], result: CaseResult) -> tuple[bool, str]:
        facts: list[str] = case.get("expected_facts") or []
        if not facts:
            return True, "sin hechos que verificar"
        if not result.all_turns_ok:
            return False, "algún turno no respondió 200"

        haystack = _recall_text(result).lower()
        faltantes = [fact for fact in facts if fact.lower() not in haystack]
        if faltantes:
            return False, f"no aparecen: {', '.join(faltantes)}"
        return True, "todos los hechos aparecen"


def _recall_text(result: CaseResult) -> str:
    """Todo el texto donde puede sobrevivir un hecho: la metadata de la sesión (si se
    consultó) y la estimación completa (fases, equipo, resumen) del último turno.
    """
    body = result.last_body or {}
    partes = [json.dumps(body.get("estimation", {}), ensure_ascii=False)]
    if result.session_metadata:
        partes.append(json.dumps(result.session_metadata, ensure_ascii=False))
    return "\n".join(partes)


# Las tres métricas, en el orden en que `evals/run.py` las reporta.
METRICS: tuple[Metric, ...] = (SchemaAdherenceMetric(), CostBoundsMetric(), ContentRecallMetric())
