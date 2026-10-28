"""Bloque 4 de LIDR_6: tres métricas nuevas sobre lo que ya miden los Bloques 1-3.

POR QUÉ UN MÓDULO PARALELO Y NO `evals/metrics.py`

`evals/metrics.py` (Bloque 1) define `Metric.evaluate(self, case: dict, result:
CaseResult) -> tuple[bool, str]`: un contrato pensado para UN caso golden con VARIOS
turnos ya corridos (`CaseResult`), comparado contra lo que ese caso esperaba (`case`).
Las tres métricas de este bloque reciben otra cosa:

- `LatencyBudgetMetric` y `CostBudgetMetric` miran UNA observación suelta (un turno,
  un punto del escenario de adjuntos del Bloque 3, cualquier cosa con `latency_ms`/
  `cost_usd`) contra un presupuesto fijo que se configura en el constructor, no un
  caso golden con rango esperado.
- `MemoryDriftMetric` mira una FOTO de la sesión en un turno dado (`session_snapshot`:
  resumen, anclas y `ProjectMetadata` tal como están DESPUÉS de ese turno), no una
  lista de respuestas HTTP.

Forzar estas tres al contrato de `evals/metrics.py` habría significado envolver cada
observación en un `CaseResult` falso solo para satisfacer la firma. Conviene más un
contrato propio (`MetricResult`, más abajo) en un módulo propio, junto a
`evals/stress/attachments.py` y `evals/stress/scenarios.py`: las tres piezas del
Bloque 3 y el Bloque 4 comparten dominio (observaciones de un escenario de estrés,
no casos golden fijos) y por eso viven en el mismo paquete `evals/stress/`.

DETERMINISMO, NO SOFISTICACIÓN

Igual que `ContentRecallMetric` del Bloque 1: comparación de texto exacta
(case-insensitive), sin embeddings ni un LLM de juez. Un hecho se considera presente
si aparece como subcadena; nada de "significa lo mismo que". Es una elección
deliberada, no una limitación que haya que disculpar: una métrica determinista da
siempre el mismo resultado sobre la misma entrada, así que una regresión real nunca se
puede confundir con que el juez tuvo un mal día.

CÓMO SE USAN (quién llama a esto)

Este módulo solo define las métricas; no corre nada contra el servicio. Quien las usa
es el futuro runner del Bloque 4 (todavía no escrito, como aclara
`evals/stress/scenarios.py`): por cada turno de un `Scenario` llama a
`/sessions/{id}/estimate`, mide `latency_ms`/`cost_usd` de esa llamada y, para la
sesión, saca del almacén en memoria (`app.sessions.store.SessionStore`) una copia de
la `Session` tal como queda después de ese turno. Esa copia es el `session_snapshot`:
`MemoryDriftMetric` necesita el texto real de `summary` y `anchors`, que la API
pública NO expone (`GET /sessions/{id}` solo da un conteo de anclas, no su contenido;
ver `app/schemas/session.py:SessionStateResponse`), así que hace falta mirar el objeto
interno, no la respuesta HTTP. El Bloque 3 ya establece este patrón: `correr_punto` en
`evals/stress/attachments.py` también mide cosas (el coste) que no viajan en el cuerpo
de la respuesta, leyendo los eventos internos de trazabilidad en vez de pedírselo a la
API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# --- El contrato de salida común a las tres métricas ---------------------------------------


@dataclass(frozen=True)
class MetricResult:
    """Resultado de UNA métrica sobre UNA observación.

    Mismo contenido que ya reporta cada métrica de `evals/metrics.py` (un nombre, si
    pasó o no, y un motivo legible), pero empaquetado distinto, por dos razones:

    1. `score` es un número (1.0 / 0.0 acá, siempre binario como pide el enunciado del
       bloque), no solo el booleano `passed`. Las dos métricas de presupuesto miden
       conceptos continuos (latencia, coste) aunque el veredicto sea binario; guardar
       el score aparte permite, más adelante, promediar entre escenarios sin tener que
       reconvertir un `bool` cada vez.
    2. Es una dataclass con CAMPOS CON NOMBRE, no la tupla `(bool, str)` de
       `Metric.evaluate`. Con tres métricas que reciben una entrada distinta cada una
       (`observation` plano para las dos de presupuesto, `session_snapshot` para la de
       memoria), conviene que el resultado lleve su propio `name` adentro: quien junta
       resultados de las tres en una lista no tiene que ir a buscar
       `metric.name` por separado para saber cuál es cuál.
    """

    name: str
    score: float
    passed: bool
    details: str


# --- Helpers de lectura: la misma observación, en varios nombres de campo ------------------
#
# El resto del framework ya mide latencia y coste, pero no siempre con el mismo nombre
# de campo: `evals/metrics.py:TurnResult` usa `latency_s`/`cost_usd` (inglés, segundos);
# `evals/stress/attachments.py:correr_punto` devuelve un dict con `latencia_s`/
# `coste_usd` (castellano, también segundos). Ninguno de los dos tiene milisegundos.
# En vez de obligar a todo el framework a agregar un campo `latency_ms` solo para este
# bloque, estos helpers leen lo que YA existe y convierten.


def _field(observation: Any, *names: str) -> Any:
    """El primer campo de `names` que tenga `observation`, probando primero como
    atributo (un dataclass, un objeto de Pydantic) y después como clave de diccionario.

    Lanza `KeyError` si no encuentra ninguno, con los nombres probados en el mensaje:
    una observación que no trae el campo que la métrica necesita es un error de quien
    construye el escenario, no un 0 silencioso que esconda el problema.
    """
    for name in names:
        if hasattr(observation, name):
            return getattr(observation, name)
        try:
            return observation[name]
        except (TypeError, KeyError, IndexError):
            continue
    probados = ", ".join(names)
    raise KeyError(f"la observación no tiene ninguno de estos campos: {probados}")


def _latency_ms(observation: Any) -> float:
    """`latency_ms` si ya viene en milisegundos; si no, lo calcula a partir de
    `latency_s`/`latencia_s` (segundos), que es como lo reporta el resto del
    framework (ver el comentario de arriba)."""
    try:
        return float(_field(observation, "latency_ms"))
    except KeyError:
        return float(_field(observation, "latency_s", "latencia_s")) * 1000.0


def _cost_usd(observation: Any) -> float:
    """`cost_usd`/`coste_usd`: el mismo dato, en los dos nombres que ya usa el
    framework (inglés en `evals/metrics.py`, castellano en `evals/stress/`)."""
    return float(_field(observation, "cost_usd", "coste_usd"))


# --- LatencyBudgetMetric --------------------------------------------------------------------


class LatencyBudgetMetric:
    """1.0 si la latencia de la observación cae dentro de `budget_ms`; 0.0 si no.

    Para un punto del escenario de adjuntos (Bloque 3) o cualquier otra observación con
    latencia: ¿la llamada respondió dentro del presupuesto, o algo (un adjunto enorme,
    un reintento, un proveedor lento) lo hizo pasarse? El presupuesto es fijo y lo
    decide quien instancia la métrica, no el caso: dos llamadas al mismo endpoint con
    presupuestos distintos (una interactiva, con tope ajustado; otra en lote, más
    permisiva) son dos instancias distintas de esta clase, no dos casos del mismo
    `Metric`.
    """

    def __init__(self, budget_ms: int) -> None:
        if budget_ms <= 0:
            raise ValueError(f"budget_ms tiene que ser positivo (recibido {budget_ms})")
        self.budget_ms = budget_ms

    @property
    def name(self) -> str:
        return f"latency_budget_{self.budget_ms}ms"

    def evaluate(self, observation: Any) -> MetricResult:
        """`observation` es cualquier objeto o dict con `latency_ms`, o con
        `latency_s`/`latencia_s` (segundos): un `TurnResult`, una fila de
        `correr_punto`, o un dict armado a mano en un test."""
        latencia = _latency_ms(observation)
        aprobado = latencia <= self.budget_ms
        detalle = (
            f"{latencia:.0f} ms, dentro del tope de {self.budget_ms} ms"
            if aprobado
            else f"{latencia:.0f} ms supera el tope de {self.budget_ms} ms"
        )
        return MetricResult(name=self.name, score=1.0 if aprobado else 0.0, passed=aprobado, details=detalle)


# --- CostBudgetMetric ------------------------------------------------------------------------


class CostBudgetMetric:
    """1.0 si el coste de la observación cae dentro de `budget_usd`; 0.0 si no.

    Mismo espíritu que `LatencyBudgetMetric`, para coste en vez de latencia. No repite
    a `CostBoundsMetric` de `evals/metrics.py`: esa mira el coste PEOR turno y el TOTAL
    de un caso golden entero (varios turnos); esta mira el coste de UNA observación
    suelta contra un único presupuesto, pensada para los puntos de un escenario de
    estrés (Bloque 2/3), no para un caso golden de varios turnos.
    """

    def __init__(self, budget_usd: float) -> None:
        if budget_usd <= 0:
            raise ValueError(f"budget_usd tiene que ser positivo (recibido {budget_usd})")
        self.budget_usd = budget_usd

    @property
    def name(self) -> str:
        return f"cost_budget_{self.budget_usd}usd"

    def evaluate(self, observation: Any) -> MetricResult:
        """`observation` es cualquier objeto o dict con `cost_usd` o `coste_usd`."""
        coste = _cost_usd(observation)
        aprobado = coste <= self.budget_usd
        detalle = (
            f"${coste:.5f}, dentro del tope de ${self.budget_usd:.5f}"
            if aprobado
            else f"${coste:.5f} supera el tope de ${self.budget_usd:.5f}"
        )
        return MetricResult(name=self.name, score=1.0 if aprobado else 0.0, passed=aprobado, details=detalle)


# --- MemoryDriftMetric -----------------------------------------------------------------------

# Las tres zonas de memoria donde puede sobrevivir un hecho, en el orden en que llegan
# al modelo (`ConversationHistory.to_history_messages`: resumen, anclas, ventana) salvo
# que `metadata` no viaja en el historial de mensajes: se inyecta aparte, en el system
# prompt de cada turno (ver `app/sessions/models.py:ProjectMetadata`).
_ZONAS_VALIDAS = ("summary", "anchors", "metadata")


def _normalize_snapshot(session_snapshot: Any) -> dict[str, Any]:
    """Convierte `session_snapshot` a un dict plano con tres claves fijas (`summary`,
    `anchors`, `metadata`), sin importar en qué forma llegó.

    Acepta dos formas:
    1. La forma REAL del dominio: un `Session` de `app.sessions.models` (o cualquier
       objeto con esa misma forma), con `.history.summary` (`str | None`),
       `.history.anchors` (lista de `Message`, cada uno con `.content`) y `.metadata`
       (un `ProjectMetadata`, o cualquier objeto/dict con sus campos). Es la forma que
       usaría el runner real: una copia de lo que el almacén en memoria tiene DESPUÉS
       de un turno, sacada directo del `SessionStore` (ver el docstring del módulo).
    2. Un dict YA plano con esas tres claves (`summary`, `anchors`, `metadata`), para
       tests que no quieren construir los modelos de Pydantic completos.
    """
    if isinstance(session_snapshot, dict) and {"summary", "anchors", "metadata"} <= session_snapshot.keys():
        resumen = session_snapshot["summary"]
        anclas = session_snapshot["anchors"]
        metadata = session_snapshot["metadata"]
    else:
        historia = getattr(session_snapshot, "history", session_snapshot)
        resumen = getattr(historia, "summary", None)
        anclas = getattr(historia, "anchors", [])
        metadata = getattr(session_snapshot, "metadata", {})

    # Cada ancla puede ser un `Message` (con `.content`) o ya un string plano.
    anclas_texto = [getattr(ancla, "content", ancla) for ancla in anclas]

    # `ProjectMetadata` es un modelo de Pydantic (`.model_dump()`); si no lo es, se
    # intenta como dict directamente, y si tampoco, se trata como un texto suelto.
    if hasattr(metadata, "model_dump"):
        metadata = metadata.model_dump()
    elif not isinstance(metadata, dict):
        metadata = {"valor": metadata}

    return {"summary": resumen, "anchors": anclas_texto, "metadata": metadata}


def _texto_de_zona(zona: str, snapshot: dict[str, Any]) -> str:
    """El texto de UNA zona del snapshot ya normalizado, listo para buscar un hecho."""
    if zona == "summary":
        return snapshot["summary"] or ""
    if zona == "anchors":
        return "\n".join(snapshot["anchors"])
    # zona == "metadata": se vuelca el dict entero a texto (nombres de campo incluidos,
    # pero eso no genera falsos positivos: un hecho de prueba razonable, como un nombre
    # de proyecto o una cifra de presupuesto, no coincide con "project_name" o
    # "assumed_team_size").
    return json.dumps(snapshot["metadata"], ensure_ascii=False, default=str)


class MemoryDriftMetric:
    """1.0 si `fact` aparece (subcadena, case-insensitive) en alguna de las zonas de
    `where` dentro de `session_snapshot`; 0.0 si no.

    QUÉ ES "DRIFT" ACÁ: el hecho de que algo declarado en el turno k debería seguir
    siendo recuperable en cualquier turno posterior N (N > k), aunque la ventana
    deslizante ya haya expulsado el mensaje original (`app/sessions/models.py:
    ConversationHistory`, `max_turns`). Esta clase no sabe nada de "turno k" ni "turno
    N": solo comprueba una foto. Esa es una decisión deliberada, no una limitación: la
    relación entre turnos (qué hecho correspondía a qué turno, y qué foto tomar para
    cada N) la conoce `Scenario.facts_expected_after(k)` de
    `evals/stress/scenarios.py`, que ya resuelve exactamente esa pregunta. El runner
    del Bloque 4 es quien une las dos piezas: por cada N, saca la foto de la sesión en
    ese punto y, por cada hecho que `facts_expected_after(k)` diga vigente para algún
    k < N, instancia `MemoryDriftMetric(fact)` y la evalúa contra esa foto.

    `where` por defecto mira las tres zonas; se puede acotar (por ejemplo, para
    comprobar específicamente que algo se promovió a ancla y no quedó solo en el
    resumen, como el presupuesto corregido del perfil `contradicting`).
    """

    def __init__(self, fact: str, where: list[str] | None = None) -> None:
        if not fact.strip():
            raise ValueError("fact no puede ser vacío")
        self.fact = fact
        # `where=None` (en vez del `["summary","anchors","metadata"]` mutable que
        # muestra el enunciado) para no repetir el error clásico de Python de un
        # argumento por defecto mutable, compartido entre instancias; el valor por
        # defecto resultante es exactamente el mismo.
        self.where = list(where) if where is not None else list(_ZONAS_VALIDAS)
        desconocidas = sorted(set(self.where) - set(_ZONAS_VALIDAS))
        if desconocidas:
            raise ValueError(f"where contiene zonas desconocidas: {desconocidas} (válidas: {_ZONAS_VALIDAS})")

    @property
    def name(self) -> str:
        return f"memory_drift[{self.fact}]"

    def evaluate(self, session_snapshot: Any) -> MetricResult:
        snapshot = _normalize_snapshot(session_snapshot)
        hecho = self.fact.lower()
        encontrado_en = [zona for zona in self.where if hecho in _texto_de_zona(zona, snapshot).lower()]
        aprobado = bool(encontrado_en)
        detalle = (
            f"aparece en: {', '.join(encontrado_en)}"
            if aprobado
            else f"no aparece en ninguna de: {', '.join(self.where)}"
        )
        return MetricResult(name=self.name, score=1.0 if aprobado else 0.0, passed=aprobado, details=detalle)
