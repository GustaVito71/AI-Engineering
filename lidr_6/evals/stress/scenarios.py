"""Bloque 2 de LIDR_6: escenarios sintéticos multi-turno.

QUÉ GENERA ESTE MÓDULO

Conversaciones de N turnos (`TURN_COUNTS`) sobre un mismo proyecto, en tres perfiles
(`PROFILES`), cada uno pensado para forzar un comportamiento distinto de la memoria de
sesión (`app/sessions/`):

- `growing` (proyecto que crece): el turno 1 fija el nombre del proyecto y cada turno
  siguiente agrega UN requisito nuevo y coherente (autenticación, multi-tenant, audit
  log, exportar a CSV...). Lo que se mide después (bloque 4): la curva de coste turno
  a turno, y si el nombre del proyecto del turno 1 sigue siendo recuperable en el
  turno 20, pese a que la ventana deslizante (`MAX_TURNS=6`) lo expulsó hace rato.
- `pivoting` (proyecto que pivota): arranca con un stack y, a mitad de camino (turno 5
  en una conversación larga; antes si la conversación es corta), lo cambia por otro
  distinto. Lo que se mide: si `mentioned_technologies` queda con el stack nuevo
  solamente, o acumula los dos (el stack viejo debería quedar en `rejected_options`,
  ver `ProjectMetadata.normalized`).
- `contradicting` (proyecto que se contradice): declara un presupuesto y, más
  adelante, lo corrige por otro distinto (30000 → 80000 euros, como en el enunciado).
  Lo que se mide: cuál de los dos números sobrevive, cuál se promueve a ancla
  (`app/sessions/compression.py`) y cuál termina solo en el resumen acumulado, si es
  que alguno se pierde del todo.

CADA PERFIL DECLARA UN FACT-TRACKER

Cada turno (`ScenarioTurn`) declara qué hechos introduce (`expected_facts`) y cuáles
deja de valer a partir de ahí (`superseded_facts`: el stack viejo tras un pivote, el
presupuesto viejo tras una corrección). `Scenario.facts_expected_after(k)` resuelve
eso a la pregunta que de verdad importa: "¿qué debería poder recordar CUALQUIER
llamada posterior al turno k?". Esa es la pregunta que alimenta a `MemoryDriftMetric`
del bloque 4: ese bloque, no este, decide si lo que el servicio recordó de verdad
coincide con lo que este módulo dice que debería recordar.

Este módulo no llama al servicio ni corre nada: solo genera datos. `evals/run.py`
corre los casos FIJOS de `golden_dataset.json`; un runner del bloque 4 (todavía no
escrito) será el que tome `all_scenarios()` y sí llame a `/sessions/{id}/estimate`
turno a turno.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

# Turnos por conversación que pide el enunciado del bloque 2.
TURN_COUNTS: tuple[int, ...] = (1, 3, 6, 10, 20)
# Los tres perfiles, en el orden en que se describen arriba.
PROFILES: tuple[str, ...] = ("growing", "pivoting", "contradicting")


@dataclass(frozen=True)
class ScenarioTurn:
    """Un turno del escenario: lo que dice el usuario y el fact-tracker de ese turno.

    `expected_facts` son los hechos que este turno introduce y que deberían poder
    recordarse en cualquier llamada posterior. `superseded_facts` son hechos de un
    turno anterior que este turno vuelve obsoletos (el stack que se abandona en un
    pivote, el presupuesto que se corrige): a partir de este turno ya no deberían
    contar como "lo vigente", aunque el servicio pueda seguir arrastrándolos (eso es
    justamente lo que el bloque 4 mide, no algo que este módulo dé por sentado).
    """

    index: int  # 1-based: el turno es el `index`-ésimo de la conversación.
    transcript: str
    expected_facts: tuple[str, ...] = ()
    superseded_facts: tuple[str, ...] = ()
    note: str = ""  # para quien lea el escenario: por qué este turno es como es.


@dataclass(frozen=True)
class Scenario:
    """Una conversación sintética completa: su perfil, el proyecto base y sus turnos.

    Los campos de proyecto (`project_type`, `detail_level`, `output_format`) se
    mandan igual en cada turno: son el formulario de `/sessions/{id}/estimate`, y
    variarlos turno a turno no es parte de lo que estos tres perfiles quieren medir.
    """

    profile: str
    n_turns: int
    project_type: str
    detail_level: str
    output_format: str
    description: str
    turns: tuple[ScenarioTurn, ...]

    @property
    def scenario_id(self) -> str:
        """Identificador corto y único: perfil + cantidad de turnos."""
        return f"{self.profile}-{self.n_turns}"

    def facts_expected_after(self, turn_index: int) -> frozenset[str]:
        """Los hechos que deberían poder recordarse en cualquier llamada DESPUÉS del
        turno `turn_index` (1-based, inclusive).

        Recorre los turnos hasta `turn_index` en orden: por cada uno, primero quita lo
        que ese turno marca como superado y después agrega lo que introduce. Así un
        pivote o una corrección que ocurre y se declara en el MISMO turno queda bien
        resuelto (no hay ambigüedad de orden dentro de un turno).
        """
        vigentes: set[str] = set()
        for turno in self.turns:
            if turno.index > turn_index:
                break
            vigentes -= set(turno.superseded_facts)
            vigentes |= set(turno.expected_facts)
        return frozenset(vigentes)

    def as_dict(self) -> dict:
        """El escenario como diccionario plano, para volcarlo a JSON (`--salida`)."""
        return asdict(self)


# --- Filler: requisitos coherentes para turnos que no son un hito del perfil -------------

# Pool compartido por los tres perfiles: requisitos de software realistas y variados,
# para que los turnos "de relleno" (los que no fijan el proyecto, ni pivotan, ni
# contradicen nada) sigan siendo un turno coherente y no un texto repetido. Con 20
# entradas alcanza para el caso más largo (N=20) sin tener que repetir ninguna.
_REQUIREMENTS_POOL: tuple[tuple[str, str], ...] = (
    ("autenticación de usuarios con email y contraseña", "autenticación"),
    ("multi-tenant: cada cliente corporativo con sus datos aislados", "multi-tenant"),
    ("un audit log de cada cambio que hace un usuario", "audit log"),
    ("poder exportar los datos a CSV", "exportar a csv"),
    ("roles y permisos distintos para administradores y usuarios", "roles y permisos"),
    ("notificaciones por email cuando cambia el estado de algo", "notificaciones"),
    ("un buscador por nombre o número de referencia", "buscador"),
    ("un dashboard con métricas de uso", "dashboard de métricas"),
    ("integración con el sistema de facturación existente", "integración con facturación"),
    ("soporte para español e inglés", "soporte multi-idioma"),
    ("un historial de versiones de cada registro", "historial de versiones"),
    ("alertas automáticas cuando algo queda atascado", "alertas automáticas"),
    ("un módulo de reportes mensuales en PDF", "reportes en pdf"),
    ("inicio de sesión único (SSO) con el proveedor corporativo", "sso"),
    ("una API pública para que terceros consulten datos", "api pública"),
    ("copias de seguridad automáticas diarias", "copias de seguridad"),
    ("un chat de soporte integrado", "chat de soporte"),
    ("firma electrónica para aprobaciones", "firma electrónica"),
    ("un modo oscuro en la interfaz", "modo oscuro"),
    ("métricas de rendimiento expuestas a Grafana", "métricas a grafana"),
)


def _filler(turno_num: int) -> tuple[str, str]:
    """El requisito de relleno para el turno `turno_num` (1-based, cualquier turno de
    cualquier perfil): cicla el pool si la conversación pide más turnos de los que
    tiene (no pasa con N<=20, pero deja el generador seguro ante un N mayor)."""
    return _REQUIREMENTS_POOL[(turno_num - 1) % len(_REQUIREMENTS_POOL)]


def _filler_turn(index: int) -> ScenarioTurn:
    """Un turno de relleno: agrega un requisito del pool, sin pivotar ni contradecir nada."""
    frase, etiqueta = _filler(index)
    return ScenarioTurn(
        index=index,
        transcript=f"También necesitamos {frase}.",
        expected_facts=(etiqueta,),
        note="relleno: agrega un requisito coherente, no es un hito del perfil",
    )


# --- Perfil 1: proyecto que crece ---------------------------------------------------------

_GROWING_PROJECT = "Portal Unificado de Clientes"


def _build_growing(n_turns: int) -> Scenario:
    """Turno 1 fija el proyecto; cada turno siguiente agrega un requisito del pool.

    Con N=1 no hay crecimiento que medir (es el control: nada que recordar más allá de
    un turno), y eso es intencional, no un caso degenerado a evitar.
    """
    turno_1 = ScenarioTurn(
        index=1,
        transcript=(
            f'Vamos a construir "{_GROWING_PROJECT}", un portal web donde los clientes '
            "puedan ver y gestionar sus pedidos."
        ),
        expected_facts=(_GROWING_PROJECT,),
        note="establece el proyecto: este nombre es lo que se mide al llegar al turno N",
    )
    turnos = [turno_1, *(_filler_turn(i) for i in range(2, n_turns + 1))]
    return Scenario(
        profile="growing",
        n_turns=n_turns,
        project_type="web_saas",
        detail_level="medium",
        output_format="phases_table",
        description=(
            f"Proyecto que crece: {n_turns} turno(s), uno agregando un requisito nuevo "
            "tras otro. Mide si el nombre del proyecto del turno 1 sobrevive al último turno."
        ),
        turns=tuple(turnos),
    )


# --- Perfil 2: proyecto que pivota ---------------------------------------------------------

_PIVOTING_PROJECT = "App de Gestión de Turnos"
_STACK_INICIAL = "react"
_STACK_NUEVO = "flutter"


def _pivot_turn(n_turns: int) -> int | None:
    """El turno en el que el proyecto pivota de stack.

    Turno 5, como en el enunciado, para toda conversación que llega ahí (N>=5: los
    casos 6, 10 y 20). Para las más cortas se escala sin perder el patrón (pivota
    cerca del final, dejando al menos un turno de "antes"); con un solo turno no hay
    "antes" que pivotar, así que no hay pivote (ver `_build_pivoting`).
    """
    if n_turns >= 5:
        return 5
    if n_turns >= 3:
        return n_turns - 1
    return None


def _build_pivoting(n_turns: int) -> Scenario:
    """Turno 1 fija el proyecto y el stack inicial; en el turno de pivote se cambia de
    stack (el viejo queda `superseded`); el resto son relleno.

    Con N=1 no hay espacio para un "antes" y un "después": el único turno fija
    directamente el stack NUEVO, sin que el viejo llegue a mencionarse (no hay nada
    que pivotar sin una conversación previa).
    """
    pivot = _pivot_turn(n_turns)

    if pivot is None:
        turno_1 = ScenarioTurn(
            index=1,
            transcript=(
                f'Vamos a construir "{_PIVOTING_PROJECT}", una app para reservar turnos, '
                f"hecha en {_STACK_NUEVO.capitalize()}."
            ),
            expected_facts=(_PIVOTING_PROJECT, _STACK_NUEVO),
            note="sin espacio para pivotar (N=1): fija directamente el stack final",
        )
        return Scenario(
            profile="pivoting",
            n_turns=n_turns,
            project_type="mobile_app",
            detail_level="medium",
            output_format="narrative",
            description=(
                "Proyecto que pivota: con un solo turno no hay pivote que mostrar, así que "
                "fija directamente el stack de destino (caso de control)."
            ),
            turns=(turno_1,),
        )

    turno_1 = ScenarioTurn(
        index=1,
        transcript=(
            f'Vamos a construir "{_PIVOTING_PROJECT}", una app para reservar turnos, hecha '
            f"en {_STACK_INICIAL.capitalize()}."
        ),
        expected_facts=(_PIVOTING_PROJECT, _STACK_INICIAL),
        note="establece el proyecto y el stack inicial",
    )
    turno_pivote = ScenarioTurn(
        index=pivot,
        transcript=(
            "Repensamos la base técnica: dejamos de usar "
            f"{_STACK_INICIAL.capitalize()} y pasamos todo a {_STACK_NUEVO.capitalize()}, "
            "para tener una sola base de código en iOS y Android."
        ),
        expected_facts=(_STACK_NUEVO,),
        superseded_facts=(_STACK_INICIAL,),
        note="el pivote: el stack viejo queda superado a partir de este turno",
    )
    turnos = [turno_1]
    turnos.extend(_filler_turn(i) for i in range(2, pivot))
    turnos.append(turno_pivote)
    turnos.extend(_filler_turn(i) for i in range(pivot + 1, n_turns + 1))

    return Scenario(
        profile="pivoting",
        n_turns=n_turns,
        project_type="mobile_app",
        detail_level="medium",
        output_format="narrative",
        description=(
            f"Proyecto que pivota: {n_turns} turno(s), cambia de "
            f"{_STACK_INICIAL.capitalize()} a {_STACK_NUEVO.capitalize()} en el turno "
            f"{pivot}. Mide si el stack viejo desaparece de `mentioned_technologies` o "
            "queda acumulado junto al nuevo."
        ),
        turns=tuple(turnos),
    )


# --- Perfil 3: proyecto que se contradice ---------------------------------------------------

_CONTRADICTING_PROJECT = "Sistema de Reservas de Clases"
_PRESUPUESTO_INICIAL = "30000"
_PRESUPUESTO_CORREGIDO = "80000"


def _contradiction_turns(n_turns: int) -> tuple[int, int] | None:
    """(turno de la declaración, turno de la corrección) del presupuesto.

    Turnos 3 y 8, como en el enunciado, para la conversación que llega ahí (N=10 y
    N=20). Para las más cortas se escala sin perder el patrón (declaración cerca del
    inicio, corrección más adelante); con 1 o 2 turnos no hay espacio para que la
    corrección sea un turno DISTINTO de la declaración, así que no hay contradicción
    (ver `_build_contradicting`).
    """
    if n_turns >= 8:
        return 3, 8
    if n_turns >= 6:
        return 2, n_turns
    if n_turns >= 3:
        return 1, n_turns
    return None


def _build_contradicting(n_turns: int) -> Scenario:
    """Turno 1 fija el proyecto; en algún turno se declara un presupuesto y, más
    adelante, se corrige por otro (30000 → 80000 euros).

    Con N=1 o N=2 no hay espacio para dos turnos distintos (declaración y corrección),
    así que el único turno relevante declara un solo presupuesto, sin contradecirlo:
    es el caso de control de este perfil, igual que el de `pivoting`.
    """
    puntos = _contradiction_turns(n_turns)

    if puntos is None:
        turno_1 = ScenarioTurn(
            index=1,
            transcript=(
                f'Queremos presupuestar "{_CONTRADICTING_PROJECT}", un sistema de reservas '
                f"de clases para un gimnasio, con un presupuesto de {_PRESUPUESTO_INICIAL} euros."
            ),
            expected_facts=(_CONTRADICTING_PROJECT, _PRESUPUESTO_INICIAL),
            note="sin espacio para contradecirse (N<=2): declara un solo presupuesto",
        )
        turnos = [turno_1] if n_turns == 1 else [turno_1, _filler_turn(2)]
        return Scenario(
            profile="contradicting",
            n_turns=n_turns,
            project_type="internal_tool",
            detail_level="medium",
            output_format="narrative",
            description=(
                "Proyecto que se contradice: con tan pocos turnos no hay espacio para la "
                "corrección, así que declara un solo presupuesto (caso de control)."
            ),
            turns=tuple(turnos),
        )

    declaracion, correccion = puntos
    if declaracion == 1:
        turno_decl = ScenarioTurn(
            index=1,
            transcript=(
                f'Queremos presupuestar "{_CONTRADICTING_PROJECT}", un sistema de reservas '
                f"de clases para un gimnasio, con un presupuesto de {_PRESUPUESTO_INICIAL} euros."
            ),
            expected_facts=(_CONTRADICTING_PROJECT, _PRESUPUESTO_INICIAL),
            note="establece el proyecto y declara el primer presupuesto en el mismo turno",
        )
    else:
        turno_decl = ScenarioTurn(
            index=declaracion,
            transcript=f"El presupuesto para este proyecto es de {_PRESUPUESTO_INICIAL} euros.",
            expected_facts=(_PRESUPUESTO_INICIAL,),
            note="declara el primer presupuesto",
        )

    turno_correccion = ScenarioTurn(
        index=correccion,
        transcript=(
            "Revisando números con finanzas, en realidad el presupuesto es de "
            f"{_PRESUPUESTO_CORREGIDO} euros, no {_PRESUPUESTO_INICIAL}."
        ),
        expected_facts=(_PRESUPUESTO_CORREGIDO,),
        superseded_facts=(_PRESUPUESTO_INICIAL,),
        note="la corrección: el presupuesto viejo queda superado a partir de este turno",
    )

    turnos = []
    if declaracion != 1:
        turnos.append(
            ScenarioTurn(
                index=1,
                transcript=f'Queremos presupuestar "{_CONTRADICTING_PROJECT}", un sistema '
                "de reservas de clases para un gimnasio.",
                expected_facts=(_CONTRADICTING_PROJECT,),
                note="establece el proyecto, antes de hablar de presupuesto",
            )
        )
    turnos.extend(_filler_turn(i) for i in range(2, declaracion))
    turnos.append(turno_decl)
    turnos.extend(_filler_turn(i) for i in range(declaracion + 1, correccion))
    turnos.append(turno_correccion)
    turnos.extend(_filler_turn(i) for i in range(correccion + 1, n_turns + 1))
    turnos.sort(key=lambda t: t.index)

    return Scenario(
        profile="contradicting",
        n_turns=n_turns,
        project_type="internal_tool",
        detail_level="medium",
        output_format="narrative",
        description=(
            f"Proyecto que se contradice: {n_turns} turno(s), declara {_PRESUPUESTO_INICIAL} "
            f"euros en el turno {declaracion} y lo corrige a {_PRESUPUESTO_CORREGIDO} en el "
            f"turno {correccion}. Mide cuál de los dos sobrevive, cuál se promueve a ancla y "
            "cuál termina solo en el resumen."
        ),
        turns=tuple(turnos),
    )


_BUILDERS = {
    "growing": _build_growing,
    "pivoting": _build_pivoting,
    "contradicting": _build_contradicting,
}


def build_scenario(profile: str, n_turns: int) -> Scenario:
    """El escenario de `profile` con `n_turns` turnos.

    Lanza `ValueError` si `profile` no es uno de `PROFILES`: un perfil mal escrito no
    debe generar silenciosamente un escenario vacío.
    """
    builder = _BUILDERS.get(profile)
    if builder is None:
        raise ValueError(f"Perfil desconocido: {profile!r} (válidos: {', '.join(PROFILES)})")
    if n_turns < 1:
        raise ValueError(f"n_turns tiene que ser al menos 1 (recibido {n_turns})")
    return builder(n_turns)


def all_scenarios() -> list[Scenario]:
    """Las 3 × 5 = 15 combinaciones (perfil, N) que pide el bloque 2, en orden estable
    (perfil por perfil, de menos a más turnos): lo que un runner del bloque 4 recorre
    para estresar la memoria de sesión con cada patrón.
    """
    return [build_scenario(profile, n) for profile in PROFILES for n in TURN_COUNTS]


def main() -> None:
    """CLI de inspección: vuelca los escenarios a JSON para mirarlos o pasarlos a otro
    script, sin correr nada contra el servicio (eso lo hace el bloque 4)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--salida",
        default=str(Path(__file__).resolve().parent / "scenarios.json"),
        help="Ruta del JSON de salida",
    )
    args = ap.parse_args()

    escenarios = [s.as_dict() for s in all_scenarios()]
    Path(args.salida).write_text(json.dumps(escenarios, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(escenarios)} escenarios ({len(PROFILES)} perfiles × {len(TURN_COUNTS)} N) en {args.salida}")


if __name__ == "__main__":
    main()
