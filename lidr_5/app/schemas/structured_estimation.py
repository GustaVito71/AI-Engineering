"""Contrato de la salida estructurada del LLM (WU8) y respuesta HTTP que la lleva.

`StructuredResult` es lo que el modelo tiene que devolver con el prompt `v3`.
Instructor envía este schema como `response_format` y valida la respuesta contra
él: si no cumple, le devuelve al modelo el error de validación y reintenta, hasta
`STRUCTURED_MAX_RETRIES` veces.

Dos clases de comprobaciones, con consecuencias distintas:

- Las del schema son errores: una respuesta que no las cumple no se puede usar, y
  se re-pregunta al modelo. Son los rangos de cada campo (mínimos y topes: una
  fase de 5.000 semanas no es una estimación), los nombres de fase únicos y la
  coherencia del rechazo: con confianza menor que `MIN_CONFIDENCE_PCT`, el resumen
  empieza con `OUT_OF_SCOPE_PREFIX` y la estimación es una sola fase
  `UNESTIMATED_PHASE` en cero; con confianza suficiente, todo es positivo.
- Las horas tienen que caber en el equipo (WU9 B, guardrail de salida): ninguna
  fase, ni el total, puede pasar de `MAX_HOURS_PER_PERSON_WEEK` por persona y
  semana con todo el equipo trabajando (`_hours_fit_team`). Es un error y se
  re-pregunta: unas horas que el equipo no puede trabajar no son una estimación.
- Tres comprobaciones de alucinación de bajo nivel (WU9 B, paso 3), todas locales y
  todas errores que se re-preguntan: el coste por hora de cada fase tiene que caer
  entre las tarifas de los prompts (`_cost_per_hour_in_rates`), la duración total no
  puede ser mayor que la suma de las fases ni menor que la más larga
  (`_duration_is_coherent`), y las cifras con unidad que cita un resumen (horas, EUR,
  semanas) tienen que ser las de los campos (`_summary_figures_exist`). Las tres
  dejan en el mensaje el valor que esperaban, para que el modelo pueda corregirlo.
- La coherencia de los totales depende del tamaño del error. Los totales los
  declara el modelo (PLAN §3.4). Una diferencia de hasta
  `TOTALS_MAX_DEVIATION_PCT` con la suma de las fases se devuelve igual, con un
  aviso para el usuario (`total_discrepancies`): re-preguntar costaría una
  llamada para corregir un redondeo que el usuario puede ver. Una diferencia
  mayor es un error y se re-pregunta (`_totals_close_to_phases`): ese total ya
  no describe la estimación.

El rechazo («Fuera de alcance:») viene de la solución de referencia
`session_4_live`: le da al modelo una forma estructurada de decir que la
descripción no alcanza para estimar, en lugar de inventar números.

`phases` va primero a propósito: WU0 midió que el modelo emite los campos en el
orden del schema, y con `phases` adelante las filas empiezan a llegar antes
(PLAN §7, reserva 1). Además, el modelo decide cada fase antes de declarar los
totales, y el resumen y la confianza llegan al final, cuando ya estimó.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

# Por debajo de esta confianza (en %), la estimación tiene que rechazarse.
MIN_CONFIDENCE_PCT = 30
OUT_OF_SCOPE_PREFIX = "Fuera de alcance:"
# La única fase de una estimación rechazada, con todo en cero.
UNESTIMATED_PHASE = "Sin estimar"
# Desvío máximo (en %) entre un total declarado (horas o coste) y la suma de las
# fases. Hasta este valor es un aviso; por encima, un error que se re-pregunta.
TOTALS_MAX_DEVIATION_PCT = 5
# Tope de horas por persona y semana de calendario que admite el guardrail de
# salida. Las horas productivas de los prompts son 32; el tope deja margen de
# redondeo (las horas se redondean hacia arriba) y no marca una semana larga.
MAX_HOURS_PER_PERSON_WEEK = 40
# Tarifa horaria mínima y máxima de los prompts (app/prompts/estimation/*/examples.yaml;
# test_structured_schema.py vigila que sigan siéndolo). El coste de una fase mezcla
# roles, así que su coste por hora cae entre las dos. El margen cubre el redondeo
# hacia arriba de horas y costes que piden los prompts.
MIN_HOURLY_RATE_EUR = 50.0
MAX_HOURLY_RATE_EUR = 62.5
RATE_TOLERANCE_PCT = 5
# Diferencia máxima, en semanas, entre una duración citada en un resumen y la de los campos.
SUMMARY_WEEKS_TOLERANCE = 0.5


class Risk(BaseModel):
    """Un riesgo de la fase y cómo mitigarlo."""

    risk: str = Field(min_length=1, max_length=300, description="Riesgo concreto de la fase.")
    mitigation: str = Field(
        min_length=1, max_length=300, description="Cómo se reduce o se contiene."
    )


class Phase(BaseModel):
    """Una fase del proyecto, con su descripción, duración, esfuerzo y coste.

    Los mínimos son 0 para admitir la fase `UNESTIMATED_PHASE` de una estimación
    rechazada; que una fase estimada tenga todo positivo lo exige
    `StructuredResult`.
    """

    name: str = Field(min_length=1, max_length=64, description="Nombre de la fase, en castellano.")
    summary: str = Field(
        min_length=10, max_length=600, description="Qué se hace en la fase, en una o dos frases."
    )
    duration_weeks: float = Field(ge=0, le=52, description="Duración en semanas de calendario.")
    hours: int = Field(ge=0, le=10_000, description="Horas de equipo de la fase.")
    cost_eur: float = Field(ge=0, le=1_000_000, description="Coste de la fase en EUR.")
    confidence_pct: int = Field(ge=0, le=100, description="Confianza en la fase, de 0 a 100.")
    assumptions: list[str] = Field(
        max_length=6,
        description="Supuestos de la fase. Lista vacía si el nivel de detalle no los pide.",
    )
    risks: list[Risk] = Field(
        max_length=6,
        description="Riesgos de la fase con su mitigación. Lista vacía si no se piden.",
    )


class TeamMember(BaseModel):
    """Un rol del equipo y cuántas personas lo cubren."""

    role: str = Field(
        min_length=1, max_length=64, description="Rol, en castellano (por ejemplo, Desarrollador)."
    )
    headcount: int = Field(gt=0, le=50, description="Cantidad de personas con ese rol.")


class Totals(BaseModel):
    """Totales del proyecto tal como los declara el modelo."""

    hours: int = Field(
        ge=0, le=50_000, description="Horas totales: suma de las horas de las fases."
    )
    cost_eur: float = Field(
        ge=0, le=2_000_000, description="Coste total en EUR: suma de los costes de las fases."
    )
    duration_weeks: float = Field(
        ge=0, le=104, description="Duración total en semanas de calendario."
    )


class StructuredResult(BaseModel):
    """Estimación completa devuelta por el modelo."""

    phases: list[Phase] = Field(
        min_length=1, max_length=8, description="Fases del proyecto, en orden."
    )
    team: list[TeamMember] = Field(
        max_length=10,
        description="Composición del equipo. Vacía solo si la estimación se rechaza.",
    )
    totals: Totals
    summary: str = Field(
        min_length=10,
        max_length=1200,
        description=(
            "Resumen de la estimación en un párrafo. Si se rechaza, empieza con "
            f"«{OUT_OF_SCOPE_PREFIX}» y explica qué falta."
        ),
    )
    confidence_pct: int = Field(
        ge=0, le=100, description="Confianza global en la estimación, de 0 a 100."
    )

    @field_validator("phases")
    @classmethod
    def _unique_phase_names(cls, phases: list[Phase]) -> list[Phase]:
        """Dos fases con el mismo nombre suelen ser una fase partida o duplicada."""
        seen: set[str] = set()
        repeated = []
        for phase in phases:
            key = phase.name.strip().casefold()
            if key in seen:
                repeated.append(phase.name)
            seen.add(key)
        if repeated:
            raise ValueError(
                f"Los nombres de fase deben ser únicos; se repite: {', '.join(repeated)}."
            )
        return phases

    @property
    def out_of_scope(self) -> bool:
        """True si el modelo rechazó estimar porque la descripción no alcanza."""
        return self.summary.startswith(OUT_OF_SCOPE_PREFIX)

    @model_validator(mode="after")
    def _confidence_matches_refusal(self) -> StructuredResult:
        """Confianza baja exige rechazar, y un rechazo exige confianza baja.

        Los mensajes se escriben para el modelo: Instructor se los devuelve tal
        cual cuando re-pregunta.
        """
        if self.confidence_pct < MIN_CONFIDENCE_PCT and not self.out_of_scope:
            raise ValueError(
                f"Con confidence_pct menor que {MIN_CONFIDENCE_PCT}, summary tiene que empezar "
                f"con «{OUT_OF_SCOPE_PREFIX}» y explicar qué falta para estimar."
            )
        if self.out_of_scope and self.confidence_pct >= MIN_CONFIDENCE_PCT:
            raise ValueError(
                f"Un summary que empieza con «{OUT_OF_SCOPE_PREFIX}» exige "
                f"confidence_pct menor que {MIN_CONFIDENCE_PCT}. Si se puede estimar, "
                "quita el prefijo."
            )
        return self

    @model_validator(mode="after")
    def _shape_matches_outcome(self) -> StructuredResult:
        """Una estimación rechazada va en cero; una estimada, con todo positivo."""
        if self.out_of_scope:
            [phase, *rest] = self.phases
            zeroed = (
                not rest
                and phase.name == UNESTIMATED_PHASE
                and phase.duration_weeks == phase.hours == phase.cost_eur == 0
                and not self.team
                and self.totals.duration_weeks == self.totals.hours == self.totals.cost_eur == 0
            )
            if not zeroed:
                raise ValueError(
                    "Una estimación fuera de alcance lleva una sola fase "
                    f"«{UNESTIMATED_PHASE}» con duration_weeks, hours y cost_eur en 0, "
                    "team vacío y todos los totales en 0."
                )
            return self
        zeroed = [
            f"{f.name} ({', '.join(missing)})"
            for f in self.phases
            if (missing := _zero_fields(f.duration_weeks, f.hours, f.cost_eur))
        ]
        if zeroed:
            raise ValueError(
                "Cada fase estimada necesita duration_weeks, hours y cost_eur mayores que 0; "
                f"no los tiene: {', '.join(zeroed)}."
            )
        if not self.team:
            raise ValueError("Una estimación necesita al menos un rol en team.")
        if missing := _zero_fields(
            self.totals.duration_weeks, self.totals.hours, self.totals.cost_eur
        ):
            raise ValueError(
                "Los totales de una estimación necesitan duration_weeks, hours y cost_eur "
                f"mayores que 0; en cero: {', '.join(f'totals.{name}' for name in missing)}. "
                "duration_weeks es la duración de calendario en semanas (suma de las fases "
                "si van una detrás de otra), no puede ser 0."
            )
        return self

    # --- Output guardrail: hours vs team -----------------------------------------
    @model_validator(mode="after")
    def _hours_fit_team(self) -> StructuredResult:
        """Las horas no pueden superar lo que el equipo trabaja en las semanas dadas.

        `team` es el equipo de todo el proyecto, así que su tamaño es un tope para
        cualquier fase (una fase puede usar solo a parte del equipo) y para el
        total (las fases pueden solaparse, pero no suman más gente que el equipo).
        Una estimación rechazada va toda en cero (`_shape_matches_outcome`, que corre
        antes) y no tiene horas que comparar.
        """
        people = sum(member.headcount for member in self.team)
        over = [
            f"{label} ({format_number(hours)} h en {format_number(weeks)} semanas)"
            for label, hours, weeks in (
                *((p.name, p.hours, p.duration_weeks) for p in self.phases),
                ("totals", self.totals.hours, self.totals.duration_weeks),
            )
            if hours > weeks * people * MAX_HOURS_PER_PERSON_WEEK
        ]
        if over:
            raise ValueError(
                f"Las horas no caben en el equipo ({people} personas, como mucho "
                f"{MAX_HOURS_PER_PERSON_WEEK} h por persona y semana): {'; '.join(over)}. "
                "Corrige las horas, las semanas o el equipo."
            )
        return self

    # --- Totals tolerance -------------------------------------------------------
    @model_validator(mode="after")
    def _totals_close_to_phases(self) -> StructuredResult:
        """Un total que se aleja más de TOTALS_MAX_DEVIATION_PCT de la suma es un error.

        Se mide contra la suma de las fases, que en una estimación es siempre
        positiva (lo exige `_shape_matches_outcome`). Una suma en cero (el
        rechazo, todo en cero) no se compara: no hay base para un porcentaje.
        La duración no se compara: las fases pueden solaparse. Las diferencias
        menores las informa `total_discrepancies`.
        """
        deviated = []
        for label, declared, total in (
            ("totals.hours", self.totals.hours, sum(f.hours for f in self.phases)),
            ("totals.cost_eur", self.totals.cost_eur, sum(f.cost_eur for f in self.phases)),
        ):
            if total and abs(declared - total) * 100 > TOTALS_MAX_DEVIATION_PCT * total:
                deviated.append(
                    f"{label} ({format_number(declared)}) se aleja más de "
                    f"{TOTALS_MAX_DEVIATION_PCT} % de la suma de las fases ({format_number(total)})"
                )
        if deviated:
            raise ValueError("; ".join(deviated) + ". Calcula cada total sumando las fases.")
        return self

    # --- Output guardrails: low-level hallucination checks ----------------------
    @model_validator(mode="after")
    def _cost_per_hour_in_rates(self) -> StructuredResult:
        """El coste por hora de cada fase tiene que caer entre las tarifas del prompt.

        Una fase mezcla roles, así que su coste por hora es un promedio entre la
        tarifa mínima y la máxima. Fuera de ese rango (con `RATE_TOLERANCE_PCT` de
        margen por el redondeo) el modelo inventó el coste o las horas.
        """
        if self.out_of_scope:
            return self
        low = MIN_HOURLY_RATE_EUR * (1 - RATE_TOLERANCE_PCT / 100)
        high = MAX_HOURLY_RATE_EUR * (1 + RATE_TOLERANCE_PCT / 100)
        off = [
            f"{p.name} ({format_number(p.cost_eur)} EUR en {format_number(p.hours)} h son "
            f"{format_number(p.cost_eur / p.hours)} EUR/h)"
            for p in self.phases
            if not low <= p.cost_eur / p.hours <= high
        ]
        if off:
            raise ValueError(
                "El coste por hora queda fuera de las tarifas (entre "
                f"{format_number(MIN_HOURLY_RATE_EUR)} y {format_number(MAX_HOURLY_RATE_EUR)} "
                f"EUR/h): {'; '.join(off)}. Recalcula el coste con las tarifas de <scope>."
            )
        return self

    @model_validator(mode="after")
    def _duration_is_coherent(self) -> StructuredResult:
        """La duración total está entre la fase más larga y la suma de las fases.

        Las fases pueden solaparse (la duración es menor que la suma) pero no más
        que hasta que la más larga las contiene, y no pueden dejar huecos que el
        modelo no declaró (mayor que la suma). Se admite `TOTALS_MAX_DEVIATION_PCT`
        de margen, como en los totales.
        """
        if self.out_of_scope:
            return self
        total = self.totals.duration_weeks
        summed = sum(p.duration_weeks for p in self.phases)
        longest = max(p.duration_weeks for p in self.phases)
        margin = TOTALS_MAX_DEVIATION_PCT / 100
        if total > summed * (1 + margin):
            raise ValueError(
                f"totals.duration_weeks ({format_number(total)}) es mayor que la suma de las "
                f"fases ({format_number(summed)}): solo puede igualarla si van una detrás de otra."
            )
        if total < longest * (1 - margin):
            raise ValueError(
                f"totals.duration_weeks ({format_number(total)}) es menor que la fase más larga "
                f"({format_number(longest)}): el proyecto dura al menos lo que su fase más larga."
            )
        return self

    @model_validator(mode="after")
    def _summary_figures_exist(self) -> StructuredResult:
        """Las cifras con unidad que cita un resumen tienen que estar en los campos.

        Solo mira horas, EUR y semanas pegadas a su unidad («390 horas», «23.950 EUR»,
        «6 semanas»); una cifra sin unidad o con otra («3 sprints», «15 %») no se
        toca. Se compara con las horas, los costes y las duraciones de las fases y del
        total, con `TOTALS_MAX_DEVIATION_PCT` de margen (`SUMMARY_WEEKS_TOLERANCE` en
        semanas).
        """
        if self.out_of_scope:
            return self
        known = {
            "h": [*(p.hours for p in self.phases), self.totals.hours],
            "eur": [*(p.cost_eur for p in self.phases), self.totals.cost_eur],
            "w": [*(p.duration_weeks for p in self.phases), self.totals.duration_weeks],
        }
        known["w"].append(sum(p.duration_weeks for p in self.phases))
        invented = [
            quote
            for text in (self.summary, *(p.summary for p in self.phases))
            for quote in _unknown_figures(text, known)
        ]
        if invented:
            raise ValueError(
                "Un resumen cita cifras que no están en la estimación: "
                f"{', '.join(dict.fromkeys(invented))}. Cita solo horas, costes y semanas de "
                "las fases o de los totales, o no cites cifras."
            )
        return self

    def total_discrepancies(self) -> list[str]:
        """Avisos para el usuario si los totales no coinciden con la suma de las fases.

        Solo llegan acá diferencias de hasta TOTALS_MAX_DEVIATION_PCT: las
        mayores ya las rechazó `_totals_close_to_phases`. Lista vacía si cuadran. Horas y coste se comparan exactos: el prompt pide
        sumar, y el redondeo ya está aplicado en cada fase. La duración no se
        compara, porque dos fases pueden solaparse en el calendario.
        """
        warnings = []
        hours = sum(f.hours for f in self.phases)
        if self.totals.hours != hours:
            warnings.append(
                f"El total de horas declarado ({format_number(self.totals.hours)}) no coincide con "
                f"la suma de las fases ({format_number(hours)})."
            )
        cost = sum(f.cost_eur for f in self.phases)
        if abs(self.totals.cost_eur - cost) >= 0.005:
            warnings.append(
                f"El coste total declarado ({format_number(self.totals.cost_eur)} EUR) no coincide "
                f"con la suma de las fases ({format_number(cost)} EUR)."
            )
        return warnings


# Un número en castellano («29.850», «1.250,50», «6», «1,5», «1.5») pegado a su unidad.
# No cuenta si sigue «por», «/» o «semanales»: «32 h por persona», «62,50 EUR/hora».
_FIGURE = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)\s*"
    r"(horas|h|eur|euros|€|semanas|semana)(?!\w)(?!\s*(?:/|por\b|semanales))",
    re.IGNORECASE,
)
_UNIT_KIND = {
    "horas": "h",
    "h": "h",
    "eur": "eur",
    "euros": "eur",
    "€": "eur",
    "semanas": "w",
    "semana": "w",
}


def _parse_number(raw: str) -> float:
    """'29.850' -> 29850; '1.250,50' -> 1250.5; '1,5' y '1.5' -> 1.5."""
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", raw):
        raw = raw.replace(".", "")
    return float(raw.replace(",", "."))


def _unknown_figures(text: str, known: dict[str, list[float]]) -> list[str]:
    """Las cifras con unidad de `text` que no se parecen a ninguna de `known[unidad]`."""
    unknown = []
    for match in _FIGURE.finditer(text):
        kind = _UNIT_KIND[match.group(2).casefold()]
        value = _parse_number(match.group(1))
        if kind == "w":
            close = any(abs(value - k) <= SUMMARY_WEEKS_TOLERANCE for k in known[kind])
        else:
            close = any(abs(value - k) * 100 <= TOTALS_MAX_DEVIATION_PCT * k for k in known[kind])
        if not close:
            unknown.append(match.group(0).strip())
    return unknown


def _zero_fields(duration_weeks: float, hours: float, cost_eur: float) -> list[str]:
    """Los campos en cero de entre duration_weeks, hours y cost_eur, en ese orden."""
    return [
        name
        for name, value in (
            ("duration_weeks", duration_weeks),
            ("hours", hours),
            ("cost_eur", cost_eur),
        )
        if not value
    ]


def format_number(n: float) -> str:
    """29850 -> '29.850'; 62.5 -> '62,50'. Formato castellano de las cifras.

    Lo usan los avisos de totales y, en `v4`, los ejemplos del prompt y el
    validador de `rendered`: los tres escriben los números igual.
    """
    if float(n).is_integer():
        return f"{int(n):,}".replace(",", ".")
    return f"{n:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


class StructuredEstimationResponse(BaseModel):
    """Respuesta de `POST /api/v1/estimate/structured`."""

    estimation: StructuredResult = Field(description="Estimación validada contra el schema.")
    prompt_version: str = Field(description="Identificador de la plantilla de prompt utilizada.")
    cached: bool = Field(
        default=False,
        description="True si la estimación salió de la caché: no hubo llamada al proveedor.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Avisos para el usuario (respaldo no disponible, totales que no cuadran). "
            "Vacío si no hay ninguno."
        ),
    )


# --- Rendered response ------------------------------------------------------
# `v4`: además de las cifras, el modelo escribe la presentación que eligió el
# usuario (`output_format`) en `rendered`. El validador comprueba la forma pedida
# y que las cifras del texto sean las de los campos: si no, Instructor re-pregunta
# y, agotados los intentos, el endpoint responde 502.

# Líneas de tabla Markdown, fila separadora e ítems de lista.
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?(\s*:?-{3,}:?\s*\|)+\s*:?-{0,}:?\s*\|?\s*$")
_NUMBERED_ITEM = re.compile(r"^\s*\d+[.)]\s+\S")
_BULLET_ITEM = re.compile(r"^\s*[-*+]\s+\S")

# Formatos que acepta `rendered`: los valores de `OutputFormat`.
_RENDERED_FORMATS = ("phases_table", "line_items", "narrative")


def _number_spellings(n: float) -> list[str]:
    """Formas válidas de escribir una cifra en `rendered`: 1.320, o 1320 sin separador."""
    spellings = [format_number(n)]
    if float(n).is_integer():
        spellings.append(str(int(n)))
    return spellings


def _mentions(text: str, n: float) -> bool:
    """True si `text` contiene la cifra `n` como número entero, no dentro de otro.

    `130` no cuenta como mención si aparece dentro de `1.130` o de `130,5`.
    """
    for spelling in _number_spellings(n):
        pattern = rf"(?<![\d.,]){re.escape(spelling)}(?![\d]|[.,]\d)"
        if re.search(pattern, text):
            return True
    return False


class RenderedResult(StructuredResult):
    """Estimación de `v4`: la de `v3` más su presentación en Markdown.

    `rendered` va al final del schema: el modelo decide las cifras antes de
    escribir el texto que las muestra.

    El validador necesita el `output_format` de la request, que no es parte de la
    respuesta: lo recibe en el contexto de validación (`context={"output_format":
    ...}`). Sin contexto solo se validan los campos, como en `v3`; el servicio
    siempre lo pasa.
    """

    rendered: str = Field(
        min_length=10,
        max_length=6000,
        description=(
            "La estimación presentada en Markdown con el formato que pide el prompt "
            "(tabla, partidas numeradas o prosa), con las mismas cifras que los campos."
        ),
    )

    @model_validator(mode="after")
    def _rendered_matches_format(self, info: ValidationInfo) -> RenderedResult:
        """Comprueba que el texto presentado (`rendered`) coincide con los datos.

        Verifica que tenga la forma del formato pedido (tabla, lista numerada o
        párrafos), que cada fase aparezca con sus mismas horas y coste, y que se
        muestren los totales. Si no cuadra, lanza un error y el modelo debe corregirlo.
        """
        output_format = (info.context or {}).get("output_format")
        if output_format is None:
            return self
        if output_format not in _RENDERED_FORMATS:
            # Error de programación, no del modelo: no se re-pregunta por esto.
            raise TypeError(f"output_format '{output_format}' no es un formato de rendered.")

        lines = self.rendered.splitlines()
        if self.out_of_scope:
            if not self.rendered.lstrip().startswith(OUT_OF_SCOPE_PREFIX):
                raise ValueError(
                    f"En una estimación fuera de alcance, rendered empieza con "
                    f"«{OUT_OF_SCOPE_PREFIX}» y explica qué falta, igual que summary."
                )
            if any(_TABLE_ROW.match(x) or _NUMBERED_ITEM.match(x) for x in lines):
                raise ValueError(
                    "En una estimación fuera de alcance, rendered es solo texto: sin tabla ni lista."
                )
            return self

        if output_format == "phases_table":
            blocks = self._table_rows(lines)
        elif output_format == "line_items":
            blocks = self._numbered_items(lines)
        else:
            blocks = self._paragraphs(lines)

        missing = [
            p.name
            for p in self.phases
            if not any(
                p.name.casefold() in b.casefold()
                and _mentions(b, p.hours)
                and _mentions(b, p.cost_eur)
                for b in blocks
            )
        ]
        if missing:
            where = {
                "phases_table": "una fila de la tabla",
                "line_items": "una partida numerada",
                "narrative": "un párrafo",
            }[output_format]
            raise ValueError(
                f"En rendered, cada fase necesita {where} con su nombre, sus horas y su "
                f"coste, con las mismas cifras que phases; falta o no coincide: "
                f"{', '.join(missing)}."
            )
        if not (
            _mentions(self.rendered, self.totals.hours)
            and _mentions(self.rendered, self.totals.cost_eur)
        ):
            raise ValueError(
                "rendered tiene que mostrar los totales de totals: "
                f"{format_number(self.totals.hours)} horas y "
                f"{format_number(self.totals.cost_eur)} EUR."
            )
        return self

    @staticmethod
    def _table_rows(lines: list[str]) -> list[str]:
        """Extrae las filas de fase de una tabla Markdown (descarta cabecera y separador).

        Lanza un error si no hay tabla con cabecera, separador y al menos una fila.
        """
        rows = [x for x in lines if _TABLE_ROW.match(x)]
        if len(rows) < 3 or not any(_TABLE_SEPARATOR.match(x) for x in rows[1:2]):
            raise ValueError(
                "Con output_format phases_table, rendered es una tabla Markdown: fila de "
                "cabecera, fila separadora (|---|) y una fila por fase."
            )
        return rows[2:]

    @staticmethod
    def _numbered_items(lines: list[str]) -> list[str]:
        """Extrae las partidas de una lista numerada (1., 2., …). Lanza un error si no hay
        ninguna.
        """
        items = [x for x in lines if _NUMBERED_ITEM.match(x)]
        if not items:
            raise ValueError(
                "Con output_format line_items, rendered es una lista numerada (1., 2., …) "
                "con una partida por fase."
            )
        return items

    @staticmethod
    def _paragraphs(lines: list[str]) -> list[str]:
        """Divide el texto en párrafos. Lanza un error si contiene tablas o listas, que no
        se permiten en formato narrativo.
        """
        if any(
            _TABLE_ROW.match(x) or _NUMBERED_ITEM.match(x) or _BULLET_ITEM.match(x) for x in lines
        ):
            raise ValueError(
                "Con output_format narrative, rendered es prosa en párrafos: sin tablas ni listas."
            )
        return [p for p in "\n".join(lines).split("\n\n") if p.strip()]


class RenderedEstimationResponse(StructuredEstimationResponse):
    """Respuesta de `POST /api/v1/estimate/rendered`."""

    estimation: RenderedResult = Field(
        description="Estimación validada contra el schema, con su presentación en rendered."
    )
