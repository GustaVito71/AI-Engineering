"""Estado de una sesión conversacional: historial, `project_metadata` y la sesión.

POR QUÉ EL ESTADO ES VOLÁTIL

Las sesiones viven en un diccionario en la memoria del proceso (ver `SessionStore`):
sin base de datos ni Redis. Se pierde al reiniciar el servicio, y solo funciona con
un worker (`uvicorn --workers=1`): con varios, cada worker tendría su propio
diccionario y una sesión "desaparecería" según qué worker atienda la petición. Se
acepta en esta fase porque lo que se quiere probar es la memoria conversacional (la
ventana y la metadata), no su persistencia, y un almacén externo añadiría una pieza
operativa sin cambiar nada de lo que se evalúa. Esta restricción vale solo para las
sesiones: la caché de respuestas de lidr_4 sigue en Redis.

EL SYSTEM PROMPT NO SE GUARDA

El system prompt se regenera en cada turno a partir del `ProjectMetadata` actual,
así que no es un mensaje del historial: es un invariante que `to_messages_list`
antepone siempre, y la ventana nunca puede descartarlo.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

# Cuántos turnos recuerda una sesión si nadie configura otra cosa (variable MAX_TURNS).
# Un "turno" es un par: lo que escribió el usuario y lo que respondió el asistente.
DEFAULT_MAX_TURNS = 6

# Quién dijo un mensaje: "user" (el usuario) o "assistant" (el modelo). Son los dos
# únicos valores que acepta la API de los proveedores de LLM dentro del historial.
Role = Literal["user", "assistant"]


def _now() -> datetime:
    """Fecha y hora actuales en UTC (hora universal, sin depender de la zona horaria)."""
    return datetime.now(UTC)


class Message(BaseModel):
    """Un mensaje del historial (usuario o asistente).

    Es la unidad más pequeña de la conversación: quién habló (`role`), qué dijo
    (`content`) y cuándo (`created_at`).
    """

    role: Role
    content: str
    created_at: datetime = Field(default_factory=_now)


# Marca con la que se presenta el resumen al modelo, y la respuesta de cortesía que lo
# cierra para que los roles sigan alternando (usuario, asistente, usuario...).
SUMMARY_PREFIX = (
    "[Resumen de la conversación anterior: los mensajes que siguen son el hilo reciente]\n"
)
SUMMARY_ACK = "Entendido. Continúo con ese contexto."


class ConversationHistory(BaseModel):
    """Memoria de la conversación: ventana reciente, anclas y resumen de lo antiguo.

    Tres capas, de lo más antiguo a lo más reciente (así llegan al modelo):
    - `summary`: texto acumulativo con lo que ya salió de la ventana (Paso 9).
    - `anchors`: pares con compromisos (alcance cerrado, presupuesto fijo, NDA...) que
      se conservan TEXTUALES y no caducan (Paso 9).
    - `messages`: la ventana deslizante, los últimos `max_turns` pares.

    `max_turns` cuenta pares. La ventana se aplica en cada `append`, así que `messages`
    nunca tiene más de `max_turns` pares. Los pares que salen se DEVUELVEN, no se
    pierden: la política de compresión decide si son ancla o van al resumen.
    """

    # `ge=1` ("mayor o igual que 1"): una ventana de 0 turnos no recordaría nada.
    max_turns: int = Field(default=DEFAULT_MAX_TURNS, ge=1)
    messages: list[Message] = Field(default_factory=list)
    anchors: list[Message] = Field(default_factory=list)
    summary: str | None = Field(default=None)

    def append(self, *, user: str, assistant: str) -> list[Message]:
        """Agrega un turno (mensaje del usuario y respuesta) y aplica la ventana.

        En palabras simples: si ya hay más turnos de los permitidos, salen los más
        viejos. Cada turno son 2 mensajes, por eso el límite se multiplica por 2.
        Devuelve los mensajes que salieron de la ventana (vacía si no salió ninguno), en
        orden y de a pares, para que se puedan comprimir en vez de olvidarse.
        """
        self.messages.append(Message(role="user", content=user))
        self.messages.append(Message(role="assistant", content=assistant))
        # `overflow` = cuántos mensajes sobran. Si es positivo, se quitan del principio
        # de la lista (los más antiguos) hasta quedar dentro del límite.
        overflow = len(self.messages) - self.max_turns * 2
        if overflow <= 0:
            return []
        evicted = self.messages[:overflow]
        del self.messages[:overflow]
        return evicted

    def to_history_messages(self) -> list[dict[str, str]]:
        """Lo que se le pasa al modelo como historial (sin system prompt): resumen, anclas
        y ventana, en ese orden.

        El servicio ya renderiza su propio system prompt y lo antepone, así que acá no
        hace falta incluirlo. El resumen va como un par usuario + asistente de cortesía
        para que los roles sigan alternando sea cual sea el proveedor.
        """
        out: list[dict[str, str]] = []
        if self.summary:
            out.append({"role": "user", "content": SUMMARY_PREFIX + self.summary})
            out.append({"role": "assistant", "content": SUMMARY_ACK})
        out.extend({"role": m.role, "content": m.content} for m in self.anchors)
        out.extend({"role": m.role, "content": m.content} for m in self.messages)
        return out

    def to_messages_list(self, system_prompt: str) -> list[dict[str, str]]:
        """El array `messages` listo para el LLM: el system prompt y luego el historial.

        El system prompt llega ya renderizado desde la metadata actual (Paso 4); acá
        solo se antepone, porque el historial no lo guarda.
        """
        return [{"role": "system", "content": system_prompt}, *self.to_history_messages()]


# Topes de la metadata. Se inyecta en el system prompt de CADA turno, así que sin
# límites un extractor descontrolado (o un usuario que provoca listas enormes)
# encarecería todas las llamadas de la sesión.
MAX_LIST_ITEMS = 30
MAX_ITEM_CHARS = 200


def _clean_list(items: list[str]) -> list[str]:
    """Limpia una lista de textos: sin espacios sobrantes, vacíos ni repetidos.

    Los repetidos se detectan sin distinguir mayúsculas y se conserva la primera
    aparición con el orden original. Cada texto se recorta a `MAX_ITEM_CHARS` y la lista
    a `MAX_LIST_ITEMS` elementos.
    """
    clean: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = " ".join(item.split())[:MAX_ITEM_CHARS]
        if text and text.lower() not in seen:
            clean.append(text)
            seen.add(text.lower())
    return clean[:MAX_LIST_ITEMS]


class ProjectMetadata(BaseModel):
    """Hechos del proyecto en curso, aparte del historial.

    Todos los campos son opcionales: en la primera llamada de la sesión todavía no
    se sabe nada, y una metadata vacía es un estado válido (el bloque del prompt va
    vacío). El extractor (Paso 4) la va actualizando turno a turno.

    Cuatro campos vienen del ENUNCIADO (nombre, equipo, tecnologías y alcance) y dos
    se agregaron para poder "olvidar" con criterio: `explicit_constraints` (límites
    que impuso el usuario: presupuesto, plazo, normativa) y `rejected_options` (lo
    que el usuario descartó, para que el modelo no lo vuelva a proponer).

    Un séptimo campo, `other_facts`, recoge los hechos relevantes para estimar que no
    encajan en ninguno de los anteriores (el cliente, un sistema existente con el que
    hay que convivir, el público objetivo, el idioma). Es una lista libre y con tope,
    para no tener que prever de antemano cada tipo de dato.
    """

    project_name: str | None = Field(default=None, max_length=120)
    assumed_team_size: int | None = Field(default=None, ge=1, le=50)
    mentioned_technologies: list[str] = Field(default_factory=list)
    # Tope: esta metadata se inyecta en el system prompt de cada turno, así que un
    # alcance sin límite encarecería todas las llamadas de la sesión.
    agreed_scope: str | None = Field(default=None, max_length=2000)
    explicit_constraints: list[str] = Field(default_factory=list)
    rejected_options: list[str] = Field(default_factory=list)
    other_facts: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        """True si todavía no se conoce ningún hecho del proyecto (primer turno)."""
        return (
            self.project_name is None
            and self.assumed_team_size is None
            and not self.mentioned_technologies
            and self.agreed_scope is None
            and not self.explicit_constraints
            and not self.rejected_options
            and not self.other_facts
        )

    def normalized(self) -> ProjectMetadata:
        """Copia limpia y coherente, lista para guardar.

        Aplica la política de olvido nº 1 (revisión explícita del usuario) de forma
        determinista, sin depender de que el modelo lo haga bien: una tecnología que
        figura entre las opciones rechazadas se quita de las mencionadas. Así, si el
        usuario dice "ya no Rails, vamos con Node", Rails no puede quedar a la vez como
        tecnología vigente y como descartada. También recorta las listas a sus topes y
        elimina repetidos y textos vacíos.
        """
        rejected = _clean_list(self.rejected_options)
        rejected_keys = {item.lower() for item in rejected}
        technologies = [
            t for t in _clean_list(self.mentioned_technologies) if t.lower() not in rejected_keys
        ]
        return ProjectMetadata(
            project_name=(self.project_name or "").strip() or None,
            assumed_team_size=self.assumed_team_size,
            mentioned_technologies=technologies,
            agreed_scope=(self.agreed_scope or "").strip() or None,
            explicit_constraints=_clean_list(self.explicit_constraints),
            rejected_options=rejected,
            other_facts=_clean_list(self.other_facts),
        )

    def merge_with(self, update: ProjectMetadata) -> ProjectMetadata:
        """Metadata nueva: los campos no nulos de `update` ganan, y las listas se unen.

        Los escalares se sobrescriben porque el extractor puede precisar el nombre o
        el equipo a medida que la conversación los aclara. Las listas se acumulan
        (sin distinguir mayúsculas, conservando el orden): agregar una tecnología al
        stack no debe hacer desaparecer las anteriores. Devuelve un objeto nuevo y no
        modifica ninguno de los dos.

        Esta unión solo AGREGA. Para quitar o mover un hecho (por ejemplo una
        tecnología que el usuario descartó) el extractor devuelve la metadata completa
        ya revisada y la sesión la guarda tal cual, pasándola por `normalized`.
        """
        return ProjectMetadata(
            project_name=update.project_name or self.project_name,
            assumed_team_size=update.assumed_team_size or self.assumed_team_size,
            mentioned_technologies=[*self.mentioned_technologies, *update.mentioned_technologies],
            agreed_scope=update.agreed_scope or self.agreed_scope,
            explicit_constraints=[*self.explicit_constraints, *update.explicit_constraints],
            rejected_options=[*self.rejected_options, *update.rejected_options],
            other_facts=[*self.other_facts, *update.other_facts],
        ).normalized()


# POR QUÉ EXISTE `ExtractedMetadata` (comentario y no docstring a propósito: el docstring de
# un modelo Pydantic viaja al proveedor dentro del esquema, y esto es para quien mantiene
# el código, no para el modelo extractor).
#
# `ProjectMetadata` es el modelo interno: sus campos son opcionales (`None`) y llevan
# límites. Ese esquema, enviado tal cual a un proveedor, resultó frágil: con Gemini la
# generación estructurada se rompió y el modelo volcó los nombres de campo y sus valores
# dentro de `rejected_options` (verificado con `test/verificar_extractor.py`). Por eso al
# modelo se le pide esta forma, la más simple posible: sin campos opcionales (los `anyOf`
# con `null`), sin límites numéricos ni de largo y con TODOS los campos obligatorios, así
# que siempre devuelve la memoria completa. «No se conoce» se escribe con una cadena
# vacía, un 0 o una lista vacía. Los límites (equipo de 1 a 50, largos máximos) se aplican
# después, en `to_project_metadata`.
class ExtractedMetadata(BaseModel):
    """Los hechos del proyecto, completos, tal como quedan después de este turno."""

    project_name: str = Field(description="Nombre del proyecto; cadena vacía si no se conoce.")
    assumed_team_size: int = Field(
        description="Personas del equipo, de todos los roles, que dijo o aceptó el usuario; 0 si no se conoce."
    )
    mentioned_technologies: list[str] = Field(
        description="Tecnologías, lenguajes o plataformas vigentes del proyecto."
    )
    agreed_scope: str = Field(
        description="Alcance acordado en pocas frases; vacío si no se conoce."
    )
    explicit_constraints: list[str] = Field(
        description="Límites que impuso el usuario: presupuesto, plazo, normativa, integraciones."
    )
    rejected_options: list[str] = Field(
        description="Opciones que el usuario descartó expresamente y no deben proponerse de nuevo."
    )
    other_facts: list[str] = Field(
        description=(
            "Otros hechos del proyecto relevantes para estimar que no encajan en los "
            "campos anteriores (cliente, sistemas existentes, público objetivo, idioma)."
        )
    )

    @classmethod
    def from_project_metadata(cls, metadata: ProjectMetadata) -> ExtractedMetadata:
        """La metadata interna en la forma plana (para mostrársela al modelo)."""
        return cls(
            project_name=metadata.project_name or "",
            assumed_team_size=metadata.assumed_team_size or 0,
            mentioned_technologies=list(metadata.mentioned_technologies),
            agreed_scope=metadata.agreed_scope or "",
            explicit_constraints=list(metadata.explicit_constraints),
            rejected_options=list(metadata.rejected_options),
            other_facts=list(metadata.other_facts),
        )

    def to_project_metadata(self) -> ProjectMetadata:
        """La metadata interna, con los límites aplicados y normalizada.

        Un equipo fuera de 1 a 50 se descarta (queda sin dato) y los textos largos se
        recortan, en vez de rechazar toda la respuesta por un campo.
        """
        team = self.assumed_team_size if 1 <= self.assumed_team_size <= 50 else None
        return ProjectMetadata(
            project_name=self.project_name.strip()[:120] or None,
            assumed_team_size=team,
            mentioned_technologies=self.mentioned_technologies,
            agreed_scope=self.agreed_scope.strip()[:2000] or None,
            explicit_constraints=self.explicit_constraints,
            rejected_options=self.rejected_options,
            other_facts=self.other_facts,
        ).normalized()


class Session(BaseModel):
    """Una sesión conversacional: su historial y su `project_metadata`.

    Junta las dos memorias de una conversación, cada una con su función:
    - `history`: los últimos mensajes tal como se dijeron (se va recortando).
    - `metadata`: los hechos del proyecto ya aclarados (no se recorta).
    """

    # UUID versión 4: un identificador aleatorio y prácticamente imposible de adivinar,
    # que además es lo único que protege el acceso a la sesión.
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    history: ConversationHistory = Field(default_factory=ConversationHistory)
    metadata: ProjectMetadata = Field(default_factory=ProjectMetadata)
    # Audiencia decidida en el último turno y la regla que la decidió (Paso 9). Solo sirve
    # para mostrarla; cada turno la vuelve a calcular.
    last_tier: str | None = None
    last_tier_rule: str | None = None
    created_at: datetime = Field(default_factory=_now)
    # Última vez que alguien usó la sesión. Sirve para que el almacén la haga caducar
    # (si pasa mucho tiempo sin uso) y para saber cuál desalojar cuando está lleno.
    last_active_at: datetime = Field(default_factory=_now)

    def touch(self, now: datetime | None = None) -> None:
        """Marca la sesión como usada ahora (renueva su plazo de caducidad)."""
        self.last_active_at = now or _now()
