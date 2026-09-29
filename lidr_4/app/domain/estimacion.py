"""El dominio: qué es una estimación y cómo se le calculan los totales.

Vive aparte de `app/schemas/`, que es lo que viaja por HTTP. La separación no es
estética: acá hay Pydantic, pero también las funciones que deciden el número
final, y esas llevan decisiones de negocio.

**Por qué los totales no son un campo del modelo.** `total_horas` y
`duracion_semanas` no vienen del LLM. Si fueran campos, el modelo podría
inventarlos —"estimo 320 horas" cuando las tareas suman 287— y no habría forma
de saber cuál de los dos números es el que se facturó. `SolicitudEstimacion` es
el `response_format` de structured output: describe lo que el modelo *debe*
producir, y no tiene forma de expresar una suma. Los totales se calculan
después, en código, y por eso son la única parte de la respuesta que no se
puede cachear mal: se derivan de las tareas, que sí.

**El `0.8` de `calcular_semanas` es una decisión de negocio**, no un ajuste
técnico: el equipo no dedica el 100% de su tiempo a la feature que se está
estimando. Está en código, no en la plantilla, justamente para que se pueda
discutir con el equipo que conoce el proyecto y testear. Si algún día cambia,
cambia acá y en el test que lo fija, no en un prompt que nadie puede versionar
con precisión.
"""

from __future__ import annotations

from enum import Enum
from math import ceil

from pydantic import BaseModel, Field, model_validator

# Horas de una semana laboral que el equipo dedica a la estimación.
# El 20% restante cubre代码 review, meetings, soporte y tiempo administrativo.
HORAS_SEMANALES = 40

# Fracción del tiempo del equipo que llega efectivamente a la feature.
# Sin esto, un equipo de 3 personas daría por hecho que 3 personas x 40 horas
# avanzan en paralelo sin ninguna fricción, y la duración saldría optimisticamente
# corta. Es un supuesto conservative a propósito: una estimación que se queda
# corta genera confianza falsa, y la que se alarga solo genera una reunión más.
FRACCION_ENFOQUE = 0.8


class Riesgo(str, Enum):
    """Nivel de riesgo de una tarea.

    El modelo lo infiere, no se lo pide al usuario: pedirle a alguien que
    cuantifique un riesgo que todavía no entiende es cambiarle la pregunta. Por
    eso es un enum cerrado —el modelo solo puede emitir estos tres— y no un
    entero del 1 al 10, que sería una precisión falsa.
    """

    BAJO = "bajo"
    MEDIO = "medio"
    ALTO = "alto"


class Tarea(BaseModel):
    """Una unidad de trabajo con su coste en horas y su dependencia de otras.

    `id` existe por `depende_de`: es lo que permite que el modelo diga "esto no
    se puede hacer hasta que lo otro esté". Sin identificador, un grafo de
    dependencias es texto libre.
    """

    id: str = Field(description='Identificador de la tarea, del tipo "T1", "T2"')
    titulo: str
    descripcion: str
    horas: int = Field(gt=0, description="Horas-persona estimadas. Estrictamente positivo.")
    depende_de: list[str] = Field(default_factory=list)
    riesgo: Riesgo


class RolEquipo(BaseModel):
    """Un rol del equipo, con su cantidad de personas."""

    rol: str
    cantidad: int = Field(gt=0)
    foco: str


class Supuesto(BaseModel):
    """Un supuesto que sostiene la estimación, con lo que pasa si es falso.

    El `impacto` es lo que convierte un supuesto decorativo en algo accionable:
    "el equipo conoce Python" no sirve; "si resulta que el legacy está en PHP,
    sumar dos semanas" sí.
    """

    texto: str
    impacto: str


class SolicitudEstimacion(BaseModel):
    """Capa 3: el `response_format` que se le pide al LLM.

    No contiene totales, y eso es deliberado — ver el docstring del módulo.
    """

    resumen: str
    tareas: list[Tarea]
    equipo: list[RolEquipo]
    supuestos: list[Supuesto]
    riesgos: list[str]
    advertencias: list[str]

    @model_validator(mode="after")
    def _validar_referencias_entre_tareas(self) -> SolicitudEstimacion:
        """Que `depende_de` apunte a tareas que existen y que no haya ciclos.

        Sin esto, el modelo puede inventar un `depende_de` apuntando a una tarea
        que no está en la lista, y nada en el sistema lo detectaría hasta que
        alguien intenta ordenar las tareas y descubre que hay un hueco. Es el
        riesgo que el plan registra de este campo.

        Se valida acá, y no en WU9 con el resto de los guardrails, porque es la
        única forma de que `calcular_semanas` pueda confiar en el orden: si una
        dependencia apunta al vacío, el grafo no se puede recorrer y la duración
        no tiene sentido.
        """
        ids = [t.id for t in self.tareas]
        duplicados = {i for i in ids if ids.count(i) > 1}
        if duplicados:
            raise ValueError(f"IDs de tarea duplicados: {sorted(duplicados)}")

        conocidas = set(ids)
        for tarea in self.tareas:
            huerfanas = [d for d in tarea.depende_de if d not in conocidas]
            if huerfanas:
                raise ValueError(
                    f"La tarea {tarea.id} depende de tareas que no existen: "
                    f"{huerfanas}. IDs válidos: {sorted(conocidas)}"
                )

        self._validar_sin_ciclos(conocidas)
        return self

    def _validar_sin_ciclos(self, conocidas: set[str]) -> None:
        """Detecta ciclos en el grafo de dependencias con recorrido en color.

        Kahn sería más rápido, pero para las decenas de tareas que produce una
        estimación el recorrido en color es más legible, y la diferencia de
        rendimiento no importa a esta escala. Un ciclo es un error de
        normalización del modelo, no una estimación válida.
        """
        # 0 = sin visitar, 1 = en la pila actual, 2 = terminado sin ciclo.
        color: dict[str, int] = {i: 0 for i in conocidas}
        adyacencia = {t.id: [d for d in t.depende_de] for t in self.tareas}

        def visitar(nodo: str, camino: list[str]) -> None:
            if color[nodo] == 1:
                raise ValueError(
                    f"Ciclo de dependencias entre tareas: {' -> '.join(camino + [nodo])}"
                )
            if color[nodo] == 2:
                return
            color[nodo] = 1
            for siguiente in adyacencia[nodo]:
                visitar(siguiente, camino + [nodo])
            color[nodo] = 2

        for tarea_id in adyacencia:
            visitar(tarea_id, [])

    @property
    def tamano_equipo(self) -> int:
        """Personas en total, sumando los roles.

        Vive como property y no como campo porque es un dato derivado: si fuera
        campo, el modelo podría emitirlo y desincronizarse de `equipo`. La
        diferencia importa —un equipo de 4 con `tamano_equipo: 2` daña la
        duración y nadie lo vería— y la regla del módulo es que los datos
        derivados se calculan en código.
        """
        return sum(rol.cantidad for rol in self.equipo)


class EstimacionCompleta(SolicitudEstimacion):
    """Lo que sale del sistema: la solicitud más los totales calculados."""

    total_horas: int
    duracion_semanas: int


def calcular_total(estimacion: SolicitudEstimacion) -> int:
    """Suma las horas de todas las tareas.

    Un `int` de horas-persona, sin decimales: las tareas traen `gt=0`, así que
    una lista vacía da 0 y una lista de cinco tareas de 8h da 40. No hay
    redondeo porque no hay división.
    """
    return sum(tarea.horas for tarea in estimacion.tareas)


def calcular_semanas(estimacion: SolicitudEstimacion) -> int:
    """Duración en semanas, a partir del total de horas y el tamaño del equipo.

    `total / (personas * 40 * 0.8)` es "cuántas semanas de capacidad hacen
    falta". El `max(1, ...)` cubre el caso de menos de una semana de trabajo:
    una feature de dos días no dura 0.05 semanas, y un `duracion_semanas: 0` en
    la respuesta sería peor que inútil, porque la UI no sabe pintar un cero.

    **Un equipo vacío se rechaza.** Sin esto la división es por cero. La
    alternativa —devolver 1— habría sido devolver un número sin sentido
    computado, que es peor que un error: es la forma exacta de una estimación
    que se ve válida en pantalla y no lo está. Con `RolEquipo.cantidad: gt=0`
    y este chequeo, una estimación sin equipo no llega nunca a calcular nada.
    """
    tamano_equipo = estimacion.tamano_equipo
    if tamano_equipo == 0:
        raise ValueError(
            "No se puede calcular la duración de un equipo vacío: se necesita al menos una persona."
        )

    capacidad_semanal = tamano_equipo * HORAS_SEMANALES * FRACCION_ENFOQUE
    return max(1, ceil(calcular_total(estimacion) / capacidad_semanal))


def completar(estimacion: SolicitudEstimacion) -> EstimacionCompleta:
    """Agrega los totales calculados a una solicitud ya validada.

    Es la frontera entre "lo que dijo el modelo" y "lo que dice el sistema". Se
    construye con `model_validate` sobre un dict en vez de copiar los campos a
    mano para que la validación de la clase base no se pueda saltar por
    accidente: si mañana se agrega un campo a `SolicitudEstimacion`, este
    `model_copy(update=...)` lo perdería en silencio y un `model_validate` lo
    지적aría.
    """
    datos = estimacion.model_dump()
    return EstimacionCompleta.model_validate(
        {
            **datos,
            "total_horas": calcular_total(estimacion),
            "duracion_semanas": calcular_semanas(estimacion),
        }
    )
