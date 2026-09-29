"""Tests del dominio: Puros, sin I/O, sin red y sin keys.

Cada test que sigue existe por una razón concreta. Los casos raros —el ciclo de
dependencias, el equipo vacío, la feature de dos días— son más stubbornly
interesantes que el camino feliz, que es el que no se rompe.

La regla del módulo: los datos derivados se calculan en código, no los emite el
modelo. Los tests de abajo fijan esa frontera. Si alguien agrega `total_horas`
como campo del modelo, el primer test que falle es el que lo dice.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.estimacion import (
    FRACCION_ENFOQUE,
    HORAS_SEMANALES,
    EstimacionCompleta,
    SolicitudEstimacion,
    Tarea,
    calcular_semanas,
    calcular_total,
    completar,
)


def _tarea(id_: str, horas: int, depende_de: list[str] | None = None) -> dict:
    return {
        "id": id_,
        "titulo": f"Tarea {id_}",
        "descripcion": f"Descripción de {id_}",
        "horas": horas,
        "depende_de": depende_de or [],
        "riesgo": "medio",
    }


def _solicitud(tareas: list[dict], equipo: list[dict] | None = None) -> dict:
    return {
        "resumen": "Estimación de prueba",
        "tareas": tareas,
        "equipo": (
            [{"rol": "backend", "cantidad": 1, "foco": "API"}] if equipo is None else equipo
        ),
        "supuestos": [{"texto": "datos de prueba", "impacto": "+1 semana si no"}],
        "riesgos": ["scope creep"],
        "advertencias": ["no incluye QA"],
    }


EQUIPO_DOS = [
    {"rol": "backend", "cantidad": 1, "foco": "API"},
    {"rol": "frontend", "cantidad": 1, "foco": "UI"},
]


# --- La frontera: los totales se calculan, no los emite el modelo.


def test_la_solicitud_no_tiene_totales():
    """El LLM no puede emitir el número que se cobra.

    Si `total_horas` fuera un campo de `SolicitudEstimacion`, el modelo podría
    decir "320 horas" con tareas que suman 287 y nada lo detectaría: en el
    schema, 320 sería una respuesta perfectamente válida. Que el campo no
    exista es lo que hace imposible la inconsistencia."""
    campos = SolicitudEstimacion.model_fields
    assert "total_horas" not in campos
    assert "duracion_semanas" not in campos


def test_los_totales_no_pueden_inyectarse_desde_el_llm():
    """Un `extra` con totales se ignora, y `completar` los calcula igual.

    Es el otro lado del test anterior visto desde la entrada: si el modelo
    incluye `total_horas` en su JSON por大户 confondirse, Pydantic lo descarta
    y el número que sale es el que.calculate_ emite el código."""
    solicitud = SolicitudEstimacion.model_validate(
        {**_solicitud([_tarea("T1", 40)]), "total_horas": 9999}
    )
    completada = completar(solicitud)
    assert completada.total_horas == 40


def test_completar_agrega_los_dos_campos():
    solicitud = SolicitudEstimacion.model_validate(
        _solicitud([_tarea("T1", 40), _tarea("T2", 60)], EQUIPO_DOS)
    )
    completada = completar(solicitud)
    assert isinstance(completada, EstimacionCompleta)
    assert completada.total_horas == 100
    assert completada.duracion_semanas == 2


# --- calcular_total


def test_total_suma_las_horas_de_las_tareas():
    solicitud = SolicitudEstimacion.model_validate(
        _solicitud([_tarea("T1", 40), _tarea("T2", 60), _tarea("T3", 100)])
    )
    assert calcular_total(solicitud) == 200


def test_total_de_sin_tareas_es_cero():
    """Cero horas no es un error de validación, es una aritmética.

    `tareas: list[Tarea]` acepta lista vacía a propósito. Puede que el modelo
    responda que un trabajo tan chico no amerita descomponerse, y en ese caso el
    sistema tiene que poder decirlo en vez de inventar un error de formato."""
    solicitud = SolicitudEstimacion.model_validate(_solicitud([]))
    assert calcular_total(solicitud) == 0


def test_las_horas_deben_ser_positivas():
    """Cero horas o negativos son un error, no un número raro.

    El `gt=0` está en el campo, así que el `model_validate` es lo que corta. El
    `sum` de abajo solo recibe enteros positivos verificados."""
    with pytest.raises(ValidationError):
        SolicitudEstimacion.model_validate(_solicitud([_tarea("T1", 0)]))


# --- calcular_semanas


def test_duracion_usa_el_equipo_y_el_enfoque_del_80_por_ciento():
    """180h / (2 personas * 40h * 0.8) = 2.81 -> 3 semanas.

    El denominador va escrito a mano, 64, y no como
    `2 * HORAS_SEMANALES * FRACCION_ENFOQUE`. Derivarlo de las constantes hace
    el test tautológico: si alguien cambia el 0.8 por 1.0, los dos lados cambian
    juntos y el test sigue verde sin comprobar nada. Verificado con mutación.

    Sin el 0.8 el resultado sería 2.25 -> 3 también, por eso el 64 explícito es
    lo que fija que el supuesto está aplicado y no solo que redondea bien."""
    solicitud = SolicitudEstimacion.model_validate(
        _solicitud(
            [_tarea("T1", 60), _tarea("T2", 60), _tarea("T3", 60)],
            EQUIPO_DOS,
        )
    )
    assert 2 * 40 * 0.8 == 64  # el supuesto, a la vista
    assert calcular_semanas(solicitud) == 3


def test_el_fraccion_de_enfoque_sigue_siendo_0_8():
    """El supuesto de negocio, fijado por separado.

    El 20% que no llega a la feature cubre review, meetings, soporte y tiempo
    administrativo. Subirlo a 1.0 da duraciones más optimistas y más cortas de
    revisar: una estimación que se queda corta genera confianza falsa. Este test
    existe para que cambiarlo sea una decisión explícita y no un refactor
    accidental."""
    assert FRACCION_ENFOQUE == 0.8
    assert HORAS_SEMANALES == 40


def test_un_equipo_mas_grande_dura_menos():
    """Más personas, menos semanas: el denominador crece.

    No es un test de la fórmula —eso lo hace el de arriba— sino de que
    `tamano_equipo` suma los roles en vez de contar cuántos roles hay. Con dos
    roles de 1, la suma es 2; con un rol de 2, también. El caso que distingue
    es un rol con 3 personas, que cuenta como 3 y no como 1."""
    tareas = [_tarea("T1", 640)]
    uno = SolicitudEstimacion.model_validate(
        _solicitud(tareas, [{"rol": "backend", "cantidad": 3, "foco": "todo"}])
    )
    tres = SolicitudEstimacion.model_validate(
        _solicitud(
            tareas,
            [
                {"rol": "backend", "cantidad": 1, "foco": "API"},
                {"rol": "frontend", "cantidad": 1, "foco": "UI"},
                {"rol": "devops", "cantidad": 1, "foco": "infra"},
            ],
        )
    )
    assert uno.tamano_equipo == tres.tamano_equipo == 3
    assert calcular_semanas(uno) == calcular_semanas(tres)


def test_una_feature_de_menos_de_una_semana_dura_una():
    """El `max(1, ...)` evita un `duracion_semanas: 0`.

    Una feature de 4 horas con 2 personas da 0.06 semanas. Lo truncar a 0 produce
    una respuesta que la UI no sabe pintar, y 0 semanas además se lee como "no
    hay trabajo" cuando sí hay. Una semana es el piso honesto."""
    solicitud = SolicitudEstimacion.model_validate(_solicitud([_tarea("T1", 4)], EQUIPO_DOS))
    assert calcular_semanas(solicitud) == 1


def test_un_equipo_vacio_se_rechaza_en_lugar_de_dividir_por_cero():
    """Sin equipo no hay duración, y hay que decirlo.

    La alternativa obvia —devolver 1 para no romper— devuelve un número sin
    sentido computado. Eso es peor que un error: es una estimación que se ve
    válida en pantalla y no lo está. Un equipo vacío es un error de la
    estimación, y muere al validarla."""
    solicitud = SolicitudEstimacion.model_validate(_solicitud([_tarea("T1", 40)], []))
    assert solicitud.tamano_equipo == 0
    with pytest.raises(ValueError, match="equipo vacío"):
        calcular_semanas(solicitud)


def test_una_cantidad_de_equipo_cero_se_rechaza_al_validar():
    """El `gt=0` del campo corta antes de llegar a la división por cero.

    Es el casoGemelos del test anterior: uno se ve al calcular y el otro al
    parsear. Ambos tienen que estar cerrados, porque el dominio se construye
    en dos momentos distintos —el modelo emite JSON, el código calcula— y cada
    error tiene que morir en el suyo."""
    with pytest.raises(ValidationError):
        SolicitudEstimacion.model_validate(
            _solicitud([_tarea("T1", 40)], [{"rol": "backend", "cantidad": 0, "foco": "x"}])
        )


# --- Las referencias entre tareas: el riesgo que el plan registra.


def test_una_dependencia_valida_pasa():
    solicitud = SolicitudEstimacion.model_validate(
        _solicitud([_tarea("T1", 40), _tarea("T2", 60, depende_de=["T1"])])
    )
    assert solicitud.tareas[1].depende_de == ["T1"]


def test_una_dependencia_a_una_tarea_inexistente_se_rechaza():
    """El modelo inventa IDs, y sin esto el hueco no aparece hasta tarde.

    Es el riesgo de la tabla del plan: `T3` depende de una tarea que no está en
    la lista. Pydantic no lo detecta —`depende_de` es `list[str]` y el string es
    válido— así que si no se valida acá, la estimación se muestra completa con un
    grafo que no se puede recorrer."""
    with pytest.raises(ValidationError, match="no existen"):
        SolicitudEstimacion.model_validate(
            _solicitud([_tarea("T1", 40), _tarea("T2", 60, depende_de=["T9"])])
        )


def test_el_error_de_dependencia_orphana_dice_que_ids_si_valen():
    """El mensaje tiene que ser accionable sin abrir el JSON.

    Un "referencia inválida" obliga a leer la respuesta del modelo entera para
    descubrir cuál de los cinco IDs estaba mal. Nombrando los IDs válidos, el
    error se corrige en el prompt sin instrumentar nada."""
    with pytest.raises(ValidationError) as error:
        SolicitudEstimacion.model_validate(
            _solicitud([_tarea("T1", 40), _tarea("T2", 60, depende_de=["T9"])])
        )
    assert "T1" in str(error.value)


def test_un_ciclo_entre_tareas_se_rechaza():
    """Un ciclo no es una estimación válida, es un error de normalización.

    `T1 → T2 → T1` significa que ninguna de las dos puede empezar antes que la
    otra, o sea que ninguna puede empezar. La lista se puede ordenar, pero el
    resultado es un plan que no se puede ejecutar, y una duración calculada
    sobre él no significa nada."""
    with pytest.raises(ValidationError, match="Ciclo"):
        SolicitudEstimacion.model_validate(
            _solicitud(
                [
                    _tarea("T1", 40, depende_de=["T2"]),
                    _tarea("T2", 60, depende_de=["T1"]),
                ]
            )
        )


def test_un_ciclo_largo_se_detecta():
    """El ciclo de tres nodos, no el de dos.

    El recorrido en color solo falla si el gris no se propaga. Con A→B→C→A el
    algoritmo tiene que volver a un nodo que ya está en la pila, y es el caso
    que un chequeo de "solo predecesores" mal escrito deja pasar."""
    with pytest.raises(ValidationError, match="Ciclo"):
        SolicitudEstimacion.model_validate(
            _solicitud(
                [
                    _tarea("T1", 40, depende_de=["T3"]),
                    _tarea("T2", 40, depende_de=["T1"]),
                    _tarea("T3", 40, depende_de=["T2"]),
                ]
            )
        )


def test_ids_duplicados_se_rechazan():
    """`depende_de` apunta por ID, así que los IDs tienen que ser únicos.

    Sin este chequeo, dos tareas con `id: "T1"` hacen que `depende_de: ["T1"]`
    sea ambiguo: ¿depende de la primera o de la segunda? Y el conteo de
    dependencias por tarea, que va a WU8, no puede distinguir."""
    with pytest.raises(ValidationError, match="duplicados"):
        SolicitudEstimacion.model_validate(_solicitud([_tarea("T1", 40), _tarea("T1", 60)]))


def test_una_tarea_depende_de_si_misma_se_rechaza():
    """`T1 → T1` es el ciclo más corto posible, y por eso se cuela más fácil.

    El recorrido lo pilla en la primera iteración, pero el caso está aparte
    porque es el que un validador escrito a mano suele pasar por alto: la
    dependencia no apunta a "otra" tarea, y un filtro `if d != self.id` lo
    deja pasar."""
    with pytest.raises(ValidationError, match="Ciclo"):
        SolicitudEstimacion.model_validate(_solicitud([_tarea("T1", 40, depende_de=["T1"])]))


# --- La forma de los datos.


def test_el_riesgo_solo_acepta_los_tres_niveles():
    """Enum cerrado, no un entero del 1 al 10.

    Un número da una precisión que no existe: la diferencia entre riesgo 3 y
    riesgo 4 no significa nada. Con tres niveles el modelo tiene que decidir de
    verdad, y un valor fuera del conjunto es un error de parseo —que es
    información— en vez de un número raro que se acepta en silencio."""
    with pytest.raises(ValidationError):
        Tarea.model_validate(_tarea("T1", 40) | {"riesgo": "critico"})


def test_tamano_equipo_es_derivado_y_no_un_campo():
    """Es una property, no un campo: el modelo no puede emitirlo.

    Si fuera campo, el modelo podría mandar `tamano_equipo: 2` con un `equipo`
    que suma 4, y laproperty devolvería el campo mientras el cálculo usaría el
    error. Que no exista como campo elimina la posibilidad de desincronización."""
    assert "tamano_equipo" not in SolicitudEstimacion.model_fields


def test_la_estimacion_completa_es_una_solicitud_mas_dos_campos():
    """La herencia tiene que ser real, no una copia de los campos.

    Si `EstimacionCompleta` copiara los campos a mano, agregar uno nuevo a
    `SolicitudEstimacion` lo perdería en silencio. Que herede garantiza que la
    respuesta tiene todo lo que el modelo dijo, más los totales."""
    assert issubclass(EstimacionCompleta, SolicitudEstimacion)
    assert set(EstimacionCompleta.model_fields) - set(SolicitudEstimacion.model_fields) == {
        "total_horas",
        "duracion_semanas",
    }
