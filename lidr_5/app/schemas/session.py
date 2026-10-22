"""Formas de los datos de los endpoints de sesiones.

QUÉ ES UN "ESQUEMA" AQUÍ

Un esquema (clase Pydantic) describe, campo por campo, cómo es la respuesta que
devuelve un endpoint. FastAPI lo usa para dos cosas: comprobar que lo que se devuelve
cumple esa forma, y publicar la forma en la documentación automática (`/docs`).

Este archivo solo tiene las formas de las sesiones. Las de las estimaciones están en
`estimation.py` y `structured_estimation.py`, que vienen de lidr_4 y no cambian.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.sessions.models import ProjectMetadata


class CreateSessionResponse(BaseModel):
    """Lo que devuelve `POST /sessions`: el identificador de la sesión recién creada.

    El cliente guarda este identificador y lo manda en cada petición posterior
    (`POST /sessions/{session_id}/estimate`) para que el servicio sepa de qué proyecto
    se está hablando y recuerde la conversación.
    """

    session_id: str = Field(
        description=(
            "Identificador de la sesión (UUID versión 4). Es el único dato que "
            "identifica la conversación: quien lo conoce puede continuarla."
        ),
        examples=["3f2b8c1e-5a47-4d0e-9b6a-1c2d3e4f5a6b"],
    )


class SessionStateResponse(BaseModel):
    """Lo que devuelve `GET /sessions/{session_id}`: el estado visible de la sesión.

    Sirve para que un cliente (por ejemplo, el panel lateral de Streamlit) muestre qué
    recuerda el servicio del proyecto sin tener que guardar su propia copia. No incluye
    el texto de los mensajes: solo cuántos turnos se recuerdan y los hechos del proyecto.
    """

    session_id: str = Field(description="Identificador de la sesión consultada.")
    turns: int = Field(
        description="Turnos (pares usuario + asistente) que la sesión recuerda ahora.",
        examples=[2],
    )
    max_turns: int = Field(
        description="Máximo de turnos que recuerda la ventana deslizante (`MAX_TURNS`).",
        examples=[6],
    )
    project_metadata: ProjectMetadata = Field(
        description="Los hechos del proyecto que el servicio tiene anotados hasta ahora."
    )
    anchors: int = Field(
        default=0,
        description="Turnos con compromisos que se conservan textuales (Paso 9).",
        examples=[1],
    )
    has_summary: bool = Field(
        default=False,
        description="True si lo que salió de la ventana ya se resumió (Paso 9).",
    )
    tier: str | None = Field(
        default=None,
        description="Audiencia usada en el último turno (executive, pm, developer o default).",
        examples=["executive"],
    )
    tier_rule: str | None = Field(
        default=None,
        description="Regla que decidió la audiencia (por ejemplo `nda_detected`).",
        examples=["nda_detected"],
    )
