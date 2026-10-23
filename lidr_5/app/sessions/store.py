"""Almacén de sesiones en memoria.

QUÉ ES

El almacén es la "estantería" donde se guardan todas las sesiones vivas del servicio:
un diccionario de Python (`dict`) que relaciona cada `session_id` con su `Session`. Es
lo que permite que, en la segunda petición de una conversación, el servicio encuentre
lo que se dijo en la primera.

POR QUÉ ES UNA CLASE Y NO UN DICCIONARIO SUELTO

Para tener un único punto que cambiar si algún día las sesiones se guardan en otro
lado (una base de datos, Redis), sin tocar los routers ni los servicios. Para el
porqué de la volatilidad, ver el docstring de `app.sessions.models`.

CADUCIDAD Y TOPE

Sin ellos, cada sesión creada ocuparía memoria para siempre. Ver `SessionStore`.

LÍMITE: UN SOLO PROCESO

No es seguro entre hilos ni entre procesos: con `uvicorn --workers=1` alcanza (el
código de los endpoints es asíncrono y corre en un solo hilo). Con varios workers,
cada uno tendría su propio almacén y la conversación se rompería.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from app.sessions.models import DEFAULT_MAX_TURNS, ConversationHistory, Session

# Valores por defecto de la caducidad y del tope (variables SESSION_TTL_SECONDS y
# MAX_SESSIONS): 24 horas sin uso y 1.000 sesiones vivas como máximo.
DEFAULT_SESSION_TTL_SECONDS = 86_400
DEFAULT_MAX_SESSIONS = 1_000


class SessionNotFoundError(KeyError):
    """El `session_id` no existe o ya caducó (el endpoint lo traduce a 404)."""


class SessionStore:
    """Guarda las sesiones en la memoria del proceso y las busca por `session_id`.

    Dos reglas evitan que la memoria crezca sin tope:
    - Caducidad: una sesión sin uso durante `ttl_seconds` se considera inexistente y se
      borra. Cada uso (`get`, `get_or_raise`) renueva el plazo.
    - Tope: al crear una sesión con el almacén lleno se descartan primero las caducadas
      y, si aún no hay lugar, la sesión menos usada recientemente.
    """

    def __init__(
        self,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        # `max_turns` es la ventana que tendrá CADA sesión nueva (variable MAX_TURNS).
        self._sessions: dict[str, Session] = {}
        self._max_turns = max_turns
        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_sessions = max_sessions
        # El reloj se puede reemplazar en los tests para simular el paso del tiempo.
        self._clock = clock or (lambda: datetime.now(UTC))

    def create(self) -> Session:
        """Crea una sesión vacía con la ventana configurada y la registra."""
        self._purge_expired()
        # Con el almacén lleno se desaloja la sesión usada hace más tiempo.
        while len(self._sessions) >= self._max_sessions:
            oldest = min(self._sessions.values(), key=lambda s: s.last_active_at)
            del self._sessions[oldest.session_id]
        now = self._clock()
        session = Session(
            history=ConversationHistory(max_turns=self._max_turns),
            created_at=now,
            last_active_at=now,
        )
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        """Devuelve la sesión, o `None` si no existe o caducó (renueva su plazo)."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        now = self._clock()
        if now - session.last_active_at > self._ttl:
            del self._sessions[session_id]
            return None
        session.touch(now)
        return session

    def get_or_raise(self, session_id: str) -> Session:
        """Devuelve la sesión, o lanza `SessionNotFoundError` si no existe o caducó.

        Es la versión para los endpoints: un identificador desconocido o vencido es un
        error del cliente y se responde con 404.
        """
        session = self.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    def _purge_expired(self) -> None:
        """Borra todas las sesiones que superaron el plazo sin uso."""
        limit = self._clock() - self._ttl
        for session_id in [k for k, v in self._sessions.items() if v.last_active_at < limit]:
            del self._sessions[session_id]

    def __len__(self) -> int:
        """Cuántas sesiones hay guardadas (se usa en la traza y en los tests)."""
        return len(self._sessions)
