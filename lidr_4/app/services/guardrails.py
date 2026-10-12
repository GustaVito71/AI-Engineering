"""Guardrails de entrada (WU9 A): lo que se le hace a la descripción antes de gastar nada.

La descripción es texto libre del usuario y termina dentro del prompt. Estas
reglas corren en `EstimationService`, antes del render y de la caché del
wrapper, así una descripción rechazada no llega al proveedor ni se sirve desde
la caché. En orden, de lo más barato a lo más caro:

1. Prompt injection (regex, local): la descripción da órdenes al modelo en vez de
   describir un proyecto. Se RECHAZA (`InputGuardrailViolation`, 422).
2. Datos personales (regex y dígitos de control, local): email, teléfono, IBAN,
   tarjeta y documentos. Se OCULTAN con un marcador (`[EMAIL]`) y la estimación
   sigue, con un aviso al usuario. Rechazar sería más simple, pero tiraría
   descripciones legítimas que solo nombran un contacto. El prompt, la caché y
   el log ven solo el texto ya oculto.
3. Moderación (OpenAI, opcional): se consulta con el texto ya oculto, para no
   mandar datos personales a un tercero. Se RECHAZA si el modelo marca contenido.

Qué NO hace, a propósito:

- Son heurísticas, no una garantía. Una inyección redactada de otra forma, en
  otro idioma o con letras de otro alfabeto que se parecen a las latinas pasa. El
  evento `entrada_rechazada` deja la regla que saltó para medir falsos positivos.
- La moderación falla ABIERTA: si OpenAI no responde, no tiene clave o tarda más
  de MODERATION_TIMEOUT, la descripción sigue y queda `moderacion_no_disponible`
  en el log. Fallar cerrado dejaría el servicio sin estimar por una avería de un
  tercero, en una capa que es un filtro extra y no la única defensa.
- Ningún evento lleva el texto de la descripción: puede traer justo lo que se
  busca ocultar.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import litellm

from app.config import Settings
from app.tracing import emit

INJECTION_MESSAGE = (
    "La descripción contiene instrucciones dirigidas al modelo en vez de describir "
    "el proyecto. Reescribila contando qué se quiere construir."
)
MODERATION_MESSAGE = (
    "La descripción incluye contenido que no se puede procesar. "
    "Reescribila describiendo solo el proyecto a estimar."
)


class InputGuardrailViolation(Exception):
    """Una regla de entrada rechazó la descripción.

    `rule` es el nombre de la regla (va al log) y `message` el texto para el
    cliente, fijo por regla: no repite nada de lo que escribió el usuario.
    """

    def __init__(self, rule: str, message: str) -> None:
        super().__init__(f"{rule}: {message}")
        self.rule = rule
        self.message = message


# --- Prompt injection -------------------------------------------------------

# Etiquetas que usan las plantillas de app/prompts/estimation. Una descripción que
# las trae intenta cerrar el bloque que la contiene (`</project_description>`) o
# imitar la estructura del prompt. test_guardrails.py compara esta lista con las
# etiquetas que usan las plantillas, para que no se desfase al publicar una versión.
PROMPT_TAGS = (
    "project_description",
    "project_type",
    "detail_level",
    "output_format",
    "scope",
    "out_of_scope",
    "rules",
    "examples",
    "example",
    "estimation",
    "totals",
    "system",
    "instructions",
    "assistant",
)

# Los patrones se escriben sin tildes ni mayúsculas: `_normalize` las quita del
# texto antes de compararlo.
_INJECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignorar_instrucciones",
        re.compile(
            r"\b(?:ignor|olvid|desobedec)\w*\s+(?:\w+\s+){0,2}"
            r"(?:instruccion|indicacion|directriz|regla|orden|prompt)\w*\s+"
            r"(?:anterior|previa|previo|de\s+arriba|del\s+sistema|del\s+prompt|iniciales)"
            r"|\b(?:ignor|olvid)\w*\s+(?:todo\s+)?(?:lo\s+)?(?:anterior|de\s+arriba)\b"
            r"|\b(?:ignore|disregard|forget)\s+(?:\w+\s+){0,3}"
            r"(?:previous|prior|above|earlier|system|your|all)\s+(?:\w+\s+){0,2}"
            r"(?:instructions?|prompts?|rules?|directions?|guidelines?)\b"
            r"|\bforget\s+everything\b"
        ),
    ),
    (
        "revelar_prompt",
        re.compile(
            r"\b(?:muestr|revel|imprim|repit|mostr|dime|dame|escrib|filtr)\w*\s+(?:\w+\s+){0,4}"
            r"(?:tu\s+(?:system\s*)?prompt|tus\s+(?:instrucciones|reglas))"
            r"|\b(?:cual(?:es)?|que)\s+(?:es|son)\s+(?:tu|tus)\s+(?:system\s*prompt|prompt|instrucciones)"
            r"|\b(?:reveal|show|print|repeat|display|output|leak)\s+(?:\w+\s+){0,4}"
            r"your\s+(?:(?:initial|hidden|system|original)\s+)?(?:instructions|prompt)\b"
            r"|\bwhat\s+(?:is|are)\s+your\s+(?:system\s+)?(?:prompt|instructions)\b"
        ),
    ),
    (
        "cambio_de_rol",
        re.compile(
            r"\bahora\s+eres\b"
            r"|\b(?:de\s+ahora\s+en\s+adelante|a\s+partir\s+de\s+ahora)\s+"
            r"(?:eres|seras|actua|responde|ignora)\b"
            r"|\byou\s+are\s+now\s+(?:a|an|the|in|my)\b"
            r"|\bfrom\s+now\s+on\s*,?\s+(?:you|ignore|act|respond)\b"
            r"|\bdan\s+mode\b|\bjailbreak\s+mode\b|\bdo\s+anything\s+now\b"
        ),
    ),
    (
        "etiquetas_del_prompt",
        re.compile(r"</?\s*(?:" + "|".join(PROMPT_TAGS) + r")\b[^>]*>"),
    ),
    (
        "marcadores_de_rol",
        re.compile(r"<\|[a-z_]+\|>|\[/?(?:inst|sys)\]|<<\s*/?\s*sys\s*>>"),
    ),
)


def _normalize(text: str) -> str:
    """Minúsculas, sin tildes ni caracteres invisibles y con los espacios colapsados.

    Sin esto, `Ignorá` y `IGNORA` esquivarían la misma regla que `ignora`, y también
    una letra con un carácter de ancho cero en el medio. NFKD lleva además las letras
    de ancho completo a su forma normal.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    kept = (
        c for c in decomposed if not unicodedata.combining(c) and unicodedata.category(c) != "Cf"
    )
    return re.sub(r"\s+", " ", "".join(kept)).casefold()


def detect_prompt_injection(text: str) -> str | None:
    """El nombre de la primera regla de inyección que salta, o None."""
    normalized = _normalize(text)
    for name, pattern in _INJECTION_RULES:
        if pattern.search(normalized):
            return name
    return None


# --- Datos personales -------------------------------------------------------


def _luhn_valid(candidate: str) -> bool:
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for position, digit in enumerate(reversed(digits)):
        if position % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _iban_valid(candidate: str) -> bool:
    compact = re.sub(r"\s", "", candidate)
    if not 15 <= len(compact) <= 34:
        return False
    rearranged = compact[4:] + compact[:4]
    return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1


_DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"


def _dni_valid(candidate: str) -> bool:
    return _DNI_LETTERS[int(candidate[:8]) % 23] == candidate[8]


def _nie_valid(candidate: str) -> bool:
    number = int(str("XYZ".index(candidate[0])) + candidate[1:8])
    return _DNI_LETTERS[number % 23] == candidate[8]


@dataclass(frozen=True)
class _PersonalDataRule:
    kind: str
    token: str
    pattern: re.Pattern[str]
    # Dígitos de control: sin ellos, cualquier número largo parecería una tarjeta.
    is_valid: Callable[[str], bool] | None = None


# El orden importa: lo más específico primero. Los marcadores no tienen dígitos,
# así que una regla posterior no vuelve a encontrar lo que ya se ocultó.
_PERSONAL_DATA_RULES: tuple[_PersonalDataRule, ...] = (
    _PersonalDataRule("email", "[EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), None),
    _PersonalDataRule(
        "iban",
        "[IBAN]",
        re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b"),
        _iban_valid,
    ),
    _PersonalDataRule(
        "telefono", "[TELÉFONO]", re.compile(r"(?<![\w+])\+\d[\d ()-]{7,16}\d(?!\w)"), None
    ),
    _PersonalDataRule(
        "tarjeta",
        "[TARJETA]",
        re.compile(r"(?<![\w.])(?:\d[ -]?){12,18}\d(?![\w.])"),
        _luhn_valid,
    ),
    _PersonalDataRule("documento", "[DOCUMENTO]", re.compile(r"\b\d{2}-\d{8}-\d\b"), None),
    _PersonalDataRule("documento", "[DOCUMENTO]", re.compile(r"\b\d{8}[A-Z]\b"), _dni_valid),
    _PersonalDataRule("documento", "[DOCUMENTO]", re.compile(r"\b[XYZ]\d{7}[A-Z]\b"), _nie_valid),
    _PersonalDataRule(
        "documento",
        "[DOCUMENTO]",
        re.compile(
            r"(?P<prefix>\b(?:DNI|CUIT|CUIL|NIE)\b[^\d\n]{0,15})\d{1,2}\.?\d{3}\.?\d{3}\b",
            re.IGNORECASE,
        ),
        None,
    ),
    _PersonalDataRule(
        "telefono",
        "[TELÉFONO]",
        re.compile(
            r"(?<![\w.])(?:"
            r"\(?0\d{2,4}\)?[ -]\d{3,4}[ -]\d{4}"
            r"|\d{2,4}[ -]\d{4}[ -]\d{4}"
            r"|[6-9]\d{2}[ -]\d{3}[ -]\d{3}"
            r"|\d{3}[ -]\d{3}[ -]\d{4}"
            r")(?![\w.])"
        ),
        None,
    ),
)

_KIND_LABELS = {
    "email": "email",
    "iban": "IBAN",
    "telefono": "teléfono",
    "tarjeta": "tarjeta",
    "documento": "documento de identidad",
}


def mask_personal_data(text: str) -> tuple[str, tuple[str, ...]]:
    """El texto con los datos personales reemplazados, y los tipos que se ocultaron.

    Mejor esfuerzo: reconoce los formatos más comunes (email, teléfonos con o sin
    prefijo, IBAN y tarjeta con su dígito de control, DNI/NIE españoles, CUIT/CUIL
    y «DNI 12.345.678»). Un número que no tiene esa forma no se toca: las cifras
    del proyecto (`20.000 usuarios`, `150.000.000 de pesos`) son parte de lo que
    se estima.
    """
    kinds: list[str] = []

    for rule in _PERSONAL_DATA_RULES:

        def replace(match: re.Match[str], rule: _PersonalDataRule = rule) -> str:
            if rule.is_valid is not None and not rule.is_valid(match.group(0)):
                return match.group(0)
            if rule.kind not in kinds:
                kinds.append(rule.kind)
            return (match.groupdict().get("prefix") or "") + rule.token

        text = rule.pattern.sub(replace, text)
    return text, tuple(kinds)


# --- Moderación -------------------------------------------------------------


class Moderator(Protocol):
    """Lo que `InputGuardrails` necesita de un servicio de moderación."""

    async def flagged_categories(self, text: str) -> list[str]:
        """Las categorías marcadas en `text`; vacío si no hay ninguna."""
        ...


class OpenAIModerator:
    """Moderación de OpenAI, por `litellm.amoderation` (el endpoint no se cobra).

    Usa la clave de OpenAI de Settings. `timeout` acota la llamada entera, con los
    reintentos del cliente de OpenAI incluidos: esta capa no puede sumar más de eso
    a la latencia de cada estimación.
    """

    def __init__(
        self, *, api_key: str, model: str, timeout: float, client: object | None = None
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        # Solo para los tests: un AsyncOpenAI con transporte falso, sin red.
        self._client = client

    async def flagged_categories(self, text: str) -> list[str]:
        extra = {"client": self._client} if self._client is not None else {}
        response = await asyncio.wait_for(
            litellm.amoderation(input=text, model=self._model, api_key=self._api_key, **extra),
            timeout=self._timeout,
        )
        result = response.results[0]
        if not result.flagged:
            return []
        # `flagged` sin ninguna categoría en true no debería pasar; si pasa, se rechaza igual.
        return sorted(name for name, hit in result.categories.items() if hit) or ["sin_categoria"]


# --- Las reglas juntas ------------------------------------------------------


@dataclass(frozen=True)
class InputCheck:
    """Lo que sigue en el pipeline: la descripción a usar y los avisos para el usuario."""

    description: str
    warnings: list[str]


class InputGuardrails:
    """Las reglas de entrada, en orden. No guarda estado entre llamadas."""

    def __init__(self, *, moderator: Moderator | None = None) -> None:
        self._moderator = moderator

    async def check(self, description: str) -> InputCheck:
        """Devuelve la descripción a usar, o lanza `InputGuardrailViolation`."""
        rule = detect_prompt_injection(description)
        if rule is not None:
            emit(
                __name__,
                "entrada_rechazada",
                level="warning",
                regla=rule,
                longitud=len(description),
            )
            raise InputGuardrailViolation(rule, INJECTION_MESSAGE)

        masked, kinds = mask_personal_data(description)
        warnings: list[str] = []
        if kinds:
            emit(__name__, "datos_personales_ocultados", tipos=list(kinds))
            labels = ", ".join(_KIND_LABELS[kind] for kind in kinds)
            warnings.append(
                f"Se ocultaron datos personales de la descripción antes de enviarla al modelo: "
                f"{labels}."
            )

        if self._moderator is not None:
            categories = await self._moderate(masked)
            if categories:
                emit(
                    __name__,
                    "entrada_rechazada",
                    level="warning",
                    regla="moderacion",
                    categorias=categories,
                    longitud=len(description),
                )
                raise InputGuardrailViolation("moderacion", MODERATION_MESSAGE)

        return InputCheck(description=masked, warnings=warnings)

    async def _moderate(self, text: str) -> list[str]:
        assert self._moderator is not None
        try:
            return await self._moderator.flagged_categories(text)
        except Exception as exc:  # noqa: BLE001 - la moderación falla abierta, ver el docstring
            emit(
                __name__,
                "moderacion_no_disponible",
                level="warning",
                tipo_error=type(exc).__name__,
                detalle=str(exc),
            )
            return []


def create_input_guardrails(settings: Settings) -> InputGuardrails | None:
    """Los guardrails que pide la configuración, o None si están apagados.

    Sin clave de OpenAI la moderación queda apagada y el resto sigue: el servicio
    tiene que arrancar igual (ver app/config.py). Cada apagado deja un evento,
    para que no sea un estado invisible.
    """
    if not settings.guardrails_enabled:
        emit(
            __name__, "guardrails_desactivados", level="warning", motivo="GUARDRAILS_ENABLED=false"
        )
        return None

    moderator: OpenAIModerator | None = None
    if settings.moderation_enabled:
        api_key = settings.active_api_key("openai")
        if api_key:
            moderator = OpenAIModerator(
                api_key=api_key,
                model=settings.moderation_model,
                timeout=settings.moderation_timeout,
            )
        else:
            emit(
                __name__,
                "moderacion_desactivada",
                level="warning",
                motivo="falta OPENAI_API_KEY",
            )
    return InputGuardrails(moderator=moderator)
