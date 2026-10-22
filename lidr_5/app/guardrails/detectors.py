"""Detectores compartidos por los guardrails de entrada y de salida.

Funciones puras sobre texto, sin eventos ni configuración: quien las usa decide
qué hace con lo que encuentran (rechazar, ocultar, avisar). Hoy las usa
`input.py`; los guardrails de salida van a reutilizar los mismos detectores sobre
lo que escribe el modelo.

- `normalize_text`: minúsculas, sin tildes ni caracteres invisibles.
- `PROMPT_TAGS`: las etiquetas de las plantillas de prompt.
- `find_prompt_leak`: si un texto vuelca partes del prompt de sistema (salida).
- `mask_personal_data`: oculta email, teléfono, IBAN, tarjeta y documentos.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

# Etiquetas que usan las plantillas de app/prompts/estimation. Una descripción que
# las trae intenta cerrar el bloque que la contiene (`</project_description>`) o
# imitar la estructura del prompt. test_detectors.py compara esta lista con las
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
    # Prompt v5 (memoria de la sesión y audiencia), bloque de adjuntos y extractor de
    # metadata: una descripción que los trae intenta imitar o cerrar esos bloques.
    "project_metadata",
    "audience",
    "attachments",
    "attachment",
    "current_metadata",
    "user_turn",
    "assistant_turn",
    # ACB (Paso 8): el feedback que el actor recibe del crítico y los bloques del prompt del
    # crítico. Una descripción que los trae intenta hacerse pasar por una revisión.
    "critic_feedback",
    "original_transcript",
    "audience_tier",
    "requested_detail_level",
    "estimation_under_review",
    # Resumidor de la conversación (Paso 9).
    "previous_summary",
    "evicted_messages",
)


def normalize_text(text: str) -> str:
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


# Frases de los prompts de sistema (normalizadas con `normalize_text`) que una salida
# normal no tiene motivo para repetir. Si el modelo las escribe, está volcando sus
# instrucciones. test_detectors.py comprueba que cada una sigue estando en alguna
# plantilla, para que no se desfasen al publicar una versión.
PROMPT_LEAK_PHRASES = (
    "senior project estimator with 15+ years of experience",
    "estimador de proyectos senior con mas de 15 anos de experiencia",
    "no elijas un numero redondo para ajustar las fases",
    "vuelve a sumar las fases y comprueba que coincidan con los totales",
)

# `<estimation>` no cuenta: el modelo lo copia de los ejemplos y el servicio lo quita.
_LEAK_TAGS = re.compile(
    r"</?\s*(?:" + "|".join(t for t in PROMPT_TAGS if t != "estimation") + r")\s*>"
)


def find_prompt_leak(text: str) -> str | None:
    """`etiqueta` o `frase` si el texto vuelca partes del prompt de sistema; None si no.

    Mejor esfuerzo: reconoce las etiquetas de las plantillas y las frases fijas de
    `PROMPT_LEAK_PHRASES`. Una paráfrasis de las instrucciones pasa.
    """
    normalized = normalize_text(text)
    if _LEAK_TAGS.search(normalized):
        return "etiqueta"
    if any(phrase in normalized for phrase in PROMPT_LEAK_PHRASES):
        return "frase"
    return None


# --- Datos personales -------------------------------------------------------


def _luhn_valid(candidate: str) -> bool:
    """Comprueba el dígito de control de un número de tarjeta (algoritmo de Luhn).

    Sirve para distinguir una tarjeta real de cualquier número largo: si el control no
    cuadra, no se trata como tarjeta.
    """
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
    """Comprueba el dígito de control de un IBAN (cuenta bancaria internacional): el resto
    de dividir por 97 tiene que ser 1.
    """
    compact = re.sub(r"\s", "", candidate)
    if not 15 <= len(compact) <= 34:
        return False
    rearranged = compact[4:] + compact[:4]
    return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1


_DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"


def _dni_valid(candidate: str) -> bool:
    """Comprueba la letra de control de un DNI español (8 dígitos y una letra)."""
    return _DNI_LETTERS[int(candidate[:8]) % 23] == candidate[8]


def _nie_valid(candidate: str) -> bool:
    """Comprueba la letra de control de un NIE español (la X, Y o Z inicial se cambia por
    0, 1 o 2).
    """
    number = int(str("XYZ".index(candidate[0])) + candidate[1:8])
    return _DNI_LETTERS[number % 23] == candidate[8]


@dataclass(frozen=True)
class _PersonalDataRule:
    """Una regla para detectar un tipo de dato personal.

    `kind` es el nombre del tipo, `token` el texto que lo reemplaza (por ejemplo
    "[TARJETA]"), `pattern` la expresión regular que lo encuentra e `is_valid` una
    comprobación opcional de dígitos de control.
    """

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

PERSONAL_DATA_LABELS = {
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
            """Reemplaza un dato encontrado por su marcador, salvo que no pase la
            comprobación de control.

            Si no pasa, devuelve el texto tal cual (era un número cualquiera). Si pasa,
            anota el tipo detectado y conserva el prefijo opcional que haya antes del
            dato.
            """
            if rule.is_valid is not None and not rule.is_valid(match.group(0)):
                return match.group(0)
            if rule.kind not in kinds:
                kinds.append(rule.kind)
            return (match.groupdict().get("prefix") or "") + rule.token

        text = rule.pattern.sub(replace, text)
    return text, tuple(kinds)
