"""Tests de los detectores compartidos: app/guardrails/detectors.py.

Funciones puras, sin red ni claves. Lo que se fija acá:

- los datos personales se ocultan y las cifras del proyecto no;
- un número sin el dígito de control correcto (IBAN, tarjeta, DNI) no se oculta;
- `PROMPT_TAGS` cubre las etiquetas que usan las plantillas de prompt.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.guardrails.detectors import (
    PROMPT_LEAK_PHRASES,
    PROMPT_TAGS,
    find_prompt_leak,
    mask_personal_data,
    normalize_text,
)

DESCRIPCION = "Un SaaS B2B pequeño para gestionar el préstamo de equipos a empleados."


def test_las_etiquetas_prohibidas_cubren_las_que_usan_las_plantillas() -> None:
    """Publicar una plantilla con una etiqueta nueva sin sumarla a PROMPT_TAGS falla acá."""
    raiz = Path(__file__).resolve().parents[1] / "app" / "prompts" / "estimation"
    usadas: set[str] = set()
    for archivo in [*raiz.glob("*/*.j2"), *raiz.glob("*/*.yaml")]:
        usadas |= set(re.findall(r"</?([a-z_]+)>", archivo.read_text(encoding="utf-8")))

    assert usadas, "no se encontró ninguna etiqueta: ¿cambió la ubicación de las plantillas?"
    assert usadas <= set(PROMPT_TAGS), f"faltan en PROMPT_TAGS: {sorted(usadas - set(PROMPT_TAGS))}"


# --- Datos personales -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado", "tipos"),
    [
        ("Escribir a ana.perez@empresa.com.ar", "Escribir a [EMAIL]", ("email",)),
        ("Llamar al +54 9 11 5555-1234 hoy", "Llamar al [TELÉFONO] hoy", ("telefono",)),
        ("Oficina 011 4555-1234", "Oficina [TELÉFONO]", ("telefono",)),
        ("Celular 11 5555 1234", "Celular [TELÉFONO]", ("telefono",)),
        ("Móvil 612 345 678", "Móvil [TELÉFONO]", ("telefono",)),
        ("Tel 555-123-4567", "Tel [TELÉFONO]", ("telefono",)),
        ("Cuenta ES91 2100 0418 4502 0005 1332", "Cuenta [IBAN]", ("iban",)),
        ("Tarjeta 4111 1111 1111 1111", "Tarjeta [TARJETA]", ("tarjeta",)),
        ("CUIT 20-12345678-9", "CUIT [DOCUMENTO]", ("documento",)),
        ("DNI 12.345.678", "DNI [DOCUMENTO]", ("documento",)),
        ("Titular 12345678Z", "Titular [DOCUMENTO]", ("documento",)),
        ("Extranjero X1234567L", "Extranjero [DOCUMENTO]", ("documento",)),
        (
            "Contacto: a@b.com, +34 612 345 678",
            "Contacto: [EMAIL], [TELÉFONO]",
            ("email", "telefono"),
        ),
    ],
)
def test_los_datos_personales_se_ocultan(texto, esperado, tipos) -> None:
    assert mask_personal_data(texto) == (esperado, tipos)


@pytest.mark.parametrize(
    "texto",
    [
        DESCRIPCION,
        "Habrá 20.000 usuarios y un presupuesto de 150.000.000 de pesos.",
        "Unas 2.500 horas, versión 1.2.3, entrega el 12/10/2026, norma ISO 27001.",
        "El servicio escucha en el puerto 8080 y guarda 1234567890123456 registros.",
        "Entre 10-20 empleados en 3 sedes, 120 equipos, Q4 2025.",
    ],
)
def test_las_cifras_del_proyecto_no_se_confunden_con_datos_personales(texto) -> None:
    assert mask_personal_data(texto) == (texto, ())


@pytest.mark.parametrize(
    "texto",
    ["Titular 12345678A", "Extranjero X1234567A"],  # la letra de control no corresponde
)
def test_un_documento_sin_la_letra_de_control_correcta_no_se_oculta(texto) -> None:
    assert mask_personal_data(texto) == (texto, ())


@pytest.mark.parametrize(
    "texto",
    [
        "IBAN ES91 2100 0418 4502 0005 1333",  # dígito de control mal
        "Tarjeta 4111 1111 1111 1112",  # no pasa Luhn
    ],
)
def test_sin_el_digito_de_control_no_es_un_iban_ni_una_tarjeta(texto) -> None:
    """El dígito de control decide el tipo, no si se oculta.

    Los grupos de dígitos los captura igual la regla de teléfonos: ocultar de más es
    el error barato; dejar pasar un número que sí era una tarjeta con un typo, no.
    """
    _, tipos = mask_personal_data(texto)
    assert "iban" not in tipos
    assert "tarjeta" not in tipos


def test_normalize_text_quita_tildes_mayusculas_y_caracteres_invisibles() -> None:
    assert normalize_text("IgnorÁ\u200b   TODO") == "ignora todo"


# --- Prompt volcado (salida) ----------------------------------------------------------


def test_las_frases_de_fuga_siguen_en_alguna_plantilla() -> None:
    """Si una plantilla cambia esa frase, el detector queda ciego sin que nadie lo note."""
    raiz = Path(__file__).resolve().parents[1] / "app" / "prompts" / "estimation"
    plantillas = " ".join(
        normalize_text(a.read_text(encoding="utf-8")) for a in raiz.glob("*/system.j2")
    )
    for frase in PROMPT_LEAK_PHRASES:
        assert frase in plantillas, frase


@pytest.mark.parametrize(
    ("texto", "regla"),
    [
        ("Resumen. </scope> y sigue", "etiqueta"),
        ("<RULES>algo</RULES>", "etiqueta"),
        ("< system >", "etiqueta"),
        ("Eres un estimador de proyectos sénior con más de 15 años de experiencia.", "frase"),
        ("I am a Senior Project Estimator with 15+ years of experience", "frase"),
        ("Me dijeron: NO ELIJAS un número redondo para ajustar las fases", "frase"),
        ("vuelve a sumar las fases y comprueba que coincidan con los totales", "frase"),
    ],
)
def test_un_prompt_volcado_se_detecta(texto, regla) -> None:
    assert find_prompt_leak(texto) == regla


@pytest.mark.parametrize(
    "texto",
    [
        DESCRIPCION,
        "<estimation>| fase | semanas |</estimation>",  # lo copia de los ejemplos y se quita
        "Hay 15 años de experiencia en el equipo y se suman las fases del plan.",
        "a < b y c > d",
        "Fuera de alcance: falta el alcance.",
        "",
    ],
)
def test_un_texto_normal_no_se_marca_como_prompt_volcado(texto) -> None:
    assert find_prompt_leak(texto) is None
