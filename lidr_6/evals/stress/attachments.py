"""Bloque 3 de LIDR_6: escenario de adjuntos grandes.

QUÉ GENERA ESTE MÓDULO

PDFs sintéticos de Lorem Ipsum calibrados a cinco tamaños (`SIZE_POINTS_BYTES`): 0 (sin
adjunto, línea base), 5 KB, 20 KB, 50 KB y 100 KB (cerca del tope por archivo de
`app.services.attachments.MAX_TEXT_CHARS`, 60.000 caracteres: a este tamaño el adjunto
entra recortado). No se usa ninguna librería de generación de PDF (reportlab, fpdf2):
se escribe el PDF a mano, con el mismo formato mínimo que ya prueba
`test/test_attachments.py::hacer_pdf` contra el extractor real (`pypdf`), solo que con
páginas largas en vez de una línea corta. El tamaño en bytes es casi proporcional al
texto incrustado (sin fuentes embebidas ni compresión), así que calibrarlo es ajustar
cuántos caracteres de Lorem Ipsum entran.

Cada PDF lleva un requisito funcional concreto al principio (`MARKER_START`) y otro,
distinto, exactamente al final del texto (`MARKER_END`) — no códigos opacos tipo
"MARCADOR-X", sino una frase que un relevamiento real traería ("el sistema debe
integrarse con SAP...", "...debe avisar por notificación push..."). Eso separa dos
preguntas distintas sobre el mismo escenario, y la frase (en vez de un código) importa
para la segunda:

1. ¿El recorte de `MAX_TEXT_CHARS` llega a tocar el adjunto? (`local_truncation_report`,
   GRATIS: llama directo a `extract_attachment`, sin red ni LLM, y busca la frase
   completa, textual, en lo que de verdad se envía). Con 100 KB el requisito del final
   queda del lado recortado; con los demás tamaños, los dos sobreviven.
2. ¿El modelo USA ese requisito al armar la estimación? (el runner HTTP, con costo
   real). Acá un código opaco no serviría: nada en el prompt le pide al modelo citarlo,
   así que buscar "MARCADOR-INICIO-7F3A" en la respuesta casi nunca da positivo aunque
   el modelo haya leído el adjunto perfectamente bien. Un requisito funcional sí puede
   aparecer de forma natural — en una fase, un supuesto, el equipo propuesto — si el
   modelo lo incorporó. Por eso el recall en la respuesta no busca la frase entera sino
   una palabra distintiva de cada una (`RECALL_KEYWORD_START` / `RECALL_KEYWORD_END`,
   p. ej. "sap" y "push": palabras que el relleno de Lorem Ipsum nunca produce, así que
   un positivo es, con alta probabilidad, el modelo reflejando el requisito real y no
   una coincidencia del filler.

CÓMO SE USA

Generar y medir la parte local (sin costo, sin red):

    python evals/stress/attachments.py --local

Correr el escenario completo contra el servicio real (un turno por tamaño, mismo
`transcript` y `project_type` en los cinco, variando solo el adjunto) y medir las tres
curvas que pide el bloque 3 (latencia, coste, recall del marcador en el summary):

    python evals/stress/attachments.py

Las claves de los proveedores se leen del `.env` de la raíz, como en el resto del
servicio.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent.parent
# Para poder hacer `from app... import ...` corriendo este archivo suelto.
sys.path.insert(0, str(RAIZ))

from app.services.attachments import MAX_TEXT_CHARS, extract_attachment

# --- Los cinco puntos del escenario -------------------------------------------------------

# En KB decimales (1 KB = 1.000 bytes), no KiB: son más fáciles de leer en un reporte y
# la diferencia no cambia nada de lo que este escenario mide.
SIZE_POINTS_KB: tuple[int, ...] = (0, 5, 20, 50, 100)

TRANSCRIPT = (
    "Necesitamos estimar un proyecto de gestión documental para una empresa mediana."
)
PROJECT_TYPE = "web_saas"
DETAIL_LEVEL = "medium"
OUTPUT_FORMAT = "phases_table"

# --- Generación del PDF sintético ---------------------------------------------------------

_LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis "
    "nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. "
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore "
    "eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt "
    "in culpa qui officia deserunt mollit anim id est laborum. "
)

# Caracteres de texto por página del PDF sintético (no afecta la extracción, solo cómo
# se reparte el texto entre páginas; ver `--- Page N ---` en los adjuntos reales).
PAGE_CHARS = 2500

# Dos requisitos funcionales, uno justo al principio del texto y otro justo al final.
# Si el PDF se recorta por `MAX_TEXT_CHARS`, el que desaparece primero es el final.
# Sin tildes a propósito: el PDF mínimo (ver `_build_pdf`) no declara codificación de
# fuente, así que el texto incrustado tiene que ser ASCII puro.
MARKER_START = (
    "Requisito: el sistema debe integrarse con el ERP SAP via el protocolo IDoc "
    "para sincronizar el inventario."
)
MARKER_END = (
    "Requisito: el sistema debe enviar notificaciones push a los celulares de "
    "los supervisores cuando se detecta un error critico."
)

# Palabras distintivas de cada requisito, para buscar RECALL (no la frase textual) en
# lo que el modelo escribe: el filler de Lorem Ipsum jamás las produce por azar, así
# que encontrarlas en la respuesta es evidencia de que el modelo usó el adjunto, no
# una coincidencia con el relleno.
RECALL_KEYWORD_START = "sap"
RECALL_KEYWORD_END = "push"


def _filler(n_chars: int) -> str:
    """`n_chars` caracteres de Lorem Ipsum, repitiendo el párrafo base lo que haga falta."""
    if n_chars <= 0:
        return ""
    repeticiones = n_chars // len(_LOREM) + 1
    return (_LOREM * repeticiones)[:n_chars]


def _texto_con_marcadores(total_chars: int) -> str:
    """El texto completo del adjunto: `MARKER_START`, relleno de Lorem Ipsum y
    `MARKER_END` exactamente al final (para que un recorte que llegue hasta ahí se lo
    lleve primero a él, nunca al del principio).
    """
    envoltura = len(MARKER_START) + len(MARKER_END) + 2  # los dos espacios separadores
    cuerpo = _filler(max(total_chars - envoltura, 0))
    return f"{MARKER_START} {cuerpo} {MARKER_END}"


def _marker_spans(texto: str) -> tuple[tuple[int, int], ...]:
    """Dónde viven `MARKER_START` y `MARKER_END` dentro de `texto` (que siempre los
    tiene al principio y al final, por construcción de `_texto_con_marcadores`), para
    pasárselo a `_build_pdf` como `protect_spans` y que ningún corte de página caiga
    a mitad de uno de los dos."""
    return (
        (0, len(MARKER_START)),
        (len(texto) - len(MARKER_END), len(texto)),
    )


def _paginar(texto: str, page_chars: int, protect_spans: tuple[tuple[int, int], ...] = ()) -> list[str]:
    """Corta `texto` en páginas de `page_chars`, salvo que el corte caiga DENTRO de un
    tramo de `protect_spans` (pares `(inicio, fin)` de índices de `texto`) — en ese caso
    empuja el corte hasta el final de ese tramo, para que quede entero en una sola
    página. Hace falta porque el extractor real intercala `--- Page N ---` entre
    páginas (ver `app.services.attachments.extract_attachment`): si un corte cayera a
    mitad de `MARKER_START` o `MARKER_END` (plausible ahora que son frases largas, no
    códigos cortos), ese separador quedaría incrustado DENTRO del requisito y rompería
    la comparación textual exacta de `local_truncation_report`, sin que el servicio
    haya recortado nada de verdad.
    """
    if not texto:
        return [""]
    paginas: list[str] = []
    i = 0
    n = len(texto)
    while i < n:
        fin = min(i + page_chars, n)
        for inicio_prot, fin_prot in protect_spans:
            if inicio_prot < fin < fin_prot:
                fin = fin_prot
        paginas.append(texto[i:fin])
        i = fin
    return paginas


def _build_pdf(
    texto: str,
    *,
    page_chars: int = PAGE_CHARS,
    protect_spans: tuple[tuple[int, int], ...] = (),
) -> bytes:
    """Un PDF mínimo con `texto` repartido en páginas de `page_chars`, en el mismo
    formato (sin fuentes embebidas, sin compresión) que `test/test_attachments.py:hacer_pdf`,
    generalizado para páginas largas en vez de una sola línea corta por página.

    Cada página es UNA sola cadena entre paréntesis en el operador `Tj` del PDF: no se
    hace salto de línea real, así que el texto no se ve "bien" si se abre el PDF, pero
    `pypdf` lo extrae igual (lee el literal del operador, no renderiza la página) y es
    lo único que importa para este escenario. `protect_spans` evita que un corte de
    página caiga a mitad de un tramo protegido (ver `_paginar`).
    """
    paginas = _paginar(texto, page_chars, protect_spans)
    objetos: list[bytes] = []
    n = len(paginas)
    ids_paginas = [4 + 2 * i for i in range(n)]
    objetos.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{i} 0 R" for i in ids_paginas)
    objetos.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    objetos.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, pagina in enumerate(paginas):
        contenido = f"BT /F1 10 Tf 40 760 Td ({pagina}) Tj ET".encode() if pagina else b""
        objetos.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>".encode()
        )
        objetos.append(
            b"<< /Length "
            + str(len(contenido)).encode()
            + b" >>\nstream\n"
            + contenido
            + b"\nendstream"
        )
    salida = bytearray(b"%PDF-1.4\n")
    posiciones = []
    for numero, cuerpo in enumerate(objetos, start=1):
        posiciones.append(len(salida))
        salida += f"{numero} 0 obj\n".encode() + cuerpo + b"\nendobj\n"
    inicio_xref = len(salida)
    salida += f"xref\n0 {len(objetos) + 1}\n".encode() + b"0000000000 65535 f \n"
    for pos in posiciones:
        salida += f"{pos:010d} 00000 n \n".encode()
    salida += (
        f"trailer\n<< /Size {len(objetos) + 1} /Root 1 0 R >>\nstartxref\n{inicio_xref}\n%%EOF\n"
    ).encode()
    return bytes(salida)


@dataclass(frozen=True)
class SyntheticAttachment:
    """Un adjunto sintético ya calibrado: sus bytes, el tamaño que se le pidió y el
    texto EXACTO que se incrustó (antes de que el servicio lo recorte, si lo recorta).
    """

    size_kb: int
    pdf_bytes: bytes
    embedded_text: str

    @property
    def filename(self) -> str:
        return f"adjunto_{self.size_kb}kb.pdf"


def build_attachment(size_kb: int, *, tolerancia_bytes: int = 300, intentos: int = 6) -> SyntheticAttachment:
    """El PDF calibrado para pesar `size_kb` KB (±`tolerancia_bytes`: 300 por defecto,
    un margen de hasta ~0,6% en el punto más chico, cómodo para un escenario de estrés
    que no necesita el byte exacto).

    La relación entre caracteres de texto y bytes del PDF es casi lineal (sin fuentes
    embebidas ni compresión: el peso es, sobre todo, el propio texto más una cabecera
    fija por página), así que ajustar la diferencia una o dos veces alcanza para
    acercarse. Cerca de un salto de página (cada `PAGE_CHARS` caracteres, el PDF agrega
    una página entera de cabecera) el ajuste puede estabilizarse un poco por encima o
    por debajo del objetivo en vez de clavarlo exacto: no converge más lejos de eso, así
    que más intentos no ayudan una vez que `len(pdf)` deja de cambiar.
    """
    target_bytes = size_kb * 1000
    chars = target_bytes  # primera aproximación: ~1 byte de PDF por carácter de texto
    pdf = b""
    texto = ""
    largo_anterior = None
    for _ in range(intentos):
        texto = _texto_con_marcadores(chars)
        pdf = _build_pdf(texto, protect_spans=_marker_spans(texto))
        if len(pdf) == largo_anterior:
            break  # punto fijo: seguir ajustando no va a acercarlo más
        largo_anterior = len(pdf)
        diferencia = target_bytes - len(pdf)
        if abs(diferencia) <= tolerancia_bytes:
            break
        chars = max(len(MARKER_START) + len(MARKER_END) + 2, chars + diferencia)
    return SyntheticAttachment(size_kb=size_kb, pdf_bytes=pdf, embedded_text=texto)


# --- Medición local: qué le llega de verdad al modelo, sin llamarlo -----------------------


def local_truncation_report(attachment: SyntheticAttachment) -> dict:
    """Lo que `extract_attachment` (el extractor real del servicio) hace con este PDF,
    SIN llamar a ningún proveedor: cuántos caracteres tiene el texto completo, cuántos
    sobreviven al recorte de `MAX_TEXT_CHARS` y si cada marcador queda dentro de lo que
    de verdad se manda al modelo. Es la parte determinista y gratis de este escenario.
    """
    extraido = extract_attachment(attachment.filename, attachment.pdf_bytes)
    enviado = extraido.text  # YA viene recortado a MAX_TEXT_CHARS, si corresponde
    completo = attachment.embedded_text
    return {
        "size_kb": attachment.size_kb,
        "pdf_bytes": len(attachment.pdf_bytes),
        "chars_del_pdf": len(completo),
        "chars_enviados_al_modelo": len(enviado),
        "truncado": len(completo) > MAX_TEXT_CHARS,
        "marcador_inicio_sobrevive": MARKER_START in enviado,
        "marcador_final_sobrevive": MARKER_END in enviado,
        "avisos": list(extraido.warnings),
    }


def imprimir_reporte_local(reportes: list[dict]) -> None:
    """La tabla de la parte local: un renglón por tamaño, sin tocar el servicio."""
    print(f"\n{'KB':>5} {'bytes PDF':>10} {'chars PDF':>10} {'chars enviados':>15} "
          f"{'truncado':>9} {'inicio':>7} {'final':>7}")
    for r in reportes:
        print(
            f"{r['size_kb']:>5} {r['pdf_bytes']:>10} {r['chars_del_pdf']:>10} "
            f"{r['chars_enviados_al_modelo']:>15} {'sí' if r['truncado'] else 'no':>9} "
            f"{'sí' if r['marcador_inicio_sobrevive'] else 'no':>7} "
            f"{'sí' if r['marcador_final_sobrevive'] else 'no':>7}"
        )


# --- Runner HTTP: las tres curvas contra el servicio real ---------------------------------


def _coste_de_los_logs(logs: list[dict]) -> float:
    """Igual que en `evals/run.py`: suma `coste_usd` de todos los eventos capturados
    durante la llamada (cubre reintentos y, si algún día se corre en modo acb, también
    las vueltas del crítico)."""
    return sum(evento.get("coste_usd", 0.0) or 0.0 for evento in logs if "coste_usd" in evento)


def correr_punto(cliente, size_kb: int) -> dict:
    """Un turno completo (sesión nueva, mismo `TRANSCRIPT` y `PROJECT_TYPE` siempre,
    el adjunto es la única variable) y lo que costó, tardó y si el modelo reflejó los
    marcadores del adjunto en su respuesta final.
    """
    from fastapi.testclient import TestClient  # import tardío: solo hace falta acá
    from structlog.testing import capture_logs

    assert isinstance(cliente, TestClient)
    session_id = cliente.post("/sessions").json()["session_id"]

    files = None
    local = None
    if size_kb > 0:
        attachment = build_attachment(size_kb)
        local = local_truncation_report(attachment)
        files = [("attachments", (attachment.filename, attachment.pdf_bytes, "application/pdf"))]

    with capture_logs() as logs:
        inicio = time.perf_counter()
        respuesta = cliente.post(
            f"/sessions/{session_id}/estimate",
            data={
                "transcript": TRANSCRIPT,
                "project_type": PROJECT_TYPE,
                "detail_level": DETAIL_LEVEL,
                "output_format": OUTPUT_FORMAT,
            },
            files=files,
        )
        latencia = time.perf_counter() - inicio
    coste = _coste_de_los_logs(logs)

    recall_inicio = recall_final = None
    if respuesta.status_code == 200 and size_kb > 0:
        # Recall por PALABRA DISTINTIVA, no por la frase completa: ver el docstring
        # del módulo (punto 2) sobre por qué buscar el requisito textual no serviría.
        cuerpo_texto = str(respuesta.json().get("estimation", {})).lower()
        recall_inicio = RECALL_KEYWORD_START in cuerpo_texto
        recall_final = RECALL_KEYWORD_END in cuerpo_texto

    return {
        "size_kb": size_kb,
        "status_code": respuesta.status_code,
        "latencia_s": round(latencia, 2),
        "coste_usd": round(coste, 6),
        "truncado": local["truncado"] if local else False,
        "marcador_inicio_enviado": local["marcador_inicio_sobrevive"] if local else None,
        "marcador_final_enviado": local["marcador_final_sobrevive"] if local else None,
        "marcador_inicio_en_respuesta": recall_inicio,
        "marcador_final_en_respuesta": recall_final,
    }


def imprimir_curvas(filas: list[dict]) -> None:
    """Las tres curvas del bloque 3: latencia, coste y recall de los marcadores, lado a
    lado por tamaño, para leerlas de un vistazo."""
    print(
        f"\n{'KB':>5} {'estado':>7} {'latencia_s':>10} {'coste_usd':>10} "
        f"{'truncado':>9} {'env.ini':>8} {'env.fin':>8} {'resp.ini':>9} {'resp.fin':>9}"
    )
    for f in filas:
        def marca(valor):
            return "-" if valor is None else ("sí" if valor else "no")

        print(
            f"{f['size_kb']:>5} {f['status_code']:>7} {f['latencia_s']:>10} "
            f"{f['coste_usd']:>10} {'sí' if f['truncado'] else 'no':>9} "
            f"{marca(f['marcador_inicio_enviado']):>8} {marca(f['marcador_final_enviado']):>8} "
            f"{marca(f['marcador_inicio_en_respuesta']):>9} {marca(f['marcador_final_en_respuesta']):>9}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--local",
        action="store_true",
        help="Solo la parte gratis (extracción y recorte); no llama al servicio ni al LLM",
    )
    ap.add_argument(
        "--salida",
        default=str(Path(__file__).resolve().parent / "resultados_adjuntos.csv"),
        help="CSV con el detalle por tamaño",
    )
    args = ap.parse_args()

    reportes_locales = [local_truncation_report(build_attachment(kb)) for kb in SIZE_POINTS_KB if kb > 0]
    imprimir_reporte_local(reportes_locales)

    if args.local:
        return

    from app.main import create_app
    from fastapi.testclient import TestClient

    cliente = TestClient(create_app())
    filas = [correr_punto(cliente, kb) for kb in SIZE_POINTS_KB]
    imprimir_curvas(filas)

    with open(args.salida, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)
    print(f"\nCSV con el detalle: {args.salida}")


if __name__ == "__main__":
    main()
