"""Benchmark del extractor de project_metadata (lidr_5).

Compara DeepSeek (deepseek-flash), OpenAI (gpt-4o-mini), Claude Haiku 5.5 y
Gemini (gemini-3.1-flash-lite) sobre el mismo conjunto de transcripciones etiquetadas.

Nombre sin prefijo test_ a propósito: pytest no debe recogerlo.

Uso (desde la raíz de lidr_5):
    python test/bench_extractor.py --dataset test/casos_extractor.jsonl
    python test/bench_extractor.py --dataset test/casos_extractor.jsonl --modelos deepseek,haiku55

Formato del dataset (JSONL, un caso por línea):
    {"id": "caso-01", "transcript": "...",
     "expected": {"project_name": "...", "assumed_team_size": 4,
                  "mentioned_technologies": ["FastAPI"], "agreed_scope": "..."}}

Las claves se leen de .env en la raíz de lidr_5. Nunca se imprimen.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SALIDA_POR_DEFECTO = Path(__file__).resolve().parent / "resultados_extractor.csv"

PROMPT = (
    "Extrae del texto los hechos del proyecto en curso. Responde SOLO con un JSON "
    "válido con estas claves:\n"
    '- "project_name": string o null\n'
    '- "assumed_team_size": entero o null\n'
    '- "mentioned_technologies": lista de strings\n'
    '- "agreed_scope": string breve o null\n'
    "Si un dato no aparece en el texto, usa null (o [] para la lista). No inventes datos.\n\n"
    "Texto:\n"
)

# Precios USD por millón de tokens: (entrada sin caché, salida).
# DeepSeek: página oficial de precios (pico 01-04 y 06-10 UTC, L-V; valle = mitad).
# OpenAI gpt-4o-mini: coincide con el coste registrado en med.csv; verificar en openai.com.
# Haiku 5.5: docs oficiales de API (EN y ES) en platform.claude.com, tramo hasta 100k tokens: 0.10 / 0.50.
# claude.com/es/pricing no lo lista. Nuestros prompts son cortos, así que aplica el tramo bajo.
# Gemini 3.1 Flash-Lite: página oficial de precios, texto (no audio): 0.25 / 1.50.
PRECIOS = {
    "deepseek": {"valle": (0.15, 0.6), "pico": (0.3, 1.2)},
    "openai": {"valle": (0.15, 0.6), "pico": (0.15, 0.6)},
    "haiku55": {"valle": (0.10, 0.50), "pico": (0.10, 0.50)},
    "gemini": {"valle": (0.25, 1.5), "pico": (0.25, 1.5)},
}

CLAVES_ENV = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "haiku55": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def cargar_env(ruta):
    env = {}
    p = Path(ruta)
    if not p.exists():
        return env
    for linea in p.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        k, v = linea.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def tramo_deepseek(ahora_utc):
    """Pico: L-V, 01:00-04:00 y 06:00-10:00 UTC. Resto: valle."""
    if ahora_utc.weekday() >= 5:
        return "valle"
    h = ahora_utc.hour
    return "pico" if (1 <= h < 4) or (6 <= h < 10) else "valle"


def _post(url, cabeceras, cuerpo):
    datos = json.dumps(cuerpo).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=datos,
        headers={**cabeceras, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        cuerpo_err = e.read().decode("utf-8", "replace")[:200]
        raise RuntimeError(f"HTTP {e.code}: {cuerpo_err}")


def llamar_openai_compatible(base, clave, modelo, texto):
    r = _post(
        base + "/chat/completions",
        {"Authorization": "Bearer " + clave},
        {
            "model": modelo,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": PROMPT + texto}],
        },
    )
    contenido = r["choices"][0]["message"]["content"] or ""
    uso = r.get("usage", {})
    return contenido, uso.get("prompt_tokens", 0), uso.get("completion_tokens", 0)


def llamar_anthropic(clave, modelo, texto):
    r = _post(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": clave, "anthropic-version": "2023-06-01"},
        {
            "model": modelo,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": PROMPT + texto}],
        },
    )
    contenido = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
    uso = r.get("usage", {})
    return contenido, uso.get("input_tokens", 0), uso.get("output_tokens", 0)


def llamar_gemini(clave, modelo, texto):
    r = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent",
        {"x-goog-api-key": clave},
        {
            "contents": [{"parts": [{"text": PROMPT + texto}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        },
    )
    partes = r["candidates"][0].get("content", {}).get("parts", [])
    contenido = "".join(p.get("text", "") for p in partes)
    uso = r.get("usageMetadata", {})
    return contenido, uso.get("promptTokenCount", 0), uso.get("candidatesTokenCount", 0)


def parsear_json(texto):
    try:
        return json.loads(texto), True
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"\{.*\}", texto or "", re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)), True
        except json.JSONDecodeError:
            pass
    return None, False


def normalizar_tec(lista):
    if not isinstance(lista, list):
        return set()
    return {str(x).strip().lower() for x in lista if str(x).strip()}


def puntuar(pred, esperado):
    """Devuelve métricas de un caso. pred puede ser None si el JSON no es válido."""
    if pred is None:
        return {"nombre_ok": 0, "equipo_ok": 0, "tech_f1": 0.0, "alcance_presente": 0}
    nombre_p = (pred.get("project_name") or "").strip().lower()
    nombre_e = (esperado.get("project_name") or "").strip().lower()
    equipo_ok = int(pred.get("assumed_team_size") == esperado.get("assumed_team_size"))

    tp = normalizar_tec(pred.get("mentioned_technologies"))
    te = normalizar_tec(esperado.get("mentioned_technologies"))
    if not tp and not te:
        f1 = 1.0
    else:
        inter = len(tp & te)
        prec = inter / len(tp) if tp else 0.0
        rec = inter / len(te) if te else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    alcance_presente = int(
        bool((pred.get("agreed_scope") or "").strip())
        == bool((esperado.get("agreed_scope") or "").strip())
    )
    return {
        "nombre_ok": int(nombre_p == nombre_e),
        "equipo_ok": equipo_ok,
        "tech_f1": round(f1, 3),
        "alcance_presente": alcance_presente,
    }


def coste_usd(modelo, tramo, tin, tout):
    tarifa = PRECIOS[modelo][tramo]
    if tarifa is None:
        return None
    return (tin * tarifa[0] + tout * tarifa[1]) / 1_000_000


def cargar_casos(ruta):
    casos = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                casos.append(json.loads(linea))
    return casos


def verificacion_cruzada(precios):
    """Contrasta PRECIOS con llmprice-kit (opcional: pip install llmprice-kit).

    Compara el tramo de valle de cada modelo. Si un modelo no existe en
    llmprice-kit, lo indica en lugar de fallar.
    """
    try:
        from llmprice import LLMPrice
    except ImportError:
        print("\n[Verificación cruzada omitida: instala llmprice-kit]")
        return

    lp = LLMPrice()
    ids = {
        "deepseek": "deepseek-flash",
        "openai": "gpt-4o-mini",
        "haiku55": "claude-haiku-5-5",
        "gemini": "gemini-3.1-flash-lite-preview",
    }
    print(f"\nVerificación cruzada con llmprice-kit ({lp.data_age()}):")
    print(
        f"{'modelo':<9} {'id llmprice':<30} {'oficial (in/out)':<22} "
        f"{'llmprice (in/out)':<22} resultado"
    )
    for modelo, tarifa_oficial in precios.items():
        oficial = tarifa_oficial["valle"]
        ref = ids[modelo]
        if oficial is None:
            print(f"{modelo:<9} {ref:<30} {'n/d':<22} {'-':<22} sin precio oficial")
            continue
        try:
            mp = lp.get(ref)
        except KeyError:
            oficial_txt = f"{oficial[0]:.2f} / {oficial[1]:.2f}"
            print(
                f"{modelo:<9} {ref:<30} {oficial_txt:<22} {'-':<22} no encontrado en llmprice-kit"
            )
            continue
        ok = (
            abs(mp.input_cost_per_1m - oficial[0]) < 1e-6
            and abs(mp.output_cost_per_1m - oficial[1]) < 1e-6
        )
        oficial_txt = f"{oficial[0]:.2f} / {oficial[1]:.2f}"
        llmprice_txt = f"{mp.input_cost_per_1m:.2f} / {mp.output_cost_per_1m:.2f}"
        resultado = "coincide" if ok else "DIFIERE"
        print(f"{modelo:<9} {mp.name:<30} {oficial_txt:<22} {llmprice_txt:<22} {resultado}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True, help="Ruta al JSONL de casos etiquetados")
    ap.add_argument("--modelos", default="deepseek,openai,haiku55,gemini")
    ap.add_argument("--env", default=str(RAIZ / ".env"))
    ap.add_argument("--salida", default=str(SALIDA_POR_DEFECTO))
    args = ap.parse_args()

    env = {**cargar_env(args.env), **os.environ}
    modelos = [m.strip() for m in args.modelos.split(",") if m.strip()]
    # Nombres de modelo: se leen de .env; si faltan, se usan los valores por defecto.
    nombres_modelo = {
        "deepseek": env.get("DEEPSEEK_MODEL") or "deepseek-flash",
        "openai": env.get("OPENAI_MODEL") or "gpt-4o-mini",
        "haiku55": env.get("ANTHROPIC_MODEL") or "claude-haiku-5-5",
        "gemini": env.get("GEMINI_MODEL") or "gemini-3.1-flash-lite",
    }
    print("Modelos: " + ", ".join(f"{k}={nombres_modelo[k]}" for k in modelos))
    for m in modelos:
        if m not in PRECIOS:
            sys.exit(f"Modelo desconocido: {m} (usa deepseek, openai, haiku55, gemini)")
        if not env.get(CLAVES_ENV[m]):
            sys.exit(f"Falta {CLAVES_ENV[m]} en {args.env}")

    if not Path(args.dataset).exists():
        sys.exit(
            f"No existe el dataset: {args.dataset}. Crea el JSONL con el formato indicado en la cabecera del script."
        )
    casos = cargar_casos(args.dataset)
    if not casos:
        sys.exit(f"El dataset está vacío: {args.dataset}")

    filas = []
    for caso in casos:
        for m in modelos:
            inicio = time.time()
            error = ""
            tin = tout = 0
            ahora = datetime.now(UTC)
            tramo = tramo_deepseek(ahora) if m == "deepseek" else "valle"
            try:
                if m == "deepseek":
                    texto, tin, tout = llamar_openai_compatible(
                        "https://api.deepseek.com",
                        env["DEEPSEEK_API_KEY"],
                        nombres_modelo["deepseek"],
                        caso["transcript"],
                    )
                elif m == "openai":
                    texto, tin, tout = llamar_openai_compatible(
                        "https://api.openai.com/v1",
                        env["OPENAI_API_KEY"],
                        nombres_modelo["openai"],
                        caso["transcript"],
                    )
                elif m == "gemini":
                    texto, tin, tout = llamar_gemini(
                        env["GEMINI_API_KEY"], nombres_modelo["gemini"], caso["transcript"]
                    )
                else:
                    texto, tin, tout = llamar_anthropic(
                        env["ANTHROPIC_API_KEY"], nombres_modelo["haiku55"], caso["transcript"]
                    )
            except Exception as e:  # noqa: BLE001 - una caída no debe parar el benchmark
                texto, error = "", str(e)
            latencia = round(time.time() - inicio, 2)

            pred, valido = parsear_json(texto)
            metricas = puntuar(pred if valido else None, caso["expected"])
            c = coste_usd(m, tramo, tin, tout)
            fila = {
                "id": caso["id"],
                "modelo": m,
                "latencia_s": latencia,
                "valido": int(valido),
                **metricas,
                "tokens_in": tin,
                "tokens_out": tout,
                "tramo": tramo,
                "coste_usd": "" if c is None else round(c, 6),
                "error": error,
            }
            filas.append(fila)
            aviso_error = ("ERROR: " + error) if error else ""
            print(
                f"{caso['id']:<12} {m:<9} valido={int(valido)} "
                f"nombre={int(metricas['nombre_ok'])} equipo={int(metricas['equipo_ok'])} "
                f"f1={metricas['tech_f1']:.2f} {latencia:5.1f}s {aviso_error}"
            )

    with open(args.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    print(f"\nResumen ({len(casos)} casos):")
    print(
        f"{'modelo':<9} {'valido%':>8} {'nombre%':>8} {'equipo%':>8} "
        f"{'tech_f1':>8} {'lat_s':>8} {'coste_total':>12}"
    )
    for m in modelos:
        grupo = [f for f in filas if f["modelo"] == m]
        n = len(grupo)
        costes = [f["coste_usd"] for f in grupo if f["coste_usd"] != ""]
        coste_txt = (f"{sum(costes):.5f}") if len(costes) == n and n else "n/d"
        valido_pct = 100 * sum(f["valido"] for f in grupo) / n
        nombre_pct = 100 * sum(f["nombre_ok"] for f in grupo) / n
        equipo_pct = 100 * sum(f["equipo_ok"] for f in grupo) / n
        tech_f1 = sum(f["tech_f1"] for f in grupo) / n
        latencia_media = sum(f["latencia_s"] for f in grupo) / n
        print(
            f"{m:<9} {valido_pct:7.0f}% {nombre_pct:7.0f}% {equipo_pct:7.0f}% "
            f"{tech_f1:8.2f} {latencia_media:8.2f} {coste_txt:>12}"
        )
    verificacion_cruzada(PRECIOS)
    print(f"\nCSV con detalle por caso: {args.salida}")


if __name__ == "__main__":
    main()
