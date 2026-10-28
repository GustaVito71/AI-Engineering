# Bloque 5 — Reporte de estrés (LIDR_6)

Generado por `evals/stress/run.py` a partir de `results.csv`. Ver el docstring del módulo para qué mide cada barrido y las limitaciones de `--http`.

## 1. Resumen

| Métrica | Valor |
| --- | --- |
| Latencia P50 (ms) | 8920.2 |
| Latencia P95 (ms) | 16801.0 |
| Coste acumulado por escenario (USD, promedio entre repeticiones) | contradicting-1: $0.00242, contradicting-3: $0.01194, contradicting-6: $0.03120, growing-1: $0.00329, growing-3: $0.01101, growing-6: $0.02591, pivoting-1: $0.00230, pivoting-3: $0.01174, pivoting-6: $0.03027 |
| Hit rate caché exacta | 77.8% |
| Hit rate caché semántica | 0.0% |
| Hit rate de cualquier caché (fallback en --http) | 77.8% |
| Recall medio del fact-tracker (todas las filas con hechos) | 92.6% |

### 1a. Desglose por escenario × tamaño de adjunto

Los tres barridos miden ejes distintos (memoria: perfil × N de turnos; adjuntos: solo tamaño, sin sesión; caché: solo perfil, un turno suelto), así que en vez de forzar una sola tabla cruzada con columnas vacías donde el eje no aplica, van tres tablas — una por barrido, cada una con sus combinaciones reales:

**Memoria conversacional (perfil × N de turnos):**

| Perfil | N | Latencia P50 | Latencia P95 | Coste total | Recall al último turno |
| --- | --- | --- | --- | --- | --- |
| growing | 1 | 9781 ms | 12212 ms | $0.00988 | 100% |
| growing | 3 | 10861 ms | 14586 ms | $0.03302 | 100% |
| growing | 6 | 8972 ms | 15119 ms | $0.07773 | 83% |
| pivoting | 1 | 3699 ms | 3791 ms | $0.00689 | 100% |
| pivoting | 3 | 8104 ms | 12890 ms | $0.03521 | 100% |
| pivoting | 6 | 10386 ms | 15488 ms | $0.09081 | 83% |
| contradicting | 1 | 3299 ms | 3909 ms | $0.00727 | 100% |
| contradicting | 3 | 8915 ms | 13713 ms | $0.03583 | 100% |
| contradicting | 6 | 9227 ms | 19640 ms | $0.08690 | 80% |

**Adjuntos (tamaño, sin escenario ni sesión):**

| Tamaño | Latencia P50 | Latencia P95 | Coste total |
| --- | --- | --- | --- |
| 0 KB | 3557 ms | 4137 ms | $0.00680 |
| 5 KB | 9004 ms | 9254 ms | $0.01362 |
| 20 KB | 9421 ms | 9454 ms | $0.01717 |
| 50 KB | 8246 ms | 9015 ms | $0.02739 |
| 100 KB | 8955 ms | 9764 ms | $0.02937 |

**Caché (perfil, un turno suelto repetido 3 veces):**

| Perfil | Latencia P50 (con hits) | Hit rate caché exacta | Hit rate caché semántica |
| --- | --- | --- | --- |
| growing | 356 ms | 3/3 (100%)\* | 0/3 |
| pivoting | 321 ms | 2/3 (67%) | 0/3 |
| contradicting | 288 ms | 2/3 (67%) | 0/3 |

\* `growing` acertó también en su repetición 1 por un hit real sobre una entrada que quedó en Redis de una corrida anterior (ver nota más abajo) — no es parte del patrón `miss→exacta→exacta` que sí muestran `pivoting` y `contradicting`.

> **Notas de esta corrida:** (1) un turno de `contradicting` N=6 repetición 2 (turno 4) devolvió 502 transitorio; los dos turnos siguientes de esa repetición no se mandaron, así que esa repetición queda con 4 de 6 turnos — no afecta las demás filas ni el resto de perfiles. (2) El perfil `growing` muestra `exacta` ya en su repetición 1: no es un error, es un acierto real sobre una entrada que quedó en Redis (TTL 24h) de una corrida corta de verificación anterior con el mismo texto; `pivoting` y `contradicting`, con texto nuevo para Redis, muestran el patrón limpio (`miss` en la 1, `exacta` en la 2 y 3).

## 2. Curvas

### 2a. Latencia vs. tokens

| Tokens (in+out) | n | Latencia media (ms) |
| --- | --- | --- |
| 5500-5999 | 21 | 3416 |
| 6000-6499 | 28 | 9857 |
| 6500-6999 | 25 | 9817 |
| 7000-7499 | 2 | 9290 |
| 8000-8499 | 2 | 8658 |
| 8500-8999 | 1 | 9282 |
| 10500-10999 | 3 | 9050 |
| 16000-16499 | 1 | 8246 |
| 16500-16999 | 2 | 8651 |
| 18000-18499 | 3 | 9098 |
| 19500-19999 | 1 | 13737 |
| 20000-20499 | 6 | 15884 |
| 20500-20999 | 1 | 13844 |
| 21000-21499 | 4 | 16031 |
| 39500-39999 | 1 | 20268 |
| 43000-43499 | 1 | 28751 |

### 2b. Coste acumulado vs. turno

| Perfil | Turno | Coste acumulado medio (USD) |
| --- | --- | --- |
| contradicting | 1 | $0.00239 |
| contradicting | 2 | $0.00742 |
| contradicting | 3 | $0.01379 |
| contradicting | 4 | $0.02075 |
| contradicting | 5 | $0.02320 |
| contradicting | 6 | $0.03221 |
| growing | 1 | $0.00312 |
| growing | 2 | $0.00654 |
| growing | 3 | $0.01059 |
| growing | 4 | $0.01422 |
| growing | 5 | $0.02152 |
| growing | 6 | $0.02591 |
| pivoting | 1 | $0.00229 |
| pivoting | 2 | $0.00579 |
| pivoting | 3 | $0.01144 |
| pivoting | 4 | $0.01832 |
| pivoting | 5 | $0.02272 |
| pivoting | 6 | $0.03027 |

### 2c. Recall vs. N

| Perfil | N (turnos) | Recall al último turno |
| --- | --- | --- |
| contradicting | 1 | 100% |
| contradicting | 3 | 100% |
| contradicting | 6 | 80% |
| growing | 1 | 100% |
| growing | 3 | 100% |
| growing | 6 | 83% |
| pivoting | 1 | 100% |
| pivoting | 3 | 100% |
| pivoting | 6 | 83% |

## 3. Dónde empieza a romperse mi CAG, y por qué

**Memoria conversacional.** El recall al último turno cae por debajo del 100% a partir de N=6 (de los N medidos en esta corrida: [1, 3, 6]); con menos turnos no se llega a perder nada. La ventana deslizante es la sospechosa natural (los hechos viejos dependen de haber sido promovidos a ancla o resumen antes de salir de la ventana; ver `app/sessions/compression.py`), pero esta corrida mide el síntoma (qué se recuerda), no la causa exacta turno por turno.

**Caché y adjuntos.** El adjunto de 100 KB vuelve a perder el requisito del final por el recorte de `MAX_TEXT_CHARS` (igual que en el Bloque 3): el límite de entrada de texto rompe antes que cualquier otra cosa medida acá. La caché exacta, pero no la semántica, acertó en el barrido de caché, como se esperaba con texto idéntico repetido (la semántica solo se consulta cuando la exacta falla, así que con repeticiones literales rara vez llega a intervenir). Más de fondo: memoria conversacional, adjuntos y caché son tres caminos que en este servicio NO se solapan (`estimate_conversation` nunca consulta ninguna caché, y `estimate_structured`, el único con caché, no acepta adjuntos ni sesión). Es ahí donde mi CAG empieza a romperse: cualquier escenario real que combine las tres cosas a la vez (una conversación larga, con adjuntos, que además debería cachear lo repetido) hoy no tiene un camino único que lo cubra.

**¿Qué dimensión domina la degradación?** Memoria, no latencia ni coste. La latencia crece con los tokens pero de forma acotada y predecible (tabla 1a: incluso N=6 se mantiene bajo 20s en P95); el coste crece linealmente con los turnos sin ningún salto (de $0.007 en N=1 a ~$0.08-0.09 en N=6, ver 2b) — ninguno de los dos "se rompe", solo escalan. El recall, en cambio, es la única métrica que deja de ser 100% y lo hace justo (y solo) a partir de N=6, sin aviso previo entre N=3 y N=6. Es la dimensión que falla de forma discreta (binaria: se olvidó o no), mientras las otras dos degradan de forma continua y manejable.

**¿Qué caso límite justificaría saltar a RAG?** El mismo punto de quiebre: una conversación que supere las ~6-10 interacciones Y dependa de un hecho mencionado una sola vez al principio (sin que se haya promovido a ancla o resumen). Ahí la memoria por ventana deslizante deja de ser confiable y un RAG (recuperar el hecho exacto por búsqueda, en vez de esperar que sobreviva en la ventana) empieza a pagar su complejidad. Por debajo de ese umbral — conversaciones cortas, o hechos que se repiten varias veces y entran al resumen — la memoria actual alcanza; migrar a RAG antes de ese punto sería complejidad sin beneficio medido en esta corrida.
