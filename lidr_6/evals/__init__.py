"""Framework de evals de LIDR_6: casos golden, métricas y generadores de estrés.

No se importa desde `app/`: es tooling para medir el servicio desde afuera, nunca
código de producción. `evals/run.py` corre los casos de `golden_dataset.json` contra
el servicio real; `evals/metrics.py` define las métricas binarias que los juzgan;
`evals/stress/` genera conversaciones sintéticas para estresar la memoria de sesión
con más turnos de los que cubre el dataset golden.
"""
