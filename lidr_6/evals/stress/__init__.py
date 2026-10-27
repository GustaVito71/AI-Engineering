"""Generadores de conversaciones sintéticas para estresar la memoria de sesión.

A diferencia de `evals/golden_dataset.json` (casos fijos, pensados para detectar una
regresión puntual), este paquete genera conversaciones MÁS LARGAS y con un patrón
deliberado, para forzar comportamientos que solo aparecen con muchos turnos: que un
hecho antiguo sobreviva a la ventana deslizante, que un cambio de rumbo no deje
restos, que una corrección se imponga a lo que dijo antes.
"""
