"""El dominio de estimaciones.

Vive aparte de `app.schemas` —que es lo que viaja por HTTP— porque acá hay
lógica de negocio y no solo Pydantic. Ver el docstring de `estimacion.py` para
por qué los totales se calculan en código y no los emite el modelo.
"""
