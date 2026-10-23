# Imagen única para la API y el frontend: mismas dependencias (uv.lock), dos
# comandos distintos en docker-compose.yml.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Primero solo las dependencias: esta capa se reutiliza mientras uv.lock no
# cambie, así un cambio de código no reinstala todo.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Sin root: el proceso no necesita privilegios. El código se copia a nombre de
# ese usuario para que pueda leerlo sean cuales sean los permisos que tengan los
# archivos en la máquina que construye la imagen (con 600, root sería el dueño y
# el usuario `app` no podría leerlos).
RUN useradd --create-home --uid 1000 app
COPY --chown=app:app app ./app
COPY --chown=app:app streamlit_app.py ./
USER app
