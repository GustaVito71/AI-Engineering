"""Lanzador del servicio: `uv run python -m app [--port N] [--reload]`.

`uvicorn app.main:app` a secas usa el 8000 (default del propio uvicorn) y
choca cuando otro servicio lo tiene tomado (p. ej. Ganttly en Docker).
Este lanzador usa APP_PORT de Settings, que por default es 8001."""
from __future__ import annotations

import argparse

import uvicorn

from .config import get_settings


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="python -m app")
    parser.add_argument("--port", type=int, default=settings.app_port)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("app.main:app", host="127.0.0.1", port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()