"""Serve the pre-built React bundle so the app is one process on one origin.

The bundle is built on the Windows dev machine (scripts/build_ui.ps1) and
committed to app/static/ui, so the Raspberry Pi needs no Node toolchain.

WHY THE DIRECTORY IS OPTIONAL
The bundle is committed, so a fresh clone already serves it at :8000/ - a
maintainer running the Windows dev workflow instead points a Vite dev server
at :5173 against this same API, and the two paths can disagree if the
committed bundle has gone stale relative to app/routers changes. mount_ui()
only becomes a no-op if app/static/ui is missing entirely (e.g. deleted, or
never checked out), in which case the API still behaves exactly as it always
has.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

UI_DIR = Path(__file__).parent / "static" / "ui"


def mount_ui(app: FastAPI, ui_dir: Path = UI_DIR) -> bool:
    """Mount `ui_dir` at / if it exists. Returns whether it was mounted.

    MUST be called after every route registration of any kind - routers
    registered via include_router() as well as routes declared directly with
    decorators (@app.get, @app.websocket, etc). A mount at / matches any
    unclaimed path, so routes registered afterwards would be shadowed.

    No SPA fallback is needed - App.tsx switches on state.ui.activePage and
    the UI has no react-router dependency, so the app lives at exactly one
    URL. Adding client-side routing means adding a 404-to-index.html handler
    here.
    """
    if not ui_dir.is_dir():
        return False
    app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")
    return True
