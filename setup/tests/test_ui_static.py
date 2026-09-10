import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import ui_static  # noqa: E402


def _bundle(tmp_path: Path) -> Path:
    """Minimal stand-in for a Vite build output."""
    ui = tmp_path / "ui"
    (ui / "assets").mkdir(parents=True)
    (ui / "index.html").write_text("<html><body>AeroSense</body></html>", encoding="utf-8")
    (ui / "assets" / "app.js").write_text("console.log('hi')", encoding="utf-8")
    return ui


def test_mount_returns_false_when_bundle_absent(tmp_path):
    """The Windows dev flow has no bundle; the API must still boot."""
    app = FastAPI()
    assert ui_static.mount_ui(app, tmp_path / "nope") is False


def test_api_still_works_when_bundle_absent(tmp_path):
    app = FastAPI()

    @app.get("/api/health")
    def health():
        return {"ok": True}

    ui_static.mount_ui(app, tmp_path / "nope")
    client = TestClient(app)
    assert client.get("/api/health").json() == {"ok": True}


def test_mount_serves_index_at_root(tmp_path):
    app = FastAPI()
    assert ui_static.mount_ui(app, _bundle(tmp_path)) is True
    res = TestClient(app).get("/")
    assert res.status_code == 200
    assert "AeroSense" in res.text


def test_mount_serves_assets(tmp_path):
    app = FastAPI()
    ui_static.mount_ui(app, _bundle(tmp_path))
    res = TestClient(app).get("/assets/app.js")
    assert res.status_code == 200
    assert "console.log" in res.text


def test_api_routes_win_over_the_static_mount(tmp_path):
    """The whole design rests on /api never being shadowed by the / mount."""
    app = FastAPI()

    @app.get("/api/health")
    def health():
        return {"ok": True}

    ui_static.mount_ui(app, _bundle(tmp_path))
    client = TestClient(app)
    assert client.get("/api/health").json() == {"ok": True}
    assert "AeroSense" in client.get("/").text


def test_a_route_registered_after_the_mount_is_shadowed(tmp_path):
    """Why mount_ui must be called last: a Mount at / matches every path,
    so any route added after it becomes unreachable. This test documents
    the failure mode rather than endorsing it."""
    app = FastAPI()
    ui_static.mount_ui(app, _bundle(tmp_path))

    @app.get("/api/health")
    def health():
        return {"ok": True}

    assert TestClient(app).get("/api/health").status_code == 404
