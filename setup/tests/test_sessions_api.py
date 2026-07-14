from fastapi.testclient import TestClient
from app.server import app
from app import db

client = TestClient(app)


def setup_function():
    db.sessions().delete_many({"title": "SESSTEST"})


def _make():
    return client.post("/api/sessions", json={
        "title": "SESSTEST", "scheduledAt": "2026-07-15T09:00:00",
        "difficulty": "hard", "assignedPlayerIds": ["p1"],
    }).json()


def test_create_session_defaults():
    s = _make()
    assert s["id"].startswith("s")
    assert s["liveData"] == []
    assert s["startedAt"] is None


def test_start_demotes_other_live_sessions():
    a = _make()
    b = _make()
    client.patch(f"/api/sessions/{a['id']}", json={"status": "live"})
    client.patch(f"/api/sessions/{b['id']}", json={"status": "live"})
    rows = {s["id"]: s for s in client.get("/api/sessions").json()}
    assert rows[a["id"]]["status"] == "scheduled"
    assert rows[b["id"]]["status"] == "live"
    assert rows[b["id"]]["startedAt"] is not None
