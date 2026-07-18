from fastapi.testclient import TestClient
from app.server import app
from app import db

client = TestClient(app)


def setup_function():
    db.sessions().delete_many({"title": "SESSTEST"})
    db.attendance().delete_many({"_id": "attSESSTEST"})


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


def test_delete_removes_session_and_its_attendance():
    s = _make()
    db.attendance().insert_one({"_id": "attSESSTEST", "sessionId": s["id"], "playerId": "p1"})
    assert client.delete(f"/api/sessions/{s['id']}").status_code == 200
    assert s["id"] not in {row["id"] for row in client.get("/api/sessions").json()}
    assert db.attendance().find_one({"_id": "attSESSTEST"}) is None


def test_delete_live_session_is_refused():
    s = _make()
    client.patch(f"/api/sessions/{s['id']}", json={"status": "live"})
    res = client.delete(f"/api/sessions/{s['id']}")
    assert res.status_code == 409
    assert s["id"] in {row["id"] for row in client.get("/api/sessions").json()}


def test_delete_missing_session_404s():
    assert client.delete("/api/sessions/nope").status_code == 404
