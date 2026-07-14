from fastapi.testclient import TestClient
from app.server import app

client = TestClient(app)


def test_difficulty_sets_engine():
    r = client.post("/api/control/difficulty", json={"difficulty": "hard"})
    assert r.status_code == 200
    assert r.json()["status"]["difficulty"] == "hard"


def test_pause_when_idle_is_noop_ok():
    r = client.post("/api/control/pause")
    assert r.status_code == 200
    assert r.json()["status"]["state"] in ("idle", "paused")
