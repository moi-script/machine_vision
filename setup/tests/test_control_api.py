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


class _FakeEngine:
    """Records control calls so we can assert start-takes-over policy without a camera."""
    def __init__(self, state):
        self._state = state
        self.calls = []

    @property
    def state(self):
        return self._state

    def stop(self):
        self.calls.append("stop")
        self._state = "idle"

    def start(self, session_id, difficulty, shots):
        self.calls.append("start")
        self._state = "running"

    def status(self):
        return {"type": "engine_status", "state": self._state,
                "difficulty": "medium", "camera": "ok"}


def test_start_takes_over_a_running_drill(monkeypatch):
    fake = _FakeEngine("running")
    monkeypatch.setattr("app.routers.control.get_engine", lambda: fake)
    r = client.post("/api/control/start", json={"sessionId": "s1", "difficulty": "medium"})
    assert r.status_code == 200
    # stop the old drill first, then start the new one — no 409
    assert fake.calls == ["stop", "start"]


def test_start_when_idle_does_not_stop_first(monkeypatch):
    fake = _FakeEngine("idle")
    monkeypatch.setattr("app.routers.control.get_engine", lambda: fake)
    r = client.post("/api/control/start", json={"sessionId": "s1", "difficulty": "medium"})
    assert r.status_code == 200
    assert fake.calls == ["start"]
