# tests/test_control_arm.py
"""Recognition-only (unarmed) start + arm-in-place, so identity acquisition can
run face recognition before the feeder fires. See app/routers/control.py."""
from app.engine import DrillEngine
import app.routers.control as control


def test_engine_starts_armed_by_default_and_arm_flips():
    e = DrillEngine()
    assert e._armed is True          # normal starts fire immediately
    e._armed = False                 # recognition-only
    e.arm()
    assert e._armed is True


def test_stop_is_safe_when_thread_never_started():
    # A concurrent start can leave a freshly-created, not-yet-started Thread in
    # _thread; stop() must not crash with "cannot join thread before started".
    import threading
    e = DrillEngine()
    e._thread = threading.Thread(target=lambda: None)  # created, never started
    e._state = "running"
    e.stop()
    assert e.state == "idle"
    assert e._thread is None


class _FakeEng:
    def __init__(self, state="idle", session_id=None):
        self.state = state
        self._session_id = session_id
        self.calls = []

    @property
    def session_id(self):
        return self._session_id

    def arm(self):
        self.calls.append("arm")

    def stop(self):
        self.calls.append("stop")
        self.state = "idle"

    def start(self, sid, diff, shots, armed=True):
        self.calls.append(("start", sid, armed))
        self.state = "running"
        self._session_id = sid

    def status(self):
        return {"state": self.state}


def test_start_same_session_arms_without_restart(monkeypatch):
    fake = _FakeEng(state="running", session_id="s1")
    monkeypatch.setattr(control, "get_engine", lambda: fake)
    control.start(control.StartBody(sessionId="s1", armed=True))
    # Armed in place — no camera teardown, no restart.
    assert fake.calls == ["arm"]


def test_start_same_session_unarmed_is_noop(monkeypatch):
    fake = _FakeEng(state="running", session_id="s1")
    monkeypatch.setattr(control, "get_engine", lambda: fake)
    control.start(control.StartBody(sessionId="s1", armed=False))
    # Already running that session, not arming — leave it recognizing.
    assert fake.calls == []


def test_start_different_session_takes_over(monkeypatch):
    fake = _FakeEng(state="running", session_id="old")
    monkeypatch.setattr(control, "get_engine", lambda: fake)
    control.start(control.StartBody(sessionId="new", armed=False))
    assert fake.calls[0] == "stop"
    assert fake.calls[1] == ("start", "new", False)


def test_start_from_idle_passes_armed_through(monkeypatch):
    fake = _FakeEng(state="idle", session_id=None)
    monkeypatch.setattr(control, "get_engine", lambda: fake)
    control.start(control.StartBody(sessionId="s1", armed=False))
    assert fake.calls == [("start", "s1", False)]
