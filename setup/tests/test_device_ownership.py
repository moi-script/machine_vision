import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import pipeline, virtual_camera as vcam  # noqa: E402
from app.routers import cameras, control  # noqa: E402
from app.server import app  # noqa: E402

client = TestClient(app)


class _FakeWorker:
    def __init__(self, frame):
        self._f = frame
        self.alive = True

    def latest_frame(self):
        return self._f.copy()


def test_grab_uses_running_worker_instead_of_opening(monkeypatch):
    frame = np.full((10, 20, 3), 7, dtype=np.uint8)
    monkeypatch.setattr(pipeline, "get", lambda cid: _FakeWorker(frame))

    def boom(cid):
        raise AssertionError("opened the device while a worker holds it")
    monkeypatch.setattr(vcam, "_open", boom)
    fid, img = vcam.grab("left")
    assert img.shape == (10, 20, 3)
    assert vcam.frozen("left")["frame_id"] == fid


def test_frame_count_of_a_device_does_not_open_it(monkeypatch):
    from app import sources
    monkeypatch.setattr(sources, "get", lambda cid: {"kind": "device", "path": "/dev/aero-left"})
    monkeypatch.setattr(vcam, "_open", lambda cid: (_ for _ in ()).throw(AssertionError("opened")))
    assert vcam.frame_count("left") == 0


def test_start_ignores_a_model_field(monkeypatch):
    seen = {}
    monkeypatch.setattr(cameras, "_engine_holds_front", lambda: False)
    monkeypatch.setattr(cameras.pipeline, "start",
                        lambda cid, raw=False, backend=None, target_fps=30.0:
                        seen.setdefault("a", (cid, raw)) and {"camera_id": cid})
    r = client.post("/api/cameras/left/start", json={"model": "pose"})
    assert r.status_code == 200
    assert seen["a"] == ("left", False)


def test_front_is_borrowed_while_engine_runs(monkeypatch):
    monkeypatch.setattr(cameras, "_engine_holds_front", lambda: True)
    monkeypatch.setattr(cameras.pipeline, "start",
                        lambda *a, **k: pytest.fail("must not open front"))
    r = client.post("/api/cameras/front/start", json={})
    assert r.status_code == 200
    assert r.json()["borrowed"] is True


def test_engine_start_frees_front_first(monkeypatch):
    order = []

    class Eng:
        state = "idle"
        session_id = None
        def start(self, *a, **k): order.append("engine")
        def status(self): return {"state": "running"}
    monkeypatch.setattr(control, "get_engine", lambda: Eng())
    monkeypatch.setattr(control.pipeline, "stop", lambda cid: order.append(f"stop:{cid}"))
    r = client.post("/api/control/start", json={"sessionId": "s1"})
    assert r.status_code == 200
    assert order == ["stop:front", "engine"]


def test_control_status():
    r = client.get("/api/control/status")
    assert r.status_code == 200
    assert "state" in r.json()
