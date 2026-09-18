import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import engine  # noqa: E402
from app.models import CameraSettings  # noqa: E402


class Cap:
    def __init__(self):
        self.props = {}

    def set(self, k, v):
        self.props[k] = v


def test_use_front_slot_defaults_on():
    assert CameraSettings().useFrontSlot is True


def test_slot_mode_opens_front_through_sources(monkeypatch):
    got = {}
    monkeypatch.setattr(engine._sources, "open_capture",
                        lambda cid: got.setdefault("cid", cid) and Cap())
    monkeypatch.setattr(engine, "_open_capture",
                        lambda s: (_ for _ in ()).throw(AssertionError("legacy path used")))
    engine._open_engine_capture(True, 1, 1280, 800, False)
    assert got["cid"] == "front"


def test_legacy_mode_keeps_the_settings_source(monkeypatch):
    cap = Cap()
    monkeypatch.setattr(engine, "_open_capture", lambda s: cap)
    out = engine._open_engine_capture(False, 1, 1280, 800, False)
    assert out is cap
    import cv2
    assert cap.props[cv2.CAP_PROP_FRAME_WIDTH] == 1280


class FakeWorker:
    def __init__(self, alive=True, frame=None):
        self.alive = alive
        self._frame = frame

    def latest_frame(self):
        return self._frame


def test_capture_frame_running_returns_buffer_unchanged(monkeypatch):
    """Existing behaviour: an engine that is running (not idle) with a
    published frame returns it straight from frame_buffer, without touching
    the pipeline worker or opening a capture at all."""
    eng = engine.DrillEngine()
    eng._state = "running"
    engine.frame_buffer.publish(b"already-published")
    from app import pipeline
    monkeypatch.setattr(pipeline, "get", lambda cid: (_ for _ in ()).throw(
                            AssertionError("pipeline touched while running")))
    monkeypatch.setattr(engine, "_open_engine_capture",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("capture opened while running")))
    try:
        assert eng.capture_frame() == b"already-published"
    finally:
        engine.frame_buffer.clear()


def test_capture_frame_reuses_front_worker_frame(monkeypatch):
    """Idle engine with a live 'front' pipeline worker: capture_frame must
    reuse the worker's latest frame and never call _open_engine_capture."""
    import numpy as np
    import cv2

    eng = engine.DrillEngine()
    eng._state = "idle"
    engine.frame_buffer.clear()
    frame = np.zeros((4, 4, 3), dtype="uint8")
    from app import pipeline
    monkeypatch.setattr(pipeline, "get",
                        lambda cid: FakeWorker(alive=True, frame=frame))
    monkeypatch.setattr(engine, "_open_engine_capture",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("opened a capture despite a live worker")))
    out = eng.capture_frame()
    assert isinstance(out, bytes)
    assert cv2.imdecode(np.frombuffer(out, dtype="uint8"), cv2.IMREAD_COLOR) is not None


def test_capture_frame_missing_file_raises_runtime_error(monkeypatch):
    """No live worker holding 'front', and the underlying source is a missing
    file: capture_frame must surface a RuntimeError (so control.py's existing
    `except RuntimeError` handling turns it into a clean 500), not a raw
    FileNotFoundError."""
    eng = engine.DrillEngine()
    eng._state = "idle"
    engine.frame_buffer.clear()
    from app import pipeline
    monkeypatch.setattr(pipeline, "get", lambda cid: None)

    def _raise(*a, **k):
        raise FileNotFoundError("front video file missing")

    monkeypatch.setattr(engine, "_open_engine_capture", _raise)
    try:
        eng.capture_frame()
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "camera unavailable" in str(exc)
    except FileNotFoundError:
        raise AssertionError("FileNotFoundError leaked past capture_frame")
