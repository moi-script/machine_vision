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
