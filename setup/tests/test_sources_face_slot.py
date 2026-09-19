import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from app import sources  # noqa: E402
from config import settings  # noqa: E402


class _NoDB:
    def __init__(self):
        self.calls = []

    def find_one(self, *a, **k):
        return None

    def update_one(self, *a, **k):
        self.calls.append((a, k))


class FakeCap:
    def __init__(self):
        self.props = {}

    def set(self, prop, val):
        self.props[prop] = val
        return True


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(sources, "_col", lambda: _NoDB())
    with sources._lock:
        sources._cache.clear()
    yield
    with sources._lock:
        sources._cache.clear()


def test_face_is_a_slot_but_not_a_court_camera():
    assert sources.FACE_ID == "face"
    assert "face" in sources.SLOT_IDS
    assert "face" not in sources.CAMERA_IDS


def test_face_defaults_to_the_udev_name_on_linux(monkeypatch):
    monkeypatch.setattr(sources.sys, "platform", "linux")
    assert sources._default_for("face") == {"kind": "device", "path": "/dev/aero-face"}


def test_face_defaults_to_index_0_on_windows(monkeypatch):
    monkeypatch.setattr(sources.sys, "platform", "win32")
    assert sources._default_for("face") == {"kind": "device", "index": 0}


@pytest.mark.parametrize("slot", ["left", "right", "back"])
def test_court_side_slots_default_to_the_udev_name_on_linux(monkeypatch, slot):
    monkeypatch.setattr(sources.sys, "platform", "linux")
    assert sources._default_for(slot) == {"kind": "device", "path": f"/dev/aero-{slot}"}


@pytest.mark.parametrize("slot", ["left", "right", "back"])
def test_court_side_slots_keep_bundled_footage_on_windows(monkeypatch, slot):
    monkeypatch.setattr(sources.sys, "platform", "win32")
    assert sources._default_for(slot) == sources._DEFAULTS[slot]


def test_front_keeps_bundled_default_on_linux(monkeypatch):
    monkeypatch.setattr(sources.sys, "platform", "linux")
    assert sources._default_for("front") == sources._DEFAULTS["front"]


def test_face_source_can_be_set():
    src = sources.set_source("face", "device", path="/dev/aero-face")
    assert src == {"kind": "device", "path": "/dev/aero-face"}


def test_unknown_slot_still_rejected():
    with pytest.raises(KeyError):
        sources.get("ceiling")


def test_missing_face_device_reports_unavailable(monkeypatch):
    monkeypatch.setattr(sources.sys, "platform", "linux")
    d = sources.describe("face")
    assert d["available"] is os.path.exists("/dev/aero-face")


@pytest.mark.parametrize("slot", ["front", "left", "right", "back", "face"])
def test_v4l2_size_is_per_slot(slot):
    cap = sources._configure_v4l2(FakeCap(), slot)
    w, h = settings.SLOT_FRAME_SIZE[slot]
    assert cap.props[cv2.CAP_PROP_FRAME_WIDTH] == w
    assert cap.props[cv2.CAP_PROP_FRAME_HEIGHT] == h


def test_v4l2_without_slot_keeps_global_size():
    cap = sources._configure_v4l2(FakeCap())
    assert cap.props[cv2.CAP_PROP_FRAME_WIDTH] == settings.FRAME_WIDTH
    assert cap.props[cv2.CAP_PROP_FRAME_HEIGHT] == settings.FRAME_HEIGHT


def test_esp32_slots_request_the_firmware_mode():
    assert settings.SLOT_FRAME_SIZE["left"] == (640, 480)
    assert settings.SLOT_FRAME_SIZE["face"] == (800, 600)
