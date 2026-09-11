import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import sources  # noqa: E402
from config import settings  # noqa: E402

BY_ID_DIR = "/dev/v4l/by-id"


class FakeCap:
    """Records every .set() so we can assert on capture configuration."""

    def __init__(self):
        self.props = {}

    def set(self, prop, val):
        self.props[prop] = val
        return True

    def isOpened(self):
        return True

    def read(self):
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        pass


class _NoDB:
    """Stands in for the camera_sources collection.

    set_source() upserts into the real aerosense database, which holds the
    operator's live camera assignments. These tests must never write there.
    """

    def find_one(self, *a, **k):
        return None

    def update_one(self, *a, **k):
        return None


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    db = _NoDB()
    monkeypatch.setattr(sources, "_col", lambda: db)
    with sources._lock:
        sources._cache.clear()
    yield db
    with sources._lock:
        sources._cache.clear()


# ── capture configuration ───────────────────────────────────

def test_v4l2_capture_requests_mjpg_and_resolution(monkeypatch):
    """Four raw 1280x800 streams exceed the Pi's shared USB3 bandwidth."""
    cap = FakeCap()
    monkeypatch.setattr(sources.sys, "platform", "linux")
    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda *a: cap)
    sources.set_source("front", "device", index=0)
    sources.open_capture("front")

    assert cap.props[sources.cv2.CAP_PROP_FOURCC] == \
        sources.cv2.VideoWriter_fourcc(*"MJPG")
    assert cap.props[sources.cv2.CAP_PROP_FRAME_WIDTH] == settings.FRAME_WIDTH
    assert cap.props[sources.cv2.CAP_PROP_FRAME_HEIGHT] == settings.FRAME_HEIGHT


def test_windows_capture_is_left_alone(monkeypatch):
    """DSHOW already works on the dev machine; do not touch it."""
    cap = FakeCap()
    monkeypatch.setattr(sources.sys, "platform", "win32")
    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda *a: cap)
    sources.set_source("front", "device", index=0)
    sources.open_capture("front")
    assert cap.props == {}


def test_file_sources_are_not_reconfigured(monkeypatch, tmp_path):
    """Forcing MJPG on a video file would corrupt decoding."""
    clip = tmp_path / "rally.mp4"
    clip.write_bytes(b"\x00")
    cap = FakeCap()
    monkeypatch.setattr(sources.sys, "platform", "linux")
    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda *a: cap)
    sources.set_source("front", "file", path=str(clip))
    sources.open_capture("front")
    assert cap.props == {}


# ── enumeration ─────────────────────────────────────────────

def test_linux_enumeration_returns_stable_paths(monkeypatch):
    names = ["usb-HBVCAM-Camera-video-index0", "usb-OV9281-video-index0"]
    monkeypatch.setattr(sources.sys, "platform", "linux")
    monkeypatch.setattr(sources.os.path, "isdir", lambda p: p == BY_ID_DIR)
    monkeypatch.setattr(sources.os, "listdir", lambda p: names)
    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda *a: FakeCap())

    found = sources.list_devices()
    assert [d["path"] for d in found] == [f"{BY_ID_DIR}/{n}" for n in names]
    assert found[0]["name"] == names[0]
    assert found[0]["width"] == 640 and found[0]["height"] == 480
    assert all("index" not in d for d in found)


def test_linux_enumeration_falls_back_to_indices(monkeypatch):
    """A Pi with no by-id directory must still enumerate something."""
    monkeypatch.setattr(sources.sys, "platform", "linux")
    monkeypatch.setattr(sources.os.path, "isdir", lambda p: False)
    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda *a: FakeCap())

    found = sources.list_devices(max_index=2)
    assert [d["index"] for d in found] == [0, 1]


def test_windows_enumeration_still_returns_indices(monkeypatch):
    monkeypatch.setattr(sources.sys, "platform", "win32")
    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda *a: FakeCap())
    found = sources.list_devices(max_index=2)
    assert [d["index"] for d in found] == [0, 1]
