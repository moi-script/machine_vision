import os

import numpy as np
import pytest

from app import engine
from config import settings


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


def test_linux_device_index_requests_mjpg(monkeypatch):
    """The engine's own capture path (app/engine.py) never got the MJPG
    treatment app/sources.py got - a single uncompressed 1280x800 V4L2
    stream alone is ~61 MB/s on the Pi."""
    cap = FakeCap()
    monkeypatch.setattr(engine.os, "name", "posix")
    monkeypatch.setattr(engine.cv2, "VideoCapture", lambda *a: cap)

    engine._open_capture(0)

    assert cap.props[engine.cv2.CAP_PROP_FOURCC] == \
        engine.cv2.VideoWriter_fourcc(*"MJPG")
    assert cap.props[engine.cv2.CAP_PROP_FRAME_WIDTH] == settings.FRAME_WIDTH
    assert cap.props[engine.cv2.CAP_PROP_FRAME_HEIGHT] == settings.FRAME_HEIGHT


def test_linux_dev_path_requests_mjpg(monkeypatch):
    """A /dev/v4l/by-id path is a device, not a regular file."""
    cap = FakeCap()
    monkeypatch.setattr(engine.os, "name", "posix")
    monkeypatch.setattr(engine.cv2, "VideoCapture", lambda *a: cap)

    engine._open_capture("/dev/v4l/by-id/usb-OV9281-video-index0")

    assert cap.props[engine.cv2.CAP_PROP_FOURCC] == \
        engine.cv2.VideoWriter_fourcc(*"MJPG")


def test_windows_capture_is_left_alone(monkeypatch):
    """DSHOW already works on the dev machine; do not touch it."""
    cap = FakeCap()
    monkeypatch.setattr(engine.os, "name", "nt")
    monkeypatch.setattr(engine.cv2, "VideoCapture", lambda *a, **k: cap)

    engine._open_capture(0)

    assert cap.props == {}


def test_file_source_is_not_reconfigured(monkeypatch, tmp_path):
    """Forcing MJPG on a video file would corrupt decoding."""
    clip = tmp_path / "rally.mp4"
    clip.write_bytes(b"\x00")
    cap = FakeCap()
    monkeypatch.setattr(engine.os, "name", "posix")
    monkeypatch.setattr(engine.cv2, "VideoCapture", lambda *a: cap)

    engine._open_capture(str(clip))

    assert cap.props == {}


@pytest.mark.parametrize("source", [0, "/dev/v4l/by-id/usb-OV9281-video-index0"])
def test_is_v4l2_device_true_for_devices(source):
    assert engine._is_v4l2_device(source) is True


def test_is_v4l2_device_false_for_real_file(tmp_path):
    clip = tmp_path / "rally.mp4"
    clip.write_bytes(b"\x00")
    assert engine._is_v4l2_device(str(clip)) is False
