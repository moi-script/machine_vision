import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import sources  # noqa: E402

BY_ID = "/dev/v4l/by-id/usb-HBVCAM-Camera-video-index0"


class _NoDB:
    """Stands in for the camera_sources collection.

    set_source() upserts into the real aerosense database, which holds the
    operator's live camera assignments. These tests must never write there.
    Records update_one calls so tests can assert on the $set/$unset shape
    without ever touching Mongo.
    """

    def __init__(self):
        self.calls = []

    def find_one(self, *a, **k):
        return None

    def update_one(self, *a, **k):
        self.calls.append((a, k))
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


def test_device_source_accepts_a_path():
    src = sources.set_source("front", "device", path=BY_ID)
    assert src["kind"] == "device"
    assert src["path"] == BY_ID
    assert "index" not in src


def test_device_source_still_accepts_an_index():
    src = sources.set_source("front", "device", index=1)
    assert src == {"kind": "device", "index": 1}


def test_device_source_needs_index_or_path():
    with pytest.raises(ValueError, match="index or path"):
        sources.set_source("front", "device")


def test_describe_names_a_path_device_by_basename(monkeypatch):
    sources.set_source("front", "device", path=BY_ID)
    monkeypatch.setattr(sources.os.path, "exists", lambda p: p == BY_ID)
    d = sources.describe("front")
    assert d["name"] == "usb-HBVCAM-Camera-video-index0"
    assert d["available"] is True


def test_describe_reports_a_missing_path_device_unavailable():
    sources.set_source("front", "device", path="/dev/v4l/by-id/not-plugged-in")
    assert sources.describe("front")["available"] is False


def test_describe_names_an_index_device_by_index():
    sources.set_source("front", "device", index=2)
    assert sources.describe("front")["name"] == "device 2"


def test_a_path_device_is_not_a_file():
    """is_file() gates frame-stepping in the UI; a camera must never be one."""
    sources.set_source("front", "device", path=BY_ID)
    assert sources.is_file("front") is False


def test_open_capture_passes_the_path_to_opencv(monkeypatch):
    seen = {}

    class FakeCap:
        def set(self, *a):
            return True

    def fake_videocapture(arg, *rest):
        seen["arg"] = arg
        return FakeCap()

    monkeypatch.setattr(sources.cv2, "VideoCapture", fake_videocapture)
    sources.set_source("front", "device", path=BY_ID)
    sources.open_capture("front")
    assert seen["arg"] == BY_ID


def _last_update(db):
    """The $set/$unset document from the most recent update_one call."""
    args, _kwargs = db.calls[-1]
    return args[1]


def test_switching_path_to_index_unsets_the_stale_path(_isolate):
    sources.set_source("front", "device", path=BY_ID)
    sources.set_source("front", "device", index=1)
    update = _last_update(_isolate)
    assert update["$set"] == {"kind": "device", "index": 1}
    assert update["$unset"] == {"path": ""}


def test_switching_index_to_path_unsets_the_stale_index(_isolate):
    sources.set_source("front", "device", index=1)
    sources.set_source("front", "device", path=BY_ID)
    update = _last_update(_isolate)
    assert update["$set"] == {"kind": "device", "path": BY_ID}
    assert update["$unset"] == {"index": ""}


def test_open_capture_prefers_index_when_a_stale_path_lingers(monkeypatch):
    """Regression test for the bug itself.

    Before the fix, set_source()'s $set-only write let a leftover `path`
    key from an earlier file/path source survive alongside a newly set
    `index`. open_capture()'s `if "path" in src` check then opened the
    stale path instead of the camera the operator just picked. Simulate
    that doc shape directly in the cache (set_source() itself must no
    longer be able to produce it) and confirm the index wins.
    """
    with sources._lock:
        sources._cache["front"] = {
            "kind": "device", "path": BY_ID, "index": 1,
        }
    seen = {}

    class FakeCap:
        def set(self, *a):
            return True

    def fake_videocapture(arg, *rest):
        seen["arg"] = arg
        return FakeCap()

    monkeypatch.setattr(sources.cv2, "VideoCapture", fake_videocapture)
    sources.open_capture("front")
    assert seen["arg"] == 1


def test_label_survives_an_identity_change(_isolate):
    sources.set_source("front", "device", index=1, label="court cam")
    sources.set_source("front", "device", path=BY_ID, label="court cam")
    update = _last_update(_isolate)
    assert update["$set"] == {
        "kind": "device", "path": BY_ID, "label": "court cam",
    }
    assert update["$unset"] == {"index": ""}
    assert "label" not in update.get("$unset", {})
