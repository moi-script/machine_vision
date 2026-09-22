import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import pipeline  # noqa: E402


@pytest.fixture(autouse=True)
def _no_mongo(monkeypatch):
    monkeypatch.setattr(pipeline, "_load_calibration", lambda cid: None)


def test_slot_models_are_fixed():
    assert pipeline.SLOT_MODELS == {
        "front": ["shuttle", "pose"],
        "left": ["landed"], "right": ["landed"], "back": ["landed"],
        "face": [],
    }


def test_models_for_raw_is_empty():
    assert pipeline.models_for("front", raw=True) == []
    assert pipeline.models_for("left") == ["landed"]


def test_models_for_unknown_slot():
    with pytest.raises(ValueError):
        pipeline.models_for("ceiling")


def test_heads_for_tick_all_when_not_alternating():
    assert pipeline.heads_for_tick(2, 0, False) == [0, 1]
    assert pipeline.heads_for_tick(2, 7, False) == [0, 1]


def test_heads_for_tick_alternates():
    assert [pipeline.heads_for_tick(2, t, True) for t in range(4)] == [[0], [1], [0], [1]]


def test_heads_for_tick_single_head_ignores_alternate():
    assert pipeline.heads_for_tick(1, 3, True) == [0]


def test_worker_builds_one_head_per_model():
    w = pipeline.Worker("front", ["shuttle", "pose"])
    assert [h.key for h in w.heads] == ["shuttle", "pose"]
    assert w.stats["model"] == "shuttle+pose"


def test_raw_worker_has_no_heads():
    w = pipeline.Worker("face", [])
    assert w.heads == []
    assert w.stats["model"] == "none"


def test_latest_frame_is_a_copy():
    w = pipeline.Worker("left", [])
    assert w.latest_frame() is None
    f = np.zeros((4, 4, 3), dtype=np.uint8)
    w._frame = f
    got = w.latest_frame()
    got[0, 0, 0] = 255
    assert f[0, 0, 0] == 0


def test_start_uses_the_slot_map(monkeypatch):
    seen = {}

    class FakeWorker:
        def __init__(self, camera_id, model_keys, backend=None, target_fps=30.0):
            seen["args"] = (camera_id, model_keys)
            self.stats = {"camera_id": camera_id}

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(pipeline, "Worker", FakeWorker)
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    pipeline.start("front")
    assert seen["args"] == ("front", ["shuttle", "pose"])
    pipeline.start("left", raw=True)
    assert seen["args"] == ("left", [])
    pipeline.stop_all()
