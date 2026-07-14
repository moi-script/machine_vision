import numpy as np
from app.engine import DrillEngine
from app import events, streamer


def test_engine_starts_idle():
    e = DrillEngine()
    assert e.state == "idle"
    assert e.status()["state"] == "idle"


def test_pause_resume_toggles_state():
    e = DrillEngine()
    e._state = "running"          # simulate a running loop
    e.pause()
    assert e.state == "paused"
    e.resume()
    assert e.state == "running"


def test_set_difficulty_updates_status():
    e = DrillEngine()
    e.set_difficulty("hard")
    assert e.status()["difficulty"] == "hard"


def test_publish_frame_encodes_jpeg_to_buffer():
    e = DrillEngine()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    e._publish(frame)
    assert streamer.buffer.latest() is not None
    assert streamer.buffer.latest()[:2] == b"\xff\xd8"  # JPEG SOI
