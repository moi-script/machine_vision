import numpy as np

from app.engine import DrillEngine
from utils.shuttle_motion import ShuttleMotionDetector

H, W = 360, 640
WARMUP = 30   # frames of empty court so MOG2 learns the background


def _court():
    """A textured but static background, like a real court."""
    rng = np.random.default_rng(0)
    return rng.integers(60, 90, (H, W, 3), dtype=np.uint8)


def _with_shuttle(bg, x, y, r=4):
    f = bg.copy()
    f[int(y) - r:int(y) + r, int(x) - r:int(x) + r] = 255
    return f


def _flight(n=25, x0=60, y0=300, vx=18, vy=-8):
    return [(x0 + vx * i, y0 + vy * i) for i in range(n)]


def _run(det, frames, boxes=()):
    return [det.update(f, boxes)[0] for f in frames]


def test_locks_onto_a_shuttle_in_flight_and_follows_it():
    bg = _court()
    det = ShuttleMotionDetector()
    _run(det, [bg] * WARMUP)
    path = _flight()
    out = _run(det, [_with_shuttle(bg, x, y) for x, y in path])

    measured = [(r, p) for r, p in zip(out, path) if r and r[2] == "det"]
    # Lock needs `confirm` frames of coherent motion, then it should hold.
    assert len(measured) >= len(path) - 6
    for r, (x, y) in measured:
        assert abs(r[0] - x) <= 6 and abs(r[1] - y) <= 6


def test_static_scene_never_locks():
    bg = _court()
    det = ShuttleMotionDetector()
    out = _run(det, [bg] * 80)
    assert all(r is None for r in out[WARMUP:])


def test_random_flicker_is_not_a_trajectory():
    """Blobs that pop up at random places are noise, not a shuttle."""
    bg = _court()
    rng = np.random.default_rng(1)
    det = ShuttleMotionDetector()
    _run(det, [bg] * WARMUP)
    frames = [_with_shuttle(bg, rng.integers(20, W - 20), rng.integers(20, H - 20))
              for _ in range(60)]
    out = _run(det, frames)
    assert all(r is None for r in out)


def test_motion_inside_a_player_box_is_ignored():
    bg = _court()
    det = ShuttleMotionDetector()
    _run(det, [bg] * WARMUP)
    path = _flight()
    player = [(0, 0, W, H)]          # the "player" covers the whole path
    out = _run(det, [_with_shuttle(bg, x, y) for x, y in path], player)
    assert all(r is None for r in out)


def test_coasts_through_a_short_occlusion():
    bg = _court()
    det = ShuttleMotionDetector()
    _run(det, [bg] * WARMUP)
    path = _flight(n=20)
    frames = [_with_shuttle(bg, x, y) for x, y in path]
    frames[14] = frames[15] = bg     # shuttle hidden for two frames
    out = _run(det, frames)
    assert out[14] is not None and out[14][2] == "coast"
    assert out[16] is not None and out[16][2] == "det"


def test_engine_motion_shuttle_returns_the_position():
    bg = _court()
    e = DrillEngine()
    assert e._motion_shuttle(bg, []) is None      # no detector: no position

    e._shuttle_motion = ShuttleMotionDetector()
    for _ in range(WARMUP):
        e._motion_shuttle(bg, [])
    got = [e._motion_shuttle(_with_shuttle(bg, x, y), []) for x, y in _flight()]
    x, y = _flight()[-1]
    assert got[-1] is not None
    assert abs(got[-1][0] - x) <= 6 and abs(got[-1][1] - y) <= 6


def test_engine_motion_shuttle_uses_the_pose_boxes():
    bg = _court()
    e = DrillEngine()
    e._shuttle_motion = ShuttleMotionDetector()
    for _ in range(WARMUP):
        e._motion_shuttle(bg, [])
    everywhere = [{"box": [0, 0, W, H]}]
    got = [e._motion_shuttle(_with_shuttle(bg, x, y), everywhere)
           for x, y in _flight()]
    assert all(g is None for g in got)


def _flicker(bg, rng):
    """A spot on a fluorescent tube: its brightness changes irregularly every
    frame. (MOG2 alone already learns a REGULAR on/off flash as background;
    irregular flicker is what gets through it.)"""
    f = bg.copy()
    f[60:72, 200:212] = rng.integers(90, 256)
    return f


def test_flicker_makes_false_blobs_without_the_filter():
    bg, rng = _court(), np.random.default_rng(3)
    det = ShuttleMotionDetector(flicker_max=1.0)
    blobs = [len(det.update(_flicker(bg, rng))[2]) for _ in range(200)]
    assert sum(blobs[WARMUP:]) > 20


def test_the_filter_removes_flicker_blobs():
    bg, rng = _court(), np.random.default_rng(3)
    det = ShuttleMotionDetector()
    blobs = [len(det.update(_flicker(bg, rng))[2]) for _ in range(200)]
    assert sum(blobs[WARMUP:]) == 0


def test_a_shuttle_still_tracks_past_a_flickering_light():
    bg, rng = _court(), np.random.default_rng(3)
    det = ShuttleMotionDetector()
    for _ in range(WARMUP + 30):
        det.update(_flicker(bg, rng))
    path = _flight()
    out = [det.update(_with_shuttle(_flicker(bg, rng), x, y))[0]
           for x, y in path]
    assert sum(1 for r in out if r and r[2] == "det") >= len(path) - 6
