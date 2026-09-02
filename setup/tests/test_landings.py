import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.landings import LandingCounter  # noqa: E402

# Two near-horizontal lines like a traced calibration. "Beyond a line" means a
# SMALLER y — further from the camera — so the outside line must sit above the
# inside one in image space, with the court below both.
LINES = {
    "outside": {"a": 0.06, "b": 460.0},   # y ~ 460..537, the far one
    "inside": {"a": 0.05, "b": 505.0},    # y ~ 505..569, the court side
}


def _confirm(counter, point, start=0):
    """Feed the same point enough times to pass the confirmation gate."""
    out = []
    for f in range(start, start + counter.confirm_frames):
        out += counter.update([point], f)
    return out


def test_a_shuttle_is_counted_once_not_once_per_frame():
    c = LandingCounter(LINES)
    _confirm(c, (600.0, 600.0))
    for f in range(10, 60):            # keep seeing the same shuttle
        c.update([(600.0, 600.0)], f)
    assert c.total == 1


def test_a_flicker_never_scores():
    # Test precision is 0.752, so a one-frame blob must not become a point.
    c = LandingCounter(LINES)
    assert c.update([(600.0, 600.0)], 0) == []
    assert c.total == 0


def test_landing_below_both_lines_is_inside_both():
    c = LandingCounter(LINES)
    events = _confirm(c, (600.0, 600.0))   # below red (y=541) and green (y=490)
    assert events[0]["green"] == "inside"
    assert events[0]["red"] == "inside"
    # Inside both increments both counters by the same amount.
    assert c.counts["green_inside"] == 1
    assert c.counts["red_inside"] == 1
    assert c.counts["green_outside"] == 0
    assert c.counts["red_outside"] == 0


def test_landing_above_both_lines_is_outside_both():
    c = LandingCounter(LINES)
    events = _confirm(c, (600.0, 400.0))   # above green (490) and red (541)
    assert events[0]["green"] == "outside"
    assert events[0]["red"] == "outside"
    assert c.counts["green_outside"] == 1
    assert c.counts["red_outside"] == 1


def test_landing_between_the_lines_splits_the_call():
    # Between green (490) and red (541): past the inside line but not the
    # outside one — the case the two counters exist to distinguish.
    c = LandingCounter(LINES)
    events = _confirm(c, (600.0, 515.0))
    assert events[0]["green"] == "outside"
    assert events[0]["red"] == "inside"
    assert c.counts["green_outside"] == 1
    assert c.counts["red_inside"] == 1


def test_lines_are_evaluated_at_the_landing_x():
    # The lines slope, so the same y is a different call at a different x.
    c = LandingCounter(LINES)
    _confirm(c, (0.0, 520.0))              # green here is 505 -> below -> inside
    assert c.counts["green_inside"] == 1
    c2 = LandingCounter(LINES)
    _confirm(c2, (1200.0, 520.0))          # green here is 565 -> above -> outside
    assert c2.counts["green_outside"] == 1


def test_separate_shuttles_each_count():
    c = LandingCounter(LINES)
    _confirm(c, (300.0, 600.0), start=0)
    _confirm(c, (900.0, 600.0), start=20)
    assert c.total == 2
    assert c.counts["green_inside"] == 2


def test_reset_clears_everything():
    c = LandingCounter(LINES)
    _confirm(c, (600.0, 600.0))
    c.reset()
    assert c.total == 0 and not any(c.counts.values())
    # And the same spot can score again after a reset, for a new rally.
    _confirm(c, (600.0, 600.0), start=100)
    assert c.total == 1
