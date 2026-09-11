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


def _seed(counter, tracks=()):
    """Run out the seed window. Whatever is passed counts as already-present."""
    for _ in range(counter.seed_frames):
        counter.update(list(tracks))


def _confirm(counter, tid, point, start=0):
    """Feed one track enough times to pass the stability gate."""
    out = []
    for f in range(start, start + counter.stable_frames):
        out += counter.update([(tid, point)], f)
    return out


def test_a_shuttle_is_counted_once_not_once_per_frame():
    c = LandingCounter(LINES)
    _seed(c)
    _confirm(c, 7, (600.0, 600.0))
    for f in range(100, 200):            # keep seeing the same track
        c.update([(7, (600.0, 600.0))], f)
    assert c.total == 1


def test_box_jitter_does_not_invent_a_second_shuttle():
    # The failure that made the old counter useless: one shuttle whose box
    # wanders further than the match radius used to score twice. The track id
    # is what settles it now.
    c = LandingCounter(LINES)
    _seed(c)
    _confirm(c, 7, (600.0, 600.0))
    for f, dx in enumerate((40, -55, 70, -30), start=100):
        c.update([(7, (600.0 + dx, 600.0 + dx))], f)
    assert c.total == 1


def test_a_flicker_never_scores():
    # Test precision is 0.752, so a one-frame blob must not become a point.
    c = LandingCounter(LINES)
    _seed(c)
    assert c.update([(7, (600.0, 600.0))], 100) == []
    assert c.total == 0


def test_shuttles_already_on_the_floor_are_not_counted():
    # The big one. lines.mp4 starts with shuttles down, and a feeder session
    # always has them after the first shots; counting those made the old total
    # meaningless.
    c = LandingCounter(LINES)
    _seed(c, [(1, (300.0, 600.0)), (2, (900.0, 600.0))])
    for f in range(100, 140):
        c.update([(1, (300.0, 600.0)), (2, (900.0, 600.0))], f)
    assert c.total == 0
    assert c.counts["green_inside"] == 0


def test_a_new_shuttle_after_seeding_does_count():
    c = LandingCounter(LINES)
    _seed(c, [(1, (300.0, 600.0))])
    _confirm(c, 2, (900.0, 600.0), start=100)
    assert c.total == 1
    assert c.active["track"] == 2


def test_a_reacquired_track_id_does_not_double_count():
    # ByteTrack can hand a re-found shuttle a fresh id; the position guard is
    # the backstop for that.
    c = LandingCounter(LINES)
    _seed(c)
    _confirm(c, 7, (600.0, 600.0), start=100)
    assert c.total == 1
    _confirm(c, 99, (605.0, 603.0), start=200)   # same spot, new id
    assert c.total == 1


def test_landing_below_both_lines_is_inside_both():
    c = LandingCounter(LINES)
    _seed(c)
    events = _confirm(c, 7, (600.0, 600.0), start=100)
    assert events[0]["green"] == "inside"
    assert events[0]["red"] == "inside"
    # Inside both increments both counters by the same amount.
    assert c.counts["green_inside"] == 1
    assert c.counts["red_inside"] == 1
    assert c.counts["green_outside"] == 0
    assert c.counts["red_outside"] == 0


def test_landing_above_both_lines_is_outside_both():
    c = LandingCounter(LINES)
    _seed(c)
    events = _confirm(c, 7, (600.0, 400.0), start=100)
    assert events[0]["green"] == "outside"
    assert events[0]["red"] == "outside"
    assert c.counts["green_outside"] == 1
    assert c.counts["red_outside"] == 1


def test_landing_between_the_lines_splits_the_call():
    # Past the inside line but not the outside one - the case the two counters
    # exist to distinguish.
    c = LandingCounter(LINES)
    _seed(c)
    events = _confirm(c, 7, (600.0, 515.0), start=100)
    assert events[0]["green"] == "outside"
    assert events[0]["red"] == "inside"


def test_lines_are_evaluated_at_the_landing_x():
    # The lines slope, so the same y is a different call at a different x.
    c = LandingCounter(LINES)
    _seed(c)
    _confirm(c, 7, (0.0, 520.0), start=100)      # green here is 505 -> inside
    assert c.counts["green_inside"] == 1
    c2 = LandingCounter(LINES)
    _seed(c2)
    _confirm(c2, 7, (1200.0, 520.0), start=100)  # green here is 565 -> outside
    assert c2.counts["green_outside"] == 1


def test_the_newest_landing_takes_over_as_active():
    c = LandingCounter(LINES)
    _seed(c)
    _confirm(c, 1, (300.0, 600.0), start=100)
    first = c.active["id"]
    _confirm(c, 2, (900.0, 600.0), start=200)
    assert c.active["id"] != first
    assert c.active["track"] == 2


def test_the_previous_active_becomes_counted_not_forgotten():
    c = LandingCounter(LINES)
    _seed(c)
    _confirm(c, 1, (300.0, 600.0), start=100)
    assert c.state_of(1) == "active"
    _confirm(c, 2, (900.0, 600.0), start=200)
    assert c.state_of(1) == "counted"     # handover
    assert c.state_of(2) == "active"


def test_an_unconfirmed_track_reads_as_pending():
    c = LandingCounter(LINES)
    _seed(c)
    c.update([(7, (600.0, 600.0))], 100)
    assert c.state_of(7) == "pending"


def test_a_counted_shuttle_never_scores_again():
    c = LandingCounter(LINES)
    _seed(c)
    _confirm(c, 1, (300.0, 600.0), start=100)
    _confirm(c, 2, (900.0, 600.0), start=200)
    for f in range(300, 400):
        c.update([(1, (300.0, 600.0)), (2, (900.0, 600.0))], f)
    assert c.total == 2
    assert c.active["track"] == 2


def test_reset_clears_everything_including_the_seed():
    c = LandingCounter(LINES)
    _seed(c)
    _confirm(c, 7, (600.0, 600.0), start=100)
    c.reset()
    assert c.total == 0 and not any(c.counts.values()) and c.active is None
    assert c.seeding                       # a new rally re-seeds


def test_update_accepts_the_boxed_triples_the_pipeline_sends():
    # The pipeline passes (track_id, point, box) so the box can be drawn; the
    # counter must ignore the third element rather than die unpacking it. This
    # killed the capture thread on the first frame that had a detection.
    c = LandingCounter(LINES)
    _seed(c)
    for f in range(100, 100 + c.stable_frames):
        c.update([(7, (600.0, 600.0), (590, 590, 610, 610))], f)
    assert c.total == 1
