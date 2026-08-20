import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feeder_court import tracks  # noqa: E402
from utils.feeder_court.motion import Candidate  # noqa: E402


def _c(frame, x, y, w=6, h=6):
    return Candidate(frame=frame, x=x, y=y, w=w, h=h, area=w * h, peak=200.0)


def test_link_joins_a_straight_fast_path_into_one_track():
    cands = [_c(i, 100 + 10 * i, 50 + 4 * i) for i in range(10)]
    out = tracks.link_tracks(cands)
    assert len(out) == 1
    assert len(out[0]) == 10


def test_link_splits_two_far_apart_objects():
    a = [_c(i, 10 * i, 20) for i in range(6)]
    b = [_c(i, 900 - 10 * i, 600) for i in range(6)]
    out = tracks.link_tracks(a + b)
    assert len(out) == 2
    assert all(len(t) == 6 for t in out)


def test_link_tolerates_a_short_gap():
    cands = [_c(i, 100 + 10 * i, 50) for i in range(10) if i != 4]
    out = tracks.link_tracks(cands, max_gap=2)
    assert len(out) == 1
    assert len(out[0]) == 9


def test_is_flight_accepts_a_fast_directed_track():
    track = [_c(i, 100 + 12 * i, 50 + 5 * i) for i in range(12)]
    assert tracks.is_flight(track) is True


def test_is_flight_rejects_a_track_that_is_too_long():
    track = [_c(i, 100 + 12 * i, 50) for i in range(200)]
    assert tracks.is_flight(track) is False


def test_is_flight_rejects_a_slow_track():
    track = [_c(i, 100 + 1 * i, 50) for i in range(12)]
    assert tracks.is_flight(track) is False


def test_is_flight_rejects_jitter_that_goes_nowhere():
    # Fast per-step motion but returns to where it started.
    xs = [100, 130, 100, 130, 100, 130, 100, 130]
    track = [_c(i, x, 50) for i, x in enumerate(xs)]
    assert tracks.is_flight(track) is False


def test_is_flight_rejects_a_track_that_is_too_short():
    track = [_c(i, 100 + 20 * i, 50) for i in range(3)]
    assert tracks.is_flight(track) is False


def test_bucket_marks_flight_frames_positive():
    out = tracks.bucket_frames([1, 2, 3], flight_frames={2}, reject_frames=set())
    assert out[2] == "positive"


def test_bucket_marks_clean_frames_negative():
    out = tracks.bucket_frames([1, 2, 3], flight_frames={2}, reject_frames=set())
    assert out[1] == "negative"
    assert out[3] == "negative"


def test_bucket_marks_ambiguous_frames_discard_not_negative():
    # A frame with a rejected candidate is UNCERTAIN. Writing it as an empty
    # label would teach the model that a real shuttle is background.
    out = tracks.bucket_frames([1, 2, 3], flight_frames={2}, reject_frames={3})
    assert out[3] == "discard"


def test_positive_wins_over_discard_when_a_frame_has_both():
    out = tracks.bucket_frames([1], flight_frames={1}, reject_frames={1})
    assert out[1] == "positive"
