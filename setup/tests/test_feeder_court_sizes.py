import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feeder_court import sizes  # noqa: E402


def test_yolo_line_converts_to_pixel_max_dimension():
    # class cx cy w h, normalised. w=0.01 of 1280 = 12.8 px, h=0.02 of 448 = 8.96 px
    line = "0 0.5 0.5 0.01 0.02"
    assert sizes.yolo_box_max_dim_px(line, img_w=1280, img_h=448) == pytest.approx(12.8)


def test_yolo_line_takes_the_larger_dimension():
    line = "0 0.5 0.5 0.001 0.05"  # 1.28 px wide, 22.4 px tall
    assert sizes.yolo_box_max_dim_px(line, img_w=1280, img_h=448) == pytest.approx(22.4)


def test_size_percentiles_reports_expected_keys():
    out = sizes.size_percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
    assert set(out) == {"p25", "median", "p75", "p95"}
    assert out["median"] == pytest.approx(3.0)


def test_gate_says_stock_when_all_clips_are_comfortably_large():
    assert sizes.gate_decision({"near": 30.0, "mid": 22.0, "far": 18.0}) == "stock"


def test_gate_says_p2_in_the_marginal_band():
    assert sizes.gate_decision({"near": 14.0, "mid": 11.0, "far": 9.0}) == "p2"


def test_gate_stops_only_when_every_clip_is_below_eight_px():
    assert sizes.gate_decision({"near": 7.0, "mid": 6.0, "far": 5.0}) == "stop"


def test_gate_does_not_stop_if_any_clip_is_workable():
    # far is hopeless but near is fine — that is a scoping decision, not a stop.
    assert sizes.gate_decision({"near": 20.0, "mid": 9.0, "far": 5.0}) == "p2"


def test_gate_uses_the_minimum_clip_to_choose_between_stock_and_p2():
    # One marginal clip forces P2 even though the others are large.
    assert sizes.gate_decision({"near": 40.0, "mid": 38.0, "far": 12.0}) == "p2"


def test_evenly_spaced_frames_spreads_across_all_spans():
    frames = sizes.evenly_spaced_frames([(0, 100), (200, 300)], count=10)
    assert len(frames) == 10
    assert all(0 <= f < 100 or 200 <= f < 300 for f in frames)
    assert len(set(frames)) == 10


def test_evenly_spaced_frames_is_sorted_and_within_bounds():
    frames = sizes.evenly_spaced_frames([(50, 90)], count=5)
    assert frames == sorted(frames)
    assert frames[0] >= 50
    assert frames[-1] < 90
