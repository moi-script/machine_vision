import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feeder_court import segments  # noqa: E402


def test_head_and_tail_motion_are_excluded():
    # 5 noisy samples, 40 calm, 5 noisy — at step 15 that is frames 0..749.
    profile = np.concatenate([
        np.full(5, 40.0),
        np.full(40, 2.0),
        np.full(5, 40.0),
    ])
    out = segments.usable_segments(profile, sample_step=15, threshold=15.0, min_frames=150)
    assert out[0] == (75, 675)


def test_short_calm_runs_are_dropped():
    profile = np.concatenate([
        np.full(10, 40.0),
        np.full(3, 2.0),      # 3 samples * 15 = 45 frames, under min_frames
        np.full(10, 40.0),
    ])
    out = segments.usable_segments(profile, sample_step=15, threshold=15.0, min_frames=150)
    assert out == []


def test_a_mid_clip_bump_splits_one_run_into_two():
    profile = np.concatenate([
        np.full(20, 2.0),
        np.full(4, 40.0),     # camera knocked
        np.full(20, 2.0),
    ])
    out = segments.usable_segments(profile, sample_step=15, threshold=15.0, min_frames=150)
    assert out == [(0, 300), (360, 660)]


def test_entirely_calm_profile_yields_one_full_span():
    profile = np.full(50, 1.0)
    out = segments.usable_segments(profile, sample_step=15, threshold=15.0, min_frames=150)
    assert len(out) == 1
    assert out[0][0] == 0


def test_entirely_noisy_profile_yields_nothing():
    profile = np.full(50, 99.0)
    assert segments.usable_segments(profile, sample_step=15) == []


def test_clamp_trims_the_recording_head_and_tail():
    out = segments.clamp_to_recording_bounds([(0, 1000)], total_frames=1000, head_trim=90, tail_trim=45)
    assert out == [(90, 955)]


def test_clamp_drops_a_span_left_too_short_by_trimming():
    # mid.mp4's junk (0, 180): after a 90-frame head trim only 90 frames remain.
    out = segments.clamp_to_recording_bounds(
        [(0, 180), (195, 1530)], total_frames=1664, head_trim=90, tail_trim=45, min_frames=150
    )
    assert out == [(195, 1530)]


def test_clamp_leaves_an_interior_span_untouched():
    out = segments.clamp_to_recording_bounds([(400, 900)], total_frames=2000, head_trim=90, tail_trim=45)
    assert out == [(400, 900)]


def test_clamp_can_drop_every_span():
    assert segments.clamp_to_recording_bounds([(0, 50)], total_frames=100, head_trim=90, tail_trim=45) == []
