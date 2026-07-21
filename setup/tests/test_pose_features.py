import math
import pytest
from utils import pose_features as pf


def _kp(overrides):
    """17 COCO keypoints, all confident at origin unless overridden by index."""
    kps = [[0.0, 0.0, 1.0] for _ in range(17)]
    for idx, val in overrides.items():
        kps[idx] = val
    return kps


def test_angle_right_angle():
    assert pf.angle((0, 1), (0, 0), (1, 0)) == pytest.approx(90.0, abs=1e-3)


def test_angle_straight_line():
    assert pf.angle((0, 2), (0, 1), (0, 0)) == pytest.approx(180.0, abs=1e-3)


def test_point_gated_by_confidence():
    kps = _kp({5: [10.0, 20.0, 0.2]})
    assert pf.point(kps, 5, 0.5) is None
    assert pf.point(kps, 5, 0.1) == (10.0, 20.0)


def test_knee_angle_straight_leg_is_180():
    # left hip(11), knee(13), ankle(15) vertically aligned -> straight leg
    kps = _kp({11: [0.0, 0.0, 1.0], 13: [0.0, 1.0, 1.0], 15: [0.0, 2.0, 1.0]})
    assert pf.knee_angle(kps) == pytest.approx(180.0, abs=1e-2)


def test_knee_angle_none_when_low_conf():
    kps = _kp({11: [0.0, 0.0, 0.1], 13: [0.0, 1.0, 0.1], 15: [0.0, 2.0, 0.1]})
    assert pf.knee_angle(kps) is None


def test_stance_width_ratio():
    # shoulders(5,6) 1 apart in x; ankles(15,16) 2 apart -> ratio 2.0
    kps = _kp({5: [0.0, 0.0, 1.0], 6: [1.0, 0.0, 1.0],
               15: [0.0, 5.0, 1.0], 16: [2.0, 5.0, 1.0]})
    assert pf.stance_width(kps) == pytest.approx(2.0, abs=1e-6)


def test_max_wrist_speed_shoulder_relative():
    # right wrist(10) moves +3 in x over dt=1; shoulder(6) static -> speed 3
    prev = _kp({6: [0.0, 0.0, 1.0], 10: [0.0, 0.0, 1.0]})
    now = _kp({6: [0.0, 0.0, 1.0], 10: [3.0, 0.0, 1.0]})
    assert pf.max_wrist_speed(now, prev, 1.0) == pytest.approx(3.0, abs=1e-6)


def test_bbox_height():
    assert pf.bbox_height([10, 20, 50, 120]) == 100.0
