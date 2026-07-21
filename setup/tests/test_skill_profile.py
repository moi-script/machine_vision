import math
import pytest
from utils.skill_profile import RunningStat


def test_running_stat_mean_and_std():
    rs = RunningStat()
    for x in (2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0):
        rs.add(x)
    assert rs.n == 8
    assert rs.mean() == pytest.approx(5.0)
    assert rs.std() == pytest.approx(2.0)  # population std


def test_running_stat_empty_is_zero():
    rs = RunningStat()
    assert rs.mean() == 0.0 and rs.std() == 0.0


def test_running_stat_merge_equivalent_to_combined():
    a = RunningStat(); [a.add(x) for x in (1, 2, 3)]
    b = RunningStat(); [b.add(x) for x in (4, 5, 6)]
    a.merge(b)
    combined = RunningStat(); [combined.add(x) for x in (1, 2, 3, 4, 5, 6)]
    assert a.n == combined.n
    assert a.mean() == pytest.approx(combined.mean())
    assert a.std() == pytest.approx(combined.std())


def test_running_stat_roundtrip_dict():
    a = RunningStat(); [a.add(x) for x in (1.5, 2.5, 3.5)]
    b = RunningStat.from_dict(a.to_dict())
    assert b.n == a.n and b.mean() == pytest.approx(a.mean())


from utils.skill_profile import SkillAccumulator


def _kp(overrides):
    kps = [[0.0, 0.0, 1.0] for _ in range(17)]
    for idx, val in overrides.items():
        kps[idx] = val
    return kps


def _acc():
    return SkillAccumulator(kp_conf=0.5, bbox_min_h=100.0,
                            react_dist=20.0, swing_speed=3.0)


def test_reaction_recorded_from_feeder_event():
    acc = _acc()
    box = [0, 0, 50, 200]  # tall enough
    kps = _kp({})
    acc.add_frame(kps, box, court_ankle=(0.0, 0.0), now=10.0)
    acc.on_feeder_fired(now=10.0, zone="back-left")
    # player hasn't moved yet
    acc.add_frame(kps, box, court_ankle=(5.0, 0.0), now=10.2)
    # now displacement exceeds react_dist (20)
    acc.add_frame(kps, box, court_ankle=(30.0, 0.0), now=10.5)
    snap = acc.snapshot()
    assert snap["stats"]["reaction"]["n"] == 1
    # reaction lag ~ 0.5s (from feeder at 10.0)
    assert snap["stats"]["reaction"]["sum"] == pytest.approx(0.5, abs=1e-6)


def test_posture_gated_out_when_bbox_too_small():
    acc = _acc()
    small_box = [0, 0, 50, 50]  # height 50 < 100 gate
    kps = _kp({11: [0, 0, 1], 13: [0, 1, 1], 15: [0, 2, 1]})
    acc.add_frame(kps, small_box, court_ankle=(0.0, 0.0), now=1.0)
    assert acc.snapshot()["sampleCounts"]["posture"] == 0


def test_posture_sampled_when_bbox_large_enough():
    acc = _acc()
    big_box = [0, 0, 50, 300]
    kps = _kp({11: [0, 0, 1], 13: [0, 1, 1], 15: [0, 2, 1]})
    acc.add_frame(kps, big_box, court_ankle=(0.0, 0.0), now=1.0)
    assert acc.snapshot()["sampleCounts"]["posture"] >= 1


def test_swing_detected_on_rising_then_falling_edge():
    acc = _acc()
    big_box = [0, 0, 50, 300]
    # frame1 wrist still, frame2 fast (rising edge > 3), frame3 slow (falling)
    f1 = _kp({6: [0, 0, 1], 10: [0, 0, 1]})
    f2 = _kp({6: [0, 0, 1], 10: [5, 0, 1]})   # speed 5 over dt 1
    f3 = _kp({6: [0, 0, 1], 10: [5, 0, 1]})   # speed 0
    acc.add_frame(f1, big_box, None, now=1.0)
    acc.add_frame(f2, big_box, None, now=2.0)
    acc.add_frame(f3, big_box, None, now=3.0)
    assert acc.snapshot()["swings"] == 1


def test_merge_snapshot_accumulates_counts():
    a = _acc(); a.on_shot(); a.on_shot(); a.on_score()
    snap1 = a.snapshot()
    b = _acc(); b.on_shot(); b.on_score()
    snap2 = b.snapshot()
    merged = SkillAccumulator.merge_snapshot({}, snap1)
    merged = SkillAccumulator.merge_snapshot(merged, snap2)
    assert merged["shots"] == 3
    assert merged["scores"] == 2


from utils.skill_profile import evaluate_rubric


def test_rubric_unranked_below_min_shots():
    prof = {"shots": 3, "scores": 2, "sampleCounts": {}, "stats": {}, "coverage": []}
    out = evaluate_rubric(prof)
    assert out["tier"] == "Unranked"
    assert out["composite"] is None


def test_rubric_high_accuracy_maps_to_a_real_tier():
    # 100 shots, 85 scores -> accuracy 85% -> high accuracy family score
    prof = {"shots": 100, "scores": 85,
            "sampleCounts": {"accuracy": 100, "move": 0, "stroke": 0, "posture": 0},
            "coverage": [], "stats": {}}
    out = evaluate_rubric(prof)
    assert out["composite"] is not None
    assert out["tier"] in {"Beginner", "Novice", "Intermediate", "Advanced", "Expert"}
    assert out["families"]["accuracy"] > 80


def test_rubric_skips_thin_families_via_redistribution():
    # only accuracy has data; move/stroke/posture are empty and must not drag it
    prof = {"shots": 100, "scores": 90,
            "sampleCounts": {"accuracy": 100, "move": 0, "stroke": 0, "posture": 0},
            "coverage": [], "stats": {}}
    out = evaluate_rubric(prof)
    # composite should equal the accuracy family score (only qualifying family)
    assert out["composite"] == pytest.approx(out["families"]["accuracy"], abs=1e-6)
