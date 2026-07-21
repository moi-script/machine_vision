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
