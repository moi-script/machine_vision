import numpy as np
import pytest
from app import face


def test_cosine_identical_is_one():
    v = [1.0, 2.0, 3.0, 4.0]
    assert face.cosine(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_orthogonal_is_zero():
    assert face.cosine([1, 0], [0, 1]) == pytest.approx(0.0, abs=1e-6)


def test_best_match_picks_closest_above_threshold():
    enrolled = {"p1": [1.0, 0.0, 0.0], "p2": [0.0, 1.0, 0.0]}
    got = face.best_match([0.9, 0.1, 0.0], enrolled, threshold=0.5)
    assert got is not None and got[0] == "p1"


def test_best_match_returns_none_below_threshold():
    enrolled = {"p1": [1.0, 0.0, 0.0]}
    assert face.best_match([0.0, 1.0, 0.0], enrolled, threshold=0.5) is None


def test_decode_data_url_bad_input_returns_none():
    assert face.decode_data_url("not-a-data-url") is None
