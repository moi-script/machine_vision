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


def test_to_grayscale_bgr_equalizes_channels():
    # A colour image must collapse to R == G == B while keeping 3 channels,
    # so SFace still gets its expected (H, W, 3) input.
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, :, 0] = 30   # B
    img[:, :, 1] = 150  # G
    img[:, :, 2] = 200  # R
    out = face.to_grayscale_bgr(img)
    assert out.shape == img.shape
    assert np.array_equal(out[:, :, 0], out[:, :, 1])
    assert np.array_equal(out[:, :, 1], out[:, :, 2])
