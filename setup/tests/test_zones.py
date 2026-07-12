import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings  # noqa: E402
from utils import zones  # noqa: E402

# A simple axis-aligned pixel quad (400x600 px) -> canonical 518x670 court.
# Order: net_left, net_right, baseline_right, baseline_left.
CORNERS = [(100.0, 100.0), (500.0, 100.0), (500.0, 700.0), (100.0, 700.0)]


@pytest.fixture(autouse=True)
def _build():
    zones.build_homography(CORNERS)
    yield


def test_corners_map_to_canonical():
    assert zones.to_court((100.0, 100.0)) == pytest.approx((0.0, 0.0), abs=1e-3)
    assert zones.to_court((500.0, 100.0)) == pytest.approx((settings.COURT_W, 0.0), abs=1e-3)
    assert zones.to_court((500.0, 700.0)) == pytest.approx((settings.COURT_W, settings.COURT_L), abs=1e-3)
    assert zones.to_court((100.0, 700.0)) == pytest.approx((0.0, settings.COURT_L), abs=1e-3)


def test_court_to_pixel_is_inverse():
    px = (300.0, 400.0)
    cx = zones.to_court(px)
    back = zones.court_to_pixel(cx)
    assert back == pytest.approx(px, abs=1e-3)


def test_build_homography_requires_four_corners():
    with pytest.raises(ValueError):
        zones.build_homography([(0, 0), (1, 1)])
