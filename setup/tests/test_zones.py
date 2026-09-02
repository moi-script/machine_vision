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


def test_zone_mapping_front_left():
    cx = settings.COURT_W * 0.16
    cy = settings.COURT_L * 0.25
    assert zones.get_zone_from_position(cx, cy) == "front_left"


def test_zone_mapping_back_right():
    cx = settings.COURT_W * 0.83
    cy = settings.COURT_L * 0.75
    assert zones.get_zone_from_position(cx, cy) == "back_right"


def test_zone_outside_returns_none():
    assert zones.get_zone_from_position(-10.0, 10.0) is None


def test_in_court_bounds():
    assert zones.in_court_bounds(10.0, 10.0) is True
    assert zones.in_court_bounds(settings.COURT_W / 2, settings.COURT_L / 2) is True
    # feeder side (negative y) and outside a sideline
    assert zones.in_court_bounds(10.0, -5.0) is False
    assert zones.in_court_bounds(settings.COURT_W + 5, 10.0) is False


def test_shuttle_side():
    assert zones.get_shuttle_side(200.0) == "player_side"
    assert zones.get_shuttle_side(-200.0) == "feeder_side"
    # within deadband counts as player side (not feeder)
    assert zones.get_shuttle_side(0.0) == "player_side"


def test_crossed_net():
    # feeder -> player crossing
    assert zones.crossed_net(-50.0, 50.0) is True
    # jitter within deadband: no crossing
    assert zones.crossed_net(-5.0, 5.0) is False
    # already on player side: no new crossing
    assert zones.crossed_net(30.0, 60.0) is False


def test_player_in_zone_court_coords():
    positions = {7: (settings.COURT_W * 0.16, settings.COURT_L * 0.25)}
    assert zones.get_player_in_zone("front_left", positions) == 7
    assert zones.get_player_in_zone("back_right", positions) is None


# ── Per-camera homography cache ──────────────────────────────
# The rig has front/left/right/back cameras, each seeing the court from its own
# angle. These guard the property that made the cache necessary: calibrating one
# camera must not disturb another.

# A different quad, so a mix-up between cameras produces different numbers
# rather than accidentally-equal ones.
CORNERS_LEFT = [(200.0, 150.0), (900.0, 150.0), (900.0, 650.0), (200.0, 650.0)]


@pytest.fixture(autouse=True)
def _clear_extra_cameras():
    yield
    for cam in list(zones.calibrated_cameras()):
        if cam != zones.DEFAULT_CAMERA:
            zones.clear_homography(cam)


def test_cameras_do_not_share_a_homography():
    zones.build_homography(CORNERS_LEFT, camera_id="left")
    # The default camera still maps its own corners to the canonical origin.
    assert zones.to_court((100.0, 100.0)) == pytest.approx((0.0, 0.0), abs=1e-3)
    # The left camera maps ITS corners there instead.
    assert zones.to_court((200.0, 150.0), camera_id="left") == pytest.approx(
        (0.0, 0.0), abs=1e-3)
    # And the same pixel means different things to the two cameras.
    assert zones.to_court((900.0, 650.0), camera_id="left") != pytest.approx(
        zones.to_court((900.0, 650.0)), abs=1e-3)


def test_court_to_pixel_is_inverse_per_camera():
    zones.build_homography(CORNERS_LEFT, camera_id="left")
    px = (450.0, 380.0)
    back = zones.court_to_pixel(zones.to_court(px, camera_id="left"), camera_id="left")
    assert back == pytest.approx(px, abs=1e-3)


def test_uncalibrated_camera_raises_rather_than_borrowing():
    # Silently falling back to another camera's geometry would produce
    # plausible, wrong court positions — the worst failure mode here.
    with pytest.raises(ValueError, match="no homography"):
        zones.to_court((300.0, 400.0), camera_id="back")


def test_failed_rebuild_leaves_previous_homography_intact():
    zones.build_homography(CORNERS_LEFT, camera_id="left")
    before = zones.to_court((450.0, 380.0), camera_id="left")
    with pytest.raises(ValueError):
        zones.build_homography([(0, 0), (1, 1)], camera_id="left")
    assert zones.to_court((450.0, 380.0), camera_id="left") == pytest.approx(
        before, abs=1e-6)


def test_clear_homography_forgets_one_camera():
    zones.build_homography(CORNERS_LEFT, camera_id="left")
    assert "left" in zones.calibrated_cameras()
    zones.clear_homography("left")
    assert "left" not in zones.calibrated_cameras()
