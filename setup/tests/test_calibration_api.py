import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.server import app  # noqa: E402
from app import calibration  # noqa: E402
from utils import zones  # noqa: E402

client = TestClient(app)

# A convex, well-spread quad on a 1280x720 frame, in
# [net_left, net_right, baseline_right, baseline_left] order.
GOOD = [[300.0, 250.0], [980.0, 250.0], [1180.0, 660.0], [100.0, 660.0]]


@pytest.fixture(autouse=True)
def _clean():
    yield
    for cid in ("front", "left", "right", "back"):
        zones.clear_homography(cid)


# ── validation ───────────────────────────────────────────────

def test_good_corners_validate():
    assert calibration.validate(GOOD, (1280, 720)) == []


def test_collinear_corners_are_rejected():
    # Three points on one line: getPerspectiveTransform accepts this happily,
    # which is exactly why it needs catching here.
    bad = [[100.0, 300.0], [500.0, 300.0], [900.0, 300.0], [500.0, 600.0]]
    errors = calibration.validate(bad, (1280, 720))
    assert any("collinear" in e for e in errors)


def test_out_of_order_corners_are_rejected():
    swapped = [GOOD[0], GOOD[2], GOOD[1], GOOD[3]]   # bow-tie, not a quad
    errors = calibration.validate(swapped, (1280, 720))
    assert any("convex" in e for e in errors)


def test_corner_outside_image_is_rejected():
    off = [[300.0, 250.0], [1400.0, 250.0], [1180.0, 660.0], [100.0, 660.0]]
    errors = calibration.validate(off, (1280, 720))
    assert any("outside" in e for e in errors)


def test_solve_raises_with_every_reason_at_once():
    bad = [[100.0, 300.0], [500.0, 300.0], [900.0, 300.0], [2000.0, 300.0]]
    with pytest.raises(calibration.CalibrationError) as exc:
        calibration.solve(bad, (1280, 720))
    assert len(exc.value.errors) >= 2


def test_reprojection_round_trips():
    _H, _H_inv, err = calibration.solve(GOOD, (1280, 720))
    assert err < 1e-3   # exact 4-point solve


def test_reproject_returns_court_features_in_pixels():
    _H, H_inv, _ = calibration.solve(GOOD, (1280, 720))
    ov = calibration.reproject(H_inv)
    assert set(["outline", "net_line", "centre_line"]).issubset(ov)
    # The outline must come back as the corners that were clicked.
    for got, want in zip(ov["outline"], GOOD):
        assert got[0] == pytest.approx(want[0], abs=0.5)
        assert got[1] == pytest.approx(want[1], abs=0.5)


# ── endpoints ────────────────────────────────────────────────

def test_list_cameras_reports_the_rig():
    r = client.get("/api/cameras")
    assert r.status_code == 200
    ids = [c["camera_id"] for c in r.json()]
    assert ids == ["front", "left", "right", "back"]


def test_unknown_camera_is_404():
    assert client.get("/api/cameras/nope/frame").status_code == 404


def test_preview_returns_overlay_without_saving():
    r = client.post("/api/cameras/front/calibration/preview",
                    json={"corners": GOOD})
    body = r.json()
    assert body["ok"] is True
    assert "outline" in body["overlay"]
    # Preview must not persist: the camera is still uncalibrated afterwards.
    assert client.get("/api/cameras/front/calibration").status_code == 404


def test_preview_reports_errors_instead_of_raising():
    bad = [[100.0, 300.0], [500.0, 300.0], [900.0, 300.0], [500.0, 600.0]]
    body = client.post("/api/cameras/front/calibration/preview",
                       json={"corners": bad}).json()
    assert body["ok"] is False
    assert body["errors"]


def test_commit_refuses_bad_geometry_with_422():
    bad = [[100.0, 300.0], [500.0, 300.0], [900.0, 300.0], [500.0, 600.0]]
    r = client.post("/api/cameras/left/calibration", json={"corners": bad})
    assert r.status_code == 422
    # And nothing reached the homography cache.
    assert "left" not in zones.calibrated_cameras()


def test_calibration_page_is_served():
    r = client.get("/calibration")
    assert r.status_code == 200
    assert "Court calibration" in r.text
