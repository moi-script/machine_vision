# ============================================================
# zones.py — Court zone detection and player-zone matching
# ============================================================

import cv2
import numpy as np

from config.settings import (
    PLAYER_ZONES, ANKLE_CONFIDENCE,
    COURT_CORNERS, COURT_W, COURT_L, NET_DEADBAND,
)

# ── Homography (image pixels <-> top-down court space) ───────
#
# One entry per camera. The rig grew from a single camera to a front/left/right/
# back set, and each position sees the court from its own angle, so each needs
# its own transform. A single module-level pair would make the last camera
# calibrated silently overwrite every other camera's geometry.
#
# DEFAULT_CAMERA keeps every existing caller working unchanged: to_court(pt)
# still means "the one camera" until a caller starts naming one.
DEFAULT_CAMERA = "front"

_H: dict[str, np.ndarray] = {}       # camera_id -> pixel -> court
_H_inv: dict[str, np.ndarray] = {}   # camera_id -> court -> pixel


def _canonical_rect():
    return np.array(
        [[0.0, 0.0], [COURT_W, 0.0], [COURT_W, COURT_L], [0.0, COURT_L]],
        dtype=np.float32,
    )


def build_homography(corners=None, camera_id: str = DEFAULT_CAMERA):
    """
    Build (and cache) the pixel->court homography from 4 court corners in
    order [net_left, net_right, baseline_right, baseline_left]. Falls back to
    settings.COURT_CORNERS. Raises ValueError if unset or degenerate.

    Cached per `camera_id`; a failed build leaves that camera's previous
    homography in place rather than half-updating it, so a rejected
    calibration cannot take a running drill's geometry down with it.
    """
    corners = corners if corners is not None else COURT_CORNERS
    if not corners or len(corners) != 4:
        raise ValueError(
            "COURT_CORNERS not set (need 4 points) — run calibrate.py first."
        )
    src = np.array(corners, dtype=np.float32)
    try:
        H = cv2.getPerspectiveTransform(src, _canonical_rect())
        H_inv = np.linalg.inv(H)
    except (cv2.error, np.linalg.LinAlgError) as exc:
        raise ValueError(
            f"Court corners look degenerate — re-run calibration. ({exc})"
        )
    _H[camera_id] = H
    _H_inv[camera_id] = H_inv
    return H


def _ensure_homography(camera_id: str = DEFAULT_CAMERA):
    if camera_id not in _H:
        # Only the default camera can fall back to config; naming an
        # uncalibrated camera is a caller error, not something to paper over
        # with another camera's geometry.
        if camera_id != DEFAULT_CAMERA:
            raise ValueError(
                f"camera {camera_id!r} has no homography — calibrate it first."
            )
        build_homography(camera_id=camera_id)


def clear_homography(camera_id: str | None = None) -> None:
    """Forget one camera's homography, or all of them when camera_id is None."""
    if camera_id is None:
        _H.clear()
        _H_inv.clear()
        return
    _H.pop(camera_id, None)
    _H_inv.pop(camera_id, None)


def calibrated_cameras() -> list[str]:
    """Camera ids that currently hold a homography."""
    return sorted(_H)


def to_court(pt, camera_id: str = DEFAULT_CAMERA):
    """Pixel (x, y) -> court (cx, cy)."""
    _ensure_homography(camera_id)
    p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    c = cv2.perspectiveTransform(p, _H[camera_id])[0][0]
    return float(c[0]), float(c[1])


def court_to_pixel(pt, camera_id: str = DEFAULT_CAMERA):
    """Court (cx, cy) -> pixel (x, y). For drawing overlays."""
    _ensure_homography(camera_id)
    p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    c = cv2.perspectiveTransform(p, _H_inv[camera_id])[0][0]
    return float(c[0]), float(c[1])


def get_ankle_position(keypoints):
    """
    Extract the most confident ankle position from YOLO pose keypoints.
    Keypoint indices: 15 = left ankle, 16 = right ankle
    Returns (ankle_x, ankle_y) or None if not confident enough
    """
    left_ankle  = keypoints[15]   # (x, y, confidence)
    right_ankle = keypoints[16]

    l_conf = float(left_ankle[2])
    r_conf = float(right_ankle[2])

    if l_conf < ANKLE_CONFIDENCE and r_conf < ANKLE_CONFIDENCE:
        return None  # neither ankle detected confidently

    # Use the more confident ankle
    if l_conf >= r_conf:
        return float(left_ankle[0]), float(left_ankle[1])
    else:
        return float(right_ankle[0]), float(right_ankle[1])


def in_court_bounds(cx, cy):
    """True if a court-space point is inside the trainee's half-court.

    This is what excludes the feeder/near side: those points map to cy < 0.
    """
    return 0.0 <= cx <= COURT_W and 0.0 <= cy <= COURT_L


def get_zone_from_position(cx, cy):
    """Court-space (cx, cy) -> zone name, or None if outside all 6 zones."""
    for zone_name, (x1, y1, x2, y2) in PLAYER_ZONES.items():
        if x1 <= cx < x2 and y1 <= cy < y2:
            return zone_name
    return None


def get_player_in_zone(zone_name, player_positions):
    """
    zone_name + court-space player positions {id: (cx, cy)} -> player id whose
    position is in that zone, else None.
    """
    if zone_name not in PLAYER_ZONES:
        return None
    x1, y1, x2, y2 = PLAYER_ZONES[zone_name]
    for player_id, (cx, cy) in player_positions.items():
        if x1 <= cx < x2 and y1 <= cy < y2:
            return player_id
    return None


def get_shuttle_side(cy):
    """Court-space y -> which side of the net the shuttle is on."""
    return "feeder_side" if cy < -NET_DEADBAND else "player_side"


def crossed_net(prev_cy, cy):
    """True on a feeder->player crossing (sign flip past the dead-band)."""
    return prev_cy < -NET_DEADBAND and cy > NET_DEADBAND
