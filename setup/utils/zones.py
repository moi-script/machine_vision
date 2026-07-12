# ============================================================
# zones.py — Court zone detection and player-zone matching
# ============================================================

import cv2
import numpy as np

from config.settings import (
    PLAYER_ZONES, NET_X, COURT_ZONE,
    ANKLE_CONFIDENCE, PLAYER_OVERLAP_RATIO,
    COURT_CORNERS, COURT_W, COURT_L, NET_DEADBAND,
)

# ── Homography (image pixels <-> top-down court space) ───────
_H = None       # pixel -> court
_H_inv = None   # court -> pixel


def _canonical_rect():
    return np.array(
        [[0.0, 0.0], [COURT_W, 0.0], [COURT_W, COURT_L], [0.0, COURT_L]],
        dtype=np.float32,
    )


def build_homography(corners=None):
    """
    Build (and cache) the pixel->court homography from 4 court corners in
    order [net_left, net_right, baseline_right, baseline_left]. Falls back to
    settings.COURT_CORNERS. Raises ValueError if unset or degenerate.
    """
    global _H, _H_inv
    corners = corners if corners is not None else COURT_CORNERS
    if not corners or len(corners) != 4:
        raise ValueError(
            "COURT_CORNERS not set (need 4 points) — run calibrate.py first."
        )
    src = np.array(corners, dtype=np.float32)
    try:
        _H = cv2.getPerspectiveTransform(src, _canonical_rect())
        _H_inv = np.linalg.inv(_H)
    except (cv2.error, np.linalg.LinAlgError) as exc:
        raise ValueError(
            f"Court corners look degenerate — re-run calibration. ({exc})"
        )
    return _H


def _ensure_homography():
    if _H is None:
        build_homography()


def to_court(pt):
    """Pixel (x, y) -> court (cx, cy)."""
    _ensure_homography()
    p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    c = cv2.perspectiveTransform(p, _H)[0][0]
    return float(c[0]), float(c[1])


def court_to_pixel(pt):
    """Court (cx, cy) -> pixel (x, y). For drawing overlays."""
    _ensure_homography()
    p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    c = cv2.perspectiveTransform(p, _H_inv)[0][0]
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


def get_zone_from_position(x, y):
    """
    Given an (x, y) position, return which of the 6 zones it belongs to.
    Returns zone name string or None if outside all zones.
    """
    for zone_name, (x1, y1, x2, y2) in PLAYER_ZONES.items():
        if x1 < x < x2 and y1 < y < y2:
            return zone_name
    return None


def get_player_in_zone(zone_name, player_positions):
    """
    Given a zone name and current player positions dict,
    return the player_id whose ankle is inside that zone.
    player_positions format: { player_id: (ankle_x, ankle_y) }
    Returns player_id or None if nobody is in that zone.
    """
    if zone_name not in PLAYER_ZONES:
        return None

    x1, y1, x2, y2 = PLAYER_ZONES[zone_name]

    for player_id, (ax, ay) in player_positions.items():
        if x1 < ax < x2 and y1 < ay < y2:
            return player_id

    return None


def is_inside_court(person_box):
    """
    Check if a detected person bounding box overlaps enough
    with the hardcoded court zone.
    Uses overlap ratio to filter out people standing outside court
    (coaches, bystanders near the boundary).
    person_box format: (x1, y1, x2, y2)
    """
    px1, py1, px2, py2 = person_box
    cx1, cy1, cx2, cy2 = COURT_ZONE

    overlap_x    = max(0, min(px2, cx2) - max(px1, cx1))
    overlap_y    = max(0, min(py2, cy2) - max(py1, cy1))
    overlap_area = overlap_x * overlap_y

    person_area  = max(1, (px2 - px1) * (py2 - py1))
    ratio        = overlap_area / person_area

    return ratio >= PLAYER_OVERLAP_RATIO


def get_shuttle_side(shuttle_x):
    """
    Determine which side of the net the shuttle is on.
    Returns 'player_side' or 'feeder_side'
    """
    if shuttle_x < NET_X:
        return "player_side"
    else:
        return "feeder_side"
