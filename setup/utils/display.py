# ============================================================
# display.py — Draw overlays, zones, scores on camera frame
# ============================================================

import cv2
import time
import numpy as np
from collections import deque
from config.settings import (
    PLAYER_ZONES, COURT_W, COURT_L,
    COLOR_COURT_ZONE, COLOR_NET, COLOR_SHUTTLE,
    COLOR_ZONE_ACTIVE, COLOR_SCORE_TEXT, COLOR_WEAK_ZONE,
    COLOR_PLAYER_1, COLOR_PLAYER_2,
    SHOW_SKELETON, SHOW_ZONES, SHOW_SHUTTLE_TRAIL, TRAIL_LENGTH
)
from utils.zones import court_to_pixel

# Shuttle trail buffer
shuttle_trail = deque(maxlen=TRAIL_LENGTH)

# Frame timestamps for smoothed FPS (rolling average over the last 30 frames)
_fps_times = deque(maxlen=30)

PLAYER_COLORS = {
    1: COLOR_PLAYER_1,
    2: COLOR_PLAYER_2,
}


def _poly(pts):
    """Court-space corner list -> int pixel polygon for cv2."""
    return np.array([[int(round(x)), int(round(y))]
                     for x, y in (court_to_pixel(p) for p in pts)], dtype=np.int32)


def draw_court_zone(frame):
    """Draw the trainee half-court outline (projected trapezoid)."""
    corners = [(0.0, 0.0), (COURT_W, 0.0), (COURT_W, COURT_L), (0.0, COURT_L)]
    cv2.polylines(frame, [_poly(corners)], True, COLOR_COURT_ZONE, 2)
    lx, ly = court_to_pixel((0.0, 0.0))
    cv2.putText(frame, "COURT", (int(lx) + 5, int(ly) + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_COURT_ZONE, 1)


def draw_net(frame):
    """Draw the net as the segment between the two net corners (y=0 edge)."""
    x1, y1 = court_to_pixel((0.0, 0.0))
    x2, y2 = court_to_pixel((COURT_W, 0.0))
    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLOR_NET, 2)
    cv2.putText(frame, "NET", (int(x1) + 5, int(y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_NET, 1)


def draw_zones(frame, active_zone=None, weak_zones=None):
    """Draw the 6 court-space zones as projected trapezoids."""
    if not SHOW_ZONES:
        return
    weak_zones = weak_zones or []

    for zone_name, (x1, y1, x2, y2) in PLAYER_ZONES.items():
        if zone_name == active_zone:
            color, thickness = COLOR_ZONE_ACTIVE, 2
        elif zone_name in weak_zones:
            color, thickness = COLOR_WEAK_ZONE, 1
        else:
            color, thickness = COLOR_COURT_ZONE, 1

        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        cv2.polylines(frame, [_poly(corners)], True, color, thickness)

        lx, ly = court_to_pixel((x1, y1))
        cv2.putText(frame, zone_name, (int(lx) + 3, int(ly) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


def draw_player(frame, player_id, box, keypoints=None, scorable=True):
    """Draw player bounding box and optional skeleton.

    scorable=False means the person was detected but has no confident ankle on
    the court (e.g. feet out of frame), so they aren't tracked for scoring. They
    are still drawn, dimmed and labelled, so it's clear they're seen.
    """
    if scorable:
        color     = PLAYER_COLORS.get(player_id, (200, 200, 200))
        thickness = 2
        label     = f"P{player_id}"
    else:
        color     = (120, 120, 120)   # dim gray
        thickness = 1
        label     = f"P{player_id} (off-court)"

    x1, y1, x2, y2 = [int(v) for v in box]

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

    # Draw skeleton if pose keypoints available
    if SHOW_SKELETON and keypoints is not None:
        SKELETON_PAIRS = [
            (5, 6),   # shoulders
            (5, 7), (7, 9),    # left arm
            (6, 8), (8, 10),   # right arm
            (5, 11), (6, 12),  # torso
            (11, 12),          # hips
            (11, 13), (13, 15), # left leg
            (12, 14), (14, 16), # right leg
        ]
        for i, j in SKELETON_PAIRS:
            xi, yi, ci = keypoints[i]
            xj, yj, cj = keypoints[j]
            if ci > 0.4 and cj > 0.4:
                cv2.line(frame,
                         (int(xi), int(yi)),
                         (int(xj), int(yj)),
                         color, 2)

        # Joint dots on every confident keypoint (skip the 5 face points 0-4)
        for idx in range(5, 17):
            jx, jy, jc = keypoints[idx]
            if jc > 0.4:
                cv2.circle(frame, (int(jx), int(jy)), 3, color, -1)

        # Emphasize the ankles — they're what the court-line decision uses
        for idx in [15, 16]:
            ax, ay, ac = keypoints[idx]
            if ac > 0.4:
                cv2.circle(frame, (int(ax), int(ay)), 7, color, -1)
                cv2.circle(frame, (int(ax), int(ay)), 7, (255, 255, 255), 1)


def draw_shuttle(frame, shuttle_x, shuttle_y):
    """Draw shuttle position and motion trail"""
    if SHOW_SHUTTLE_TRAIL:
        shuttle_trail.append((int(shuttle_x), int(shuttle_y)))
        for i, (tx, ty) in enumerate(shuttle_trail):
            alpha = int(255 * (i / len(shuttle_trail)))
            cv2.circle(frame, (tx, ty), 3,
                       (0, alpha, 255), -1)

    cv2.circle(frame, (int(shuttle_x), int(shuttle_y)),
               8, COLOR_SHUTTLE, -1)
    cv2.putText(frame, "shuttle",
                (int(shuttle_x) + 10, int(shuttle_y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_SHUTTLE, 1)


def draw_fps(frame):
    """Draw a smoothed FPS reading in the top-right corner, above the scoreboard.

    Averages over the last ~30 frames so the number stays steady instead of
    flickering frame to frame.
    """
    _fps_times.append(time.time())

    if len(_fps_times) >= 2:
        span = _fps_times[-1] - _fps_times[0]
        fps = (len(_fps_times) - 1) / span if span > 0 else 0.0
    else:
        fps = 0.0

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (frame.shape[1] - 215, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_SCORE_TEXT, 1)


def draw_scoreboard(frame, player_scores, difficulty, active_target=None):
    """Draw live scoreboard in top-right corner"""
    panel_x = frame.shape[1] - 220
    panel_y = 32
    panel_w = 210
    panel_h = 30 + (len(player_scores.scores) * 60)

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Title
    cv2.putText(frame, f"DRILL: {difficulty.upper()}",
                (panel_x + 5, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                COLOR_SCORE_TEXT, 1)

    y_offset = panel_y + 40

    for pid, data in player_scores.scores.items():
        color       = PLAYER_COLORS.get(pid, (200, 200, 200))
        total_score = data["total_score"]
        total_shots = data["total_shots"]
        accuracy    = player_scores.get_total_accuracy(pid)

        label = f"P{pid}"
        if pid == active_target:
            label += " ◀ TARGET"

        cv2.putText(frame, label,
                    (panel_x + 5, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame,
                    f"  Score: {total_score}/{total_shots} ({accuracy:.0f}%)",
                    (panel_x + 5, y_offset + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    COLOR_SCORE_TEXT, 1)

        # Weak zones indicator
        weak = player_scores.get_weak_zones(pid)
        if weak:
            cv2.putText(frame,
                        f"  Weak: {', '.join(weak[:2])}",
                        (panel_x + 5, y_offset + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        COLOR_WEAK_ZONE, 1)

        y_offset += 60


def draw_status(frame, status_text, color=(255, 255, 255)):
    """Draw current drill status at the bottom of frame"""
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 40), (frame.shape[1], h), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
