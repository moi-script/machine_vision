"""Pure geometry helpers over YOLOv8 pose keypoints.

`keypoints` is a list of 17 [x, y, confidence] in COCO order. Every helper is
confidence-gated: it returns None when a required keypoint is below `min_conf`,
so callers never build a metric from a jittery low-confidence point."""
from __future__ import annotations
import math

# COCO keypoint indices we use
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


def point(keypoints, idx, min_conf):
    """(x, y) of keypoint idx if its confidence >= min_conf, else None."""
    kp = keypoints[idx]
    if float(kp[2]) < min_conf:
        return None
    return float(kp[0]), float(kp[1])


def angle(a, b, c):
    """Interior angle in degrees at vertex b, formed by a-b-c. None if degenerate."""
    if a is None or b is None or c is None:
        return None
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return None
    cosv = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))


def _side_knee(keypoints, hip_i, knee_i, ankle_i, min_conf):
    return angle(point(keypoints, hip_i, min_conf),
                 point(keypoints, knee_i, min_conf),
                 point(keypoints, ankle_i, min_conf))


def knee_angle(keypoints, min_conf=0.5):
    """Mean hip-knee-ankle angle over whichever legs are confidently visible."""
    vals = [v for v in (
        _side_knee(keypoints, L_HIP, L_KNEE, L_ANKLE, min_conf),
        _side_knee(keypoints, R_HIP, R_KNEE, R_ANKLE, min_conf),
    ) if v is not None]
    return sum(vals) / len(vals) if vals else None


def _midpoint(p, q):
    return ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)


def torso_lean(keypoints, min_conf=0.5):
    """Angle (deg) of the shoulder-mid -> hip-mid line away from vertical."""
    ls = point(keypoints, L_SHOULDER, min_conf)
    rs = point(keypoints, R_SHOULDER, min_conf)
    lh = point(keypoints, L_HIP, min_conf)
    rh = point(keypoints, R_HIP, min_conf)
    if None in (ls, rs, lh, rh):
        return None
    sm = _midpoint(ls, rs)
    hm = _midpoint(lh, rh)
    dx = hm[0] - sm[0]
    dy = hm[1] - sm[1]
    if dx == 0 and dy == 0:
        return None
    # angle from the vertical axis (0 deg = perfectly upright)
    return abs(math.degrees(math.atan2(dx, dy)))


def stance_width(keypoints, min_conf=0.5):
    """Ankle horizontal separation divided by shoulder separation (body-scaled)."""
    la = point(keypoints, L_ANKLE, min_conf)
    ra = point(keypoints, R_ANKLE, min_conf)
    ls = point(keypoints, L_SHOULDER, min_conf)
    rs = point(keypoints, R_SHOULDER, min_conf)
    if None in (la, ra, ls, rs):
        return None
    shoulder = abs(ls[0] - rs[0])
    if shoulder == 0:
        return None
    return abs(la[0] - ra[0]) / shoulder


def _wrist_speed(kp_now, kp_prev, dt, wrist_i, shoulder_i, min_conf):
    wn = point(kp_now, wrist_i, min_conf)
    sn = point(kp_now, shoulder_i, min_conf)
    wp = point(kp_prev, wrist_i, min_conf)
    sp = point(kp_prev, shoulder_i, min_conf)
    if None in (wn, sn, wp, sp) or dt <= 0:
        return None
    # shoulder-relative displacement removes whole-body translation
    rel_now = (wn[0] - sn[0], wn[1] - sn[1])
    rel_prev = (wp[0] - sp[0], wp[1] - sp[1])
    return math.hypot(rel_now[0] - rel_prev[0], rel_now[1] - rel_prev[1]) / dt


def max_wrist_speed(kp_now, kp_prev, dt, min_conf=0.5):
    """Max shoulder-relative wrist speed across both arms, or None."""
    vals = [v for v in (
        _wrist_speed(kp_now, kp_prev, dt, L_WRIST, L_SHOULDER, min_conf),
        _wrist_speed(kp_now, kp_prev, dt, R_WRIST, R_SHOULDER, min_conf),
    ) if v is not None]
    return max(vals) if vals else None


def bbox_height(box):
    """Pixel height of an [x1, y1, x2, y2] box."""
    return float(box[3]) - float(box[1])
