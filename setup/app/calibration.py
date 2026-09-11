"""Solve, validate and reproject a court calibration.

WHY VALIDATION LIVES HERE AND NOT IN THE ROUTER
`POST /api/calibration` used to save first and build the homography afterwards,
inside `_notify_engine()`, which swallows exceptions. Degenerate corners were
therefore accepted silently and surfaced later as wrong line calls rather than
as an error at the point of entry. Preview and commit both call `solve()`, so a
client that skips preview cannot store something preview would have refused.

WHY REPROJECTION ERROR NEEDS MORE THAN FOUR POINTS
With exactly four points `getPerspectiveTransform` passes through them exactly
and the error is zero by construction — it measures nothing. Six points make the
solve least-squares (`findHomography`), and the residual becomes a real measure
of how well the clicks agree with a rectangle.
"""
from __future__ import annotations

import numpy as np
import cv2

from config.settings import COURT_W, COURT_L

CORNER_LABELS = ("net_left", "net_right", "baseline_right", "baseline_left")

# Court-space positions of the four corners, in CORNER_LABELS order.
CORNER_COURT = np.array(
    [[0.0, 0.0], [COURT_W, 0.0], [COURT_W, COURT_L], [0.0, COURT_L]],
    dtype=np.float32,
)

MAX_REPROJ_PX = 6.0
# Triangle area below this (px^2) means three points are effectively collinear
# and the quad is degenerate, which getPerspectiveTransform will accept happily.
MIN_TRIANGLE_AREA = 500.0


class CalibrationError(ValueError):
    """Raised with a list of human-readable reasons."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _triangle_area(a, b, c) -> float:
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0


def _is_convex_quad(pts) -> bool:
    """All cross products the same sign => convex and consistently wound."""
    signs = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(cross > 0)
    return all(signs) or not any(signs)


def validate(corners, image_size) -> list[str]:
    """Everything wrong with these corners, as messages. Empty means usable."""
    errors: list[str] = []
    if corners is None or len(corners) != 4:
        return [f"need exactly 4 corners, got {0 if corners is None else len(corners)}"]

    w, h = image_size
    for i, (x, y) in enumerate(corners):
        if not (0 <= x <= w and 0 <= y <= h):
            errors.append(f"{CORNER_LABELS[i]} ({x:.0f},{y:.0f}) is outside the {w}x{h} image")

    for i in range(4):
        for j in range(i + 1, 4):
            for k in range(j + 1, 4):
                if _triangle_area(corners[i], corners[j], corners[k]) < MIN_TRIANGLE_AREA:
                    errors.append(
                        f"{CORNER_LABELS[i]}, {CORNER_LABELS[j]} and {CORNER_LABELS[k]} "
                        "are nearly collinear — the homography would be degenerate")
                    break

    if not _is_convex_quad(corners):
        errors.append(
            "the four corners do not form a convex quad — they were probably "
            f"clicked out of order (expected {', '.join(CORNER_LABELS)})")
    return errors


def solve(corners, image_size):
    """Validate then solve. Returns (H, H_inv, reprojection_error_px).

    Raises CalibrationError listing every problem, so the operator sees all of
    them at once rather than fixing one and being told about the next.
    """
    errors = validate(corners, image_size)
    if errors:
        raise CalibrationError(errors)

    src = np.asarray(corners, dtype=np.float32)
    try:
        H = cv2.getPerspectiveTransform(src, CORNER_COURT)
        H_inv = np.linalg.inv(H)
    except (cv2.error, np.linalg.LinAlgError) as exc:
        raise CalibrationError([f"could not solve the homography ({exc})"])

    # Round-trip the clicked corners: pixel -> court -> pixel. With a 4-point
    # exact solve this is ~0 and only catches numerical trouble; it becomes a
    # real quality signal once extra points make the solve over-determined.
    back = cv2.perspectiveTransform(
        cv2.perspectiveTransform(src.reshape(-1, 1, 2), H), H_inv).reshape(-1, 2)
    err = float(np.mean(np.linalg.norm(back - src, axis=1)))
    return H, H_inv, err


def court_lines(singles: bool = False) -> dict[str, list]:
    """Court features in COURT-SPACE metres, for reprojecting onto the frame.

    Court space runs x across (0..COURT_W) and y along (0..COURT_L), with y=0 at
    the net — the convention `crossed_net()` already keys on.
    """
    w, l = COURT_W, COURT_L
    inset = (w - 5.18) / 2.0 if not singles else 0.0
    short_service = 1.98          # from the net
    long_service_doubles = l - 0.76
    return {
        "outline": [[0, 0], [w, 0], [w, l], [0, l]],
        "net_line": [[0, 0], [w, 0]],
        "baseline": [[0, l], [w, l]],
        "singles_left": [[inset, 0], [inset, l]],
        "singles_right": [[w - inset, 0], [w - inset, l]],
        "short_service": [[0, short_service], [w, short_service]],
        "long_service_doubles": [[0, long_service_doubles], [w, long_service_doubles]],
        "centre_line": [[w / 2.0, short_service], [w / 2.0, l]],
    }


def reproject(H_inv, singles: bool = False) -> dict[str, list]:
    """Court features mapped back into image pixels, for the overlay.

    This is what turns calibration from "hope it is right" into "look, it is
    right": the operator sees the whole court drawn over the real frame, not
    just the four points they clicked.
    """
    out: dict[str, list] = {}
    for name, pts in court_lines(singles).items():
        arr = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        px = cv2.perspectiveTransform(arr, H_inv).reshape(-1, 2)
        out[name] = [[round(float(x), 1), round(float(y), 1)] for x, y in px]
    return out


# ── Line-band calibration, for the side and back cameras ────
#
# The front camera sees four court corners, so it gets a homography. A camera
# looking down a sideline does not: measured on lines.mp4, every visible court
# line sits inside ~110 px of a 720 px frame, and four points squeezed into that
# band give a homography that is numerically valid and useless in the near-far
# direction. What those views CAN answer is which side of a line the shuttle
# landed, so they are calibrated by tracing two lines instead - the inside line
# and the outside line - and a landing between them is IN.
#
# Strokes are fitted rather than used as polylines: a court line is straight, so
# least squares through every traced point averages out hand jitter, and
# retracing the same line twice makes the fit better rather than ambiguous.

LINE_ROLES = ("inside", "outside")


def fit_lines(strokes: list[dict]) -> dict[str, dict]:
    """[{role, points:[[x,y],...]}] -> {role: {a, b, n}} for y = a*x + b."""
    pooled: dict[str, list] = {}
    for s in strokes:
        role = s.get("role")
        pts = s.get("points") or []
        if role not in LINE_ROLES or len(pts) < 2:
            continue
        pooled.setdefault(role, []).extend(pts)

    fits: dict[str, dict] = {}
    errors: list[str] = []
    for role, pts in pooled.items():
        arr = np.asarray(pts, dtype=float)
        if np.ptp(arr[:, 0]) < 20:
            errors.append(f"the {role} line is too steep or too short to fit")
            continue
        a, b = np.polyfit(arr[:, 0], arr[:, 1], 1)
        resid = float(np.mean(np.abs(arr[:, 1] - (a * arr[:, 0] + b))))
        fits[role] = {"a": float(a), "b": float(b), "n": len(arr),
                      "residual_px": round(resid, 2)}

    missing = [r for r in LINE_ROLES if r not in fits]
    if missing:
        errors.append(f"missing traced line(s): {', '.join(missing)}")
    if errors:
        raise CalibrationError(errors)

    # Two lines that cross inside the frame are not a band; almost always the
    # two roles were traced onto the same court line.
    a1, b1 = fits["inside"]["a"], fits["inside"]["b"]
    a2, b2 = fits["outside"]["a"], fits["outside"]["b"]
    if abs(a1 - a2) > 1e-9:
        x_cross = (b2 - b1) / (a1 - a2)
        if 0 <= x_cross <= 1280:
            raise CalibrationError([
                "the two lines cross inside the frame - they are probably "
                "tracing the same court line"])
    return fits


def band_call(fits: dict[str, dict], point) -> str:
    """IN when a landing sits between the two fitted lines."""
    x, y = point
    ys = sorted(f["a"] * x + f["b"] for f in fits.values())
    return "IN" if ys[0] <= y <= ys[-1] else "OUT"


def line_overlay(fits: dict[str, dict], width: int = 1280) -> dict[str, list]:
    """Each fitted line as two endpoints, for drawing across the frame."""
    return {role: [[0.0, round(f["b"], 1)],
                   [float(width - 1), round(f["a"] * (width - 1) + f["b"], 1)]]
            for role, f in fits.items()}
