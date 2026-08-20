# ============================================================
# motion.py — Isolate fast movers on a drifting camera.
#
# Median-background subtraction was tried first and FAILED on this footage: the
# camera drifts by tens of pixels even inside a "stable" window, so static court
# lines light up in the difference. Measured ~58 candidates/frame, whose
# strongest members were visibly line edges and player limbs.
#
# This instead aligns t-K and t+K onto t by phase correlation and takes the
# MINIMUM of the two differences: a pixel must differ from BOTH neighbours to
# survive. Static structure cancels, and so do the lens smudge blobs, which are
# fixed in image coordinates. Measured: 6.0 candidates/frame.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Candidate:
    frame: int
    x: int
    y: int
    w: int
    h: int
    area: int
    peak: float


def translation_align(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Shift `src` onto `ref`, estimating translation by phase correlation.

    Translation only. The drift here is a slowly sagging tripod, not a pan, and a
    full homography would be both slower and less stable on such a dark scene.
    """
    shift, _ = cv2.phaseCorrelate(ref.astype(np.float32), src.astype(np.float32))
    dx, dy = shift
    matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(
        src,
        matrix,
        (src.shape[1], src.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def three_frame_diff(prev: np.ndarray, cur: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Per-pixel min(|cur - prev'|, |cur - nxt'|) after aligning both neighbours."""
    p = translation_align(prev, cur)
    n = translation_align(nxt, cur)
    d = cv2.min(cv2.absdiff(cur, p), cv2.absdiff(cur, n))
    return cv2.GaussianBlur(d, (3, 3), 0)


def candidate_boxes(
    diff: np.ndarray,
    frame_index: int,
    thresh: int = 22,
    min_dim: int = 3,
    max_dim: int = 45,
    min_area: int = 5,
    max_area: int = 700,
    ar_range: tuple[float, float] = (0.3, 3.0),
    min_fill: float = 0.32,
) -> list[Candidate]:
    """Shuttle-shaped connected components of a difference image.

    The filters encode what a shuttlecock is NOT: not enormous (a player), not a
    long thin sliver (a court-line edge), not sparse (noise speckle).
    """
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    out = []
    for k in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[k])
        if not (min_dim <= w <= max_dim and min_dim <= h <= max_dim):
            continue
        if not (min_area <= area <= max_area):
            continue
        if not (ar_range[0] <= w / h <= ar_range[1]):
            continue
        if area / (w * h) < min_fill:
            continue
        out.append(
            Candidate(
                frame=frame_index,
                x=x,
                y=y,
                w=w,
                h=h,
                area=area,
                peak=float(diff[y:y + h, x:x + w].max()),
            )
        )
    return out
