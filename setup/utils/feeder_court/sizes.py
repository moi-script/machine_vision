# ============================================================
# sizes.py — Measure real shuttlecock pixel size, and gate on it.
#
# Automated probing of these clips returned an IDENTICAL size distribution for
# near and far (median 8 px both), which is a red flag rather than a result: it
# means the probe was likely measuring noise and court-line edges, not shuttles.
# Only hand-drawn boxes settle it.
#
# 8 px matters because YOLOv8n's finest stride is 8 — an 8 px object occupies a
# single grid cell at native resolution. Below that, no amount of training helps.
# ============================================================

from __future__ import annotations

import numpy as np

STOP_PX = 8.0     # below this everywhere -> the architecture cannot work
STOCK_PX = 16.0   # at or above this everywhere -> no small-object head needed
REQUIRED_CLIPS = ("near", "mid", "far")


def yolo_box_max_dim_px(line: str, img_w: int, img_h: int) -> float:
    """Largest side, in pixels, of one YOLO-format label line."""
    parts = line.split()
    w = float(parts[3]) * img_w
    h = float(parts[4]) * img_h
    return max(w, h)


def size_percentiles(dims: list[float]) -> dict[str, float]:
    arr = np.asarray(dims, dtype=float)
    return {
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def gate_decision(medians_by_clip: dict[str, float]) -> str:
    """Apply the spec's calibration gate.

    Requires all three clips (near, mid, far) to be present.

    Returns "stop", "p2", or "stock".

    "stop" requires EVERY clip to be under 8 px. If one distance still works, the
    right answer is to narrow the deployment envelope, not to abandon the model —
    so that case returns an architecture instead.

    The choice between "stock" and "p2" is driven by the WORST clip, because the
    model has to handle every distance it will be deployed at.
    """
    missing = [c for c in REQUIRED_CLIPS if c not in medians_by_clip]
    if missing:
        raise ValueError(
            f"cannot decide the gate: no measurement for {missing}. "
            "An unmeasured clip is not the same as a passing one - a clip with "
            "zero hand-drawn boxes may be exactly the distance that forces a stop."
        )

    medians = list(medians_by_clip.values())

    if all(m < STOP_PX for m in medians):
        return "stop"
    if min(medians) >= STOCK_PX:
        return "stock"
    return "p2"


def evenly_spaced_frames(spans: list[tuple[int, int]], count: int) -> list[int]:
    """`count` frame indices spread evenly across the union of `spans`.

    Sampling evenly rather than randomly guarantees coverage of the whole
    timeline, so the calibration set cannot accidentally concentrate on one rally.
    """
    lengths = [b - a for a, b in spans]
    total = sum(lengths)
    if total < count:
        raise ValueError(f"need {count} frames but spans only hold {total}")

    positions = np.linspace(0, total - 1, count).astype(int)
    frames = []
    for pos in positions:
        remaining = int(pos)
        for (a, b), length in zip(spans, lengths):
            if remaining < length:
                frames.append(a + remaining)
                break
            remaining -= length
    return sorted(frames)
