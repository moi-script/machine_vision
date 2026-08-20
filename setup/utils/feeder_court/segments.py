# ============================================================
# segments.py — Find the usable window of each clip.
#
# Every clip in new_badminton_source has a heavy camera-motion segment at head
# and tail: the rig being carried into and out of position. near.mp4 literally
# ends on a close-up of a laptop keyboard. Training on those frames would teach
# the model a keyboard is a badminton court.
#
# Also splits the middle wherever motion spikes, because the three-frame
# differencing in motion.py assumes only small drift between neighbouring frames.
# ============================================================

from __future__ import annotations

import cv2
import numpy as np


def motion_profile(
    video_path: str,
    sample_step: int = 15,
    size: tuple[int, int] = (160, 90),
) -> np.ndarray:
    """Mean absolute difference between consecutively sampled frames.

    Index i of the result corresponds to frame (i + 1) * sample_step.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    values = []
    prev = None
    for i in range(0, total, sample_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size).astype(np.float32)
        if prev is not None:
            values.append(float(np.abs(gray - prev).mean()))
        prev = gray
    cap.release()
    return np.array(values, dtype=float)


def usable_segments(
    profile: np.ndarray,
    sample_step: int,
    threshold: float = 15.0,
    min_frames: int = 150,
) -> list[tuple[int, int]]:
    """Contiguous frame spans whose sampled motion stays below `threshold`.

    Spans shorter than `min_frames` are dropped — too short to yield a useful
    background model or a complete shuttle flight.
    """
    calm = np.asarray(profile, dtype=float) < threshold
    spans: list[tuple[int, int]] = []
    start = None

    for idx, is_calm in enumerate(calm):
        if is_calm and start is None:
            start = idx
        elif not is_calm and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, len(calm)))

    out = []
    for a, b in spans:
        start_frame = a * sample_step
        end_frame = b * sample_step
        if end_frame - start_frame >= min_frames:
            out.append((start_frame, end_frame))
    return out
