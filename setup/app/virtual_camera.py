"""Video files standing in for the four-camera rig.

WHY THIS EXISTS
The rig is front/left/right/back, and none of it is wired up yet. Calibration,
the court overlay and the scoring flow all need four simultaneous views to be
exercised at all, so this serves frames from the four `angle_*.mp4` recordings
under the same camera ids the real rig will use. Swapping in real capture later
is a change to `_open()` and nothing else.

Frames are pulled on demand rather than streamed: calibration clicks on a frozen
frame, and freezing is the point — points clicked against a moving image would
refer to a view that no longer exists by the time they are posted.
"""
from __future__ import annotations

import base64
import os
import threading
import time

import cv2

# Rig positions, in the order the calibration page shows them. front sees the
# net, left/right see their sidelines, back sees the baseline.
CAMERA_IDS = ("front", "left", "right", "back")

_VID_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "vid_source")

# Stand-in footage. These are four angles of the same hall, which is what makes
# them usable as a four-camera rig without hardware.
_SOURCES: dict[str, str] = {
    "front": os.path.join(_VID_DIR, "angle_1.mp4"),
    "left": os.path.join(_VID_DIR, "angle_2.mp4"),
    "right": os.path.join(_VID_DIR, "angle_3.mp4"),
    "back": os.path.join(_VID_DIR, "Angle_4.mp4"),
}

_lock = threading.Lock()
_frozen: dict[str, dict] = {}   # camera_id -> {"frame_id", "image", "at"}


def source_path(camera_id: str) -> str:
    if camera_id not in _SOURCES:
        raise KeyError(camera_id)
    return os.path.abspath(_SOURCES[camera_id])


def available() -> dict[str, bool]:
    from app import sources
    return {cid: sources.describe(cid).get("available", False) for cid in CAMERA_IDS}


def _open(camera_id: str):
    """Delegates to the source registry so calibration and the live pipeline
    always read the same thing. A slot pointed at a USB camera is calibrated
    against that camera, not against the bundled stand-in it replaced."""
    from app import sources
    return sources.open_capture(camera_id)


def grab(camera_id: str, frame_index: int | None = None):
    """Read one frame. Returns (frame_id, BGR image).

    The frame is cached under a generated id so a later POST can prove its
    clicked points refer to this exact image.
    """
    # A running worker already holds this device. Opening it a second time
    # fails on V4L2 (busy) or steals frames, so freeze its newest frame.
    from app import pipeline
    w = pipeline.get(camera_id)
    live = w.latest_frame() if (w is not None and w.alive) else None
    if live is not None:
        frame_id = f"{camera_id}-live-{int(time.time() * 1000)}"
        with _lock:
            _frozen[camera_id] = {"frame_id": frame_id, "image": live,
                                  "at": time.time(), "index": 0}
        return frame_id, live

    cap = _open(camera_id)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # A live device reports no frame count and cannot seek; just take the
        # next frame it offers.
        seekable = total > 0
        if frame_index is None:
            # A quarter in, not frame 0. Recordings routinely start with the
            # lens covered or the rig being carried into position — lines.mp4's
            # first ~130 frames measure a Laplacian variance of 4 against a
            # median of 1300, i.e. unusable for clicking court lines on.
            idx = total // 4 if seekable else 0
        else:
            idx = max(0, min(frame_index, max(total - 1, 0))) if seekable else 0
        if idx and seekable:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"could not read frame {idx} from {camera_id}")
    finally:
        cap.release()

    frame_id = f"{camera_id}-{idx}-{int(time.time() * 1000)}"
    with _lock:
        _frozen[camera_id] = {"frame_id": frame_id, "image": frame, "at": time.time(),
                              "index": idx}
    return frame_id, frame


def frozen(camera_id: str) -> dict | None:
    with _lock:
        return _frozen.get(camera_id)


def frame_count(camera_id: str) -> int:
    from app import sources
    if sources.get(camera_id)["kind"] == "device":
        return 0          # live devices have no length; don't open a busy one
    cap = _open(camera_id)
    try:
        return max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
    finally:
        cap.release()


def enhance(img):
    """CLAHE boost. This footage is dim enough that court lines are barely above
    the floor in brightness, and a line you cannot see is a line you cannot click."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def to_jpeg_b64(img, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")
