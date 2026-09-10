"""What each camera slot is actually reading from.

A slot is one of the rig positions (front/left/right/back). Its source is either
a video file — bundled footage or something the user uploaded — or a capture
device index for a USB or built-in camera. Everything downstream (calibration
frame grabs, the live pipeline) opens a slot through here, so switching a slot
from a file to a real camera changes one Mongo document and nothing else.

WHY THE REGISTRY IS PERSISTED
The rig will be four cameras, wired once and left alone. Re-picking sources on
every restart would make calibration meaningless, since a calibration belongs to
a physical viewpoint rather than to a slot name.
"""
from __future__ import annotations

import os
import sys
import threading

import cv2

from app import db

CAMERA_IDS = ("front", "left", "right", "back")

# Keys that identify *which* physical source a slot points at. Exactly one
# combination of these is live at a time (a path xor an index); the other
# must be cleared on write or a stale one left by a previous $set can hijack
# open_capture()'s precedence check.
_IDENTITY = ("path", "index")

_VID_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "vid_source")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "uploads")

# Bundled stand-ins, used until a slot is pointed somewhere else. Four angles of
# the same hall, which is what makes them usable as a four-camera rig with no
# hardware.
_DEFAULTS: dict[str, dict] = {
    "front": {"kind": "file", "path": os.path.join(_VID_DIR, "angle_1.mp4")},
    "left": {"kind": "file", "path": os.path.join(_VID_DIR, "angle_2.mp4")},
    "right": {"kind": "file", "path": os.path.join(_VID_DIR, "angle_3.mp4")},
    "back": {"kind": "file", "path": os.path.join(_VID_DIR, "Angle_4.mp4")},
}

_lock = threading.Lock()
_cache: dict[str, dict] = {}


def _col():
    return db.get_db()["camera_sources"]


def get(camera_id: str) -> dict:
    """Current source for a slot. Falls back to the bundled stand-in."""
    if camera_id not in CAMERA_IDS:
        raise KeyError(camera_id)
    with _lock:
        if camera_id in _cache:
            return dict(_cache[camera_id])
    doc = None
    try:
        doc = _col().find_one({"_id": camera_id})
    except Exception:
        pass  # Mongo down: the bundled default still lets the app run
    src = {k: v for k, v in (doc or {}).items() if k in ("kind", "path", "index", "label")}
    if not src:
        src = dict(_DEFAULTS[camera_id])
    # Normalise: the defaults are built with ".." segments, while the pickers
    # list absolute paths. Unnormalised, the two never compare equal and the UI
    # cannot show which source a slot is actually on.
    if src.get("kind") == "file" and src.get("path"):
        src["path"] = os.path.abspath(src["path"])
    with _lock:
        _cache[camera_id] = dict(src)
    return dict(src)


def set_source(camera_id: str, kind: str, path: str | None = None,
               index: int | None = None, label: str | None = None) -> dict:
    if camera_id not in CAMERA_IDS:
        raise KeyError(camera_id)
    if kind == "file":
        if not path or not os.path.exists(path):
            raise ValueError(f"no such video file: {path!r}")
        src = {"kind": "file", "path": os.path.abspath(path)}
    elif kind == "device":
        # A device is addressed either by capture index (Windows) or by a
        # stable /dev/v4l/by-id path (Linux, where udev reorders indices
        # across reboots).
        if index is None and not path:
            raise ValueError("device source needs an index or path")
        src = ({"kind": "device", "path": path} if path
               else {"kind": "device", "index": int(index)})
    else:
        raise ValueError(f"unknown source kind {kind!r}")
    if label:
        src["label"] = label

    with _lock:
        _cache[camera_id] = dict(src)
    try:
        stale = {k: "" for k in _IDENTITY if k not in src}
        update = {"$set": src}
        if stale:
            update["$unset"] = stale
        _col().update_one({"_id": camera_id}, update, upsert=True)
    except Exception:
        pass
    return src


def describe(camera_id: str) -> dict:
    src = get(camera_id)
    if src["kind"] == "file":
        return {**src, "name": os.path.basename(src["path"]),
                "available": os.path.exists(src["path"])}
    if "path" in src:
        return {**src, "name": os.path.basename(src["path"]),
                "available": os.path.exists(src["path"])}
    return {**src, "name": f"device {src['index']}", "available": True}


def open_capture(camera_id: str) -> cv2.VideoCapture:
    """Open a slot's source. The one seam between files and real cameras."""
    src = get(camera_id)
    if src["kind"] == "device":
        # index takes precedence over path: set_source() unsets whichever
        # identity key it isn't setting, so a doc should never carry both,
        # but if a stale one ever lingers (an old doc written before that
        # guard existed, a manual edit), the explicit index an operator
        # just picked must win over a leftover path rather than silently
        # keep playing whatever the path pointed at.
        if "index" in src:
            # DirectShow: MSMF takes ~10 s to open the built-in camera on
            # this laptop, and enumerates USB devices in a different order,
            # so an index that works under one backend can point elsewhere
            # under the other.
            if sys.platform == "win32":
                return cv2.VideoCapture(src["index"], cv2.CAP_DSHOW)
            return cv2.VideoCapture(src["index"])
        return cv2.VideoCapture(src["path"])
    if not os.path.exists(src["path"]):
        raise FileNotFoundError(src["path"])
    return cv2.VideoCapture(src["path"])


def is_file(camera_id: str) -> bool:
    return get(camera_id)["kind"] == "file"


def list_devices(max_index: int = 4) -> list[dict]:
    """Probe capture indices. Slow-ish, so the UI should call it on demand."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if sys.platform == "win32" \
            else cv2.VideoCapture(i)
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    found.append({"index": i, "width": frame.shape[1],
                                  "height": frame.shape[0]})
        finally:
            cap.release()
    return found


def list_uploads() -> list[dict]:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    out = []
    for name in sorted(os.listdir(UPLOAD_DIR)):
        p = os.path.join(UPLOAD_DIR, name)
        if os.path.isfile(p) and name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            out.append({"name": name, "path": os.path.abspath(p),
                        "size_mb": round(os.path.getsize(p) / 1e6, 1)})
    return out


def list_bundled() -> list[dict]:
    out = []
    root = os.path.abspath(_VID_DIR)
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            if name.lower().endswith(".mp4"):
                p = os.path.join(dirpath, name)
                out.append({"name": os.path.relpath(p, root).replace("\\", "/"),
                            "path": p,
                            "size_mb": round(os.path.getsize(p) / 1e6, 1)})
    return out
