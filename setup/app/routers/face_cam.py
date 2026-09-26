"""Face enrollment from the registration camera.

Two boards can serve it, behind the same endpoints so the UI never changes:

- "usb":  the AERO-FACE ESP32-S3 board, a UVC webcam on the Pi opened as the
  "face" slot by one raw pipeline worker. The preview stream and the enrollment
  grabs both read that worker, so they never fight over the device.
- "wifi": the ESP32-CAM station on the LAN (app/esp32_camera_client.py). The
  browser shows the board's own MJPEG preview; enrollment GETs /capture.

settings.FACE_CAM_SOURCE picks one, or "auto" prefers USB and falls back to
Wi-Fi. Enrollment behaviour is the same for both: several good shots,
L2-averaged into the single faceEmbedding the player schema holds.
"""
from __future__ import annotations

import base64
import time

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

from app import db, face, pipeline, sources
from app import esp32_camera_client as esp32
from config import settings as _settings

router = APIRouter(prefix="/api/face-cam", tags=["face-cam"])

MIN_GOOD_SHOTS = 3
MAX_ATTEMPTS = MIN_GOOD_SHOTS * 3
SHOT_GAP_S = 0.2          # distinct frames at 10 fps, not the same one thrice
FIRST_FRAME_WAIT_S = 3.0


def _wanted() -> str:
    return getattr(_settings, "FACE_CAM_SOURCE", "auto")


def _mode() -> str | None:
    """Which board serves registration right now: "usb", "wifi" or None.

    USB is recognised by its device (udev names the AERO-FACE board by its USB
    serial), Wi-Fi by the ESP32-CAM answering /health at ESP32_CAM_IP."""
    want = _wanted()
    if want != "wifi" and sources.describe(sources.FACE_ID).get("available"):
        return "usb"
    if want != "usb" and esp32.check_health():
        return "wifi"
    return None


def _unavailable() -> HTTPException:
    want = _wanted()
    usb = f"USB face camera ({sources.describe(sources.FACE_ID).get('name', 'face')})"
    wifi = f"Wi-Fi ESP32-CAM ({getattr(_settings, 'ESP32_CAM_IP', None) or 'no IP set'})"
    if want == "usb":
        return HTTPException(503, f"face camera not connected - {usb}")
    if want == "wifi":
        return HTTPException(503, f"face camera unreachable - {wifi}")
    return HTTPException(503, f"no face camera - neither {usb} nor {wifi} is up")


def _ensure_worker():
    w = pipeline.get(sources.FACE_ID)
    if w is None or not w.alive:
        pipeline.start(sources.FACE_ID, raw=True, target_fps=15.0)
        w = pipeline.get(sources.FACE_ID)
    return w


def _usb_grabber():
    w = _ensure_worker()
    deadline = time.time() + FIRST_FRAME_WAIT_S
    while w.latest_frame() is None and time.time() < deadline:
        time.sleep(0.05)
    if w.latest_frame() is None:
        raise HTTPException(502, "face camera gave no frame")
    return w.latest_frame


def _wifi_grab():
    try:
        jpeg = esp32.capture_snapshot(use_flash=False)
    except esp32.ESP32CaptureError as exc:
        raise HTTPException(502, str(exc)) from exc
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def _grayscale() -> bool:
    # Match the on-court camera's modality so enrolled and live embeddings compare.
    from app.routers.settings import load_settings
    return load_settings().camera.grayscale


def _average_embedding(embeddings: list[list[float]]) -> list[float]:
    vecs = np.asarray(embeddings, dtype=np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    mean = vecs.mean(axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm if norm > 0 else mean).tolist()


@router.get("/health")
def health():
    mode = _mode()
    if mode is None:
        raise _unavailable()
    if mode == "wifi":
        return {"status": "ok", "mode": "wifi", "streamUrl": esp32.stream_url(),
                "source": {"name": "esp32-cam", "ip": _settings.ESP32_CAM_IP}}
    return {"status": "ok", "mode": "usb", "streamUrl": "/api/face-cam/stream",
            "source": sources.describe(sources.FACE_ID)}


@router.get("/stream")
def stream():
    if _mode() == "wifi":
        return RedirectResponse(esp32.stream_url())
    _ensure_worker()
    from app.routers.cameras import stream as camera_stream
    return camera_stream(sources.FACE_ID)


@router.post("/stop")
def stop():
    # A no-op in Wi-Fi mode: the board's preview ends when the page closes it.
    pipeline.stop(sources.FACE_ID)
    return {"running": False}


@router.post("/players/{pid}/enroll")
def enroll(pid: str, shots: int = MIN_GOOD_SHOTS):
    if db.players().find_one({"_id": pid}) is None:
        raise HTTPException(404, "player not found")
    if not face.models_available():
        raise HTTPException(503, "face models unavailable - run fetch_face_models.py")
    if shots < 1:
        raise HTTPException(400, "shots must be >= 1")

    mode = _mode()
    if mode is None:
        raise _unavailable()
    grab = _usb_grabber() if mode == "usb" else _wifi_grab

    grayscale = _grayscale()
    embeddings: list[list[float]] = []
    best_jpeg_b64: str | None = None
    attempts = 0
    while len(embeddings) < shots and attempts < max(MAX_ATTEMPTS, shots * 3):
        attempts += 1
        img = grab()
        time.sleep(SHOT_GAP_S)
        if img is None:
            continue
        emb = face.detect_and_embed(img, grayscale=grayscale)
        if emb is None:
            continue
        embeddings.append(emb)
        if best_jpeg_b64 is None:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                best_jpeg_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    if len(embeddings) < min(MIN_GOOD_SHOTS, shots):
        raise HTTPException(422, f"only got {len(embeddings)} usable face(s) out of "
                                 f"{attempts} attempts - check lighting/framing and retry")

    final = _average_embedding(embeddings) if len(embeddings) > 1 else embeddings[0]
    db.players().update_one({"_id": pid},
                            {"$set": {"faceEmbedding": final, "faceEnrolled": True}})
    return {"ok": True, "mode": mode, "shotsUsed": len(embeddings), "attempts": attempts,
            "imageDataUrl": f"data:image/jpeg;base64,{best_jpeg_b64}" if best_jpeg_b64 else None}
