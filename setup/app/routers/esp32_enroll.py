"""
esp32_enroll.py

ESP32-CAM-based face enrollment. Additive: mounted as its own router
next to players.py, does not modify app/players.py, app/db.py, or
app/face.py.

Drop this file at:  app/routers/esp32_enroll.py
Wiring instructions: esp32_server_wireup.txt

Mirrors the exact behaviour of players.enroll_face() (same grayscale
handling, same db write, same 404/503/422 semantics) — the only
difference is the image comes from the ESP32-CAM instead of an
uploaded data URL, and we average several shots into one embedding
for a more robust enrollment than a single frame gives you.
"""

from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException
import numpy as np

from app import db, face
from app.esp32_camera_client import capture_snapshot, check_health, ESP32CaptureError

router = APIRouter(prefix="/api/esp32", tags=["esp32-enrollment"])

MIN_GOOD_SHOTS = 3
MAX_ATTEMPTS = MIN_GOOD_SHOTS * 3  # generous retry budget for bad frames


def _jpeg_bytes_to_bgr(jpeg_bytes: bytes):
    """Reuse face.decode_data_url so decoding stays identical to the
    upload path — just base64-wrap the raw JPEG the ESP32 returns."""
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return face.decode_data_url(f"data:image/jpeg;base64,{b64}")


def _average_embedding(embeddings: list[list[float]]) -> list[float]:
    """L2-normalize each shot, average, then re-normalize the result.
    This is what turns several imperfect angles into one embedding
    that's more robust than any single frame — while still fitting the
    existing single-`faceEmbedding`-field schema."""
    vecs = np.asarray(embeddings, dtype=np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    mean = vecs.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    return mean.tolist()


@router.get("/health")
async def esp32_health():
    """Connectivity check for the enrollment station camera."""
    ok = await check_health()
    if not ok:
        raise HTTPException(503, "ESP32-CAM unreachable")
    return {"status": "ok"}


@router.post("/players/{pid}/enroll")
async def enroll_face_via_esp32(pid: str, shots: int = MIN_GOOD_SHOTS):
    if db.players().find_one({"_id": pid}) is None:
        raise HTTPException(404, "player not found")
    if not face.models_available():
        raise HTTPException(503, "face models unavailable — run fetch_face_models.py")
    if shots < 1:
        raise HTTPException(400, "shots must be >= 1")

    # Same grayscale handling as the upload path in players.py, so an
    # ESP32 enrollment matches the feeder camera's colour modality.
    from app.routers.settings import load_settings
    grayscale = load_settings().camera.grayscale

    embeddings: list[list[float]] = []
    attempts = 0
    # Keep the first frame that actually had a usable face — returned to the
    # frontend so it can build the same grayscale-cutout card thumbnail the
    # webcam flow produces, via the existing toGrayscaleCutout() on a color crop.
    best_jpeg_b64: str | None = None

    while len(embeddings) < shots and attempts < max(MAX_ATTEMPTS, shots * 3):
        attempts += 1
        try:
            jpeg_bytes = await capture_snapshot(use_flash=True)
        except ESP32CaptureError as exc:
            raise HTTPException(502, str(exc)) from exc

        img = _jpeg_bytes_to_bgr(jpeg_bytes)
        if img is None:
            continue

        emb = face.detect_and_embed(img, grayscale=grayscale)
        if emb is None:
            continue  # no face in this frame — try again rather than save junk

        embeddings.append(emb)
        if best_jpeg_b64 is None:
            best_jpeg_b64 = base64.b64encode(jpeg_bytes).decode("ascii")

    if len(embeddings) < min(MIN_GOOD_SHOTS, shots):
        raise HTTPException(
            422,
            f"only got {len(embeddings)} usable face(s) out of {attempts} "
            f"attempts — check lighting/framing and retry",
        )

    final_embedding = _average_embedding(embeddings) if len(embeddings) > 1 else embeddings[0]

    db.players().update_one(
        {"_id": pid},
        {"$set": {"faceEmbedding": final_embedding, "faceEnrolled": True}},
    )
    return {
        "ok": True,
        "shotsUsed": len(embeddings),
        "attempts": attempts,
        "imageDataUrl": f"data:image/jpeg;base64,{best_jpeg_b64}" if best_jpeg_b64 else None,
    }