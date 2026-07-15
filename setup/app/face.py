"""Face detection + recognition via OpenCV YuNet (detect) and SFace (embed).

Models live in setup/models/ (gitignored); fetch with fetch_face_models.py.
Everything is guarded so a missing model never crashes callers — recognition
simply stays unavailable."""
from __future__ import annotations
import base64
import os

import cv2
import numpy as np

_MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
_YUNET = os.path.join(_MODELS, "face_detection_yunet_2023mar.onnx")
_SFACE = os.path.join(_MODELS, "face_recognition_sface_2021dec.onnx")

_detector = None
_recognizer = None


def models_available() -> bool:
    return os.path.isfile(_YUNET) and os.path.isfile(_SFACE)


def _load():
    global _detector, _recognizer
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(_YUNET, "", (320, 320),
                                              score_threshold=0.9)
        _recognizer = cv2.FaceRecognizerSF.create(_SFACE, "")
    return _detector, _recognizer


def decode_data_url(data_url: str):
    """Decode a base64 data URL (or raw base64) to a BGR image, or None."""
    try:
        b64 = data_url.split(",", 1)[-1]
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def detect_and_embed(bgr):
    """Return the 128-d embedding of the largest detected face, or None."""
    if bgr is None or not models_available():
        return None
    try:
        det, rec = _load()
        h, w = bgr.shape[:2]
        det.setInputSize((w, h))
        _, faces = det.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        # largest face by box area (cols 2,3 are width,height)
        face_row = max(faces, key=lambda f: float(f[2]) * float(f[3]))
        aligned = rec.alignCrop(bgr, face_row)
        feat = rec.feature(aligned)  # shape (1, 128) float32
        return np.asarray(feat, dtype=np.float32).flatten().tolist()
    except Exception:
        return None


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def best_match(embedding, enrolled: dict, threshold: float = 0.363):
    """Return (playerId, score) of the closest enrolled face above threshold."""
    best_id, best_score = None, -1.0
    for pid, emb in enrolled.items():
        s = cosine(embedding, emb)
        if s > best_score:
            best_id, best_score = pid, s
    if best_id is not None and best_score >= threshold:
        return best_id, best_score
    return None
