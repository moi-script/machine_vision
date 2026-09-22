import os
import sys

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.routers import face_cam  # noqa: E402
from app.server import app  # noqa: E402

client = TestClient(app)


class W:
    alive = True

    def latest_frame(self):
        return np.full((60, 80, 3), 128, dtype=np.uint8)


def test_health_503_when_device_missing(monkeypatch):
    monkeypatch.setattr(face_cam.sources, "describe", lambda cid: {"available": False})
    assert client.get("/api/face-cam/health").status_code == 503


def test_health_ok_gives_relative_stream_url(monkeypatch):
    monkeypatch.setattr(face_cam.sources, "describe", lambda cid: {"available": True, "name": "aero-face"})
    r = client.get("/api/face-cam/health")
    assert r.status_code == 200
    assert r.json()["streamUrl"] == "/api/face-cam/stream"


def test_enroll_averages_shots_from_the_worker(monkeypatch):
    monkeypatch.setattr(face_cam, "_ensure_worker", lambda: W())
    monkeypatch.setattr(face_cam.time, "sleep", lambda s: None)
    monkeypatch.setattr(face_cam.face, "models_available", lambda: True)
    monkeypatch.setattr(face_cam.face, "detect_and_embed",
                        lambda img, grayscale=False: [1.0, 0.0, 0.0])
    monkeypatch.setattr(face_cam, "_grayscale", lambda: False)
    saved = {}

    class Players:
        def find_one(self, q):
            return {"_id": q["_id"]}

        def update_one(self, q, u):
            saved.update(u["$set"])
    monkeypatch.setattr(face_cam.db, "players", lambda: Players())
    r = client.post("/api/face-cam/players/p1/enroll?shots=3")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["shotsUsed"] == 3
    assert body["imageDataUrl"].startswith("data:image/jpeg;base64,")
    assert saved["faceEnrolled"] is True


def test_enroll_422_when_no_face(monkeypatch):
    monkeypatch.setattr(face_cam, "_ensure_worker", lambda: W())
    monkeypatch.setattr(face_cam.time, "sleep", lambda s: None)
    monkeypatch.setattr(face_cam.face, "models_available", lambda: True)
    monkeypatch.setattr(face_cam.face, "detect_and_embed", lambda img, grayscale=False: None)
    monkeypatch.setattr(face_cam, "_grayscale", lambda: False)

    class Players:
        def find_one(self, q):
            return {"_id": "p1"}
    monkeypatch.setattr(face_cam.db, "players", lambda: Players())
    assert client.post("/api/face-cam/players/p1/enroll").status_code == 422


def test_old_esp32_routes_are_gone():
    assert client.get("/api/esp32/health").status_code in (404, 405)
