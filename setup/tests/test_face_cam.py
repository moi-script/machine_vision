import os
import sys

import cv2
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


class Players:
    def __init__(self, saved=None):
        self.saved = saved if saved is not None else {}

    def find_one(self, q):
        return {"_id": q["_id"]}

    def update_one(self, q, u):
        self.saved.update(u["$set"])


def _enroll_stubs(monkeypatch, embed=lambda img, grayscale=False: [1.0, 0.0, 0.0]):
    monkeypatch.setattr(face_cam.time, "sleep", lambda s: None)
    monkeypatch.setattr(face_cam.face, "models_available", lambda: True)
    monkeypatch.setattr(face_cam.face, "detect_and_embed", embed)
    monkeypatch.setattr(face_cam, "_grayscale", lambda: False)


def _boards(monkeypatch, want, usb, wifi):
    monkeypatch.setattr(face_cam._settings, "FACE_CAM_SOURCE", want)
    monkeypatch.setattr(face_cam.sources, "describe",
                        lambda cid: {"available": usb, "name": "aero-face"})
    monkeypatch.setattr(face_cam.esp32, "check_health", lambda: wifi)


def test_health_503_when_device_missing(monkeypatch):
    _boards(monkeypatch, "auto", usb=False, wifi=False)
    assert client.get("/api/face-cam/health").status_code == 503


def test_health_ok_gives_relative_stream_url(monkeypatch):
    _boards(monkeypatch, "auto", usb=True, wifi=True)
    r = client.get("/api/face-cam/health")
    assert r.status_code == 200
    assert r.json()["mode"] == "usb"
    assert r.json()["streamUrl"] == "/api/face-cam/stream"


def test_auto_falls_back_to_wifi(monkeypatch):
    _boards(monkeypatch, "auto", usb=False, wifi=True)
    monkeypatch.setattr(face_cam._settings, "ESP32_CAM_IP", "10.0.0.9")
    r = client.get("/api/face-cam/health")
    assert r.status_code == 200
    assert r.json()["mode"] == "wifi"
    assert r.json()["streamUrl"] == "http://10.0.0.9:81/stream"


def test_forced_modes_ignore_the_other_board(monkeypatch):
    _boards(monkeypatch, "usb", usb=False, wifi=True)
    assert client.get("/api/face-cam/health").status_code == 503
    _boards(monkeypatch, "wifi", usb=True, wifi=False)
    assert client.get("/api/face-cam/health").status_code == 503


def test_enroll_averages_shots_from_the_worker(monkeypatch):
    monkeypatch.setattr(face_cam, "_mode", lambda: "usb")
    monkeypatch.setattr(face_cam, "_ensure_worker", lambda: W())
    _enroll_stubs(monkeypatch)
    saved = {}
    monkeypatch.setattr(face_cam.db, "players", lambda: Players(saved))
    r = client.post("/api/face-cam/players/p1/enroll?shots=3")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "usb"
    assert body["shotsUsed"] == 3
    assert body["imageDataUrl"].startswith("data:image/jpeg;base64,")
    assert saved["faceEnrolled"] is True


def test_enroll_from_the_wifi_board(monkeypatch):
    monkeypatch.setattr(face_cam, "_mode", lambda: "wifi")
    ok, jpg = cv2.imencode(".jpg", np.full((60, 80, 3), 128, dtype=np.uint8))
    calls = []

    def snap(use_flash=False):
        calls.append(use_flash)
        return jpg.tobytes()
    monkeypatch.setattr(face_cam.esp32, "capture_snapshot", snap)
    _enroll_stubs(monkeypatch)
    saved = {}
    monkeypatch.setattr(face_cam.db, "players", lambda: Players(saved))
    r = client.post("/api/face-cam/players/p1/enroll?shots=3")
    assert r.status_code == 200, r.text
    assert r.json()["mode"] == "wifi"
    assert calls == [False, False, False]      # three captures, flash off
    assert saved["faceEnrolled"] is True


def test_enroll_502_when_wifi_capture_fails(monkeypatch):
    monkeypatch.setattr(face_cam, "_mode", lambda: "wifi")

    def snap(use_flash=False):
        raise face_cam.esp32.ESP32CaptureError("could not reach ESP32-CAM")
    monkeypatch.setattr(face_cam.esp32, "capture_snapshot", snap)
    _enroll_stubs(monkeypatch)
    monkeypatch.setattr(face_cam.db, "players", lambda: Players())
    assert client.post("/api/face-cam/players/p1/enroll").status_code == 502


def test_enroll_503_when_no_board(monkeypatch):
    _boards(monkeypatch, "auto", usb=False, wifi=False)
    _enroll_stubs(monkeypatch)
    monkeypatch.setattr(face_cam.db, "players", lambda: Players())
    assert client.post("/api/face-cam/players/p1/enroll").status_code == 503


def test_enroll_422_when_no_face(monkeypatch):
    monkeypatch.setattr(face_cam, "_mode", lambda: "usb")
    monkeypatch.setattr(face_cam, "_ensure_worker", lambda: W())
    _enroll_stubs(monkeypatch, embed=lambda img, grayscale=False: None)
    monkeypatch.setattr(face_cam.db, "players", lambda: Players())
    assert client.post("/api/face-cam/players/p1/enroll").status_code == 422


def test_old_esp32_routes_are_gone():
    # The Wi-Fi board is served through /api/face-cam now, not its own router.
    assert client.get("/api/esp32/health").status_code in (404, 405)
