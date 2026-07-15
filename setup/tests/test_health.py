# tests/test_health.py
from fastapi.testclient import TestClient
from app.server import app

client = TestClient(app)


def test_health_reports_mongo():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert "mongo" in body and "engine" in body and "camera" in body


def test_health_reports_face_flag():
    r = client.get("/api/health")
    assert "face" in r.json()
    assert isinstance(r.json()["face"], bool)
