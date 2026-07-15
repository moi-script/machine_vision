from fastapi.testclient import TestClient
from app.server import app
from app import db, face as face_mod

client = TestClient(app)


def _make_player():
    return client.post("/api/players", json={"name": "FACETEST"}).json()["id"]


def setup_function():
    db.players().delete_many({"name": "FACETEST"})


def test_enroll_stores_embedding_and_is_hidden(monkeypatch):
    pid = _make_player()
    # bypass the real model: pretend a face was found
    monkeypatch.setattr(face_mod, "models_available", lambda: True)
    monkeypatch.setattr(face_mod, "decode_data_url", lambda s: "img")
    monkeypatch.setattr(face_mod, "detect_and_embed", lambda img: [0.1] * 128)

    r = client.post(f"/api/players/{pid}/face", json={"imageDataUrl": "data:image/jpeg;base64,AAAA"})
    assert r.status_code == 200 and r.json()["ok"] is True

    # stored on the doc
    assert db.players().find_one({"_id": pid}).get("faceEnrolled") is True
    assert len(db.players().find_one({"_id": pid})["faceEmbedding"]) == 128
    # but NOT exposed via the API
    assert "faceEmbedding" not in client.get(f"/api/players/{pid}").json()
    assert all("faceEmbedding" not in p for p in client.get("/api/players").json())


def test_enroll_no_face_returns_422(monkeypatch):
    pid = _make_player()
    monkeypatch.setattr(face_mod, "models_available", lambda: True)
    monkeypatch.setattr(face_mod, "decode_data_url", lambda s: "img")
    monkeypatch.setattr(face_mod, "detect_and_embed", lambda img: None)
    r = client.post(f"/api/players/{pid}/face", json={"imageDataUrl": "data:image/jpeg;base64,AAAA"})
    assert r.status_code == 422
