# tests/test_players_api.py
from fastapi.testclient import TestClient
from app.server import app
from app import db

client = TestClient(app)


def setup_function():
    db.players().delete_many({"name": "APITEST"})


def test_create_and_list_player():
    r = client.post("/api/players", json={"name": "APITEST", "age": 21})
    assert r.status_code == 200
    pid = r.json()["id"]
    assert pid.startswith("p")
    assert r.json()["stats"]["totalShots"] == 0

    ids = [p["id"] for p in client.get("/api/players").json()]
    assert pid in ids


def test_patch_and_soft_delete():
    pid = client.post("/api/players", json={"name": "APITEST"}).json()["id"]
    client.patch(f"/api/players/{pid}", json={"age": 30})
    assert client.get(f"/api/players/{pid}").json()["age"] == 30
    client.delete(f"/api/players/{pid}")
    assert client.get(f"/api/players/{pid}").json()["isActive"] is False
