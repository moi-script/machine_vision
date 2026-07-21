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


def test_skill_endpoint_returns_unranked_for_fresh_player():
    pid = client.post("/api/players", json={"name": "APITEST"}).json()["id"]
    r = client.get(f"/api/players/{pid}/skill")
    assert r.status_code == 200
    body = r.json()
    assert body["tier"] == "Unranked"
    assert body["history"] == []


def test_skill_endpoint_404_for_missing_player():
    assert client.get("/api/players/nope/skill").status_code == 404


def test_get_player_includes_skillprofile_when_present():
    pid = client.post("/api/players", json={"name": "APITEST"}).json()["id"]
    db.players().update_one({"_id": pid},
        {"$set": {"skillProfile": {"computed": {"tier": "Novice"}}}})
    got = client.get(f"/api/players/{pid}").json()
    assert got["skillProfile"]["computed"]["tier"] == "Novice"
