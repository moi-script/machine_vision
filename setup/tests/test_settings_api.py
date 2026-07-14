from fastapi.testclient import TestClient
from app.server import app
from app import db

client = TestClient(app)


def test_get_settings_creates_defaults():
    db.settings_col().delete_many({})
    body = client.get("/api/settings").json()
    assert body["drill"]["intervals"]["easy"] == 5.0
    assert len(body["court"]["corners"]) == 4


def test_put_settings_persists():
    body = client.get("/api/settings").json()
    body["detection"]["shuttleConf"] = 0.55
    client.put("/api/settings", json=body)
    assert client.get("/api/settings").json()["detection"]["shuttleConf"] == 0.55


def test_calibration_updates_corners():
    corners = [[1, 2], [3, 4], [5, 6], [7, 8]]
    client.post("/api/calibration", json={"corners": corners})
    assert client.get("/api/settings").json()["court"]["corners"] == corners
