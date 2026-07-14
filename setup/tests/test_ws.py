from fastapi.testclient import TestClient
from app.server import app
from app.events import hub


def test_ws_receives_broadcast():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        hub.broadcast({"type": "shot", "zone": "front_left"})
        msg = ws.receive_json()
        assert msg["type"] == "shot"
