import queue
from fastapi.testclient import TestClient
from app.server import app
from app.events import hub


def test_ws_receives_broadcast():
    hub.reset()                                  # clear state left by earlier tests
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            hub.broadcast({"type": "shot", "zone": "front_left"})
            msg = None
            for _ in range(10):                  # skip any stale non-shot frame
                msg = ws.receive_json()
                if msg["type"] == "shot":
                    break
            assert msg is not None and msg["type"] == "shot"
