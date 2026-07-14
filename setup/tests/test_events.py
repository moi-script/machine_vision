import asyncio
from app.events import EventHub


def test_broadcast_from_sync_thread_enqueues():
    hub = EventHub()
    hub.broadcast({"type": "shot", "zone": "front_left"})
    assert hub.queue_size() == 1


def test_snapshot_tracks_engine_status():
    hub = EventHub()
    hub.broadcast({"type": "engine_status", "state": "running"})
    assert hub.snapshot_last()["state"] == "running"
