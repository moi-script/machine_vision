"""WebSocket broadcast hub. Engine (sync thread) -> browser clients."""
import asyncio
import queue
from typing import Any


class EventHub:
    def __init__(self):
        self._clients: set[Any] = set()
        self._q: "queue.Queue[dict]" = queue.Queue()
        self._last_status: dict | None = None

    # --- called from the engine's background (sync) thread ---
    def broadcast(self, message: dict) -> None:
        if message.get("type") == "engine_status":
            self._last_status = message
        self._q.put(message)

    def queue_size(self) -> int:
        return self._q.qsize()

    def snapshot_last(self) -> dict | None:
        return self._last_status

    # --- async side, on the FastAPI event loop ---
    async def connect(self, ws) -> None:
        await ws.accept()
        self._clients.add(ws)
        if self._last_status is not None:
            await ws.send_json(self._last_status)

    def disconnect(self, ws) -> None:
        self._clients.discard(ws)

    async def pump(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            try:
                msg = await loop.run_in_executor(None, self._q.get, True, 0.5)
            except queue.Empty:
                continue
            dead = []
            for ws in list(self._clients):
                try:
                    await ws.send_json(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.disconnect(ws)


hub = EventHub()
