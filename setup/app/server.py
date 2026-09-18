"""FastAPI application entrypoint for AeroSense."""
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app import db
from app.events import hub
app = FastAPI(title="AeroSense Backend")
app.add_middleware(
    CORSMiddleware,
    # Local-only app: allow any localhost port so a Vite dev server that
    # auto-increments off 5173 (when the port is taken) still reaches the API.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routers import (
    players, sessions, settings as settings_router, control, cameras,
    face_cam, aim as aim_router,
)

app.include_router(players.router)
app.include_router(sessions.router)
app.include_router(settings_router.router)
app.include_router(control.router)
app.include_router(control.media_router)
app.include_router(cameras.router)
app.include_router(face_cam.router)
app.include_router(aim_router.router)


@app.get("/api/health")
def health():
    from app.engine import get_engine
    from app import face
    st = get_engine().status()
    return {"mongo": db.ping(), "db": db.db_name(), "engine": st["state"],
            "camera": st["camera"], "face": face.models_available()}


@app.on_event("startup")
async def _start_pump():
    asyncio.create_task(hub.pump())


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await hub.connect(ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive; ignore client messages
    except WebSocketDisconnect:
        hub.disconnect(ws)


# Serve the built UI last: a mount at / claims every unclaimed path, so this
# must come after every route registration above - routers *and* the
# decorated routes declared directly in this file (health, startup, ws).
# No-op when the bundle is absent (dev flow).
from app.ui_static import mount_ui  # noqa: E402

_ui_mounted = mount_ui(app)
print(f"[UI] bundle {'mounted at /' if _ui_mounted else 'absent - API only'}", flush=True)
