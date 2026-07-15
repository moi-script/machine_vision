"""Engine control + video stream endpoints."""
import threading

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.engine import get_engine
from app.streamer import buffer

router = APIRouter(prefix="/api/control", tags=["control"])

# FastAPI runs these sync endpoints in a threadpool, so the shared engine
# singleton can be mutated concurrently — e.g. the identity-acquisition overlay
# starts the engine while the "go live" path also starts it (and React
# StrictMode fires effects twice in dev). Serialize every state transition so a
# check-then-act (state → stop → start) can't interleave and leave a
# half-initialized worker thread behind.
_ctl_lock = threading.Lock()


class StartBody(BaseModel):
    sessionId: str | None = None
    difficulty: str = "medium"
    shots: int = 0
    # False starts recognition-only (identity acquisition); the feeder stays
    # idle until an /arm (or an armed start of the same session) promotes it.
    armed: bool = True


class DifficultyBody(BaseModel):
    difficulty: str


@router.post("/start")
def start(body: StartBody):
    with _ctl_lock:
        eng = get_engine()
        # Same session already running? It was likely started unarmed for
        # identity acquisition — arm it in place instead of tearing down the
        # camera and losing the recognized-track cache built up during scanning.
        if eng.state != "idle" and eng.session_id == body.sessionId:
            if body.armed:
                eng.arm()
            return {"ok": True, "status": eng.status()}
        # A different session (or paused) — the newest start wins; stop first so
        # starting takes over cleanly instead of colliding with a 409.
        if eng.state != "idle":
            eng.stop()
        try:
            eng.start(body.sessionId, body.difficulty, body.shots, armed=body.armed)
        except RuntimeError as exc:
            # A real failure to start (camera unavailable / not calibrated).
            raise HTTPException(409, str(exc))
        return {"ok": True, "status": eng.status()}


@router.post("/arm")
def arm():
    with _ctl_lock:
        eng = get_engine(); eng.arm()
        return {"ok": True, "status": eng.status()}


@router.post("/pause")
def pause():
    with _ctl_lock:
        eng = get_engine(); eng.pause()
        return {"ok": True, "status": eng.status()}


@router.post("/resume")
def resume():
    with _ctl_lock:
        eng = get_engine(); eng.resume()
        return {"ok": True, "status": eng.status()}


@router.post("/stop")
def stop():
    with _ctl_lock:
        eng = get_engine(); eng.stop()
        return {"ok": True, "status": eng.status()}


@router.post("/difficulty")
def difficulty(body: DifficultyBody):
    eng = get_engine(); eng.set_difficulty(body.difficulty)
    return {"ok": True, "status": eng.status()}


media_router = APIRouter(tags=["media"])


@media_router.get("/api/stream")
def stream():
    return StreamingResponse(
        buffer.frames(),
        media_type="multipart/x-mixed-replace; boundary=frame")


@media_router.get("/api/frame")
def frame():
    try:
        jpeg = get_engine().capture_frame()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    return Response(content=jpeg, media_type="image/jpeg")
