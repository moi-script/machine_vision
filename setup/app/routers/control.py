"""Engine control + video stream endpoints."""
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.engine import get_engine
from app.streamer import buffer

router = APIRouter(prefix="/api/control", tags=["control"])


class StartBody(BaseModel):
    sessionId: str | None = None
    difficulty: str = "medium"
    shots: int = 0


class DifficultyBody(BaseModel):
    difficulty: str


@router.post("/start")
def start(body: StartBody):
    eng = get_engine()
    try:
        eng.start(body.sessionId, body.difficulty, body.shots)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return {"ok": True, "status": eng.status()}


@router.post("/pause")
def pause():
    eng = get_engine(); eng.pause()
    return {"ok": True, "status": eng.status()}


@router.post("/resume")
def resume():
    eng = get_engine(); eng.resume()
    return {"ok": True, "status": eng.status()}


@router.post("/stop")
def stop():
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
