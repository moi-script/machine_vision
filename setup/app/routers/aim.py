"""Live servo control for the Settings > Aim calibration panel."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import aim

router = APIRouter(prefix="/api/aim", tags=["aim"])


class MoveBody(BaseModel):
    x: int
    y: int


@router.get("/health")
def health():
    a = aim.get_aimer()
    return {"connected": a.ping(), "port": a.port}


@router.post("/move")
def move(body: MoveBody):
    got = aim.get_aimer().move(body.x, body.y)
    if got is None:
        raise HTTPException(503, "servo controller not responding")
    return {"x": got[0], "y": got[1]}


@router.post("/center")
def center():
    got = aim.get_aimer().center()
    if got is None:
        raise HTTPException(503, "servo controller not responding")
    return {"x": got[0], "y": got[1]}
