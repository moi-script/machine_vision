"""Runtime settings + court calibration endpoints."""
from fastapi import APIRouter
from pydantic import BaseModel

from app import db
from app.models import Settings

router = APIRouter(tags=["settings"])

_DOC_ID = "runtime"


def load_settings() -> Settings:
    doc = db.settings_col().find_one({"_id": _DOC_ID})
    if not doc:
        s = Settings.defaults()
        save_settings(s)
        return s
    doc.pop("_id", None)
    return Settings(**doc)


def save_settings(s: Settings) -> None:
    db.settings_col().update_one(
        {"_id": _DOC_ID}, {"$set": s.model_dump()}, upsert=True)


def _notify_engine() -> None:
    try:
        from app.engine import get_engine
        get_engine().reload_settings()
    except Exception:
        pass  # engine not started / not present yet


@router.get("/api/settings")
def get_settings():
    return load_settings().model_dump()


@router.put("/api/settings")
def put_settings(body: Settings):
    save_settings(body)
    _notify_engine()
    return body.model_dump()


class CalibrationBody(BaseModel):
    corners: list[list[float]]


@router.post("/api/calibration")
def calibrate(body: CalibrationBody):
    s = load_settings()
    s.court.corners = body.corners
    save_settings(s)
    _notify_engine()
    return s.model_dump()
