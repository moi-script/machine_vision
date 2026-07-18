"""Session CRUD + status transitions, and attendance read."""
import os
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException

from app import db
from app.models import SessionIn, SessionPatch

router = APIRouter(tags=["sessions"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _out(doc: dict) -> dict:
    doc = dict(doc)
    doc["id"] = doc.pop("_id")
    return doc


@router.get("/api/sessions")
def list_sessions():
    return db.list_docs(db.sessions())


@router.post("/api/sessions")
def create_session(body: SessionIn):
    sid = body.id or ("s" + os.urandom(4).hex())
    doc = {
        "_id": sid,
        "title": body.title,
        "coachId": body.coachId,
        "difficulty": body.difficulty,
        "status": body.status,
        "scheduledAt": body.scheduledAt,
        "startedAt": None,
        "endedAt": None,
        "assignedPlayerIds": body.assignedPlayerIds,
        "liveData": [],
        "notes": body.notes,
    }
    db.sessions().insert_one(doc)
    return _out(doc)


@router.patch("/api/sessions/{sid}")
def patch_session(sid: str, body: SessionPatch):
    doc = db.sessions().find_one({"_id": sid})
    if not doc:
        raise HTTPException(404, "session not found")

    changes = {k: v for k, v in body.model_dump().items()
               if v is not None and k != "status"}

    if body.status is not None:
        changes["status"] = body.status
        if body.status == "live":
            changes["startedAt"] = _now()
            # only one live session at a time
            db.sessions().update_many(
                {"status": "live", "_id": {"$ne": sid}},
                {"$set": {"status": "scheduled", "startedAt": None,
                          "endedAt": None, "liveData": []}},
            )
        elif body.status == "completed":
            changes["endedAt"] = _now()
        elif body.status == "scheduled":
            changes["startedAt"] = None
            changes["endedAt"] = None
            changes["liveData"] = []

    db.sessions().update_one({"_id": sid}, {"$set": changes})
    return _out(db.sessions().find_one({"_id": sid}))


@router.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    doc = db.sessions().find_one({"_id": sid})
    if not doc:
        raise HTTPException(404, "session not found")
    if doc.get("status") == "live":
        raise HTTPException(409, "cannot delete a live match — complete it first")
    db.sessions().delete_one({"_id": sid})
    db.attendance().delete_many({"sessionId": sid})
    return {"ok": True}


@router.get("/api/attendance")
def list_attendance():
    return db.list_docs(db.attendance())
