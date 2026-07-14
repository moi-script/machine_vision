"""Player CRUD endpoints."""
import os
from datetime import date
from fastapi import APIRouter, HTTPException

from app import db
from app.models import PlayerIn, PlayerPatch

router = APIRouter(prefix="/api/players", tags=["players"])

_AVATAR_COLORS = [
    "#ff3366", "#ff5e00", "#00e676", "#00f0ff",
    "#b026ff", "#00b8ff", "#ff9100", "#ff007b",
]


def _empty_zones():
    zones = ["front-left", "front-center", "front-right",
             "back-left", "back-center", "back-right"]
    return {z: {"shots": 0, "scores": 0} for z in zones}


def _new_id() -> str:
    return "p" + os.urandom(4).hex()


@router.get("")
def list_players():
    return db.list_docs(db.players())


@router.post("")
def create_player(body: PlayerIn):
    count = db.players().count_documents({})
    doc = {
        "_id": _new_id(),
        "name": body.name,
        "age": body.age,
        "role": body.role,
        "jerseyNumber": body.jerseyNumber,
        "imageUrl": body.imageUrl,
        "faceEnrolled": body.faceEnrolled,
        "isActive": body.isActive,
        "avatarColor": _AVATAR_COLORS[count % len(_AVATAR_COLORS)],
        "joinedAt": date.today().isoformat(),
        "trainingDays": [],
        "stats": {"totalShots": 0, "totalScores": 0, "zones": _empty_zones()},
    }
    db.players().insert_one(doc)
    doc["id"] = doc.pop("_id")
    return doc


@router.get("/{pid}")
def get_player(pid: str):
    doc = db.players().find_one({"_id": pid})
    if not doc:
        raise HTTPException(404, "player not found")
    doc["id"] = doc.pop("_id")
    return doc


@router.patch("/{pid}")
def patch_player(pid: str, body: PlayerPatch):
    changes = {k: v for k, v in body.model_dump().items() if v is not None}
    res = db.players().update_one({"_id": pid}, {"$set": changes})
    if res.matched_count == 0:
        raise HTTPException(404, "player not found")
    return get_player(pid)


@router.delete("/{pid}")
def deactivate_player(pid: str):
    res = db.players().update_one({"_id": pid}, {"$set": {"isActive": False}})
    if res.matched_count == 0:
        raise HTTPException(404, "player not found")
    return {"ok": True}
