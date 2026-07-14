"""MongoDB connection + thin repository helpers for AeroSense."""
import os
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

_MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017/")
_MONGO_DB = os.environ.get("MONGO_DB", "aerosense")

_client: MongoClient | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(_MONGO_URL, serverSelectionTimeoutMS=2000)
    return _client


def get_db() -> Database:
    return get_client()[_MONGO_DB]


def ping() -> bool:
    try:
        get_client().admin.command("ping")
        return True
    except Exception:
        return False


def players() -> Collection:
    return get_db()["players"]


def sessions() -> Collection:
    return get_db()["sessions"]


def attendance() -> Collection:
    return get_db()["attendance"]


def settings_col() -> Collection:
    return get_db()["settings"]


def list_docs(col: Collection) -> list[dict]:
    """Return all docs with Mongo `_id` surfaced as `id`."""
    out = []
    for doc in col.find():
        doc = dict(doc)
        if "id" not in doc:
            doc["id"] = doc.pop("_id")
        else:
            doc.pop("_id", None)
        out.append(doc)
    return out
