"""MongoDB connection + thin repository helpers for AeroSense."""
import os
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection


def _load_env_file() -> None:
    """Load KEY=VALUE lines from setup/.env into os.environ (without overriding
    vars already set in the shell). python-dotenv isn't a dependency, so this is
    hand-rolled. This pins MONGO_DB to the committed value so the database can't
    silently drift between runs — the cause of "registered players vanish after
    a restart" (data written to one db name, read back from another)."""
    path = os.path.join(os.path.dirname(__file__), "..", ".env")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


_load_env_file()

_MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017/")
_MONGO_DB = os.environ.get("MONGO_DB", "aerosense")
# Surface the resolved target on every start so which database is in use is
# never a mystery — cross-check this against your Mongo viewer.
print(f"[DB] MongoDB {_MONGO_URL} -> database '{_MONGO_DB}'", flush=True)

_client: MongoClient | None = None


def db_name() -> str:
    return _MONGO_DB


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
    """Return all docs with Mongo `_id` surfaced as a string `id`.

    Docs inserted without an explicit `_id` (e.g. attendance) get a Mongo
    ObjectId, which is not JSON-serializable; stringify it so the API can
    return it. String ids (players/sessions use `p1`/`s1`) pass through
    unchanged.
    """
    out = []
    for doc in col.find():
        doc = dict(doc)
        if "id" not in doc:
            doc["id"] = str(doc.pop("_id"))
        else:
            doc.pop("_id", None)
        out.append(doc)
    return out
