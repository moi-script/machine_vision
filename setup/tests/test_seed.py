from seed import seed
from app import db


def test_seed_inserts_demo_data():
    seed()
    assert db.players().count_documents({}) >= 8
    assert db.sessions().count_documents({}) >= 5
    p1 = db.players().find_one({"_id": "p1"})
    assert p1["name"] == "Marco Dela Cruz"
