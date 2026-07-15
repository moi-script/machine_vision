from seed import seed
from app import db


def test_seed_inserts_demo_data():
    seed()
    assert db.players().count_documents({}) >= 8
    assert db.sessions().count_documents({}) >= 5
    p1 = db.players().find_one({"_id": "p1"})
    assert p1["name"] == "Marco Dela Cruz"


def test_clear_db_empties_collections():
    from seed import clear_db
    from app import db
    db.players().insert_one({"_id": "tmp_clear", "name": "x"})
    clear_db()
    assert db.players().count_documents({}) == 0
    assert db.sessions().count_documents({}) == 0
    assert db.attendance().count_documents({}) == 0
