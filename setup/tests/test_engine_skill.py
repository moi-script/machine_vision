from app.engine import DrillEngine
from app import db


def test_flush_writes_skillprofile_to_mongo():
    # seed a player
    pid = "pskilltest1"
    db.players().update_one({"_id": pid},
        {"$set": {"_id": pid, "name": "SKILLTEST",
                  "stats": {"totalShots": 0, "totalScores": 0}}},
        upsert=True)
    try:
        eng = DrillEngine()
        acc = eng._skill_for(pid)
        for _ in range(60):
            acc.on_shot()
        for _ in range(45):
            acc.on_score()
        eng._flush_skill_profiles()
        doc = db.players().find_one({"_id": pid})
        assert "skillProfile" in doc
        assert doc["skillProfile"]["computed"]["tier"] in {
            "Beginner", "Novice", "Intermediate", "Advanced", "Expert"}
        assert doc["skillProfile"]["cumulative"]["shots"] == 60
        assert len(doc["skillHistory"]) == 1
    finally:
        db.players().delete_one({"_id": pid})


def test_skill_for_reuses_same_accumulator():
    eng = DrillEngine()
    a = eng._skill_for("pX")
    b = eng._skill_for("pX")
    assert a is b
