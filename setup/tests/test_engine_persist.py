from app.engine import DrillEngine
from app import db


def test_zone_key_conversion():
    e = DrillEngine()
    assert e._zone_to_frontend("front_left") == "front-left"
    assert e._zone_to_frontend("back_center") == "back-center"


def test_persist_score_increments_session_and_player():
    db.sessions().delete_many({"_id": "s_ptest"})
    db.players().delete_many({"_id": "p_ptest"})
    db.sessions().insert_one({"_id": "s_ptest", "liveData": []})
    db.players().insert_one({"_id": "p_ptest",
        "stats": {"totalShots": 0, "totalScores": 0,
                  "zones": {"front-left": {"shots": 0, "scores": 0}}}})

    e = DrillEngine()
    e._session_id = "s_ptest"
    e._persist_shot("p_ptest", "front_left")
    e._persist_score("p_ptest", "front_left")

    p = db.players().find_one({"_id": "p_ptest"})
    assert p["stats"]["totalShots"] == 1
    assert p["stats"]["totalScores"] == 1
    assert p["stats"]["zones"]["front-left"]["scores"] == 1

    s = db.sessions().find_one({"_id": "s_ptest"})
    entry = next(d for d in s["liveData"] if d["playerId"] == "p_ptest")
    assert entry["scores"] == 1 and entry["shots"] == 1

    db.sessions().delete_many({"_id": "s_ptest"})
    db.players().delete_many({"_id": "p_ptest"})


def test_single_assigned_player_is_attributed():
    db.sessions().delete_many({"_id": "s_attr"})
    db.sessions().insert_one({"_id": "s_attr", "liveData": [],
                              "assignedPlayerIds": ["p_attr"]})

    e = DrillEngine()
    e._session_id = "s_attr"
    e._load_attribution()

    assert e._attributed_player == "p_attr"
    # Any YOLO track id rolls up to the enrolled athlete.
    assert e._athlete_id(1) == "p_attr"
    assert e._athlete_id(7) == "p_attr"

    db.sessions().delete_many({"_id": "s_attr"})


def test_multiple_assigned_players_disable_attribution():
    db.sessions().delete_many({"_id": "s_multi"})
    db.sessions().insert_one({"_id": "s_multi", "liveData": [],
                              "assignedPlayerIds": ["p_a", "p_b"]})

    e = DrillEngine()
    e._session_id = "s_multi"
    e._load_attribution()

    assert e._attributed_player is None
    # With attribution off, scoring stays keyed by the raw track id.
    assert e._athlete_id(3) == "3"

    db.sessions().delete_many({"_id": "s_multi"})
