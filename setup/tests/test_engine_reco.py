# tests/test_engine_reco.py
from app.engine import DrillEngine


def test_athlete_id_prefers_recognized_track():
    e = DrillEngine()
    e._attributed_player = "p_single"
    e._track_identity = {7: "p_recognized"}
    assert e._athlete_id(7) == "p_recognized"      # recognition wins
    assert e._athlete_id(9) == "p_single"          # falls back to single-assigned
    e._attributed_player = None
    assert e._athlete_id(9) == "9"                 # finally the track id


def test_recognize_sets_identity_and_broadcasts(monkeypatch):
    import app.engine as eng
    e = DrillEngine()
    e._enrolled = {"p1": [0.1] * 128}
    e._track_identity = {}
    e._last_reco = {}
    # stub face detection to always "recognize" p1
    monkeypatch.setattr(eng.face, "detect_and_embed", lambda crop: [0.1] * 128)
    monkeypatch.setattr(eng.face, "best_match", lambda emb, enrolled, **k: ("p1", 0.9))
    sent = []
    monkeypatch.setattr(eng.hub, "broadcast", lambda m: sent.append(m))
    import numpy as np
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    e._recognize(3, [10, 10, 120, 180], frame)
    assert e._track_identity.get(3) == "p1"
    assert any(m.get("type") == "identity" and m.get("playerId") == "p1" for m in sent)
