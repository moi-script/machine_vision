import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import aim  # noqa: E402
from app.models import AimSettings, Settings  # noqa: E402
from config import settings as cfg  # noqa: E402


def test_defaults_cover_every_engine_zone():
    s = AimSettings()
    assert set(s.zoneAngles) == set(cfg.PLAYER_ZONES)
    assert all(a.x == 90 and a.y == 90 for a in s.zoneAngles.values())
    assert (s.jitterDeg, s.angleMin, s.angleMax) == (4, 30, 150)


def test_settings_has_aim_and_old_docs_still_load():
    assert Settings().aim.jitterDeg == 4
    # a pi-kiosk-v1 settings doc has no "aim" key
    assert Settings(**{"camera": {}}).aim.angleMax == 150


def test_pick_angles_stays_within_jitter():
    s = AimSettings()
    s.zoneAngles["back_left"].x = 60
    s.zoneAngles["back_left"].y = 110
    rng = random.Random(1)
    for _ in range(200):
        x, y = aim.pick_angles("back_left", s, rng)
        assert 56 <= x <= 64 and 106 <= y <= 114


def test_pick_angles_varies():
    s = AimSettings()
    rng = random.Random(2)
    assert len({aim.pick_angles("front_left", s, rng) for _ in range(30)}) > 5


def test_pick_angles_clamps():
    s = AimSettings(jitterDeg=0)
    s.zoneAngles["front_right"].x = 10
    s.zoneAngles["front_right"].y = 175
    assert aim.pick_angles("front_right", s, random.Random(0)) == (30, 150)


def test_unknown_zone_aims_center():
    s = AimSettings(jitterDeg=0)
    assert aim.pick_angles("nowhere", s, random.Random(0)) == (90, 90)


class FakeSerial:
    def __init__(self, replies):
        self.replies = list(replies)
        self.written = []
        self.is_open = True

    def write(self, b):
        self.written.append(b.decode())

    def readline(self):
        return (self.replies.pop(0) + "\n").encode() if self.replies else b""

    def reset_input_buffer(self):
        pass

    def close(self):
        self.is_open = False


def test_aimer_waits_for_ready_then_moves():
    fake = FakeSerial(["READY", "OK 70 100"])
    a = aim.Aimer("X", opener=lambda port, baud: fake)
    assert a.move(70, 100) == (70, 100)
    assert fake.written == ["A 70 100\n"]


def test_aimer_offline_returns_none_without_raising():
    def opener(port, baud):
        raise OSError("no such port")
    a = aim.Aimer("X", opener=opener)
    assert a.move(90, 90) is None
    assert a.connected is False


def test_aimer_drops_port_on_bad_reply_and_reconnects():
    fakes = [FakeSerial(["READY", "garbage"]), FakeSerial(["READY", "OK 80 80"])]
    a = aim.Aimer("X", opener=lambda port, baud: fakes.pop(0))
    assert a.move(80, 80) is None
    assert a.move(80, 80) == (80, 80)


def test_fire_feeder_broadcasts_angles(monkeypatch):
    from app import engine
    sent = []
    monkeypatch.setattr(engine.hub, "broadcast", sent.append)
    monkeypatch.setattr(aim, "aim_zone", lambda zone, cfg: (61, 99))

    class NowThread:
        def __init__(self, target, **k): self.t = target
        def start(self): self.t()
    monkeypatch.setattr(engine.threading, "Thread", NowThread)
    engine.DrillEngine()._fire_feeder("back_left")
    assert sent[-1]["type"] == "feeder"
    assert (sent[-1]["zone"], sent[-1]["x"], sent[-1]["y"]) == ("back_left", 61, 99)
