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


def test_aimer_waits_out_reset_before_ready():
    # A Nano/Uno resets on port-open and stays silent for a while before
    # printing READY — empty readlines (read timeouts) must not be mistaken
    # for "no READY coming".
    fake = FakeSerial(["", "", "", "READY", "OK 70 100"])
    opens = []

    def opener(port, baud):
        opens.append((port, baud))
        return fake

    a = aim.Aimer("X", opener=opener)
    assert a.move(70, 100) == (70, 100)
    assert len(opens) == 1


def test_aimer_already_booted_pings_before_trusting_port(monkeypatch):
    # No READY ever comes (board was already running, no reset on this
    # open) — once the deadline passes, a P/PONG ping confirms the board
    # before the first real command is trusted.
    times = iter([0.0, 0.1, 0.2, 0.3, 10.0])
    monkeypatch.setattr(aim.time, "time", lambda: next(times))
    monkeypatch.setattr(aim, "READY_TIMEOUT_S", 1.0)
    fake = FakeSerial(["", "", "", "PONG", "OK 80 80"])
    a = aim.Aimer("X", opener=lambda port, baud: fake)
    assert a.move(80, 80) == (80, 80)


def test_aimer_no_ready_no_pong_goes_offline(monkeypatch):
    monkeypatch.setattr(aim, "READY_TIMEOUT_S", 0.05)
    fake = FakeSerial([])  # readline() always returns "" (no data at all)
    a = aim.Aimer("X", opener=lambda port, baud: fake)
    assert a.move(80, 80) is None
    assert a.connected is False
    assert fake.is_open is False


def test_aimer_backs_off_after_failed_connect(monkeypatch):
    opens = []

    def opener(port, baud):
        opens.append(1)
        raise OSError("no such port")

    times = iter([0.0, 5.0, 15.0, 15.5])
    monkeypatch.setattr(aim.time, "time", lambda: next(times))
    a = aim.Aimer("X", opener=opener)

    assert a.move(1, 1) is None
    assert len(opens) == 1          # first attempt: opener called, fails

    assert a.move(1, 1) is None
    assert len(opens) == 1          # still within RETRY_BACKOFF_S: no retry

    assert a.move(1, 1) is None
    assert len(opens) == 2          # backoff window elapsed: retried


def test_fire_feeder_skips_aim_when_one_already_in_flight(monkeypatch):
    from app import engine
    sent = []
    monkeypatch.setattr(engine.hub, "broadcast", sent.append)
    calls = []
    monkeypatch.setattr(aim, "aim_zone",
                         lambda zone, cfg: calls.append(1) or (61, 99))

    class NowThread:
        def __init__(self, target, **k): self.t = target
        def start(self): self.t()
    monkeypatch.setattr(engine.threading, "Thread", NowThread)

    e = engine.DrillEngine()
    e._aim_lock.acquire()  # simulate an aim already in flight
    e._fire_feeder("back_left")
    assert calls == []
    assert sent[-1]["type"] == "feeder"
    assert (sent[-1]["zone"], sent[-1]["x"], sent[-1]["y"]) == \
        ("back_left", None, None)


def test_get_aimer_is_a_singleton_under_concurrency(monkeypatch):
    import threading as _threading

    monkeypatch.setattr(aim, "_aimer", None)
    results = []
    barrier = _threading.Barrier(8)

    def worker():
        barrier.wait()
        results.append(aim.get_aimer())

    threads = [_threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 8
    assert len({id(r) for r in results}) == 1
