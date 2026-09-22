"""Feeder aim: zone -> servo angles, sent to the Arduino running
firmware/servo_aim/servo_aim.ino.

The engine already picks a random zone per shot; this turns that zone into
that zone's calibrated (x, y) plus a little uniform jitter, so the player can
predict neither the zone nor the exact spot. A missing or wedged Arduino must
never stop a drill: every failure here logs and returns None.
"""
from __future__ import annotations

import random
import threading
import time

from config import settings as _settings

READY_TIMEOUT_S = 3.0    # a Nano resets when the port opens; boot takes ~1.5 s
REPLY_TIMEOUT_S = 0.5
RETRY_BACKOFF_S = 10.0   # cool-down after a failed connect (wrong device /
                         # wedged firmware) before _ensure() opens the port
                         # again — without this, every call pays the full
                         # ~3.5s READY+PONG probe, and _fire_feeder spawns an
                         # aim thread per shot, so blocked threads pile up.


def pick_angles(zone: str, cfg, rng: random.Random) -> tuple[int, int]:
    base = cfg.zoneAngles.get(zone)
    bx, by = (base.x, base.y) if base is not None else (90, 90)
    j = float(cfg.jitterDeg)
    x = round(bx + rng.uniform(-j, j))
    y = round(by + rng.uniform(-j, j))
    clamp = lambda v: max(cfg.angleMin, min(cfg.angleMax, v))  # noqa: E731
    return clamp(x), clamp(y)


def _open_serial(port: str, baud: int):
    import serial
    return serial.Serial(port, baud, timeout=REPLY_TIMEOUT_S)


def _parse_ok(reply: str | None) -> tuple[int, int] | None:
    parts = (reply or "").split()
    if len(parts) == 3 and parts[0] == "OK":
        return int(parts[1]), int(parts[2])
    return None


class Aimer:
    def __init__(self, port: str, baud: int = 115200, opener=None):
        self.port = port
        self.baud = baud
        self._opener = opener or _open_serial
        self._ser = None
        self._lock = threading.Lock()
        self._failed_at: float | None = None

    @property
    def connected(self) -> bool:
        return self._ser is not None

    def _drop(self):
        try:
            if self._ser is not None:
                self._ser.close()
        except Exception:
            pass
        self._ser = None

    def _ensure(self) -> bool:
        """Open the port and confirm the board is actually there.

        Called with self._lock already held. A Nano/Uno resets when the port
        opens and stays silent for ~1-2s before printing READY, so an empty
        readline (a read timeout) must not be read as "no READY coming" — it
        just means the board hasn't booted yet. Keep reading until READY
        shows up or the full READY_TIMEOUT_S deadline passes. If the deadline
        passes with no READY, the board may simply already be running (no
        reset on this open): confirm with a P/PONG ping before trusting the
        port, and give up (close it) if that fails too.
        """
        if self._ser is not None:
            return True
        if (self._failed_at is not None
                and time.time() - self._failed_at < RETRY_BACKOFF_S):
            # Still cooling down from a recent failed connect — don't pay
            # the ~3.5s READY+PONG probe again on every single call.
            return False
        try:
            ser = self._opener(self.port, self.baud)
        except Exception as exc:
            print(f"[AIM] servo port {self.port} unavailable: {exc}", flush=True)
            self._failed_at = time.time()
            return False
        deadline = time.time() + READY_TIMEOUT_S
        got_ready = False
        while time.time() < deadline:
            line = ser.readline().decode(errors="ignore").strip()
            if line == "READY":
                got_ready = True
                break
            # empty (read timeout) or other noise while the board boots —
            # keep waiting out the deadline instead of bailing on line 1.
        if not got_ready:
            try:
                ser.reset_input_buffer()
                ser.write(b"P\n")
                reply = ser.readline().decode(errors="ignore").strip()
            except Exception as exc:
                print(f"[AIM] servo port {self.port} ping failed: {exc}",
                      flush=True)
                try:
                    ser.close()
                except Exception:
                    pass
                self._failed_at = time.time()
                return False
            if reply != "PONG":
                print(f"[AIM] servo port {self.port} gave no READY/PONG "
                      f"(got {reply!r}) — treating as offline", flush=True)
                try:
                    ser.close()
                except Exception:
                    pass
                self._failed_at = time.time()
                return False
        self._ser = ser
        self._failed_at = None
        return True

    def _exchange(self, cmd: str, expect) -> str | None:
        """Ensure -> write -> read -> validate, all under one lock hold, so a
        concurrent caller's successful exchange can never be torn down mid-
        way by another thread's _drop(). `expect(reply)` decides validity; an
        invalid reply (including "") drops the port and returns None."""
        with self._lock:
            if not self._ensure():
                return None
            try:
                self._ser.reset_input_buffer()
                self._ser.write((cmd + "\n").encode())
                reply = self._ser.readline().decode(errors="ignore").strip()
            except Exception as exc:
                print(f"[AIM] serial error: {exc}", flush=True)
                self._drop()
                return None
            if not expect(reply):
                if reply:
                    print(f"[AIM] unexpected reply {reply!r}", flush=True)
                self._drop()
                return None
            return reply

    def move(self, x: int, y: int) -> tuple[int, int] | None:
        reply = self._exchange(f"A {int(x)} {int(y)}",
                                lambda r: _parse_ok(r) is not None)
        return _parse_ok(reply)

    def center(self) -> tuple[int, int] | None:
        reply = self._exchange("C", lambda r: _parse_ok(r) is not None)
        return _parse_ok(reply)

    def ping(self) -> bool:
        return self._exchange("P", lambda r: r == "PONG") == "PONG"


_aimer: Aimer | None = None
_aimer_lock = threading.Lock()
_rng = random.Random()


def get_aimer() -> Aimer:
    global _aimer
    if _aimer is None:
        with _aimer_lock:
            if _aimer is None:
                _aimer = Aimer(_settings.SERVO_PORT, _settings.SERVO_BAUD)
    return _aimer


def aim_zone(zone: str, cfg) -> tuple[int, int] | None:
    x, y = pick_angles(zone, cfg, _rng)
    return get_aimer().move(x, y)
