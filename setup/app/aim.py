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


class Aimer:
    def __init__(self, port: str, baud: int = 115200, opener=None):
        self.port = port
        self.baud = baud
        self._opener = opener or _open_serial
        self._ser = None
        self._lock = threading.Lock()

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
        if self._ser is not None:
            return True
        try:
            ser = self._opener(self.port, self.baud)
        except Exception as exc:
            print(f"[AIM] servo port {self.port} unavailable: {exc}", flush=True)
            return False
        deadline = time.time() + READY_TIMEOUT_S
        while time.time() < deadline:
            line = ser.readline().decode(errors="ignore").strip()
            if line == "READY":
                break
            if not line:
                # Already booted (no reset on open): no READY coming. Carry on.
                break
        self._ser = ser
        return True

    def _command(self, cmd: str) -> str | None:
        with self._lock:
            if not self._ensure():
                return None
            try:
                self._ser.reset_input_buffer()
                self._ser.write((cmd + "\n").encode())
                return self._ser.readline().decode(errors="ignore").strip()
            except Exception as exc:
                print(f"[AIM] serial error: {exc}", flush=True)
                self._drop()
                return None

    def move(self, x: int, y: int) -> tuple[int, int] | None:
        reply = self._command(f"A {int(x)} {int(y)}")
        parts = (reply or "").split()
        if len(parts) == 3 and parts[0] == "OK":
            return int(parts[1]), int(parts[2])
        if reply is not None:
            print(f"[AIM] unexpected reply {reply!r}", flush=True)
            with self._lock:
                self._drop()
        return None

    def center(self) -> tuple[int, int] | None:
        reply = self._command("C")
        parts = (reply or "").split()
        return (int(parts[1]), int(parts[2])) if len(parts) == 3 and parts[0] == "OK" else None

    def ping(self) -> bool:
        return self._command("P") == "PONG"


_aimer: Aimer | None = None
_rng = random.Random()


def get_aimer() -> Aimer:
    global _aimer
    if _aimer is None:
        _aimer = Aimer(_settings.SERVO_PORT, _settings.SERVO_BAUD)
    return _aimer


def aim_zone(zone: str, cfg) -> tuple[int, int] | None:
    x, y = pick_angles(zone, cfg, _rng)
    return get_aimer().move(x, y)
