# ============================================================
# test_shuttle_worker.py — unit tests for the background shuttle runner
#
# No camera or network needed: detect_fn is a fake we control.
#
# Run:
#   python -m pytest tests/test_shuttle_worker.py -s
#   python tests/test_shuttle_worker.py            # no pytest
# ============================================================

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.shuttle_worker import ShuttleWorker  # noqa: E402


def _wait_until(predicate, timeout=2.0):
    """Poll predicate until true or timeout; returns the final bool."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_processes_submitted_frame():
    """A submitted frame is detected and its result becomes available."""
    worker = ShuttleWorker(detect_fn=lambda f: ("pos", f))
    worker.start()
    try:
        assert worker.get() is None            # nothing processed yet
        worker.submit(42)
        assert _wait_until(lambda: worker.get() == ("pos", 42))
    finally:
        worker.stop()


def test_drops_stale_frames_keeps_newest():
    """While the worker is busy, only the newest queued frame is processed next."""
    release = threading.Event()
    seen    = []

    def slow_detect(frame):
        seen.append(frame)
        release.wait(timeout=2.0)   # block on the FIRST frame until released
        return frame

    worker = ShuttleWorker(detect_fn=slow_detect)
    worker.start()
    try:
        worker.submit(1)                                  # worker grabs 1, blocks
        assert _wait_until(lambda: seen == [1])
        worker.submit(2)                                  # queued...
        worker.submit(3)                                  # ...replaces 2
        release.set()                                     # let it finish frame 1
        # Next frame processed must be 3 (the newest), never 2.
        assert _wait_until(lambda: worker.get() == 3)
        assert 2 not in seen
    finally:
        worker.stop()


def test_detect_exception_does_not_kill_worker():
    """A detect_fn that raises yields a None result but keeps the worker alive."""
    calls = {"n": 0}

    def flaky(frame):
        calls["n"] += 1
        if frame == "boom":
            raise RuntimeError("network blip")
        return frame

    worker = ShuttleWorker(detect_fn=flaky)
    worker.start()
    try:
        worker.submit("boom")
        assert _wait_until(lambda: calls["n"] >= 1)
        worker.submit("ok")                               # worker still running
        assert _wait_until(lambda: worker.get() == "ok")
    finally:
        worker.stop()


if __name__ == "__main__":
    test_processes_submitted_frame()
    test_drops_stale_frames_keeps_newest()
    test_detect_exception_does_not_kill_worker()
    print("OK: all shuttle_worker tests passed")
