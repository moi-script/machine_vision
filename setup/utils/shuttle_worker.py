# ============================================================
# shuttle_worker.py — Run shuttle detection off the main loop
#
# Shuttle detection (serverless HTTP or local weights) can take ~1s per frame,
# which would drag the whole render/drill loop down to <1 FPS if called inline.
#
# ShuttleWorker runs that detection on a background thread. The main loop hands
# it the newest frame (submit) and reads the most recent result (get) without
# ever blocking. Only the latest submitted frame is kept, so the worker always
# processes the freshest frame and silently drops stale ones. The trade-off is
# that get() may return a position that is a detection-latency behind reality.
# ============================================================

import threading
import time


class ShuttleWorker:
    """Background runner for a single-frame shuttle detection function.

    Args:
        detect_fn: callable(frame) -> shuttle result (e.g. (x, y) or None).
                   Must be self-contained; exceptions are swallowed so a bad
                   frame or network blip never kills the drill.
    """

    def __init__(self, detect_fn):
        self._detect_fn     = detect_fn
        self._lock          = threading.Lock()
        self._latest_frame  = None   # newest frame awaiting processing
        self._latest_result = None   # most recent detection result
        self._running       = False
        self._thread        = None

    def start(self):
        """Spin up the worker thread (idempotent)."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, name="ShuttleWorker", daemon=True)
        self._thread.start()

    def submit(self, frame):
        """Offer a frame for detection. Replaces any not-yet-processed frame."""
        with self._lock:
            self._latest_frame = frame

    def get(self):
        """Return the most recent detection result (may be slightly stale)."""
        with self._lock:
            return self._latest_result

    def stop(self):
        """Signal the thread to exit and wait briefly for it to finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self):
        while self._running:
            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None

            if frame is None:
                # Nothing new to process yet — yield briefly and re-check.
                time.sleep(0.005)
                continue

            try:
                result = self._detect_fn(frame)
            except Exception:  # noqa: BLE001 - never let the worker die
                result = None

            with self._lock:
                self._latest_result = result
