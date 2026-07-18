"""Shared MJPEG frame buffer for the live video endpoint."""
import threading
import time

BOUNDARY = "frame"


class FrameBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None

    def publish(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg

    def latest(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def clear(self) -> None:
        """Drop the retained frame when the engine stops.

        Without this the last published JPEG is served indefinitely, so a dead
        engine still looks like a live feed — a frozen frame that hides the fact
        that nothing is running.
        """
        with self._lock:
            self._jpeg = None

    def frames(self):
        """Yield multipart JPEG chunks for an MJPEG HTTP response."""
        header = (b"--" + BOUNDARY.encode() +
                  b"\r\nContent-Type: image/jpeg\r\nContent-Length: ")
        while True:
            jpeg = self.latest()
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield (header + str(len(jpeg)).encode() + b"\r\n\r\n" +
                   jpeg + b"\r\n")
            time.sleep(0.1)  # ~10 fps cap


buffer = FrameBuffer()
