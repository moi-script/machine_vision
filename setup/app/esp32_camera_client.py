"""HTTP client for the Wi-Fi ESP32-CAM registration station.

The board serves three things: GET /capture (one JPEG), GET /health, and an
MJPEG preview at :ESP32_CAM_STREAM_PORT/stream. app/routers/face_cam.py uses
this when the face camera resolves to "wifi"; the USB AERO-FACE board never
touches it. Face detection stays on the Pi (app/face.py), so the board must
send clean frames - no boxes or landmarks drawn in.
"""
from __future__ import annotations

import httpx

from config import settings as _settings


class ESP32CaptureError(Exception):
    """Raised when the ESP32-CAM can't be reached or returns no image."""


def _ip() -> str | None:
    return getattr(_settings, "ESP32_CAM_IP", None) or None


def capture_snapshot(use_flash: bool = False) -> bytes:
    """One JPEG from the board. Flash stays off by default: the coach framed
    the shot on the unlit preview, and that is the exposure to enroll."""
    ip = _ip()
    if not ip:
        raise ESP32CaptureError("ESP32_CAM_IP is not set in config/settings.py")
    timeout_s = getattr(_settings, "ESP32_CAM_TIMEOUT_S", 5.0)
    params = {"flash": "1"} if use_flash else {}
    try:
        resp = httpx.get(f"http://{ip}/capture", params=params, timeout=timeout_s)
    except httpx.HTTPError as exc:
        raise ESP32CaptureError(f"could not reach ESP32-CAM at {ip}: {exc}") from exc
    if resp.status_code != 200 or not resp.content:
        raise ESP32CaptureError(
            f"ESP32-CAM returned status {resp.status_code} with no usable image")
    return resp.content


def stream_url() -> str | None:
    """The board's live MJPEG preview, or None if no IP is configured. Built
    from the same ESP32_CAM_IP as captures, so the two cannot drift apart."""
    ip = _ip()
    if not ip:
        return None
    port = getattr(_settings, "ESP32_CAM_STREAM_PORT", 81)
    return f"http://{ip}:{port}/stream"


def check_health(timeout_s: float = 2.0) -> bool:
    ip = _ip()
    if not ip:
        return False
    try:
        return httpx.get(f"http://{ip}/health", timeout=timeout_s).status_code == 200
    except httpx.HTTPError:
        return False
