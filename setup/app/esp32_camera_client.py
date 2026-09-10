"""
esp32_camera_client.py

Standalone client for pulling a snapshot from the ESP32-CAM enrollment
station over HTTP. This is new — it does not modify anything in app/.

Drop this file at:  app/esp32_camera_client.py

Requires ESP32_CAM_IP (and optionally ESP32_CAM_TIMEOUT_S) as plain
module-level constants in config/settings.py, same pattern as
CAMERA_INDEX / FRAME_WIDTH / etc. See esp32_settings_additions.txt.
"""

from __future__ import annotations

import httpx

from config import settings as _settings  # module, not an object — matches
                                           # how CAMERA_INDEX etc. are defined


class ESP32CaptureError(Exception):
    """Raised when the ESP32-CAM can't be reached or returns no image."""


async def capture_snapshot(use_flash: bool = True) -> bytes:
    """
    Fetch a single JPEG frame from the ESP32-CAM enrollment station.

    Returns raw JPEG bytes on success. Raises ESP32CaptureError on any
    network failure, timeout, or non-200 response so callers can turn
    it into a clean HTTP error for the UI instead of a stack trace.
    """
    ip = getattr(_settings, "ESP32_CAM_IP", None)
    if not ip:
        raise ESP32CaptureError(
            "ESP32_CAM_IP is not set in config/settings.py"
        )

    timeout_s = getattr(_settings, "ESP32_CAM_TIMEOUT_S", 5.0)
    url = f"http://{ip}/capture"
    params = {"flash": "1"} if use_flash else {}

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(url, params=params)
    except httpx.RequestError as exc:
        raise ESP32CaptureError(f"could not reach ESP32-CAM at {ip}: {exc}") from exc

    if resp.status_code != 200 or not resp.content:
        raise ESP32CaptureError(
            f"ESP32-CAM returned status {resp.status_code} with no usable image"
        )

    return resp.content


def stream_url() -> str | None:
    """URL of the board's live MJPEG preview, or None if no IP is configured.

    The browser points an <img> straight at this, so it is the one piece of
    ESP32 addressing that has to leave the backend. Built from the same
    ESP32_CAM_IP the capture path uses, so the preview can never drift onto a
    stale address while captures work (or vice versa).
    """
    ip = getattr(_settings, "ESP32_CAM_IP", None)
    if not ip:
        return None
    port = getattr(_settings, "ESP32_CAM_STREAM_PORT", 81)
    return f"http://{ip}:{port}/stream"


async def check_health() -> bool:
    """Quick connectivity check — used by a control.py debug endpoint."""
    ip = getattr(_settings, "ESP32_CAM_IP", None)
    if not ip:
        return False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"http://{ip}/health")
        return resp.status_code == 200
    except httpx.RequestError:
        return False