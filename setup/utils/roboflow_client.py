# ============================================================
# roboflow_client.py — Roboflow Workflow integration
#
# Calls the hosted Roboflow *serverless* Workflow
#   "shuttlecock vshuttlecock-m9ihi-nimwo-1-yolo11s-t1 Logic"
# to detect the shuttlecock in a single image / frame.
#
# The official `inference-sdk` does not support Python 3.13 (this repo runs
# 3.13), so this client calls the serverless REST endpoint directly with
# `requests` — modelled on the SDK's run_workflow() call. Interface stays the
# same if you later move to the SDK on an older Python.
#
#   POST https://serverless.roboflow.com/<workspace>/workflows/<workflow_id>
#   { "api_key": "...", "inputs": { "image": {"type": "url"|"base64", ...} } }
#
# The API key is read from the ROBOFLOW_API_KEY environment variable.
#   Get yours at https://app.roboflow.com/settings/api  — NEVER hard-code it.
#
# ⚠️  LIVE VIDEO NOTE
#   Each call here is one HTTP round-trip. That is fine for stills and
#   testing, but it is NOT the path for real-time per-frame webcam
#   inference. For live video use Roboflow's InferencePipeline (runs the
#   workflow locally) or the WebRTC streaming path instead.
# ============================================================

from __future__ import annotations

import base64
import os
import time
from typing import Any

import requests

# --- Workflow coordinates (source of truth: the run endpoint you gave) ---
WORKSPACE_NAME = "mois-workspace"
WORKFLOW_ID    = "shuttlecock-vshuttlecock-m9ihi-nimwo-1-yolo11s-t1-logic"
API_URL        = "https://serverless.roboflow.com"
RUN_URL        = f"{API_URL}/{WORKSPACE_NAME}/workflows/{WORKFLOW_ID}"

# Direct model (bypasses the workflow + its "Logic" block). This runs on the
# free serverless plan and returns a plain object-detection response, so it is
# the reliable path while the workflow's compile bug is unresolved.
#   response: {"image": {...}, "predictions": [{x,y,width,height,confidence,class,...}], ...}
MODEL_ID   = "shuttlecock-m9ihi-nimwo/1"
DETECT_URL = f"{API_URL}/{MODEL_ID}"

# The workflow's image input is named "image".
IMAGE_INPUT_NAME = "image"

# Optional non-image inputs the workflow declares (discovered from the live
# endpoint's input schema). Pass any of these via `parameters=`.
#   confidence, iou_threshold, class_agnostic_nms, max_detections
KNOWN_PARAMETERS = (
    "confidence",
    "iou_threshold",
    "class_agnostic_nms",
    "max_detections",
)

# --- Defaults for the request wrapper ---
DEFAULT_TIMEOUT = 30.0   # seconds per attempt (connect + read)
DEFAULT_RETRIES = 2      # extra attempts after the first (so 3 tries total)
DEFAULT_BACKOFF = 1.5    # seconds; grows as backoff * (2 ** attempt)


def _load_dotenv() -> None:
    """
    Minimal .env loader (no external dependency): read KEY=VALUE lines from a
    .env at the project root and set them in os.environ *without* overriding
    variables that are already set in the real environment.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".env")
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


_load_dotenv()


class RoboflowConfigError(RuntimeError):
    """Raised when required configuration is missing (API key, bad input)."""


class RoboflowInferenceError(RuntimeError):
    """Raised when the workflow call fails after all retries / times out."""


def _image_input(image: Any) -> dict[str, str]:
    """
    Convert `image` into the workflow input dict Roboflow expects.

    Accepts:
      - an https URL string            -> {"type": "url", "value": ...}
      - a local image file path (str)  -> base64 of the file contents
      - a raw base64 string            -> {"type": "base64", "value": ...}
      - a numpy array (OpenCV frame)   -> JPEG-encoded then base64
    """
    # numpy array (e.g. an OpenCV BGR frame) — encode to JPEG, then base64.
    if hasattr(image, "shape") and hasattr(image, "dtype"):
        import cv2  # local import: only needed for array inputs

        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            raise RoboflowConfigError("Failed to JPEG-encode the image array.")
        return {"type": "base64", "value": base64.b64encode(buf).decode("ascii")}

    if isinstance(image, str):
        lowered = image.lower()
        if lowered.startswith("https://"):
            return {"type": "url", "value": image}
        if lowered.startswith("http://"):
            raise RoboflowConfigError(
                "Roboflow rejects plain http:// URLs — use https:// or base64."
            )
        if os.path.isfile(image):
            with open(image, "rb") as fh:
                return {"type": "base64", "value": base64.b64encode(fh.read()).decode("ascii")}
        # Otherwise assume it is already a base64 string.
        return {"type": "base64", "value": image}

    raise RoboflowConfigError(
        f"Unsupported image input type: {type(image).__name__}. "
        "Pass an https URL, a file path, a base64 string, or a numpy array."
    )


def run_shuttlecock_workflow(
    image: Any,
    *,
    api_key: str | None = None,
    parameters: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
) -> list[dict[str, Any]]:
    """
    Run the shuttlecock detection workflow on one image.

    Args:
        image:      https URL, local file path, base64 string, or a numpy
                    array (OpenCV frame) to send as the "image" input.
        api_key:    Roboflow API key. Falls back to $ROBOFLOW_API_KEY.
        parameters: Extra workflow inputs, merged into "inputs". The workflow
                    declares these tunables besides the image (see
                    KNOWN_PARAMETERS): confidence, iou_threshold,
                    class_agnostic_nms, max_detections. Empty by default.
        timeout:    Max seconds to wait for a single attempt.
        retries:    Number of retries after the first failed attempt.
        backoff:    Base seconds for exponential backoff between retries.

    Returns:
        A list with one entry per input image (a single-image call returns a
        one-element list). Each entry is a dict keyed by the workflow's own
        output names — parse it defensively (see helpers below); do not assume
        specific keys.

    Raises:
        RoboflowConfigError:    Missing API key or unusable image input.
        RoboflowInferenceError: The call failed after all retries.
    """
    api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RoboflowConfigError(
            "No Roboflow API key found. Set the ROBOFLOW_API_KEY environment "
            "variable (from https://app.roboflow.com/settings/api) or pass "
            "api_key=... explicitly."
        )

    payload: dict[str, Any] = {
        "api_key": api_key,
        "inputs": {IMAGE_INPUT_NAME: _image_input(image)},
    }
    if parameters:
        payload["inputs"].update(parameters)

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(RUN_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            last_exc = exc
        except ValueError as exc:  # non-JSON body
            last_exc = RoboflowInferenceError("Workflow returned a non-JSON response.")
            last_exc.__cause__ = exc
        else:
            # Serverless returns {"outputs": [ {…} ]}; normalise to a list.
            outputs = data.get("outputs", data) if isinstance(data, dict) else data
            if not isinstance(outputs, list):
                outputs = [outputs]
            return outputs

        if attempt < retries:
            time.sleep(backoff * (2 ** attempt))

    raise RoboflowInferenceError(
        f"Roboflow workflow '{WORKFLOW_ID}' failed after {retries + 1} attempts."
    ) from last_exc


def run_shuttlecock_model(
    image: Any,
    *,
    api_key: str | None = None,
    confidence: int | None = None,
    overlap: int | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
) -> dict[str, Any]:
    """
    Run the trained shuttlecock model DIRECTLY (no workflow), over the free
    serverless detect API. Use this when the workflow's Logic block isn't
    needed, or while the workflow is broken.

    Args:
        image:      https URL, local file path, base64 string, or numpy frame.
        api_key:    Roboflow API key. Falls back to $ROBOFLOW_API_KEY.
        confidence: Min confidence as a PERCENT 0–100 (hosted API convention),
                    not 0–1. Omit to use the model default.
        overlap:    Max IoU overlap for NMS, percent 0–100. Omit for default.

    Returns:
        The raw detection dict: {"image": {...},
        "predictions": [{"x","y","width","height","confidence","class",...}]}.
        Use extract_shuttle_xy() to pull the top detection's centre.

    Raises:
        RoboflowConfigError / RoboflowInferenceError (same as the workflow fn).
    """
    api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RoboflowConfigError(
            "No Roboflow API key found. Set ROBOFLOW_API_KEY (env or .env) or "
            "pass api_key=..."
        )

    params: dict[str, Any] = {"api_key": api_key}
    if confidence is not None:
        params["confidence"] = confidence
    if overlap is not None:
        params["overlap"] = overlap

    inp = _image_input(image)
    data = None
    headers = None
    if inp["type"] == "url":
        params["image"] = inp["value"]
    else:
        # base64 body per the hosted detect API convention.
        data = inp["value"]
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(DETECT_URL, params=params, data=data,
                                 headers=headers, timeout=timeout)
            resp.raise_for_status()
            result = resp.json()
        except requests.RequestException as exc:
            last_exc = exc
        except ValueError as exc:
            last_exc = RoboflowInferenceError("Model returned a non-JSON response.")
            last_exc.__cause__ = exc
        else:
            return result

        if attempt < retries:
            time.sleep(backoff * (2 ** attempt))

    raise RoboflowInferenceError(
        f"Roboflow model '{MODEL_ID}' failed after {retries + 1} attempts."
    ) from last_exc


# ── Defensive response helpers ───────────────────────────────
# The workflow's output names are whatever you configured in Roboflow, so we
# never hard-code them. These helpers inspect the real structure instead.


def _looks_like_base64_image(value: Any) -> str | None:
    """Return a base64 string if `value` looks like an image blob, else None."""
    # Roboflow may return image outputs as a raw base64 string, or as a dict
    # like {"type": "base64", "value": "..."} / {"value": "..."}.
    if isinstance(value, dict):
        if value.get("type") in ("base64", "image") and isinstance(value.get("value"), str):
            return value["value"]
        inner = value.get("value")
        if isinstance(inner, str) and len(inner) > 256:
            return inner
        return None
    if isinstance(value, str) and len(value) > 256:
        # Long strings in a detection workflow are almost always image blobs.
        return value
    return None


def summarize_outputs(entry: dict[str, Any]) -> dict[str, Any]:
    """
    Return a small, log-safe view of one result entry: every key with its
    type, and for image-shaped values just a size marker (never the blob,
    never raw polygon `points`).
    """
    summary: dict[str, Any] = {}
    for key, value in entry.items():
        blob = _looks_like_base64_image(value)
        if blob is not None:
            summary[key] = f"<image base64, {len(blob)} chars>"
        elif isinstance(value, list):
            summary[key] = f"<list, {len(value)} items>"
        elif isinstance(value, dict):
            summary[key] = f"<dict, keys={sorted(value)[:8]}>"
        else:
            summary[key] = value
    return summary


def save_image_outputs(entry: dict[str, Any], out_dir: str, prefix: str = "rf") -> list[str]:
    """
    Decode any image-shaped (base64) outputs in `entry` and write them to
    `out_dir`. Returns the list of file paths written. The blobs are decoded,
    written, and dropped — never logged or retained.
    """
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    for key, value in entry.items():
        blob = _looks_like_base64_image(value)
        if blob is None:
            continue
        try:
            raw = base64.b64decode(blob, validate=False)
        except Exception:  # noqa: BLE001 - skip anything that isn't real base64
            continue
        path = os.path.join(out_dir, f"{prefix}_{key}.jpg")
        with open(path, "wb") as fh:
            fh.write(raw)
        written.append(path)
    return written


def _iter_detection_lists(entry: dict[str, Any]):
    """Yield lists of detection dicts (those carrying x/y coordinates)."""
    for value in entry.values():
        items = value
        # Object-detection blocks often nest under {"predictions": [...]}.
        if isinstance(value, dict) and isinstance(value.get("predictions"), list):
            items = value["predictions"]
        if isinstance(items, list) and items and isinstance(items[0], dict):
            if {"x", "y"} <= set(items[0].keys()):
                yield items


def extract_shuttle_xy(entry: dict[str, Any]) -> tuple[float, float] | None:
    """
    Best-effort: return the (x, y) centre of the highest-confidence detection
    in this result entry, or None if there are no detections. Only the fields
    we use (x, y, confidence) are read — bulky fields like polygon `points`
    are ignored.
    """
    best: tuple[float, float] | None = None
    best_conf = -1.0
    for detections in _iter_detection_lists(entry):
        for det in detections:
            conf = float(det.get("confidence", 0.0))
            if conf >= best_conf and "x" in det and "y" in det:
                best_conf = conf
                best = (float(det["x"]), float(det["y"]))
    return best
