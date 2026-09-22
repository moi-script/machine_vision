"""Per-camera court calibration for the four-camera rig.

Extends the single-camera `POST /api/calibration` in settings.py, which stays
working as an alias for the front camera. See docs/calibration-api.md.
"""
from __future__ import annotations

import datetime as _dt

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import db, pipeline, sources, virtual_camera as vcam
from app.calibration import (CORNER_LABELS, CalibrationError, fit_lines,
                             line_overlay, reproject, solve)
from app.routers import control
from utils import zones

router = APIRouter(tags=["cameras"])

# MJPEG part separators, built here so no escape survives a source rewrite.
BOUNDARY = b"--frame" + bytes([13, 10]) + b"Content-Type: image/jpeg" + bytes([13, 10, 13, 10])
TAIL = bytes([13, 10])


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _doc(camera_id: str) -> dict | None:
    return db.calibrations().find_one({"_id": camera_id})


class PreviewBody(BaseModel):
    frame_id: str | None = None
    corners: list[list[float]] = Field(..., min_length=4, max_length=4)
    singles: bool = False


class CommitBody(PreviewBody):
    operator: str | None = None
    note: str | None = None


@router.get("/api/cameras")
def list_cameras():
    """Rig status in one call, for the calibration page's camera list."""
    present = vcam.available()
    out = []
    for cid in vcam.CAMERA_IDS:
        doc = _doc(cid)
        out.append({
            "camera_id": cid,
            "source_available": present.get(cid, False),
            "calibrated": bool(doc),
            "mode": (doc or {}).get("mode", "corners" if doc else None),
            "reprojection_error_px": (doc or {}).get("reprojection_error_px"),
            "calibrated_at": (doc or {}).get("calibrated_at"),
            "health": "ok" if doc else "uncalibrated",
        })
    return out


@router.get("/api/cameras/{camera_id}/frame")
def get_frame(camera_id: str, index: int | None = None, enhance: bool = True):
    """Freeze one frame and hand it over with an id.

    The id matters: points clicked against a moving image would refer to a view
    that no longer exists by the time they are posted.
    """
    if camera_id not in vcam.CAMERA_IDS:
        raise HTTPException(404, f"unknown camera {camera_id!r}")
    if camera_id == "front" and _engine_holds_front():
        raise HTTPException(409, "a session is using the front camera - stop it to calibrate")
    try:
        frame_id, frame = vcam.grab(camera_id, index)
    except FileNotFoundError as exc:
        raise HTTPException(503, f"no source for {camera_id}: {exc}")
    except vcam.CameraStarting:
        raise HTTPException(503, "camera is starting - try again")
    except Exception as exc:
        raise HTTPException(500, str(exc))

    shown = vcam.enhance(frame) if enhance else frame
    h, w = frame.shape[:2]
    return {"camera_id": camera_id, "frame_id": frame_id,
            "width": w, "height": h,
            "frame_count": vcam.frame_count(camera_id),
            "jpeg_b64": vcam.to_jpeg_b64(shown)}


@router.post("/api/cameras/{camera_id}/calibration/preview")
def preview(camera_id: str, body: PreviewBody):
    """Solve WITHOUT saving, and return the court reprojected onto the frame."""
    if camera_id not in vcam.CAMERA_IDS:
        raise HTTPException(404, f"unknown camera {camera_id!r}")
    frozen = vcam.frozen(camera_id)
    size = (frozen["image"].shape[1], frozen["image"].shape[0]) if frozen else (1280, 720)

    try:
        _H, H_inv, err = solve(body.corners, size)
    except CalibrationError as exc:
        return {"ok": False, "errors": exc.errors, "reprojection_error_px": None}

    warnings = []
    if frozen and body.frame_id and frozen["frame_id"] != body.frame_id:
        warnings.append("these points were clicked on a different frame than the "
                        "one currently frozen for this camera")
    return {"ok": True,
            "reprojection_error_px": round(err, 3),
            "corner_labels": list(CORNER_LABELS),
            "overlay": reproject(H_inv, body.singles),
            "warnings": warnings}


@router.post("/api/cameras/{camera_id}/calibration")
def commit(camera_id: str, body: CommitBody):
    """Validate, persist, and load the homography into the live cache."""
    if camera_id not in vcam.CAMERA_IDS:
        raise HTTPException(404, f"unknown camera {camera_id!r}")
    frozen = vcam.frozen(camera_id)
    size = (frozen["image"].shape[1], frozen["image"].shape[0]) if frozen else (1280, 720)

    try:
        _H, _H_inv, err = solve(body.corners, size)
    except CalibrationError as exc:
        # 422, not 500: the request is well-formed but the geometry is unusable.
        raise HTTPException(422, {"errors": exc.errors})

    # Only after the solve succeeds. Storing first was how bad corners used to
    # reach Mongo and surface later as wrong line calls.
    zones.build_homography(body.corners, camera_id=camera_id)

    doc = {
        "_id": camera_id,
        "camera_id": camera_id,
        "corners": [[float(x), float(y)] for x, y in body.corners],
        "corner_labels": list(CORNER_LABELS),
        "singles": body.singles,
        "reprojection_error_px": round(err, 3),
        "reference_frame_id": body.frame_id or (frozen or {}).get("frame_id"),
        "image_size": list(size),
        "operator": body.operator,
        "note": body.note,
        "calibrated_at": _now(),
        "schema_version": 2,
    }
    db.calibrations().update_one({"_id": camera_id}, {"$set": doc}, upsert=True)
    db.calibrations().insert_one({**doc, "_id": f"{camera_id}:{doc['calibrated_at']}",
                                  "history": True})
    return doc


@router.get("/api/cameras/{camera_id}/calibration")
def get_calibration(camera_id: str):
    doc = _doc(camera_id)
    if not doc:
        raise HTTPException(404, f"{camera_id} is not calibrated")
    return doc


@router.delete("/api/cameras/{camera_id}/calibration")
def delete_calibration(camera_id: str):
    db.calibrations().delete_one({"_id": camera_id})
    zones.clear_homography(camera_id)
    return {"camera_id": camera_id, "calibrated": False}


class LineStroke(BaseModel):
    role: str                      # "inside" | "outside"
    points: list[list[float]]


class LinesBody(BaseModel):
    frame_id: str | None = None
    strokes: list[LineStroke]
    operator: str | None = None
    note: str | None = None


@router.post("/api/cameras/{camera_id}/calibration/lines/preview")
def preview_lines(camera_id: str, body: LinesBody):
    """Fit the traced lines without saving, and return them for the overlay."""
    if camera_id not in vcam.CAMERA_IDS:
        raise HTTPException(404, f"unknown camera {camera_id!r}")
    try:
        fits = fit_lines([s.model_dump() for s in body.strokes])
    except CalibrationError as exc:
        return {"ok": False, "errors": exc.errors}
    frozen = vcam.frozen(camera_id)
    w = frozen["image"].shape[1] if frozen else 1280
    return {"ok": True, "fits": fits, "overlay": line_overlay(fits, w),
            "residual_px": {r: f["residual_px"] for r, f in fits.items()}}


@router.post("/api/cameras/{camera_id}/calibration/lines")
def commit_lines(camera_id: str, body: LinesBody):
    """Persist a two-line band calibration for a side or back camera."""
    if camera_id not in vcam.CAMERA_IDS:
        raise HTTPException(404, f"unknown camera {camera_id!r}")
    try:
        fits = fit_lines([s.model_dump() for s in body.strokes])
    except CalibrationError as exc:
        raise HTTPException(422, {"errors": exc.errors})

    frozen = vcam.frozen(camera_id)
    size = ([frozen["image"].shape[1], frozen["image"].shape[0]]
            if frozen else [1280, 720])
    doc = {
        "_id": camera_id,
        "camera_id": camera_id,
        "mode": "lines",
        "lines": fits,
        "strokes": [s.model_dump() for s in body.strokes],
        "image_size": size,
        "reference_frame_id": body.frame_id or (frozen or {}).get("frame_id"),
        "operator": body.operator,
        "note": body.note,
        "calibrated_at": _now(),
        "schema_version": 2,
    }
    db.calibrations().update_one({"_id": camera_id}, {"$set": doc}, upsert=True)
    db.calibrations().insert_one({**doc, "_id": f"{camera_id}:{doc['calibrated_at']}",
                                  "history": True})
    return doc


# -- sources: files and capture devices ----------------------

class SourceBody(BaseModel):
    kind: str                      # "file" | "device"
    path: str | None = None
    index: int | None = None
    label: str | None = None


@router.get("/api/sources")
def list_sources():
    """Everything a slot could be pointed at, for the source picker."""
    return {"bundled": sources.list_bundled(),
            "uploads": sources.list_uploads(),
            "current": {cid: sources.describe(cid) for cid in sources.CAMERA_IDS}}


@router.get("/api/devices")
def list_devices(max_index: int = 4):
    """Probe capture indices. Slow, so the UI asks for it explicitly."""
    return sources.list_devices(max_index)


@router.post("/api/uploads")
async def upload_video(file: UploadFile = File(...)):
    """Accept a video from the user's machine and make it selectable."""
    import os as _os
    if not file.filename or not file.filename.lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(400, "expected a .mp4/.avi/.mov/.mkv video")
    _os.makedirs(sources.UPLOAD_DIR, exist_ok=True)
    safe = _os.path.basename(file.filename).replace("..", "_")
    dest = _os.path.join(sources.UPLOAD_DIR, safe)
    with open(dest, "wb") as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)

    # Reject anything OpenCV cannot decode now, rather than at stream time.
    import cv2 as _cv2
    cap = _cv2.VideoCapture(dest)
    ok = cap.isOpened() and cap.read()[0]
    frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
    w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH)) if ok else 0
    h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT)) if ok else 0
    cap.release()
    if not ok:
        _os.remove(dest)
        raise HTTPException(400, "could not decode that video")
    return {"name": safe, "path": _os.path.abspath(dest), "frames": frames,
            "width": w, "height": h,
            "size_mb": round(_os.path.getsize(dest) / 1e6, 1)}


@router.post("/api/cameras/{camera_id}/source")
def set_source(camera_id: str, body: SourceBody):
    if camera_id not in sources.CAMERA_IDS:
        raise HTTPException(404, f"unknown camera {camera_id!r}")
    try:
        sources.set_source(camera_id, body.kind, body.path, body.index, body.label)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    # A slot pointed somewhere new must be re-read, not left showing the old feed.
    pipeline.stop(camera_id)
    return sources.describe(camera_id)


# -- live pipeline -------------------------------------------

class StartBody(BaseModel):
    # What runs is fixed per slot (pipeline.SLOT_MODELS). `raw` asks for the
    # bare feed (calibration). An old client's `model` field is ignored.
    raw: bool = False
    backend: str | None = None
    target_fps: float = 30.0


def _engine_holds_front() -> bool:
    """True while a drill/identity session owns the front camera."""
    try:
        from app.engine import get_engine
        return get_engine().state != "idle"
    except Exception:
        return False


@router.get("/api/models")
def list_models():
    return {"catalog": pipeline.model_catalog(), "slots": pipeline.SLOT_MODELS}


@router.post("/api/cameras/{camera_id}/start")
def start_camera(camera_id: str, body: StartBody):
    if camera_id not in sources.CAMERA_IDS:
        raise HTTPException(404, f"unknown camera {camera_id!r}")
    # Share control.py's lock: without it, an engine control.start and this
    # front start_camera can interleave and both open the device.
    with control._ctl_lock:
        if camera_id == "front" and _engine_holds_front():
            # The engine owns the device; its annotated frames are what we show.
            return {"camera_id": "front", "running": True, "borrowed": True,
                    "model": "engine", "error": None}
        try:
            return pipeline.start(camera_id, body.raw, body.backend, body.target_fps)
        except ValueError as exc:
            raise HTTPException(400, str(exc))


@router.post("/api/cameras/{camera_id}/stop")
def stop_camera(camera_id: str):
    pipeline.stop(camera_id)
    return {"camera_id": camera_id, "running": False}


@router.get("/api/pipeline")
def pipeline_status():
    return pipeline.all_stats()


@router.get("/api/cameras/{camera_id}/stream")
def stream(camera_id: str):
    """MJPEG. Works in a plain <img> tag, so the page needs no player."""
    if camera_id == "front" and _engine_holds_front():
        from app.streamer import buffer

        def borrowed_frames():
            # buffer.frames() loops forever; stop once the engine gives the
            # camera back, or this generator would outlive the drill and keep
            # serving a stale/frozen feed to whoever is still connected.
            for chunk in buffer.frames():
                yield chunk
                if not _engine_holds_front():
                    return

        return StreamingResponse(
            borrowed_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
    worker = pipeline.get(camera_id)
    if worker is None:
        raise HTTPException(409, f"{camera_id} is not running - start it first")

    def frames():
        import time as _t
        blank = 0
        while True:
            w = pipeline.get(camera_id)
            if w is None:
                return
            jpg = w.latest_jpeg()
            if jpg is None:
                blank += 1
                if blank > 200:      # ~10 s of nothing: the worker is dead
                    return
                _t.sleep(0.05)
                continue
            blank = 0
            yield (BOUNDARY + jpg + TAIL)
            _t.sleep(0.03)

    return StreamingResponse(
        frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/live", response_class=HTMLResponse, include_in_schema=False)
def live_page():
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "static", "live.html")
    with open(path, encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


@router.get("/calibration", response_class=HTMLResponse, include_in_schema=False)
def calibration_page():
    """Self-contained calibration UI.

    Served from the backend rather than the Vite app so the rig can be
    calibrated and the flow exercised with no frontend build and no cameras.
    """
    import os
    # __file__ is app/routers/cameras.py, so one dirname lands on app/.
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "static", "calibration.html")
    with open(path, encoding="utf-8") as fh:
        return HTMLResponse(fh.read())
