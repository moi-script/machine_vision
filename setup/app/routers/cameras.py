"""Per-camera court calibration for the four-camera rig.

Extends the single-camera `POST /api/calibration` in settings.py, which stays
working as an alias for the front camera. See docs/calibration-api.md.
"""
from __future__ import annotations

import datetime as _dt

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app import db, virtual_camera as vcam
from app.calibration import CORNER_LABELS, CalibrationError, reproject, solve
from utils import zones

router = APIRouter(tags=["cameras"])


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
    try:
        frame_id, frame = vcam.grab(camera_id, index)
    except FileNotFoundError as exc:
        raise HTTPException(503, f"no source for {camera_id}: {exc}")
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
