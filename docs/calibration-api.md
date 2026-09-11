# Calibration API — design sketch

Extends the existing single-camera calibration to the 3–4 camera rig, and adds the
verification step that makes a calibration trustworthy rather than merely stored.

## 1. What already exists

| Piece | Where | Does |
|---|---|---|
| `POST /api/calibration` | `app/routers/settings.py:52` | takes `{corners: [[x,y] x4]}`, saves to Mongo, notifies the engine |
| `CourtSettings.corners` | `app/models.py:39` | one corner list, defaulted from `config.settings.COURT_CORNERS` |
| `build_homography()` | `utils/zones.py:25` | 4 corners → pixel↔court transform, raises on degenerate input |
| `to_court`, `in_court_bounds`, `crossed_net` | `utils/zones.py` | the consumers |
| live reload | `app/engine.py:413` | rebuilds geometry mid-drill without dropping the session |
| `calibrate.py` | repo root of `setup/` | click 4 corners, paste into settings |

Corner order is `[net_left, net_right, baseline_right, baseline_left]`, mapped onto
the canonical rect `[0,0] [W,0] [W,L] [0,L]`. The trainee half-court is `cy ∈ [0, L]`;
the feeder side extrapolates to negative `cy`, which is what `crossed_net()` keys on.

## 2. What is missing

1. **One camera.** `CourtSettings.corners` is a single list. Three cameras need three
   calibrations, addressed independently.
2. **No verification.** The endpoint persists whatever it is given. `build_homography`
   raises on degenerate corners, but only later, inside `_notify_engine()`, which
   swallows exceptions — so a bad calibration is accepted silently and the failure
   surfaces as wrong line calls rather than as an error.
3. **No reprojection error.** Nothing measures how well the solved homography actually
   fits the clicked points, so "slightly wrong" is indistinguishable from "right".
4. **No reference frame.** Drift tracking needs the frame the calibration was solved
   against; without it a moved rig cannot be detected, only guessed at.
5. **No provenance.** No timestamp, no operator, no history. A calibration that was
   good last week and bad today leaves no trail.

## 3. Endpoints

All under `/api/cameras/{camera_id}`. `camera_id` is one of the rig positions —
`front`, `left`, `right`, `back`.

### `GET /api/cameras`

Rig status in one call, for the calibration page to render its camera list.

```json
[
  {"camera_id": "front", "calibrated": true,  "reprojection_error_px": 1.8,
   "calibrated_at": "2026-09-02T11:20:31Z", "drift_px": 0.6, "health": "ok"},
  {"camera_id": "left",  "calibrated": false, "health": "uncalibrated"}
]
```

### `GET /api/cameras/{camera_id}/frame`

One JPEG from that camera, for the operator to click on. Freezes a frame server-side
and returns it with an id, so the points that come back are known to refer to *that*
image rather than to whatever the camera is showing a second later.

```json
{"frame_id": "front-1788347796", "width": 1280, "height": 720,
 "jpeg_b64": "..."}
```

### `POST /api/cameras/{camera_id}/calibration/preview`

**The endpoint that makes this reliable.** Solves without saving and returns what the
UI needs to draw the court back over the frame, so the operator can see whether it
landed on the real lines before committing.

```json
// request
{"frame_id": "front-1788347796",
 "corners": [[210,455],[1104,498],[1240,690],[62,640]]}

// response
{"ok": true,
 "reprojection_error_px": 1.8,
 "overlay": {
   "court_outline":  [[210,455],[1104,498],[1240,690],[62,640]],
   "net_line":       [[210,455],[1104,498]],
   "service_lines":  [[[168,512],[1140,551]], ...],
   "centre_line":    [[651,470],[651,665]],
   "zones":          {"front_left": [[...]], ...}
 },
 "warnings": []}
```

`ok: false` with `errors` when it cannot be used at all:

```json
{"ok": false,
 "errors": ["corners are near-collinear — the homography is degenerate"],
 "reprojection_error_px": null}
```

### `POST /api/cameras/{camera_id}/calibration`

Commits a previewed calibration. Rejects on the same rules the preview applies, so a
client that skips preview cannot store something the preview would have refused.

```json
{"frame_id": "front-1788347796",
 "corners": [[210,455],[1104,498],[1240,690],[62,640]],
 "operator": "moises",
 "note": "left post moved after session 3"}
```

Returns the stored record. Side effects: writes the Mongo doc, stores the reference
frame, and calls the existing `_notify_engine()` so a running drill picks it up.

### `GET /api/cameras/{camera_id}/calibration`

The current record, including the reference frame id.

### `GET /api/cameras/{camera_id}/calibration/history`

Previous calibrations, newest first. Cheap to keep and the fastest way to answer
"did the geometry change between session 2 and session 5".

### `POST /api/cameras/{camera_id}/calibration/health`

Drift check. Registers the current frame against the stored reference and reports how
far the camera has moved.

```json
{"drift_px": 0.6, "match_confidence": 0.94, "health": "ok"}
{"drift_px": 14.2, "match_confidence": 0.71, "health": "drifted",
 "detail": "geometry auto-corrected; recalibrate when convenient"}
{"drift_px": null, "match_confidence": 0.18, "health": "lost",
 "detail": "cannot register to the reference frame — recalibrate before scoring"}
```

`lost` must stop scoring rather than degrade it. A system that knows it is lost is
safe; one that keeps calling lines from a stale homography is not.

## 4. Mongo document

Collection `calibrations`, one current doc per camera plus history.

```json
{
  "_id": "front",
  "camera_id": "front",
  "corners": [[210,455],[1104,498],[1240,690],[62,640]],
  "corner_labels": ["net_left","net_right","baseline_right","baseline_left"],
  "court": {"width": 6.1, "length": 13.4, "net_deadband": 0.15},
  "reprojection_error_px": 1.8,
  "reference_frame_id": "front-1788347796",
  "image_size": [1280, 720],
  "operator": "moises",
  "note": "left post moved after session 3",
  "calibrated_at": "2026-09-02T11:20:31Z",
  "schema_version": 2
}
```

`corner_labels` is stored rather than assumed. The order is load-bearing —
`build_homography` maps position 0 to `[0,0]` — and an order swap produces a
homography that is valid, wrong, and completely silent.

## 5. Validation

Applied identically by `preview` and `POST`. In rough order of how often each will fire:

| Check | Rejects |
|---|---|
| 4 points, inside image bounds | typos, stale frame ids |
| no 3 points collinear (area of every triple > threshold) | a degenerate quad `getPerspectiveTransform` will happily accept |
| quad is convex, wound consistently | corners clicked out of order |
| reprojection error ≤ `MAX_REPROJ_PX` (start at 4 px) | sloppy clicking |
| aspect ratio within tolerance of `COURT_W / COURT_L` | wrong lines identified as sidelines |
| `frame_id` matches a frame this camera served | points clicked on a different view |

Reprojection error is the honest one: map the 4 clicked points through the solved
homography into court space, map them back, and measure the pixel distance. With only
4 points and an exact 4-point solve that error is zero by construction, so the check
only means something once the operator clicks **more than 4** points and the solve
becomes least-squares (`cv2.findHomography` rather than `getPerspectiveTransform`).

**Recommendation: ask for 6 points, not 4** — the four court corners plus the two
service-line/sideline intersections. The extra two turn an exact solve into an
over-determined one, which is what makes the error number real, and they cost the
operator about ten seconds.

## 6. Drift tracking

Per camera, roughly once a second rather than per frame:

1. Register the current frame to the stored reference (ORB features + RANSAC
   homography, or ECC for a small search). Both are milliseconds and neither needs
   training data.
2. `drift_px` = median displacement of matched inliers.
3. Below `DRIFT_OK_PX` (~3): nothing to do.
4. Between that and `DRIFT_MAX_PX` (~25): compose the drift homography with the stored
   calibration and carry on, flagged `drifted`.
5. Above that, or match confidence below threshold: `lost`. Stop scoring, ask for
   recalibration.

This is what covers the movable-prototype problem. It uses all the static texture in
the scene — floor markings, barriers, fixtures — not just four lines, so it is both
more accurate and cheaper than re-detecting court geometry per frame.

## 7. Migration

The existing single-camera path stays working:

1. Add the `calibrations` collection alongside `CourtSettings.corners`.
2. On first read, if `calibrations` has no `front` doc but `CourtSettings.corners` is
   set, copy it in with `schema_version: 1` and `reprojection_error_px: null`.
3. Keep `POST /api/calibration` as an alias for `POST /api/cameras/front/calibration`,
   marked deprecated.
4. `build_homography(corners)` already takes corners as an argument, so per-camera
   support needs a homography cache keyed by `camera_id` rather than the current
   module-level `_H` — that is the one real change inside `utils/zones.py`.

Point 4 is the only part that touches working geometry code. Everything else is
additive.
