# Court-Space Homography Geometry — Design Spec

**Date:** 2026-07-12
**Component:** `setup/` (Badminton Feeder Trainer)
**Status:** Approved design, ready for implementation plan

## 1. Problem

The current geometry assumes a **centered side-view camera**: the net is a
vertical pixel line (`NET_X`), player vs. feeder side is a left/right split
(`shuttle_x < NET_X`), and the court + 6 target zones are **axis-aligned pixel
rectangles** (`COURT_ZONE`, `PLAYER_ZONES`, `is_inside_court` overlap ratio).

The real deployment is different: the camera is **mounted on the feeder**, at
the net, looking **across the net at the trainee on the far side**. In that
view:

- The net is not a vertical line — it appears low/wide (near the camera) and at
  whatever angle the mounting gives.
- The far half-court is a **perspective trapezoid** (far baseline small/high,
  net edge wide/low), so axis-aligned pixel rectangles do not map to real court
  positions.
- Left/right `NET_X` logic is the wrong axis entirely.

Result: zone attribution, net-crossing, and side detection are all wrong for
the actual camera placement.

## 2. Goal

Replace the pixel-based geometry with a **top-down "court space"** model built
from a homography. All spatial logic (in-court test, zone lookup, side, net
crossing) is computed in court space, making it **independent of camera angle**
(feeder-mounted, side, tilted, or off-corner all work). The scored trainee is
always on the **far side only**; the feeder/near side is excluded
automatically.

Non-goals (out of scope): changing the shuttle *detection* source
(`SHUTTLE_SOURCE` stays as-is), multi-court support, players on the near side,
or auto-detecting the court corners (calibration stays manual).

## 3. Court coordinate system

The trainee's **far half-court** is rectified to a canonical rectangle:

- Origin `(0, 0)` = net-line **left** corner.
- `x` → right along the net; `y` → away from the net toward the far baseline.
- Dimensions in court units (default, singles-half proportions):
  - `COURT_W = 518`  (width, sideline to sideline)
  - `COURT_L = 670`  (net to far baseline)
- Net line = `y = 0`; far baseline = `y = COURT_L`; sidelines = `x = 0` and
  `x = COURT_W`.
- Feeder/near side is `y < 0` (extrapolated; only the sign is trusted, see §6).

## 4. Homography & calibration

**Calibration (`calibrate.py`)** captures exactly **4 clicks** on the trainee's
half-court, in this fixed order:

1. `net_left`      — net meets the left sideline
2. `net_right`     — net meets the right sideline
3. `baseline_right`— far baseline meets the right sideline
4. `baseline_left` — far baseline meets the left sideline

It prints a ready-to-paste `COURT_CORNERS = [(x,y), (x,y), (x,y), (x,y)]`
block and (optionally) shows a rectified preview for a sanity check. The old
`NET_X` / bounding-box output is removed.

Two of the four points (`net_left`, `net_right`) lie **on the net**, so the net
line is data, never assumed.

**Transform (`utils/zones.py`)**:

- `build_homography()` — reads `COURT_CORNERS` + `COURT_W/COURT_L`, returns `H`
  via `cv2.getPerspectiveTransform(src_corners, canonical_rect)`. Built once and
  cached. Also caches `H_inv` for drawing.
- `to_court(pixel_pt) -> (cx, cy)` — via `cv2.perspectiveTransform`.
- `court_to_pixel(court_pt) -> (px, py)` — via `H_inv`, for overlays.

`canonical_rect = [(0,0), (COURT_W,0), (COURT_W,COURT_L), (0,COURT_L)]`
(matches the click order).

## 5. Zones (court space, 3×2 grid)

`PLAYER_ZONES` is redefined in **court units** as 6 rectangles
`(x1, y1, x2, y2)`, derived from a 3-column × 2-row split
(columns = `COURT_W/3`, rows = `COURT_L/2`):

| Zone           | x range            | y range          |
|----------------|--------------------|------------------|
| `front_left`   | 0 … 172.67         | 0 … 335          |
| `front_center` | 172.67 … 345.33    | 0 … 335          |
| `front_right`  | 345.33 … 518       | 0 … 335          |
| `back_left`    | 0 … 172.67         | 335 … 670        |
| `back_center`  | 172.67 … 345.33    | 335 … 670        |
| `back_right`   | 345.33 … 518       | 335 … 670        |

"front" = nearer the net (`y` small); "back" = nearer the far baseline.
This fixes the current copy-paste bug where `back_right` duplicated
`back_mid`.

## 6. Spatial logic (all in court space)

Rewritten in `utils/zones.py`; callers pass court coords:

- `in_court_bounds(cx, cy)` — `0 ≤ cx ≤ COURT_W and 0 ≤ cy ≤ COURT_L`. Takes
  **court** coords (caller transforms once; no double transform). Replaces the
  old pixel overlap-ratio `is_inside_court`. **This is what excludes the feeder
  side:** near-side people map to `cy < 0` → False → never scored.
- `get_zone_from_position(cx, cy)` — first matching zone rectangle, else None.
- `get_player_in_zone(zone, player_positions)` — `player_positions` now holds
  **court** coords `{id: (cx, cy)}`.
- `get_shuttle_side(cy)` — `cy < -NET_DEADBAND` → `"feeder_side"`, else
  `"player_side"`.
- `crossed_net(prev_cy, cy)` — True when `y` goes from `< -NET_DEADBAND` to
  `> +NET_DEADBAND` (feeder → player). Its inverse detects the return.

`NET_DEADBAND` (default `15` court units) absorbs jitter around `y = 0` so a
shuttle/player at the net doesn't flicker sides.

**Extrapolation note:** points far outside the calibrated quad (feeder side)
have unreliable *magnitudes* under `H`, but the **sign of `cy`** near the net is
stable. We only rely on the sign there; this is documented in code.

## 7. Integration (`main.py`)

- `detect_players()` — for each tracked person, compute the pixel ankle
  (`get_ankle_position`, unchanged), transform once via `to_court(ankle)` →
  `(cx, cy)`; keep only those with `in_court_bounds(cx, cy)` true; store court
  coords in `player_positions`.
- Shuttle handling in the loop — transform `shuttle_pos` → `(cx, cy)`; feed
  into `crossed_net`, `get_shuttle_side`, `get_zone_from_position`. Replaces the
  `NET_X`-based `check_net_crossing` / `check_return`.
- `SHUTTLE_SOURCE = "serverless"` detection is untouched — only the *geometry*
  interpretation of the detected point changes.

## 8. Overlays (`utils/display.py`)

- `draw_net()` — draw the segment between the two net corners
  (`net_left`–`net_right`), not a vertical `NET_X`.
- `draw_zones()` / court box — transform each court-space zone rectangle's
  corners back to pixels via `court_to_pixel()` so zones render as the correct
  trapezoids on the live frame.

## 9. Settings changes (`config/settings.py`)

- **Remove:** `NET_X`, `PLAYER_SIDE`, `FEEDER_SIDE`, `COURT_ZONE`,
  `PLAYER_OVERLAP_RATIO`, and the pixel-rectangle `PLAYER_ZONES`.
- **Add:** `COURT_CORNERS` (4 pixel points, placeholder until calibrated),
  `COURT_W`, `COURT_L`, `NET_DEADBAND`, and `PLAYER_ZONES` redefined in court
  units (§5).

## 10. Testing

**Unit tests (`tests/test_zones.py`, new)** — pure functions, no camera, run on
Python 3.13:

- **Homography round-trip:** each `COURT_CORNERS` pixel → its canonical corner.
- **Zone mapping:** representative court points → expected zone (e.g.
  `(W*0.16, L*0.25)` → `front_left`).
- **Side + crossing:** `cy < 0` → feeder, `cy > 0` → player; `crossed_net`
  True on a real sign flip; jitter within `±NET_DEADBAND` → no crossing.
- **Inside-court:** point just outside a sideline → False; center → True;
  a feeder-side point (`cy < 0`) → False.

**Error handling:**

- Unset/placeholder `COURT_CORNERS` (not 4 points) → `build_homography()`
  raises a clear "run calibrate.py first" error at startup.
- Degenerate corners (collinear/duplicate → non-invertible) → caught, reported
  as "corners look degenerate, re-calibrate."

**Live verification:** `calibrate.py` → `main.py`; confirm the 3×2 grid renders
as the correct trapezoid and a player standing in each zone is attributed
correctly, while someone on the feeder side is ignored.

## 11. Affected files (summary)

| File | Change |
|------|--------|
| `config/settings.py` | Remove pixel geometry; add court-space config |
| `utils/zones.py` | Homography build/transform + all spatial logic in court space |
| `main.py` | Transform ankle/shuttle to court space; new crossing/side/zone calls |
| `calibrate.py` | Guided 4-corner capture; print `COURT_CORNERS` |
| `utils/display.py` | Draw net segment + trapezoidal zones via inverse homography |
| `tests/test_zones.py` | New unit tests for court-space geometry |
