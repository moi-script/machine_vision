# Court-Space Homography Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the side-view pixel geometry (`NET_X`, axis-aligned pixel zones) with a 4-corner homography "court space" so the badminton trainer works with the feeder-mounted end-on camera.

**Architecture:** Calibration captures the 4 corners of the trainee's far half-court. A homography maps image pixels → a canonical top-down court rectangle. All spatial logic (in-court test, zone lookup, side, net crossing) runs in court coordinates, making it camera-angle-independent. The feeder/near side maps to `cy < 0` and is filtered out automatically.

**Tech Stack:** Python 3.13, OpenCV (`cv2`), NumPy, ultralytics YOLO (unchanged), pytest.

## Global Constraints

- Python 3.13 (repo default interpreter); geometry code must import and tests must pass on it.
- Court units: `COURT_W = 518`, `COURT_L = 670` (arbitrary units; only ratios matter).
- `NET_DEADBAND = 15` court units around `y = 0`.
- Calibration click order (fixed): `net_left, net_right, baseline_right, baseline_left`.
- Canonical rectangle order matches click order: `[(0,0), (COURT_W,0), (COURT_W,COURT_L), (0,COURT_L)]`.
- Zone names (6): `front_left, front_center, front_right, back_left, back_center, back_right` ("front" = near net, small `y`).
- Never commit secrets (`.env`) or weights (`*.pt`) — already gitignored.
- Do not change `SHUTTLE_SOURCE` detection behavior; only the geometry interpretation of detected points changes.

## File Structure

| File | Responsibility |
|------|----------------|
| `config/settings.py` | Court-space config constants (corners, dims, deadband, court-space zones) |
| `utils/zones.py` | Homography build/transform + all spatial logic in court space |
| `main.py` | Transform ankle/shuttle to court space; net-crossing/side/zone calls |
| `utils/display.py` | Draw net segment + trapezoidal court/zones via inverse homography |
| `calibrate.py` | Guided 4-corner capture; print `COURT_CORNERS` |
| `tests/test_zones.py` | Unit tests for court-space geometry (new) |

---

### Task 1: Homography transform core

Add court-space config and the homography build/transform functions. Purely **additive** — the existing side-view code stays intact and the app still runs, so this task is independently reviewable.

**Files:**
- Modify: `config/settings.py` (append court-space config; leave existing constants)
- Modify: `utils/zones.py` (add imports + transform functions)
- Create: `tests/test_zones.py`

**Interfaces:**
- Produces:
  - `settings.COURT_CORNERS: list[tuple[float,float]] | None`
  - `settings.COURT_W: float`, `settings.COURT_L: float`, `settings.NET_DEADBAND: float`
  - `zones.build_homography(corners: list | None = None) -> np.ndarray` (sets module globals `_H`, `_H_inv`; raises `ValueError` on missing/degenerate corners)
  - `zones.to_court(pt: tuple[float,float]) -> tuple[float,float]`
  - `zones.court_to_pixel(pt: tuple[float,float]) -> tuple[float,float]`

- [ ] **Step 1: Ensure pytest is available**

Run: `python -m pytest --version`
If it prints a version, skip. Otherwise:
Run: `python -m pip install --user pytest`
Expected: `Successfully installed pytest-...`

- [ ] **Step 2: Append court-space config to `config/settings.py`**

Add at the end of the file:

```python
# ============================================================
# Court-space geometry (top-down homography model)
# ============================================================

# 4 pixel corners of the TRAINEE's far half-court, in this order:
#   net_left, net_right, baseline_right, baseline_left
# Filled in by calibrate.py. None until calibrated (build_homography errors).
COURT_CORNERS = None   # e.g. [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]

# Court-space dimensions (arbitrary units; only ratios matter for zones).
COURT_W = 518.0        # width  (net_left -> net_right)
COURT_L = 670.0        # length (net -> far baseline)

# Dead-band (court units) around the net line y=0 to debounce side/crossing.
NET_DEADBAND = 15.0
```

- [ ] **Step 3: Write the failing test file `tests/test_zones.py`**

```python
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings  # noqa: E402
from utils import zones  # noqa: E402

# A simple axis-aligned pixel quad (400x600 px) -> canonical 518x670 court.
# Order: net_left, net_right, baseline_right, baseline_left.
CORNERS = [(100.0, 100.0), (500.0, 100.0), (500.0, 700.0), (100.0, 700.0)]


@pytest.fixture(autouse=True)
def _build():
    zones.build_homography(CORNERS)
    yield


def test_corners_map_to_canonical():
    assert zones.to_court((100.0, 100.0)) == pytest.approx((0.0, 0.0), abs=1e-3)
    assert zones.to_court((500.0, 100.0)) == pytest.approx((settings.COURT_W, 0.0), abs=1e-3)
    assert zones.to_court((500.0, 700.0)) == pytest.approx((settings.COURT_W, settings.COURT_L), abs=1e-3)
    assert zones.to_court((100.0, 700.0)) == pytest.approx((0.0, settings.COURT_L), abs=1e-3)


def test_court_to_pixel_is_inverse():
    px = (300.0, 400.0)
    cx = zones.to_court(px)
    back = zones.court_to_pixel(cx)
    assert back == pytest.approx(px, abs=1e-3)


def test_build_homography_requires_four_corners():
    with pytest.raises(ValueError):
        zones.build_homography([(0, 0), (1, 1)])
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python -m pytest tests/test_zones.py -v`
Expected: FAIL — `AttributeError: module 'utils.zones' has no attribute 'build_homography'`

- [ ] **Step 5: Add transform functions to `utils/zones.py`**

At the top of `utils/zones.py`, update the imports block to include the new settings and `cv2`/`numpy`:

```python
import cv2
import numpy as np

from config.settings import (
    PLAYER_ZONES, NET_X, COURT_ZONE,
    ANKLE_CONFIDENCE, PLAYER_OVERLAP_RATIO,
    COURT_CORNERS, COURT_W, COURT_L, NET_DEADBAND,
)
```

Then add, immediately below the imports:

```python
# ── Homography (image pixels <-> top-down court space) ───────
_H = None       # pixel -> court
_H_inv = None   # court -> pixel


def _canonical_rect():
    return np.array(
        [[0.0, 0.0], [COURT_W, 0.0], [COURT_W, COURT_L], [0.0, COURT_L]],
        dtype=np.float32,
    )


def build_homography(corners=None):
    """
    Build (and cache) the pixel->court homography from 4 court corners in
    order [net_left, net_right, baseline_right, baseline_left]. Falls back to
    settings.COURT_CORNERS. Raises ValueError if unset or degenerate.
    """
    global _H, _H_inv
    corners = corners if corners is not None else COURT_CORNERS
    if not corners or len(corners) != 4:
        raise ValueError(
            "COURT_CORNERS not set (need 4 points) — run calibrate.py first."
        )
    src = np.array(corners, dtype=np.float32)
    try:
        _H = cv2.getPerspectiveTransform(src, _canonical_rect())
        _H_inv = np.linalg.inv(_H)
    except (cv2.error, np.linalg.LinAlgError) as exc:
        raise ValueError(
            f"Court corners look degenerate — re-run calibration. ({exc})"
        )
    return _H


def _ensure_homography():
    if _H is None:
        build_homography()


def to_court(pt):
    """Pixel (x, y) -> court (cx, cy)."""
    _ensure_homography()
    p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    c = cv2.perspectiveTransform(p, _H)[0][0]
    return float(c[0]), float(c[1])


def court_to_pixel(pt):
    """Court (cx, cy) -> pixel (x, y). For drawing overlays."""
    _ensure_homography()
    p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    c = cv2.perspectiveTransform(p, _H_inv)[0][0]
    return float(c[0]), float(c[1])
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_zones.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Verify the existing app still imports (nothing removed yet)**

Run: `python -c "import config.settings; import utils.zones; import utils.display; print('imports OK')"`
Expected: `imports OK`

- [ ] **Step 8: Commit**

```bash
git add setup/config/settings.py setup/utils/zones.py setup/tests/test_zones.py
git commit -m "feat(geometry): add court-space homography transform core"
```

---

### Task 2: Court-space spatial logic + court-space zones

Add the court-space spatial functions and redefine `PLAYER_ZONES` in court units. Keep the old constants (`NET_X`, `COURT_ZONE`, `PLAYER_OVERLAP_RATIO`) and the old `is_inside_court` in place for now so `main.py`/`display.py` still import — they get removed in Task 3.

**Files:**
- Modify: `config/settings.py` (replace `PLAYER_ZONES` with court-space zones)
- Modify: `utils/zones.py` (add `in_court_bounds`, `crossed_net`; rewrite `get_zone_from_position`, `get_player_in_zone`, `get_shuttle_side` to court semantics; keep `is_inside_court`)
- Modify: `tests/test_zones.py` (add zone/side/crossing tests)

**Interfaces:**
- Consumes: `zones.to_court`, `settings.COURT_W/COURT_L/NET_DEADBAND` (Task 1).
- Produces:
  - `zones.in_court_bounds(cx: float, cy: float) -> bool`
  - `zones.get_zone_from_position(cx: float, cy: float) -> str | None` (court coords)
  - `zones.get_player_in_zone(zone_name: str, player_positions: dict[int, tuple[float,float]]) -> int | None` (court coords)
  - `zones.get_shuttle_side(cy: float) -> str` (`"feeder_side"` | `"player_side"`)
  - `zones.crossed_net(prev_cy: float, cy: float) -> bool`
  - `settings.PLAYER_ZONES: dict[str, tuple[float,float,float,float]]` in court units

- [ ] **Step 1: Replace `PLAYER_ZONES` in `config/settings.py`**

Find the existing `PLAYER_ZONES = { ... }` block (the 6 pixel rectangles) and replace it entirely with:

```python
# --- 6 target zones in COURT SPACE (x1, y1, x2, y2); 3 cols x 2 rows ---
# "front" = nearer the net (small y); "back" = nearer the far baseline.
_CW3 = COURT_W / 3.0
_CL2 = COURT_L / 2.0
PLAYER_ZONES = {
    "front_left":   (0.0,       0.0,   _CW3,      _CL2),
    "front_center": (_CW3,      0.0,   2 * _CW3,  _CL2),
    "front_right":  (2 * _CW3,  0.0,   COURT_W,   _CL2),
    "back_left":    (0.0,       _CL2,  _CW3,      COURT_L),
    "back_center":  (_CW3,      _CL2,  2 * _CW3,  COURT_L),
    "back_right":   (2 * _CW3,  _CL2,  COURT_W,   COURT_L),
}
```

Note: this block must appear **after** `COURT_W`/`COURT_L` are defined (they are, from Task 1). Leave `COURT_ZONE`, `NET_X`, `PLAYER_SIDE`, `FEEDER_SIDE`, `PLAYER_OVERLAP_RATIO` untouched for now.

- [ ] **Step 2: Add failing tests to `tests/test_zones.py`**

Append these tests:

```python
def test_zone_mapping_front_left():
    cx = settings.COURT_W * 0.16
    cy = settings.COURT_L * 0.25
    assert zones.get_zone_from_position(cx, cy) == "front_left"


def test_zone_mapping_back_right():
    cx = settings.COURT_W * 0.83
    cy = settings.COURT_L * 0.75
    assert zones.get_zone_from_position(cx, cy) == "back_right"


def test_zone_outside_returns_none():
    assert zones.get_zone_from_position(-10.0, 10.0) is None


def test_in_court_bounds():
    assert zones.in_court_bounds(10.0, 10.0) is True
    assert zones.in_court_bounds(settings.COURT_W / 2, settings.COURT_L / 2) is True
    # feeder side (negative y) and outside a sideline
    assert zones.in_court_bounds(10.0, -5.0) is False
    assert zones.in_court_bounds(settings.COURT_W + 5, 10.0) is False


def test_shuttle_side():
    assert zones.get_shuttle_side(200.0) == "player_side"
    assert zones.get_shuttle_side(-200.0) == "feeder_side"
    # within deadband counts as player side (not feeder)
    assert zones.get_shuttle_side(0.0) == "player_side"


def test_crossed_net():
    # feeder -> player crossing
    assert zones.crossed_net(-50.0, 50.0) is True
    # jitter within deadband: no crossing
    assert zones.crossed_net(-5.0, 5.0) is False
    # already on player side: no new crossing
    assert zones.crossed_net(30.0, 60.0) is False


def test_player_in_zone_court_coords():
    positions = {7: (settings.COURT_W * 0.16, settings.COURT_L * 0.25)}
    assert zones.get_player_in_zone("front_left", positions) == 7
    assert zones.get_player_in_zone("back_right", positions) is None
```

- [ ] **Step 3: Run tests to verify the new ones fail**

Run: `python -m pytest tests/test_zones.py -v`
Expected: the new tests FAIL (e.g. `get_shuttle_side` gets 1 arg but old signature differs / `in_court_bounds` missing / `crossed_net` missing). Task 1 tests still PASS.

- [ ] **Step 4: Rewrite spatial logic in `utils/zones.py`**

Replace the existing `get_zone_from_position`, `get_player_in_zone`, and `get_shuttle_side` function bodies, and add `in_court_bounds` + `crossed_net`. Leave `get_ankle_position` and `is_inside_court` as they are.

```python
def in_court_bounds(cx, cy):
    """True if a court-space point is inside the trainee's half-court.

    This is what excludes the feeder/near side: those points map to cy < 0.
    """
    return 0.0 <= cx <= COURT_W and 0.0 <= cy <= COURT_L


def get_zone_from_position(cx, cy):
    """Court-space (cx, cy) -> zone name, or None if outside all 6 zones."""
    for zone_name, (x1, y1, x2, y2) in PLAYER_ZONES.items():
        if x1 <= cx < x2 and y1 <= cy < y2:
            return zone_name
    return None


def get_player_in_zone(zone_name, player_positions):
    """
    zone_name + court-space player positions {id: (cx, cy)} -> player id whose
    position is in that zone, else None.
    """
    if zone_name not in PLAYER_ZONES:
        return None
    x1, y1, x2, y2 = PLAYER_ZONES[zone_name]
    for player_id, (cx, cy) in player_positions.items():
        if x1 <= cx < x2 and y1 <= cy < y2:
            return player_id
    return None


def get_shuttle_side(cy):
    """Court-space y -> which side of the net the shuttle is on."""
    return "feeder_side" if cy < -NET_DEADBAND else "player_side"


def crossed_net(prev_cy, cy):
    """True on a feeder->player crossing (sign flip past the dead-band)."""
    return prev_cy < -NET_DEADBAND and cy > NET_DEADBAND
```

- [ ] **Step 5: Run tests to verify all pass**

Run: `python -m pytest tests/test_zones.py -v`
Expected: PASS (all tests, ~10 passed)

- [ ] **Step 6: Verify imports still work**

Run: `python -c "import utils.zones, utils.display, config.settings; print('imports OK')"`
Expected: `imports OK`
(Note: `main.py` live drawing/zone logic is now transitional — it is fully corrected in Task 3. This step only confirms nothing errors at import.)

- [ ] **Step 7: Commit**

```bash
git add setup/config/settings.py setup/utils/zones.py setup/tests/test_zones.py
git commit -m "feat(geometry): court-space zones, bounds, side and net-crossing logic"
```

---

### Task 3: Integrate into main.py + display.py + calibrate.py; remove old geometry

The cutover. All consumers move to court space in one atomic change, then the dead side-view constants and `is_inside_court` are removed. After this task the app imports cleanly, unit tests pass, and (with a calibrated `COURT_CORNERS`) it runs correctly.

**Files:**
- Modify: `main.py` (imports, `detect_players`, shuttle handling, `check_net_crossing`, `check_return`)
- Modify: `utils/display.py` (imports, `draw_court_zone`, `draw_net`, `draw_zones`)
- Modify: `calibrate.py` (4-corner capture + `COURT_CORNERS` output)
- Modify: `config/settings.py` (remove `NET_X`, `COURT_ZONE`, `PLAYER_SIDE`, `FEEDER_SIDE`, `PLAYER_OVERLAP_RATIO`)
- Modify: `utils/zones.py` (remove now-unused `is_inside_court`)

**Interfaces:**
- Consumes: everything from Tasks 1–2 (`to_court`, `court_to_pixel`, `in_court_bounds`, `crossed_net`, `get_shuttle_side`, `get_zone_from_position`, `get_player_in_zone`, `build_homography`).

- [ ] **Step 1: Update `main.py` imports**

Replace the `from config.settings import (...)` block with (drop `NET_X`):

```python
from config.settings import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    PLAYER_ZONES, DIFFICULTY,
    PERSON_CONFIDENCE, SHUTTLE_CONFIDENCE,
    RETURN_CONFIRM_FRAMES, GRAYSCALE
)
```

Replace the `from utils.zones import (...)` block with (drop `is_inside_court`, add court helpers):

```python
from utils.zones import (
    get_ankle_position, get_zone_from_position,
    get_player_in_zone, get_shuttle_side,
    to_court, in_court_bounds, crossed_net, build_homography,
)
```

- [ ] **Step 2: Transform ankles to court space in `main.py` `detect_players`**

In `detect_players`, replace the court filter + ankle store. Find:

```python
        # Filter: only count players inside the court zone
        if not is_inside_court(box):
            continue

        kps    = keypoints_data[i].tolist()
        ankle  = get_ankle_position(keypoints_data[i])

        scores.init_player(tid)

        if ankle:
            player_positions[tid] = ankle
```

Replace with:

```python
        kps   = keypoints_data[i].tolist()
        ankle = get_ankle_position(keypoints_data[i])

        # Transform the pixel ankle into court space; keep only players on the
        # trainee's half-court (feeder/near side maps to cy < 0 -> excluded).
        if not ankle:
            continue
        cx, cy = to_court(ankle)
        if not in_court_bounds(cx, cy):
            continue

        scores.init_player(tid)
        player_positions[tid] = (cx, cy)
```

- [ ] **Step 3: Rewrite net-crossing/return helpers in `main.py`**

Rename the shuttle-x state to court-y and use `crossed_net` / `get_shuttle_side`. Replace the `prev_shuttle_x` global and both helper functions.

Find `prev_shuttle_x      = None` in the state block and rename to:

```python
prev_shuttle_cy     = None
```

Replace `check_net_crossing`:

```python
def check_net_crossing(shuttle_cy):
    """
    Detects when the shuttle crosses from feeder side -> player side in court
    space. Returns True on the frame the crossing is first confirmed.
    """
    global prev_shuttle_cy, shuttle_crossed

    if prev_shuttle_cy is None:
        prev_shuttle_cy = shuttle_cy
        return False

    just_crossed = crossed_net(prev_shuttle_cy, shuttle_cy)
    prev_shuttle_cy = shuttle_cy

    if just_crossed and not shuttle_crossed:
        shuttle_crossed = True
        return True
    return False
```

Replace `check_return`:

```python
def check_return(shuttle_cy):
    """
    Detects when the shuttle crosses back player side -> feeder side. Requires
    RETURN_CONFIRM_FRAMES consecutive feeder-side frames to avoid flicker.
    """
    global return_side_count

    if get_shuttle_side(shuttle_cy) == "feeder_side":
        return_side_count += 1
    else:
        return_side_count = 0

    return return_side_count >= RETURN_CONFIRM_FRAMES
```

In `start_new_shot`, update the reset line `prev_shuttle_x = None` to `prev_shuttle_cy = None`, and update its `global` declaration accordingly (change `prev_shuttle_x` to `prev_shuttle_cy`).

- [ ] **Step 4: Use court coords for the shuttle in the main loop (`main.py` `run`)**

In `run()`, where the shuttle is processed, transform once and feed court coords in. Find:

```python
            if shuttle_pos:
                sx, sy = shuttle_pos

                # Phase 1: wait for shuttle to cross net into player zone
                if not shuttle_crossed:
                    if check_net_crossing(sx):
                        zone = get_zone_from_position(sx, sy)
```

Replace the first lines so both phases use court coords:

```python
            if shuttle_pos:
                sx, sy = shuttle_pos
                scx, scy = to_court((sx, sy))

                # Phase 1: wait for shuttle to cross net into player zone
                if not shuttle_crossed:
                    if check_net_crossing(scy):
                        zone = get_zone_from_position(scx, scy)
```

Then in Phase 2 of the same block, change `if check_return(sx):` to `if check_return(scy):`.

- [ ] **Step 5: Build the homography at startup (`main.py` `run`)**

At the very start of `run()` (before `open_camera()`), add a clear failure if the court isn't calibrated:

```python
    try:
        build_homography()
    except ValueError as exc:
        print(f"[CALIBRATION] {exc}")
        return
```

- [ ] **Step 6: Update `utils/display.py` imports**

Replace the `from config.settings import (...)` block (drop `NET_X`, `COURT_ZONE`; add court dims + `court_to_pixel`):

```python
import cv2
import numpy as np
from collections import deque
from config.settings import (
    PLAYER_ZONES, COURT_W, COURT_L,
    COLOR_COURT_ZONE, COLOR_NET, COLOR_SHUTTLE,
    COLOR_ZONE_ACTIVE, COLOR_SCORE_TEXT, COLOR_WEAK_ZONE,
    COLOR_PLAYER_1, COLOR_PLAYER_2,
    SHOW_SKELETON, SHOW_ZONES, SHOW_SHUTTLE_TRAIL, TRAIL_LENGTH
)
from utils.zones import court_to_pixel
```

- [ ] **Step 7: Rewrite the court/net/zone drawing in `utils/display.py`**

Replace `draw_court_zone`, `draw_net`, and `draw_zones` with court-space versions that project court points back to pixels:

```python
def _poly(pts):
    """Court-space corner list -> int pixel polygon for cv2."""
    return np.array([[int(round(x)), int(round(y))]
                     for x, y in (court_to_pixel(p) for p in pts)], dtype=np.int32)


def draw_court_zone(frame):
    """Draw the trainee half-court outline (projected trapezoid)."""
    corners = [(0.0, 0.0), (COURT_W, 0.0), (COURT_W, COURT_L), (0.0, COURT_L)]
    cv2.polylines(frame, [_poly(corners)], True, COLOR_COURT_ZONE, 2)
    lx, ly = court_to_pixel((0.0, 0.0))
    cv2.putText(frame, "COURT", (int(lx) + 5, int(ly) + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_COURT_ZONE, 1)


def draw_net(frame):
    """Draw the net as the segment between the two net corners (y=0 edge)."""
    x1, y1 = court_to_pixel((0.0, 0.0))
    x2, y2 = court_to_pixel((COURT_W, 0.0))
    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLOR_NET, 2)
    cv2.putText(frame, "NET", (int(x1) + 5, int(y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_NET, 1)


def draw_zones(frame, active_zone=None, weak_zones=None):
    """Draw the 6 court-space zones as projected trapezoids."""
    if not SHOW_ZONES:
        return
    weak_zones = weak_zones or []

    for zone_name, (x1, y1, x2, y2) in PLAYER_ZONES.items():
        if zone_name == active_zone:
            color, thickness = COLOR_ZONE_ACTIVE, 2
        elif zone_name in weak_zones:
            color, thickness = COLOR_WEAK_ZONE, 1
        else:
            color, thickness = COLOR_COURT_ZONE, 1

        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        cv2.polylines(frame, [_poly(corners)], True, color, thickness)

        lx, ly = court_to_pixel((x1, y1))
        cv2.putText(frame, zone_name, (int(lx) + 3, int(ly) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
```

- [ ] **Step 8: Rewrite `calibrate.py` for 4-corner capture**

Replace the hint list and the summary section so it captures exactly 4 ordered corners and prints `COURT_CORNERS`. In `on_mouse_click`, replace the `hints` list with:

```python
        hints = [
            "→ 1: NET_LEFT  (net meets the LEFT sideline)",
            "→ 2: NET_RIGHT (net meets the RIGHT sideline)",
            "→ 3: BASELINE_RIGHT (far baseline meets RIGHT sideline)",
            "→ 4: BASELINE_LEFT  (far baseline meets LEFT sideline)",
        ]
```

Replace the printed instructions in `run_calibration` (the "Suggested order" block) with:

```python
    print("\n[CALIBRATION] Click the 4 corners of the TRAINEE's half-court:")
    print("  1. NET_LEFT      (net meets the left sideline)")
    print("  2. NET_RIGHT     (net meets the right sideline)")
    print("  3. BASELINE_RIGHT(far baseline meets the right sideline)")
    print("  4. BASELINE_LEFT (far baseline meets the left sideline)")
    print("\n  Press S to save frame | Q to quit\n")
```

Replace the `# Print summary` section (everything from `print("\n[SUMMARY]...` to the end of `run_calibration`) with:

```python
    print("\n[SUMMARY] Your clicked points:")
    for i, (px, py) in enumerate(click_points):
        print(f"  Point {i+1}: ({px}, {py})")

    if len(click_points) >= 4:
        c = click_points[:4]
        print("\n[COURT_CORNERS suggestion] paste into config/settings.py:")
        print(f"COURT_CORNERS = [{c[0]}, {c[1]}, {c[2]}, {c[3]}]")
    else:
        print("\n[WARN] Need 4 corner clicks to produce COURT_CORNERS.")
```

- [ ] **Step 9: Remove dead constants from `config/settings.py`**

Delete these now-unused lines/blocks: `COURT_ZONE = (...)`, `NET_X = ...`, `PLAYER_SIDE = ...`, `FEEDER_SIDE = ...`, and `PLAYER_OVERLAP_RATIO = ...`. Leave `ANKLE_CONFIDENCE`, `SHUTTLE_CONFIDENCE`, `PERSON_CONFIDENCE`, and everything else.

- [ ] **Step 10: Remove `is_inside_court` from `utils/zones.py`**

Delete the entire `is_inside_court` function (no longer referenced). Also remove `NET_X, COURT_ZONE, PLAYER_OVERLAP_RATIO` from the `from config.settings import (...)` block in `utils/zones.py`, leaving:

```python
from config.settings import (
    PLAYER_ZONES, ANKLE_CONFIDENCE,
    COURT_CORNERS, COURT_W, COURT_L, NET_DEADBAND,
)
```

- [ ] **Step 11: Verify unit tests and imports**

Run: `python -m pytest tests/test_zones.py -v`
Expected: PASS (all tests)

Run: `python -c "import main; print('main imports OK')"`
Expected: `main imports OK` (after ultralytics loads; ignore the model-load prints)

- [ ] **Step 12: Commit**

```bash
git add setup/main.py setup/utils/display.py setup/calibrate.py setup/config/settings.py setup/utils/zones.py
git commit -m "feat(geometry): cut over to court-space; remove side-view constants"
```

---

### Task 4: Docs + verification

Update the README so the calibration/settings docs match the new model, and run the final checks.

**Files:**
- Modify: `README.md` (calibration + settings sections)

- [ ] **Step 1: Update `README.md` calibration section**

In the "Calibrate your camera" section, replace the click-the-4-corners/net instructions with:

```markdown
### 2. Calibrate your camera (feeder-mounted, end-on view)
```bash
python calibrate.py
```
- Click the 4 corners of the **trainee's half-court**, in order:
  1. net meets the **left** sideline
  2. net meets the **right** sideline
  3. far baseline meets the **right** sideline
  4. far baseline meets the **left** sideline
- Copy the printed `COURT_CORNERS = [...]` into `config/settings.py`.

All court geometry (net line, sides, 6 zones) is derived from these 4
corners via a homography, so it works for any camera angle. The person on
the feeder/near side of the net is automatically ignored.
```

Also remove any lingering README references to `COURT_ZONE` / `NET_X` in that section.

- [ ] **Step 2: Full test run**

Run: `python -m pytest tests/test_zones.py -v`
Expected: PASS (all tests)

- [ ] **Step 3: Import smoke check**

Run: `python -c "import main, calibrate, utils.display, utils.zones, config.settings; print('all imports OK')"`
Expected: `all imports OK`

- [ ] **Step 4: Live verification (manual, needs camera)**

1. `python calibrate.py` → click the 4 far-half-court corners → paste `COURT_CORNERS` into `config/settings.py`.
2. `python main.py --difficulty medium`.
3. Confirm: the net line and 6 zones render as the correct trapezoids over the far court; a person standing in each zone is attributed to that zone; a person on the feeder/near side is **not** counted.

- [ ] **Step 5: Commit**

```bash
git add setup/README.md
git commit -m "docs(geometry): update calibration docs for court-space model"
```

---

## Self-Review Notes

- **Spec coverage:** court coord system (Task 1) ✓; homography + calibration order (Tasks 1, 3) ✓; 6 court zones + bug fix (Task 2) ✓; in-court/zone/side/crossing logic incl. feeder-side exclusion (Task 2) ✓; main.py integration (Task 3) ✓; display trapezoids + net segment (Task 3) ✓; settings replacement (Tasks 2, 3) ✓; unit tests + error handling (Tasks 1, 2) ✓; docs (Task 4) ✓.
- **Degenerate-corner + unset-corner errors:** covered by `build_homography` (Task 1) and tested (`test_build_homography_requires_four_corners`).
- **Type consistency:** `player_positions` is `{id: (cx, cy)}` in both producer (`detect_players`, Task 3) and consumer (`get_player_in_zone`, Task 2). Shuttle uses `scy`/`scx` court coords consistently across `check_net_crossing`, `check_return`, `get_zone_from_position`.
- **No placeholders:** every code step contains complete code.
