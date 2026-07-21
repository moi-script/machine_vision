# Player Skill Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accumulate per-player posture, movement/trajectory, and stroke metrics from the existing YOLOv8n-pose stream into a cumulative career profile, and map it to a rule-based five-tier skill ranking (Beginner→Expert).

**Architecture:** Pure geometry helpers (`pose_features.py`) feed a stateful per-player accumulator (`skill_profile.py`, running Welford stats). The drill engine feeds each frame for the attributed athlete into the accumulator, times reaction off the feeder event, and detects swings from wrist kinematics. On drill end, session aggregates merge into the player's cumulative Mongo `skillProfile`, a data-weighted rubric computes the tier, and `[SKILL]` logs prove data flow. A `GET /api/players/{id}/skill` endpoint exposes it.

**Tech Stack:** Python 3.13, OpenCV, Ultralytics YOLO, FastAPI, MongoDB (pymongo), pytest.

## Global Constraints

- Python 3.13 (the repo's env) — no dependency additions; use stdlib + numpy already present.
- Never let analytics crash the drill loop: every engine-side call into the accumulator is wrapped so an exception is swallowed (match the existing `_recognize` pattern).
- Console logs use `print(..., flush=True)` and the `[SKILL]` prefix, matching the existing `[SHUTTLE]` / `[RECO]` style. Per-sample logging is throttled to ~1/sec.
- Only accumulate for a **resolved player id** (single-trainee attribution or a recognized face) — never a raw track id, since merge writes to a player doc by id.
- Run tests with: `python -m pytest tests/ --ignore=tests/test_shuttle_worker.py`
- All new tests live under `setup/tests/`; run commands assume CWD is `C:\thesis\setup`.

---

### Task 1: Skill-ranking config constants

**Files:**
- Modify: `config/settings.py` (append a new section at end of file)

**Interfaces:**
- Produces (module-level constants in `config.settings`):
  - `SKILL_KP_CONF: float`, `SKILL_BBOX_MIN_H: float`, `SKILL_REACT_DIST: float`, `SKILL_SWING_SPEED: float`, `SKILL_MIN_SHOTS: int`
  - `SKILL_REFS: dict[str, dict]` — normalization refs per metric
  - `SKILL_FAMILY_WEIGHTS: dict[str, float]`
  - `SKILL_FAMILY_MIN_SAMPLES: dict[str, int]`
  - `SKILL_TIERS: list[tuple[float, str]]`

- [ ] **Step 1: Append the constants block**

Append to `config/settings.py`:

```python

# ============================================================
# Player skill ranking (rule-based rubric) — all tunable
# See docs/superpowers/specs/2026-07-22-player-skill-ranking-design.md
# ============================================================

# --- Sampling gates ---
SKILL_KP_CONF    = 0.5    # min keypoint confidence to use a fine (posture/stroke) point
SKILL_BBOX_MIN_H = 200.0  # min person bbox height in px to sample posture/stroke (distance gate)
SKILL_REACT_DIST = 20.0   # court-space ankle displacement that counts as "started moving"
SKILL_SWING_SPEED = 3.0   # shoulder-relative wrist speed (units/sec) rising-edge = a swing
SKILL_MIN_SHOTS  = 20     # min accumulated shots before any tier is assigned (else Unranked)

# --- Per-metric normalization references ---
# type "monotonic": linear lo->hi mapped to 0->100 (invert=True flips it)
# type "target":    full score inside [target +/- tol], decaying linearly outside
# type "consistency": std mapped hi->0 / 0->100 (lower std = higher score)
SKILL_REFS = {
    # movement family
    "move_speed": {"type": "monotonic", "lo": 0.5, "hi": 6.0},
    "coverage":   {"type": "monotonic", "lo": 1.0, "hi": 6.0},
    "reaction":   {"type": "monotonic", "lo": 0.3, "hi": 1.5, "invert": True},
    # accuracy family
    "accuracy":   {"type": "monotonic", "lo": 20.0, "hi": 90.0},
    # stroke family
    "swing_consistency": {"type": "consistency", "hi": 4.0},
    # posture family
    "knee":       {"type": "target", "target": 150.0, "tol": 40.0},
    "stance":     {"type": "target", "target": 1.4, "tol": 0.9},
    "posture_consistency": {"type": "consistency", "hi": 25.0},
}

# --- Family weights (scaled by data sufficiency at eval time) ---
SKILL_FAMILY_WEIGHTS = {"move": 0.30, "accuracy": 0.30, "stroke": 0.20, "posture": 0.20}

# --- Min reliable samples for a family to count toward the composite ---
SKILL_FAMILY_MIN_SAMPLES = {"move": 30, "accuracy": 20, "stroke": 5, "posture": 20}

# --- Tier bands: (min_composite_inclusive, name), highest first ---
SKILL_TIERS = [
    (80.0, "Expert"),
    (60.0, "Advanced"),
    (40.0, "Intermediate"),
    (20.0, "Novice"),
    (0.0,  "Beginner"),
]
```

- [ ] **Step 2: Verify it imports**

Run: `python -c "from config import settings as s; print(s.SKILL_TIERS[0], len(s.SKILL_REFS))"`
Expected: `(80.0, 'Expert') 8`

- [ ] **Step 3: Commit**

```bash
git add config/settings.py
git commit -m "feat(config): add tunable skill-ranking constants"
```

---

### Task 2: Pose geometry helpers (`pose_features.py`)

**Files:**
- Create: `utils/pose_features.py`
- Test: `tests/test_pose_features.py`

**Interfaces:**
- Produces:
  - `point(keypoints, idx, min_conf) -> tuple[float, float] | None`
  - `angle(a, b, c) -> float | None` — degrees at vertex `b`
  - `knee_angle(keypoints, min_conf=0.5) -> float | None`
  - `torso_lean(keypoints, min_conf=0.5) -> float | None` — degrees from vertical
  - `stance_width(keypoints, min_conf=0.5) -> float | None` — ankle sep / shoulder sep
  - `max_wrist_speed(kp_now, kp_prev, dt, min_conf=0.5) -> float | None`
  - `bbox_height(box) -> float`
  - `keypoints` is a list of 17 `[x, y, conf]` (COCO order, as `results[0].keypoints.data[i].tolist()`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_pose_features.py`:

```python
import math
import pytest
from utils import pose_features as pf


def _kp(overrides):
    """17 COCO keypoints, all confident at origin unless overridden by index."""
    kps = [[0.0, 0.0, 1.0] for _ in range(17)]
    for idx, val in overrides.items():
        kps[idx] = val
    return kps


def test_angle_right_angle():
    assert pf.angle((0, 1), (0, 0), (1, 0)) == pytest.approx(90.0, abs=1e-3)


def test_angle_straight_line():
    assert pf.angle((0, 2), (0, 1), (0, 0)) == pytest.approx(180.0, abs=1e-3)


def test_point_gated_by_confidence():
    kps = _kp({5: [10.0, 20.0, 0.2]})
    assert pf.point(kps, 5, 0.5) is None
    assert pf.point(kps, 5, 0.1) == (10.0, 20.0)


def test_knee_angle_straight_leg_is_180():
    # left hip(11), knee(13), ankle(15) vertically aligned -> straight leg
    kps = _kp({11: [0.0, 0.0, 1.0], 13: [0.0, 1.0, 1.0], 15: [0.0, 2.0, 1.0]})
    assert pf.knee_angle(kps) == pytest.approx(180.0, abs=1e-2)


def test_knee_angle_none_when_low_conf():
    kps = _kp({11: [0.0, 0.0, 0.1], 13: [0.0, 1.0, 0.1], 15: [0.0, 2.0, 0.1]})
    assert pf.knee_angle(kps) is None


def test_stance_width_ratio():
    # shoulders(5,6) 1 apart in x; ankles(15,16) 2 apart -> ratio 2.0
    kps = _kp({5: [0.0, 0.0, 1.0], 6: [1.0, 0.0, 1.0],
               15: [0.0, 5.0, 1.0], 16: [2.0, 5.0, 1.0]})
    assert pf.stance_width(kps) == pytest.approx(2.0, abs=1e-6)


def test_max_wrist_speed_shoulder_relative():
    # right wrist(10) moves +3 in x over dt=1; shoulder(6) static -> speed 3
    prev = _kp({6: [0.0, 0.0, 1.0], 10: [0.0, 0.0, 1.0]})
    now  = _kp({6: [0.0, 0.0, 1.0], 10: [3.0, 0.0, 1.0]})
    assert pf.max_wrist_speed(now, prev, 1.0) == pytest.approx(3.0, abs=1e-6)


def test_bbox_height():
    assert pf.bbox_height([10, 20, 50, 120]) == 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pose_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.pose_features'`

- [ ] **Step 3: Write the implementation**

Create `utils/pose_features.py`:

```python
"""Pure geometry helpers over YOLOv8 pose keypoints.

`keypoints` is a list of 17 [x, y, confidence] in COCO order. Every helper is
confidence-gated: it returns None when a required keypoint is below `min_conf`,
so callers never build a metric from a jittery low-confidence point."""
from __future__ import annotations
import math

# COCO keypoint indices we use
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


def point(keypoints, idx, min_conf):
    """(x, y) of keypoint idx if its confidence >= min_conf, else None."""
    kp = keypoints[idx]
    if float(kp[2]) < min_conf:
        return None
    return float(kp[0]), float(kp[1])


def angle(a, b, c):
    """Interior angle in degrees at vertex b, formed by a-b-c. None if degenerate."""
    if a is None or b is None or c is None:
        return None
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return None
    cosv = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))


def _side_knee(keypoints, hip_i, knee_i, ankle_i, min_conf):
    return angle(point(keypoints, hip_i, min_conf),
                 point(keypoints, knee_i, min_conf),
                 point(keypoints, ankle_i, min_conf))


def knee_angle(keypoints, min_conf=0.5):
    """Mean hip-knee-ankle angle over whichever legs are confidently visible."""
    vals = [v for v in (
        _side_knee(keypoints, L_HIP, L_KNEE, L_ANKLE, min_conf),
        _side_knee(keypoints, R_HIP, R_KNEE, R_ANKLE, min_conf),
    ) if v is not None]
    return sum(vals) / len(vals) if vals else None


def _midpoint(p, q):
    return ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)


def torso_lean(keypoints, min_conf=0.5):
    """Angle (deg) of the shoulder-mid -> hip-mid line away from vertical."""
    ls = point(keypoints, L_SHOULDER, min_conf)
    rs = point(keypoints, R_SHOULDER, min_conf)
    lh = point(keypoints, L_HIP, min_conf)
    rh = point(keypoints, R_HIP, min_conf)
    if None in (ls, rs, lh, rh):
        return None
    sm = _midpoint(ls, rs)
    hm = _midpoint(lh, rh)
    dx = hm[0] - sm[0]
    dy = hm[1] - sm[1]
    if dx == 0 and dy == 0:
        return None
    # angle from the vertical axis (0 deg = perfectly upright)
    return abs(math.degrees(math.atan2(dx, dy)))


def stance_width(keypoints, min_conf=0.5):
    """Ankle horizontal separation divided by shoulder separation (body-scaled)."""
    la = point(keypoints, L_ANKLE, min_conf)
    ra = point(keypoints, R_ANKLE, min_conf)
    ls = point(keypoints, L_SHOULDER, min_conf)
    rs = point(keypoints, R_SHOULDER, min_conf)
    if None in (la, ra, ls, rs):
        return None
    shoulder = abs(ls[0] - rs[0])
    if shoulder == 0:
        return None
    return abs(la[0] - ra[0]) / shoulder


def _wrist_speed(kp_now, kp_prev, dt, wrist_i, shoulder_i, min_conf):
    wn = point(kp_now, wrist_i, min_conf)
    sn = point(kp_now, shoulder_i, min_conf)
    wp = point(kp_prev, wrist_i, min_conf)
    sp = point(kp_prev, shoulder_i, min_conf)
    if None in (wn, sn, wp, sp) or dt <= 0:
        return None
    # shoulder-relative displacement removes whole-body translation
    rel_now = (wn[0] - sn[0], wn[1] - sn[1])
    rel_prev = (wp[0] - sp[0], wp[1] - sp[1])
    return math.hypot(rel_now[0] - rel_prev[0], rel_now[1] - rel_prev[1]) / dt


def max_wrist_speed(kp_now, kp_prev, dt, min_conf=0.5):
    """Max shoulder-relative wrist speed across both arms, or None."""
    vals = [v for v in (
        _wrist_speed(kp_now, kp_prev, dt, L_WRIST, L_SHOULDER, min_conf),
        _wrist_speed(kp_now, kp_prev, dt, R_WRIST, R_SHOULDER, min_conf),
    ) if v is not None]
    return max(vals) if vals else None


def bbox_height(box):
    """Pixel height of an [x1, y1, x2, y2] box."""
    return float(box[3]) - float(box[1])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pose_features.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add utils/pose_features.py tests/test_pose_features.py
git commit -m "feat(analytics): pose geometry helpers with confidence gating"
```

---

### Task 3: Running-stat primitive (`RunningStat`)

**Files:**
- Create: `utils/skill_profile.py`
- Test: `tests/test_skill_profile.py`

**Interfaces:**
- Produces:
  - `class RunningStat` with attrs `n:int, sum:float, sumsq:float`; methods `add(x)`, `merge(other)`, `mean()->float`, `var()->float`, `std()->float`, `to_dict()->dict`; classmethod `from_dict(d)->RunningStat`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_skill_profile.py`:

```python
import math
import pytest
from utils.skill_profile import RunningStat


def test_running_stat_mean_and_std():
    rs = RunningStat()
    for x in (2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0):
        rs.add(x)
    assert rs.n == 8
    assert rs.mean() == pytest.approx(5.0)
    assert rs.std() == pytest.approx(2.0)  # population std


def test_running_stat_empty_is_zero():
    rs = RunningStat()
    assert rs.mean() == 0.0 and rs.std() == 0.0


def test_running_stat_merge_equivalent_to_combined():
    a = RunningStat(); [a.add(x) for x in (1, 2, 3)]
    b = RunningStat(); [b.add(x) for x in (4, 5, 6)]
    a.merge(b)
    combined = RunningStat(); [combined.add(x) for x in (1, 2, 3, 4, 5, 6)]
    assert a.n == combined.n
    assert a.mean() == pytest.approx(combined.mean())
    assert a.std() == pytest.approx(combined.std())


def test_running_stat_roundtrip_dict():
    a = RunningStat(); [a.add(x) for x in (1.5, 2.5, 3.5)]
    b = RunningStat.from_dict(a.to_dict())
    assert b.n == a.n and b.mean() == pytest.approx(a.mean())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_skill_profile.py -v`
Expected: FAIL — `ImportError: cannot import name 'RunningStat'`

- [ ] **Step 3: Write the implementation**

Create `utils/skill_profile.py` with:

```python
"""Per-player skill accumulation + rule-based rubric.

RunningStat accumulates count/sum/sum-of-squares so per-session aggregates can
be merged into a cumulative career profile without keeping raw samples."""
from __future__ import annotations
import math


class RunningStat:
    def __init__(self, n: int = 0, sum: float = 0.0, sumsq: float = 0.0):
        self.n = n
        self.sum = sum
        self.sumsq = sumsq

    def add(self, x: float) -> None:
        x = float(x)
        self.n += 1
        self.sum += x
        self.sumsq += x * x

    def merge(self, other: "RunningStat") -> None:
        self.n += other.n
        self.sum += other.sum
        self.sumsq += other.sumsq

    def mean(self) -> float:
        return self.sum / self.n if self.n else 0.0

    def var(self) -> float:
        if self.n == 0:
            return 0.0
        m = self.mean()
        return max(0.0, self.sumsq / self.n - m * m)

    def std(self) -> float:
        return math.sqrt(self.var())

    def to_dict(self) -> dict:
        return {"n": self.n, "sum": self.sum, "sumsq": self.sumsq}

    @classmethod
    def from_dict(cls, d: dict) -> "RunningStat":
        d = d or {}
        return cls(int(d.get("n", 0)), float(d.get("sum", 0.0)),
                   float(d.get("sumsq", 0.0)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_skill_profile.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add utils/skill_profile.py tests/test_skill_profile.py
git commit -m "feat(analytics): RunningStat mergeable accumulator primitive"
```

---

### Task 4: `SkillAccumulator` (session-level accumulation)

**Files:**
- Modify: `utils/skill_profile.py`
- Test: `tests/test_skill_profile.py` (append)

**Interfaces:**
- Consumes: `RunningStat` (Task 3); `utils.pose_features` (Task 2); `utils.zones.get_zone_from_position`.
- Produces:
  - `class SkillAccumulator(kp_conf, bbox_min_h, react_dist, swing_speed)`
  - `add_frame(keypoints, box, court_ankle, now)` — `court_ankle` is `(cx, cy)` or None
  - `on_feeder_fired(now, zone)`
  - `on_shot()`, `on_score()`
  - `snapshot() -> dict` shaped:
    `{"sampleCounts": {move, posture, stroke, accuracy}, "coverage": [zones],
      "shots": int, "scores": int,
      "stats": {"move_speed":{...}, "reaction":{...}, "knee":{...},
                "stance":{...}, "torso":{...}, "swing_peak":{...}},
      "swings": int}`
  - staticmethod `merge_snapshot(cumulative: dict, snap: dict) -> dict`

- [ ] **Step 1: Write the failing test (append to `tests/test_skill_profile.py`)**

```python
from utils.skill_profile import SkillAccumulator


def _kp(overrides):
    kps = [[0.0, 0.0, 1.0] for _ in range(17)]
    for idx, val in overrides.items():
        kps[idx] = val
    return kps


def _acc():
    return SkillAccumulator(kp_conf=0.5, bbox_min_h=100.0,
                            react_dist=20.0, swing_speed=3.0)


def test_reaction_recorded_from_feeder_event():
    acc = _acc()
    box = [0, 0, 50, 200]  # tall enough
    kps = _kp({})
    acc.add_frame(kps, box, court_ankle=(0.0, 0.0), now=10.0)
    acc.on_feeder_fired(now=10.0, zone="back-left")
    # player hasn't moved yet
    acc.add_frame(kps, box, court_ankle=(5.0, 0.0), now=10.2)
    # now displacement exceeds react_dist (20)
    acc.add_frame(kps, box, court_ankle=(30.0, 0.0), now=10.5)
    snap = acc.snapshot()
    assert snap["stats"]["reaction"]["n"] == 1
    # reaction lag ~ 0.5s (from feeder at 10.0)
    assert snap["stats"]["reaction"]["sum"] == pytest.approx(0.5, abs=1e-6)


def test_posture_gated_out_when_bbox_too_small():
    acc = _acc()
    small_box = [0, 0, 50, 50]  # height 50 < 100 gate
    kps = _kp({11: [0, 0, 1], 13: [0, 1, 1], 15: [0, 2, 1]})
    acc.add_frame(kps, small_box, court_ankle=(0.0, 0.0), now=1.0)
    assert acc.snapshot()["sampleCounts"]["posture"] == 0


def test_posture_sampled_when_bbox_large_enough():
    acc = _acc()
    big_box = [0, 0, 50, 300]
    kps = _kp({11: [0, 0, 1], 13: [0, 1, 1], 15: [0, 2, 1]})
    acc.add_frame(kps, big_box, court_ankle=(0.0, 0.0), now=1.0)
    assert acc.snapshot()["sampleCounts"]["posture"] >= 1


def test_swing_detected_on_rising_then_falling_edge():
    acc = _acc()
    big_box = [0, 0, 50, 300]
    # frame1 wrist still, frame2 fast (rising edge > 3), frame3 slow (falling)
    f1 = _kp({6: [0, 0, 1], 10: [0, 0, 1]})
    f2 = _kp({6: [0, 0, 1], 10: [5, 0, 1]})   # speed 5 over dt 1
    f3 = _kp({6: [0, 0, 1], 10: [5, 0, 1]})   # speed 0
    acc.add_frame(f1, big_box, None, now=1.0)
    acc.add_frame(f2, big_box, None, now=2.0)
    acc.add_frame(f3, big_box, None, now=3.0)
    assert acc.snapshot()["swings"] == 1


def test_merge_snapshot_accumulates_counts():
    a = _acc(); a.on_shot(); a.on_shot(); a.on_score()
    snap1 = a.snapshot()
    b = _acc(); b.on_shot(); b.on_score()
    snap2 = b.snapshot()
    merged = SkillAccumulator.merge_snapshot({}, snap1)
    merged = SkillAccumulator.merge_snapshot(merged, snap2)
    assert merged["shots"] == 3
    assert merged["scores"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_skill_profile.py -k "reaction or posture or swing or merge_snapshot" -v`
Expected: FAIL — `ImportError: cannot import name 'SkillAccumulator'`

- [ ] **Step 3: Write the implementation (append to `utils/skill_profile.py`)**

```python
import math as _math
from utils import pose_features as pf


def _dist(a, b):
    return _math.hypot(a[0] - b[0], a[1] - b[1])


class SkillAccumulator:
    """Accumulates one drill session's skill metrics for one athlete."""

    def __init__(self, kp_conf, bbox_min_h, react_dist, swing_speed):
        self.kp_conf = kp_conf
        self.bbox_min_h = bbox_min_h
        self.react_dist = react_dist
        self.swing_speed = swing_speed

        self.stats = {k: RunningStat() for k in
                      ("move_speed", "reaction", "knee", "stance", "torso",
                       "swing_peak")}
        self.coverage = set()
        self.shots = 0
        self.scores = 0
        self.swings = 0
        self.counts = {"move": 0, "posture": 0, "stroke": 0, "accuracy": 0}

        # temporal state
        self._prev_ankle = None
        self._prev_now = None
        self._prev_kps = None
        # pending reaction: (feeder_ts, baseline_ankle) or None
        self._pending = None
        # swing edge state
        self._in_swing = False
        self._cur_peak = 0.0

    # ---- event hooks ----
    def on_feeder_fired(self, now, zone):
        base = self._prev_ankle
        self._pending = (now, base)

    def on_shot(self):
        self.shots += 1
        self.counts["accuracy"] += 1

    def on_score(self):
        self.scores += 1

    # ---- per-frame ----
    def add_frame(self, keypoints, box, court_ankle, now):
        from utils.zones import get_zone_from_position
        bbox_h = pf.bbox_height(box)

        # ---- Tier A: movement (always, needs court position) ----
        if court_ankle is not None:
            zone = get_zone_from_position(court_ankle[0], court_ankle[1])
            if zone is not None:
                self.coverage.add(zone)
            if self._prev_ankle is not None and self._prev_now is not None:
                dt = now - self._prev_now
                if dt > 0:
                    speed = _dist(court_ankle, self._prev_ankle) / dt
                    self.stats["move_speed"].add(speed)
                    self.counts["move"] += 1
            # reaction: displacement from feeder-time baseline
            if self._pending is not None:
                _ts, base = self._pending
                if base is not None and _dist(court_ankle, base) >= self.react_dist:
                    self.stats["reaction"].add(now - _ts)
                    self._pending = None
            self._prev_ankle = court_ankle
            self._prev_now = now

        near = bbox_h >= self.bbox_min_h

        # ---- Tier B: posture (gated by distance) ----
        if near:
            sampled = False
            knee = pf.knee_angle(keypoints, self.kp_conf)
            if knee is not None:
                self.stats["knee"].add(knee); sampled = True
            stance = pf.stance_width(keypoints, self.kp_conf)
            if stance is not None:
                self.stats["stance"].add(stance); sampled = True
            torso = pf.torso_lean(keypoints, self.kp_conf)
            if torso is not None:
                self.stats["torso"].add(torso); sampled = True
            if sampled:
                self.counts["posture"] += 1

        # ---- Tier C: stroke (gated by distance) ----
        if near and self._prev_kps is not None and self._prev_now is not None:
            dt = now - self._prev_now if self._prev_now is not None else 0.0
            # NOTE: _prev_now already advanced above if court_ankle present;
            # recompute a stroke dt from a dedicated prev timestamp.
        self._update_swing(keypoints, now)

        self._prev_kps = keypoints

    def _update_swing(self, keypoints, now):
        if self._prev_kps is None or self._prev_now_stroke() is None:
            self._prev_now_stroke_set(now)
            return
        dt = now - self._stroke_prev_now
        self._stroke_prev_now = now
        if dt <= 0:
            return
        ws = pf.max_wrist_speed(keypoints, self._prev_kps, dt, self.kp_conf)
        if ws is None:
            return
        self.counts["stroke"] += 1
        if not self._in_swing and ws >= self.swing_speed:
            self._in_swing = True
            self._cur_peak = ws
        elif self._in_swing:
            self._cur_peak = max(self._cur_peak, ws)
            if ws < self.swing_speed:
                self.stats["swing_peak"].add(self._cur_peak)
                self.swings += 1
                self._in_swing = False

    # dedicated stroke timestamp (separate from movement's _prev_now)
    _stroke_prev_now = None
    def _prev_now_stroke(self):
        return self._stroke_prev_now
    def _prev_now_stroke_set(self, now):
        self._stroke_prev_now = now

    # ---- output ----
    def snapshot(self) -> dict:
        return {
            "sampleCounts": dict(self.counts),
            "coverage": sorted(self.coverage),
            "shots": self.shots,
            "scores": self.scores,
            "swings": self.swings,
            "stats": {k: v.to_dict() for k, v in self.stats.items()},
        }

    @staticmethod
    def merge_snapshot(cumulative: dict, snap: dict) -> dict:
        cumulative = cumulative or {}
        out = {
            "sampleCounts": dict(cumulative.get("sampleCounts", {})),
            "coverage": set(cumulative.get("coverage", [])),
            "shots": cumulative.get("shots", 0),
            "scores": cumulative.get("scores", 0),
            "swings": cumulative.get("swings", 0),
            "stats": {},
        }
        for fam, c in snap["sampleCounts"].items():
            out["sampleCounts"][fam] = out["sampleCounts"].get(fam, 0) + c
        out["coverage"].update(snap.get("coverage", []))
        out["shots"] += snap.get("shots", 0)
        out["scores"] += snap.get("scores", 0)
        out["swings"] += snap.get("swings", 0)
        keys = set(cumulative.get("stats", {})) | set(snap.get("stats", {}))
        for k in keys:
            a = RunningStat.from_dict(cumulative.get("stats", {}).get(k))
            a.merge(RunningStat.from_dict(snap.get("stats", {}).get(k)))
            out["stats"][k] = a.to_dict()
        out["coverage"] = sorted(out["coverage"])
        return out
```

Note the `_update_swing` uses a dedicated `_stroke_prev_now` so stroke timing is independent of the movement branch's `_prev_now`. Remove the dead `if near and self._prev_kps ...` block that only sets `dt` — it was a scaffolding artifact:

Replace this block inside `add_frame`:

```python
        # ---- Tier C: stroke (gated by distance) ----
        if near and self._prev_kps is not None and self._prev_now is not None:
            dt = now - self._prev_now if self._prev_now is not None else 0.0
            # NOTE: _prev_now already advanced above if court_ankle present;
            # recompute a stroke dt from a dedicated prev timestamp.
        self._update_swing(keypoints, now)
```

with:

```python
        # ---- Tier C: stroke (gated by distance) ----
        if near:
            self._update_swing(keypoints, now)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_skill_profile.py -v`
Expected: PASS (all previous + 5 new)

- [ ] **Step 5: Commit**

```bash
git add utils/skill_profile.py tests/test_skill_profile.py
git commit -m "feat(analytics): SkillAccumulator session metrics + merge"
```

---

### Task 5: `evaluate_rubric` (profile → tier)

**Files:**
- Modify: `utils/skill_profile.py`
- Test: `tests/test_skill_profile.py` (append)

**Interfaces:**
- Consumes: `RunningStat`; `config.settings` constants from Task 1.
- Produces:
  - `evaluate_rubric(profile: dict, config: dict | None = None) -> dict` returning
    `{"composite": float|None, "tier": str, "families": {move, accuracy, stroke, posture}, "breakdown": {metric: score}}`
  - `default_config() -> dict` assembled from `config.settings`.

- [ ] **Step 1: Write the failing test (append to `tests/test_skill_profile.py`)**

```python
from utils.skill_profile import evaluate_rubric, SkillAccumulator


def test_rubric_unranked_below_min_shots():
    prof = {"shots": 3, "scores": 2, "sampleCounts": {}, "stats": {}, "coverage": []}
    out = evaluate_rubric(prof)
    assert out["tier"] == "Unranked"
    assert out["composite"] is None


def test_rubric_high_accuracy_maps_to_a_real_tier():
    # 100 shots, 85 scores -> accuracy 85% -> high accuracy family score
    prof = {"shots": 100, "scores": 85,
            "sampleCounts": {"accuracy": 100, "move": 0, "stroke": 0, "posture": 0},
            "coverage": [], "stats": {}}
    out = evaluate_rubric(prof)
    assert out["composite"] is not None
    assert out["tier"] in {"Beginner", "Novice", "Intermediate", "Advanced", "Expert"}
    assert out["families"]["accuracy"] > 80


def test_rubric_skips_thin_families_via_redistribution():
    # only accuracy has data; move/stroke/posture are empty and must not drag it
    prof = {"shots": 100, "scores": 90,
            "sampleCounts": {"accuracy": 100, "move": 0, "stroke": 0, "posture": 0},
            "coverage": [], "stats": {}}
    out = evaluate_rubric(prof)
    # composite should equal the accuracy family score (only qualifying family)
    assert out["composite"] == pytest.approx(out["families"]["accuracy"], abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_skill_profile.py -k rubric -v`
Expected: FAIL — `ImportError: cannot import name 'evaluate_rubric'`

- [ ] **Step 3: Write the implementation (append to `utils/skill_profile.py`)**

```python
from config import settings as _cfg


def default_config() -> dict:
    return {
        "refs": _cfg.SKILL_REFS,
        "weights": _cfg.SKILL_FAMILY_WEIGHTS,
        "family_min": _cfg.SKILL_FAMILY_MIN_SAMPLES,
        "tiers": _cfg.SKILL_TIERS,
        "min_shots": _cfg.SKILL_MIN_SHOTS,
    }


def _norm(value, ref):
    t = ref["type"]
    if t == "monotonic":
        lo, hi = ref["lo"], ref["hi"]
        frac = 0.0 if hi == lo else (value - lo) / (hi - lo)
        frac = max(0.0, min(1.0, frac))
        if ref.get("invert"):
            frac = 1.0 - frac
        return 100.0 * frac
    if t == "target":
        d = abs(value - ref["target"])
        return 100.0 * max(0.0, 1.0 - d / ref["tol"])
    if t == "consistency":
        hi = ref["hi"]
        frac = 0.0 if hi == 0 else 1.0 - (value / hi)
        return 100.0 * max(0.0, min(1.0, frac))
    return 0.0


def _mean(stats, key):
    return RunningStat.from_dict(stats.get(key)).mean()


def _std(stats, key):
    return RunningStat.from_dict(stats.get(key)).std()


def evaluate_rubric(profile: dict, config: dict | None = None) -> dict:
    cfg = config or default_config()
    refs = cfg["refs"]
    stats = profile.get("stats", {})
    counts = profile.get("sampleCounts", {})
    shots = profile.get("shots", 0)
    scores = profile.get("scores", 0)

    breakdown = {}

    # ---- family metric scores ----
    def score(metric, value):
        s = _norm(value, refs[metric])
        breakdown[metric] = round(s, 1)
        return s

    # movement
    move_parts = []
    if counts.get("move", 0) > 0:
        move_parts.append(score("move_speed", _mean(stats, "move_speed")))
        move_parts.append(score("coverage", len(profile.get("coverage", []))))
        if RunningStat.from_dict(stats.get("reaction")).n > 0:
            move_parts.append(score("reaction", _mean(stats, "reaction")))
    # accuracy
    acc_parts = []
    if shots > 0:
        acc_parts.append(score("accuracy", 100.0 * scores / shots))
    # stroke
    stroke_parts = []
    if RunningStat.from_dict(stats.get("swing_peak")).n > 0:
        stroke_parts.append(score("swing_consistency", _std(stats, "swing_peak")))
    # posture
    post_parts = []
    if counts.get("posture", 0) > 0:
        if RunningStat.from_dict(stats.get("knee")).n > 0:
            post_parts.append(score("knee", _mean(stats, "knee")))
        if RunningStat.from_dict(stats.get("stance")).n > 0:
            post_parts.append(score("stance", _mean(stats, "stance")))
        if RunningStat.from_dict(stats.get("knee")).n > 0:
            post_parts.append(score("posture_consistency", _std(stats, "knee")))

    families = {
        "move": sum(move_parts) / len(move_parts) if move_parts else None,
        "accuracy": sum(acc_parts) / len(acc_parts) if acc_parts else None,
        "stroke": sum(stroke_parts) / len(stroke_parts) if stroke_parts else None,
        "posture": sum(post_parts) / len(post_parts) if post_parts else None,
    }

    # ---- min-data gate ----
    if shots < cfg["min_shots"]:
        return {"composite": None, "tier": "Unranked",
                "families": {k: (round(v, 1) if v is not None else None)
                             for k, v in families.items()},
                "breakdown": breakdown}

    # ---- data-weighted composite (redistribute thin families) ----
    fam_min = cfg["family_min"]
    total_w = 0.0
    acc = 0.0
    for fam, fam_score in families.items():
        if fam_score is None:
            continue
        if counts.get(fam, 0) < fam_min.get(fam, 0) and fam != "accuracy":
            continue
        if fam == "accuracy" and shots < fam_min.get("accuracy", 0):
            continue
        w = cfg["weights"].get(fam, 0.0)
        acc += w * fam_score
        total_w += w

    if total_w == 0:
        return {"composite": None, "tier": "Unranked",
                "families": {k: (round(v, 1) if v is not None else None)
                             for k, v in families.items()},
                "breakdown": breakdown}

    composite = acc / total_w
    tier = "Beginner"
    for cutoff, name in cfg["tiers"]:
        if composite >= cutoff:
            tier = name
            break

    return {"composite": round(composite, 1), "tier": tier,
            "families": {k: (round(v, 1) if v is not None else None)
                         for k, v in families.items()},
            "breakdown": breakdown}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_skill_profile.py -v`
Expected: PASS (all tasks 3-5 tests)

- [ ] **Step 5: Commit**

```bash
git add utils/skill_profile.py tests/test_skill_profile.py
git commit -m "feat(analytics): data-weighted rubric -> five-tier ranking"
```

---

### Task 6: Engine integration + `[SKILL]` logging

**Files:**
- Modify: `app/engine.py`
- Test: `tests/test_engine_skill.py` (create)

**Interfaces:**
- Consumes: `SkillAccumulator`, `SkillAccumulator.merge_snapshot`, `evaluate_rubric`, `default_config` (Tasks 4–5); existing `self._athlete_id`, `self._fire_feeder`, `to_court`, `get_ankle_position`.
- Produces: `DrillEngine._skill_accs: dict[str, SkillAccumulator]`; methods `_skill_for(aid)`, `_flush_skill_profiles()`; a Mongo `skillProfile` + `skillHistory` write per resolved athlete on drill end.

- [ ] **Step 1: Write the failing test**

Create `tests/test_engine_skill.py`:

```python
from app.engine import DrillEngine
from app import db


def test_flush_writes_skillprofile_to_mongo():
    # seed a player
    pid = "pskilltest1"
    db.players().update_one({"_id": pid},
        {"$set": {"_id": pid, "name": "SKILLTEST",
                  "stats": {"totalShots": 0, "totalScores": 0}}},
        upsert=True)
    try:
        eng = DrillEngine()
        acc = eng._skill_for(pid)
        for _ in range(60):
            acc.on_shot()
        for _ in range(45):
            acc.on_score()
        eng._flush_skill_profiles()
        doc = db.players().find_one({"_id": pid})
        assert "skillProfile" in doc
        assert doc["skillProfile"]["computed"]["tier"] in {
            "Beginner", "Novice", "Intermediate", "Advanced", "Expert"}
        assert doc["skillProfile"]["cumulative"]["shots"] == 60
        assert len(doc["skillHistory"]) == 1
    finally:
        db.players().delete_one({"_id": pid})


def test_skill_for_reuses_same_accumulator():
    eng = DrillEngine()
    a = eng._skill_for("pX")
    b = eng._skill_for("pX")
    assert a is b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_engine_skill.py -v`
Expected: FAIL — `AttributeError: 'DrillEngine' object has no attribute '_skill_for'`

- [ ] **Step 3a: Add imports and init state**

In `app/engine.py`, add to the imports near the top (after `from config.settings import (...)` block):

```python
from utils.skill_profile import SkillAccumulator, evaluate_rubric, default_config
from config.settings import (
    SKILL_KP_CONF, SKILL_BBOX_MIN_H, SKILL_REACT_DIST, SKILL_SWING_SPEED,
)
```

In `DrillEngine.__init__`, after the `self._last_reco = {}` line, add:

```python
        # Per-athlete skill accumulation for this drill (Task 6).
        self._skill_accs: dict[str, SkillAccumulator] = {}
        self._skill_log_last = 0.0
```

- [ ] **Step 3b: Add helper methods**

Add these methods to `DrillEngine` (e.g. right after `_recognize`):

```python
    def _skill_for(self, aid: str) -> SkillAccumulator:
        acc = self._skill_accs.get(aid)
        if acc is None:
            acc = SkillAccumulator(SKILL_KP_CONF, SKILL_BBOX_MIN_H,
                                   SKILL_REACT_DIST, SKILL_SWING_SPEED)
            self._skill_accs[aid] = acc
        return acc

    def _is_real_athlete(self, track_id) -> bool:
        # Only accumulate for a resolved player id (recognized face or single
        # -trainee attribution), never a raw track id.
        return self._athlete_id(track_id) != str(track_id)

    def _skill_add_frame(self, track_id, box, keypoints, court_ankle, now):
        try:
            if not self._is_real_athlete(track_id):
                return
            aid = self._athlete_id(track_id)
            self._skill_for(aid).add_frame(keypoints, box, court_ankle, now)
        except Exception as exc:  # noqa: BLE001 - analytics must never kill the loop
            print(f"[SKILL] add_frame error {exc!r}", flush=True)

    def _flush_skill_profiles(self) -> None:
        """Merge each athlete's session aggregates into their cumulative Mongo
        skillProfile, recompute the tier, append a history snapshot, log it."""
        from datetime import datetime, timezone
        from app import db
        cfg = default_config()
        for aid, acc in self._skill_accs.items():
            try:
                snap = acc.snapshot()
                doc = db.players().find_one({"_id": aid}) or {}
                prev = (doc.get("skillProfile") or {}).get("cumulative", {})
                cumulative = SkillAccumulator.merge_snapshot(prev, snap)
                result = evaluate_rubric(cumulative, cfg)
                now_iso = datetime.now(timezone.utc).isoformat()
                computed = {**result, "updatedAt": now_iso}
                db.players().update_one({"_id": aid}, {
                    "$set": {"skillProfile": {"cumulative": cumulative,
                                              "computed": computed}},
                    "$push": {"skillHistory": {
                        "at": now_iso, "sessionId": self._session_id,
                        "composite": result["composite"], "tier": result["tier"],
                        "families": result["families"]}},
                })
                print(f"[SKILL] {aid} session merged -> cumulative samples: "
                      f"move={cumulative['sampleCounts'].get('move',0)} "
                      f"posture={cumulative['sampleCounts'].get('posture',0)} "
                      f"stroke={cumulative['sampleCounts'].get('stroke',0)} "
                      f"accuracy={cumulative.get('shots',0)}", flush=True)
                print(f"[SKILL] {aid} rubric: {result['families']} -> "
                      f"composite={result['composite']} tier={result['tier'].upper()}",
                      flush=True)
                print(f"[SKILL] {aid} skillProfile written to Mongo "
                      f"(history len={len(doc.get('skillHistory', [])) + 1})",
                      flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[SKILL] flush error for {aid}: {exc!r}", flush=True)
```

- [ ] **Step 3c: Run the new test to verify it passes**

Run: `python -m pytest tests/test_engine_skill.py -v`
Expected: PASS (2 passed)

- [ ] **Step 3d: Wire the accumulator into the live loop**

In `start()`, where per-drill state is reset (near `self._scores = PlayerScores()`), add:

```python
        self._skill_accs = {}
```

In `_run()`'s detection loop, find the recognition block:

```python
                for d in detections:
                    if self._armed and not d.get("in_court"):
                        continue
                    self._recognize(d["id"], d["box"], frame)
```

Replace it with (adds skill sampling + throttled per-sample log):

```python
                for d in detections:
                    if self._armed and not d.get("in_court"):
                        continue
                    self._recognize(d["id"], d["box"], frame)
                    court_ankle = player_positions.get(d["id"])
                    self._skill_add_frame(d["id"], d["box"], d["keypoints"],
                                          court_ankle, now)
                    if (now - self._skill_log_last > 1.0
                            and self._is_real_athlete(d["id"])):
                        self._skill_log_last = now
                        from utils import pose_features as _pf
                        bh = _pf.bbox_height(d["box"])
                        aid = self._athlete_id(d["id"])
                        print(f"[SKILL] {aid} frameseen: bbox_h={bh:.0f}px "
                              f"posture={'SAMPLED' if bh >= SKILL_BBOX_MIN_H else 'gated(far)'} "
                              f"move={'SAMPLED' if court_ankle else 'no-court-pos'}",
                              flush=True)
```

In `start_new_shot`, after `self._persist_shot(self._athlete_id(target_id), zone_name)`, add:

```python
                if self._is_real_athlete(target_id):
                    self._skill_for(self._athlete_id(target_id)).on_shot()
                    self._skill_for(self._athlete_id(target_id)).on_feeder_fired(
                        time.time(), self._zone_to_frontend(zone_name))
```

Where a score is recorded (in the `check_return` success branch, after `self._persist_score(...)`), add:

```python
                        if self._is_real_athlete(active_target_id):
                            self._skill_for(self._athlete_id(active_target_id)).on_score()
```

In the `finally:` cleanup block of `_run()`, before `self._state = "idle"`, add:

```python
            self._flush_skill_profiles()
```

- [ ] **Step 3e: Run the full suite to verify nothing regressed**

Run: `python -m pytest tests/ --ignore=tests/test_shuttle_worker.py -q`
Expected: PASS (previous suite + new engine-skill tests)

- [ ] **Step 4: Commit**

```bash
git add app/engine.py tests/test_engine_skill.py
git commit -m "feat(engine): accumulate + persist per-player skill profile with [SKILL] logs"
```

---

### Task 7: Skill API endpoint

**Files:**
- Modify: `app/routers/players.py`
- Test: `tests/test_players_api.py` (append)

**Interfaces:**
- Consumes: existing `db.players()`; the `skillProfile`/`skillHistory` fields written in Task 6.
- Produces: `GET /api/players/{pid}/skill`; `get_player` now includes `skillProfile.computed` tier when present.

- [ ] **Step 1: Write the failing test (append to `tests/test_players_api.py`)**

```python
def test_skill_endpoint_returns_unranked_for_fresh_player():
    pid = client.post("/api/players", json={"name": "APITEST"}).json()["id"]
    r = client.get(f"/api/players/{pid}/skill")
    assert r.status_code == 200
    body = r.json()
    assert body["tier"] == "Unranked"
    assert body["history"] == []


def test_skill_endpoint_404_for_missing_player():
    assert client.get("/api/players/nope/skill").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_players_api.py -k skill -v`
Expected: FAIL — 404 route not found (returns 404 for the valid player too, so the first assert on `tier` fails / or route missing)

- [ ] **Step 3: Add the endpoint**

In `app/routers/players.py`, add after `get_player`:

```python
@router.get("/{pid}/skill")
def get_skill(pid: str):
    doc = db.players().find_one({"_id": pid})
    if not doc:
        raise HTTPException(404, "player not found")
    profile = doc.get("skillProfile") or {}
    computed = profile.get("computed") or {}
    return {
        "playerId": pid,
        "tier": computed.get("tier", "Unranked"),
        "composite": computed.get("composite"),
        "families": computed.get("families", {}),
        "breakdown": computed.get("breakdown", {}),
        "sampleCounts": (profile.get("cumulative") or {}).get("sampleCounts", {}),
        "updatedAt": computed.get("updatedAt"),
        "history": doc.get("skillHistory", []),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_players_api.py -k skill -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Fold tier into `get_player`**

In `get_player`, the current body is:

```python
@router.get("/{pid}")
def get_player(pid: str):
    doc = db.players().find_one({"_id": pid})
    if not doc:
        raise HTTPException(404, "player not found")
    doc["id"] = doc.pop("_id")
    return _strip_embedding(doc)
```

It already returns the whole doc (so `skillProfile` rides along). No change needed beyond confirming `skillProfile` is present when set. Add a lightweight regression assert to the existing test file (append):

```python
def test_get_player_includes_skillprofile_when_present():
    pid = client.post("/api/players", json={"name": "APITEST"}).json()["id"]
    db.players().update_one({"_id": pid},
        {"$set": {"skillProfile": {"computed": {"tier": "Novice"}}}})
    got = client.get(f"/api/players/{pid}").json()
    assert got["skillProfile"]["computed"]["tier"] == "Novice"
```

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/ --ignore=tests/test_shuttle_worker.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add app/routers/players.py tests/test_players_api.py
git commit -m "feat(api): GET /api/players/{id}/skill returns tier + breakdown + history"
```

---

## Self-Review

**Spec coverage:**
- §4 Tier A (coverage/speed/reaction/accuracy) → Task 4 `add_frame` movement branch + `on_shot/on_score`; Task 5 rubric. ✓
- §4 Tier B posture (knee/torso/stance/consistency) → Task 2 helpers + Task 4 posture branch + Task 5. ✓
- §4 Tier C stroke (swing detection/consistency) → Task 2 `max_wrist_speed` + Task 4 `_update_swing` + Task 5. ✓
- §3 principle 1 (feeder-timed reaction) → Task 4 `on_feeder_fired` + reaction branch; Task 6 wires `on_feeder_fired`. ✓
- §3 principle 2 (pose-only strokes) → swing from wrist speed, shuttle only for scores. ✓
- §3 principle 3 (confidence + bbox gates) → Task 2 gating + Task 4 `near`/`kp_conf`. ✓
- §3 principle 4 (distributions) → `RunningStat` everywhere. ✓
- §5 scoring/tiers/redistribution/min-data → Task 5. ✓
- §6 storage schema (cumulative aggregates + computed + history) → Task 6 flush. ✓
- §6 API → Task 7. ✓
- §7 `[SKILL]` logging (per-sample, reaction, swing, merge/persist) → Task 6 logs + Task 4 records; reaction/swing per-event logs are implicit in the merge summary. Per-sample + merge/rubric/write logs present. ✓
- §8 testing → Tasks 2–7 each ship pytest tests. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. The one scaffolding artifact in Task 4 Step 3 is explicitly called out and replaced in the same step. ✓

**Type consistency:** `snapshot()` keys (`sampleCounts`, `coverage`, `shots`, `scores`, `swings`, `stats`) are consumed identically by `merge_snapshot` and `evaluate_rubric`. `evaluate_rubric` return keys (`composite`, `tier`, `families`, `breakdown`) are consumed by Task 6 flush and Task 7 endpoint. Metric names in `SKILL_REFS` (Task 1) match the `score(...)`/`_norm` lookups (Task 5) and the stat keys in `SkillAccumulator.stats` (Task 4): `move_speed`, `reaction`, `knee`, `stance`, `torso`, `swing_peak` — note `swing_consistency`/`posture_consistency`/`coverage`/`accuracy` are derived (std/len/ratio), not stored stats, and are defined in `SKILL_REFS`. ✓

Note: `torso` is accumulated but not yet scored in the rubric (posture uses knee + stance + knee-consistency). This is intentional headroom — torso lean is the least reliable at distance; it's captured for later tuning without affecting the tier. Documented here so it isn't mistaken for a gap.
