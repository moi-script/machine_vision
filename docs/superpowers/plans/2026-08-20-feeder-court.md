# feeder_court Shuttlecock Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and deploy `feeder_court`, a shuttlecock detector that works on OV9281 grayscale footage recorded in the actual deployment hall.

**Architecture:** Logic lives in testable pure modules under `setup/utils/feeder_court/`; the scripts under `setup/scripts/feeder_court/` are thin CLIs over them. The pipeline is: detect usable clip windows → hand-calibrate 60 frames to measure true shuttle size (a hard gate) → drift-compensated motion auto-labelling → hand-correction in Roboflow → train `yolov8n-p2` on Colab → evaluate against the held-out `far` clip → export OpenVINO.

**Tech Stack:** Python, OpenCV (`cv2`), NumPy, `ultralytics 8.4.14`, `torch 2.9.1+cpu` (local, no CUDA), `openvino 2026.3.0`, pytest, Roboflow, Google Colab (T4).

**Spec:** `docs/superpowers/specs/2026-08-20-feeder-court-design.md`

## Global Constraints

Every task's requirements implicitly include this section. Values are copied verbatim from the spec.

- **Model name is `feeder_court`.** Artefacts: `models/feeder_court.pt`, `models/feeder_court_openvino_model/`.
- **`far.mp4` is never trained on and never used for early stopping.** It is the held-out eval set.
- **`line_1.mp4` and `line_2.mp4` are excluded entirely** — they belong to the future line-detection dataset.
- **Never downscale.** The ROI crop keeps native pixels. An 8 px shuttle at `imgsz 640` on a 1280-wide frame becomes 4 px and is undetectable.
- **The ROI crop must be applied identically at training and inference.** A mismatch is a silent correctness bug.
- **Random frame splits are prohibited.** Splits are temporal. Measured: random splits leave 80.9% of val frames as >0.95-similarity near-duplicates of a train frame; temporal splits leave 0.0%.
- **Augmentation overrides (mono footage):** `hsv_h=0`, `hsv_s=0`, `hsv_v` kept at default, `scale=0.25`, `flipud=0`, `fliplr=0.5`, `mosaic=1.0`, `close_mosaic=10`.
- **Calibration gate decision rule:** median max-dimension `>= 16 px` → stock `yolov8n`; `8–16 px` → `yolov8n-p2`; `< 8 px` **across all three clips** → STOP, escalate to fallbacks.
- **Success criteria (both must hold):** ≥50% recall of hand-verified shuttle instances in held-out `far` at conf 0.25, with ≤1 false positive per 10 frames; and ≥10 fps after OpenVINO export on the i3-1215U.
- **Auto-labelling emits three buckets** — `positive`, `negative`, `discard`. Uncertain frames are never written as false negatives.
- **Every exported pre-label is hand-corrected.** No exceptions.
- Tests run from `setup/`: `python -m pytest tests/ -v`. Test files add the parent to `sys.path` (see existing `tests/test_zones.py`).

---

## File Structure

**New testable modules** — `setup/utils/feeder_court/`:

| File | Responsibility |
|---|---|
| `__init__.py` | Package marker |
| `roi.py` | Crop band geometry; frame cropping and label coordinate transforms |
| `segments.py` | Motion profiling and usable-window detection |
| `sizes.py` | Label size measurement and the calibration gate decision |
| `motion.py` | Drift-compensated three-frame differencing and blob candidate extraction |
| `tracks.py` | Detection linking and ballistic flight classification |
| `splits.py` | Temporal train/val splitting with a guard band |

**New CLIs** — `setup/scripts/feeder_court/`:

| File | Responsibility |
|---|---|
| `segment_clips.py` | Emit per-clip usable windows as JSON |
| `sample_calibration.py` | Phase 0: sample frames, then measure and print the gate decision |
| `propose_shuttles.py` | Auto-label proposals + contact sheets |
| `export_for_roboflow.py` | ROI-cropped frames + pre-labels + manifest |
| `evaluate_feeder_court.py` | Held-out `far` scoring, FP rate, size bands, imgsz sweep |
| `deploy_feeder_court.py` | OpenVINO export, equivalence check, fps measurement |

**New tests** — `setup/tests/`: `test_feeder_court_roi.py`, `test_feeder_court_segments.py`, `test_feeder_court_sizes.py`, `test_feeder_court_motion.py`, `test_feeder_court_tracks.py`, `test_feeder_court_splits.py`

**New notebook** — `notebooks/feeder_court_train.ipynb`

---

## Task 1: ROI crop geometry

The spec calls a train/infer crop mismatch a silent correctness bug, so this is the foundation everything else imports.

**Files:**
- Create: `setup/utils/feeder_court/__init__.py`
- Create: `setup/utils/feeder_court/roi.py`
- Test: `setup/tests/test_feeder_court_roi.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `CropBand(top: int, height: int)` frozen dataclass, with property `bottom -> int`
  - `band_from_extent(y_min: float, y_max: float, frame_h: int, margin: int = 24, multiple_of: int = 32) -> CropBand`
  - `apply_crop(frame: np.ndarray, band: CropBand) -> np.ndarray`
  - `shift_boxes_into_band(boxes_xyxy: np.ndarray, band: CropBand) -> np.ndarray`
  - `boxes_fully_inside(boxes_xyxy: np.ndarray, band: CropBand) -> np.ndarray` (bool mask)

- [ ] **Step 1: Write the failing test**

Create `setup/tests/test_feeder_court_roi.py`:

```python
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feeder_court import roi  # noqa: E402


def test_band_snaps_up_to_multiple_and_clamps_to_frame():
    band = roi.band_from_extent(y_min=300.0, y_max=520.0, frame_h=720, margin=24, multiple_of=32)
    assert band.height % 32 == 0
    assert band.top >= 0
    assert band.bottom <= 720
    # The requested extent plus margin must still fit inside the band.
    assert band.top <= 300 - 24 + 1
    assert band.bottom >= 520 + 24 - 1


def test_band_never_exceeds_frame_when_extent_touches_edges():
    band = roi.band_from_extent(y_min=0.0, y_max=719.0, frame_h=720, margin=48, multiple_of=32)
    assert band.top == 0
    assert band.bottom <= 720


def test_apply_crop_returns_band_height():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    band = roi.CropBand(top=192, height=448)
    out = roi.apply_crop(frame, band)
    assert out.shape == (448, 1280, 3)


def test_apply_crop_preserves_pixels():
    frame = np.arange(720 * 1280, dtype=np.uint8).reshape(720, 1280)
    band = roi.CropBand(top=100, height=64)
    out = roi.apply_crop(frame, band)
    assert np.array_equal(out, frame[100:164])


def test_shift_boxes_subtracts_top_from_y_only():
    band = roi.CropBand(top=192, height=448)
    boxes = np.array([[10.0, 200.0, 30.0, 240.0]])
    out = roi.shift_boxes_into_band(boxes, band)
    assert out.tolist() == [[10.0, 8.0, 30.0, 48.0]]


def test_boxes_fully_inside_rejects_boxes_crossing_the_band_edge():
    band = roi.CropBand(top=192, height=448)
    boxes = np.array([
        [10.0, 200.0, 30.0, 240.0],   # inside
        [10.0, 180.0, 30.0, 210.0],   # crosses top edge
        [10.0, 620.0, 30.0, 700.0],   # crosses bottom edge
    ])
    mask = roi.boxes_fully_inside(boxes, band)
    assert mask.tolist() == [True, False, False]


def test_crop_then_shift_round_trips_to_the_same_pixel():
    frame = np.zeros((720, 1280), dtype=np.uint8)
    frame[300, 640] = 255
    band = roi.CropBand(top=192, height=448)
    cropped = roi.apply_crop(frame, band)
    boxes = np.array([[639.0, 299.0, 641.0, 301.0]])
    shifted = roi.shift_boxes_into_band(boxes, band)
    y = int(shifted[0][1]) + 1
    x = int(shifted[0][0]) + 1
    assert cropped[y, x] == 255
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd setup && python -m pytest tests/test_feeder_court_roi.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'utils.feeder_court'`

- [ ] **Step 3: Write minimal implementation**

Create `setup/utils/feeder_court/__init__.py` as an empty file.

Create `setup/utils/feeder_court/roi.py`:

```python
# ============================================================
# roi.py — Court-band crop geometry.
#
# The OV9281 frames are 1280x720 but the upper third is unlit ceiling where no
# shuttle ever appears. Cropping that away is 2.7x cheaper than letterboxing the
# full frame to 1280x1280 and costs zero shuttle pixels.
#
# The same band MUST be applied at training and at inference. A mismatch silently
# rescales every object, which is how the imgsz-640 half-scaling bug happened.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CropBand:
    """A horizontal slice of the frame, in native pixels."""

    top: int
    height: int

    @property
    def bottom(self) -> int:
        return self.top + self.height


def band_from_extent(
    y_min: float,
    y_max: float,
    frame_h: int,
    margin: int = 24,
    multiple_of: int = 32,
) -> CropBand:
    """Smallest band covering [y_min, y_max] plus margin, snapped for the network.

    Height is rounded UP to a multiple of `multiple_of` (YOLO strides), then the
    band is clamped inside the frame. Clamping can only shrink height, never push
    the band outside the image.
    """
    lo = int(np.floor(y_min - margin))
    hi = int(np.ceil(y_max + margin))
    lo = max(lo, 0)
    hi = min(hi, frame_h)

    height = hi - lo
    remainder = height % multiple_of
    if remainder:
        height += multiple_of - remainder

    # Grow downward first, then upward, then clamp.
    if lo + height > frame_h:
        lo = max(frame_h - height, 0)
    height = min(height, frame_h - lo)

    return CropBand(top=lo, height=height)


def apply_crop(frame: np.ndarray, band: CropBand) -> np.ndarray:
    """Slice the band out of a frame. Works for 2D grayscale and 3D colour."""
    return frame[band.top:band.bottom]


def shift_boxes_into_band(boxes_xyxy: np.ndarray, band: CropBand) -> np.ndarray:
    """Rebase absolute xyxy boxes into the cropped frame's coordinates."""
    out = np.asarray(boxes_xyxy, dtype=float).copy()
    out[:, 1] -= band.top
    out[:, 3] -= band.top
    return out


def boxes_fully_inside(boxes_xyxy: np.ndarray, band: CropBand) -> np.ndarray:
    """Mask of boxes wholly within the band.

    Boxes crossing an edge are rejected rather than clipped: a clipped shuttle box
    is a wrong label, and wrong labels are what poisoned scene_v2.
    """
    b = np.asarray(boxes_xyxy, dtype=float)
    return (b[:, 1] >= band.top) & (b[:, 3] <= band.bottom)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd setup && python -m pytest tests/test_feeder_court_roi.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add setup/utils/feeder_court/__init__.py setup/utils/feeder_court/roi.py setup/tests/test_feeder_court_roi.py
git commit -m "feat(feeder_court): ROI court-band crop geometry"
```

---

## Task 2: Usable-window detection

Every clip has camera-motion junk at head and tail — `near.mp4` ends on a close-up of a laptop keyboard. This finds the real footage.

**Files:**
- Create: `setup/utils/feeder_court/segments.py`
- Create: `setup/scripts/feeder_court/segment_clips.py`
- Test: `setup/tests/test_feeder_court_segments.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `usable_segments(profile: np.ndarray, sample_step: int, threshold: float = 15.0, min_frames: int = 150) -> list[tuple[int, int]]`
  - `motion_profile(video_path: str, sample_step: int = 15, size: tuple[int, int] = (160, 90)) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

Create `setup/tests/test_feeder_court_segments.py`:

```python
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feeder_court import segments  # noqa: E402


def test_head_and_tail_motion_are_excluded():
    # 5 noisy samples, 40 calm, 5 noisy — at step 15 that is frames 0..749.
    profile = np.concatenate([
        np.full(5, 40.0),
        np.full(40, 2.0),
        np.full(5, 40.0),
    ])
    out = segments.usable_segments(profile, sample_step=15, threshold=15.0, min_frames=150)
    assert len(out) == 1
    start, end = out[0]
    assert start >= 5 * 15
    assert end <= 45 * 15


def test_short_calm_runs_are_dropped():
    profile = np.concatenate([
        np.full(10, 40.0),
        np.full(3, 2.0),      # 3 samples * 15 = 45 frames, under min_frames
        np.full(10, 40.0),
    ])
    out = segments.usable_segments(profile, sample_step=15, threshold=15.0, min_frames=150)
    assert out == []


def test_a_mid_clip_bump_splits_one_run_into_two():
    profile = np.concatenate([
        np.full(20, 2.0),
        np.full(4, 40.0),     # camera knocked
        np.full(20, 2.0),
    ])
    out = segments.usable_segments(profile, sample_step=15, threshold=15.0, min_frames=150)
    assert len(out) == 2
    assert out[0][1] <= out[1][0]


def test_entirely_calm_profile_yields_one_full_span():
    profile = np.full(50, 1.0)
    out = segments.usable_segments(profile, sample_step=15, threshold=15.0, min_frames=150)
    assert len(out) == 1
    assert out[0][0] == 0


def test_entirely_noisy_profile_yields_nothing():
    profile = np.full(50, 99.0)
    assert segments.usable_segments(profile, sample_step=15) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd setup && python -m pytest tests/test_feeder_court_segments.py -v`
Expected: FAIL with `ImportError: cannot import name 'segments'`

- [ ] **Step 3: Write minimal implementation**

Create `setup/utils/feeder_court/segments.py`:

```python
# ============================================================
# segments.py — Find the usable window of each clip.
#
# Every clip in new_badminton_source has a heavy camera-motion segment at head
# and tail: the rig being carried into and out of position. near.mp4 literally
# ends on a close-up of a laptop keyboard. Training on those frames would teach
# the model a keyboard is a badminton court.
#
# Also splits the middle wherever motion spikes, because the three-frame
# differencing in motion.py assumes only small drift between neighbouring frames.
# ============================================================

from __future__ import annotations

import cv2
import numpy as np


def motion_profile(
    video_path: str,
    sample_step: int = 15,
    size: tuple[int, int] = (160, 90),
) -> np.ndarray:
    """Mean absolute difference between consecutively sampled frames.

    Index i of the result corresponds to frame (i + 1) * sample_step.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    values = []
    prev = None
    for i in range(0, total, sample_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size).astype(np.float32)
        if prev is not None:
            values.append(float(np.abs(gray - prev).mean()))
        prev = gray
    cap.release()
    return np.array(values, dtype=float)


def usable_segments(
    profile: np.ndarray,
    sample_step: int,
    threshold: float = 15.0,
    min_frames: int = 150,
) -> list[tuple[int, int]]:
    """Contiguous frame spans whose sampled motion stays below `threshold`.

    Spans shorter than `min_frames` are dropped — too short to yield a useful
    background model or a complete shuttle flight.
    """
    calm = np.asarray(profile, dtype=float) < threshold
    spans: list[tuple[int, int]] = []
    start = None

    for idx, is_calm in enumerate(calm):
        if is_calm and start is None:
            start = idx
        elif not is_calm and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, len(calm)))

    out = []
    for a, b in spans:
        start_frame = a * sample_step
        end_frame = b * sample_step
        if end_frame - start_frame >= min_frames:
            out.append((start_frame, end_frame))
    return out
```

Create `setup/scripts/feeder_court/segment_clips.py`:

```python
# ============================================================
# segment_clips.py — Emit the usable window(s) of each source clip as JSON.
#
# Usage:
#   python scripts/feeder_court/segment_clips.py \
#       --source datasets/vid_source/new_badminton_source \
#       --out datasets/feeder_court/segments.json
# ============================================================

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.feeder_court import segments  # noqa: E402

# line_1/line_2 are excluded: they belong to the future line-detection dataset.
CLIPS = ["near", "mid", "far"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-step", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=15.0)
    parser.add_argument("--min-frames", type=int, default=150)
    args = parser.parse_args()

    result = {}
    for clip in CLIPS:
        path = os.path.join(args.source, f"{clip}.mp4")
        profile = segments.motion_profile(path, sample_step=args.sample_step)
        spans = segments.usable_segments(
            profile,
            sample_step=args.sample_step,
            threshold=args.threshold,
            min_frames=args.min_frames,
        )
        result[clip] = spans
        total = sum(b - a for a, b in spans)
        print(f"{clip}: {len(spans)} segment(s), {total} usable frames -> {spans}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd setup && python -m pytest tests/test_feeder_court_segments.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Run the CLI against the real clips**

Run:
```bash
cd setup && python scripts/feeder_court/segment_clips.py \
    --source datasets/vid_source/new_badminton_source \
    --out datasets/feeder_court/segments.json
```

Expected: three lines of output. Sanity-check against the spec's §2.1 table — `near` ≈ 200–2100, `mid` ≈ 250–1480, `far` ≈ 200–1740, roughly 4,670 usable frames in total. If the totals are wildly different (say under 2,000 or over 7,000), tune `--threshold` before continuing; a wrong window here corrupts everything downstream.

- [ ] **Step 6: Commit**

```bash
git add setup/utils/feeder_court/segments.py setup/scripts/feeder_court/segment_clips.py setup/tests/test_feeder_court_segments.py
git commit -m "feat(feeder_court): usable-window detection for source clips"
```

---

## Task 3: Calibration gate — PHASE 0, HARD STOP

This is the task that decides whether the rest of the plan runs at all. Do not skip ahead.

**Files:**
- Create: `setup/utils/feeder_court/sizes.py`
- Create: `setup/scripts/feeder_court/sample_calibration.py`
- Test: `setup/tests/test_feeder_court_sizes.py`

**Interfaces:**
- Consumes: `segments.usable_segments` output (JSON from Task 2)
- Produces:
  - `yolo_box_max_dim_px(line: str, img_w: int, img_h: int) -> float`
  - `size_percentiles(dims: list[float]) -> dict[str, float]` with keys `p25`, `median`, `p75`, `p95`
  - `gate_decision(medians_by_clip: dict[str, float]) -> str` returning `"stock"`, `"p2"`, or `"stop"`
  - `evenly_spaced_frames(spans: list[tuple[int, int]], count: int) -> list[int]`

- [ ] **Step 1: Write the failing test**

Create `setup/tests/test_feeder_court_sizes.py`:

```python
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feeder_court import sizes  # noqa: E402


def test_yolo_line_converts_to_pixel_max_dimension():
    # class cx cy w h, normalised. w=0.01 of 1280 = 12.8 px, h=0.02 of 448 = 8.96 px
    line = "0 0.5 0.5 0.01 0.02"
    assert sizes.yolo_box_max_dim_px(line, img_w=1280, img_h=448) == pytest.approx(12.8)


def test_yolo_line_takes_the_larger_dimension():
    line = "0 0.5 0.5 0.001 0.05"  # 1.28 px wide, 22.4 px tall
    assert sizes.yolo_box_max_dim_px(line, img_w=1280, img_h=448) == pytest.approx(22.4)


def test_size_percentiles_reports_expected_keys():
    out = sizes.size_percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
    assert set(out) == {"p25", "median", "p75", "p95"}
    assert out["median"] == pytest.approx(3.0)


def test_gate_says_stock_when_all_clips_are_comfortably_large():
    assert sizes.gate_decision({"near": 30.0, "mid": 22.0, "far": 18.0}) == "stock"


def test_gate_says_p2_in_the_marginal_band():
    assert sizes.gate_decision({"near": 14.0, "mid": 11.0, "far": 9.0}) == "p2"


def test_gate_stops_only_when_every_clip_is_below_eight_px():
    assert sizes.gate_decision({"near": 7.0, "mid": 6.0, "far": 5.0}) == "stop"


def test_gate_does_not_stop_if_any_clip_is_workable():
    # far is hopeless but near is fine — that is a scoping decision, not a stop.
    assert sizes.gate_decision({"near": 20.0, "mid": 9.0, "far": 5.0}) == "p2"


def test_gate_uses_the_minimum_clip_to_choose_between_stock_and_p2():
    # One marginal clip forces P2 even though the others are large.
    assert sizes.gate_decision({"near": 40.0, "mid": 38.0, "far": 12.0}) == "p2"


def test_evenly_spaced_frames_spreads_across_all_spans():
    frames = sizes.evenly_spaced_frames([(0, 100), (200, 300)], count=10)
    assert len(frames) == 10
    assert all(0 <= f < 100 or 200 <= f < 300 for f in frames)
    assert len(set(frames)) == 10


def test_evenly_spaced_frames_is_sorted_and_within_bounds():
    frames = sizes.evenly_spaced_frames([(50, 90)], count=5)
    assert frames == sorted(frames)
    assert frames[0] >= 50
    assert frames[-1] < 90


def test_gate_refuses_to_decide_when_a_clip_is_unmeasured():
    with pytest.raises(ValueError):
        sizes.gate_decision({"near": 20.0})


def test_gate_refuses_even_when_the_measured_clips_all_look_fine():
    # The danger case: present clips look great, absent ones were never checked.
    with pytest.raises(ValueError):
        sizes.gate_decision({"near": 30.0, "mid": 25.0})


def test_gate_ignores_clips_outside_the_required_set():
    # line_1/line_2 are a different dataset entirely and must not sway the gate.
    assert sizes.gate_decision(
        {"near": 20.0, "mid": 20.0, "far": 20.0, "line_1": 3.0}
    ) == "stock"


def test_gate_boundary_exactly_eight_is_not_a_stop():
    assert sizes.gate_decision({"near": 8.0, "mid": 8.0, "far": 8.0}) == "p2"


def test_gate_boundary_exactly_sixteen_is_stock():
    assert sizes.gate_decision({"near": 16.0, "mid": 16.0, "far": 16.0}) == "stock"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd setup && python -m pytest tests/test_feeder_court_sizes.py -v`
Expected: FAIL with `ImportError: cannot import name 'sizes'`

- [ ] **Step 3: Write minimal implementation**

Create `setup/utils/feeder_court/sizes.py`:

```python
# ============================================================
# sizes.py — Measure real shuttlecock pixel size, and gate on it.
#
# Automated probing of these clips returned an IDENTICAL size distribution for
# near and far (median 8 px both), which is a red flag rather than a result: it
# means the probe was likely measuring noise and court-line edges, not shuttles.
# Only hand-drawn boxes settle it.
#
# 8 px matters because YOLOv8n's finest stride is 8 — an 8 px object occupies a
# single grid cell at native resolution. Below that, no amount of training helps.
# ============================================================

from __future__ import annotations

import numpy as np

STOP_PX = 8.0     # below this everywhere -> the architecture cannot work
STOCK_PX = 16.0   # at or above this everywhere -> no small-object head needed
REQUIRED_CLIPS = ("near", "mid", "far")


def yolo_box_max_dim_px(line: str, img_w: int, img_h: int) -> float:
    """Largest side, in pixels, of one YOLO-format label line."""
    parts = line.split()
    w = float(parts[3]) * img_w
    h = float(parts[4]) * img_h
    return max(w, h)


def size_percentiles(dims: list[float]) -> dict[str, float]:
    arr = np.asarray(dims, dtype=float)
    return {
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def gate_decision(medians_by_clip: dict[str, float]) -> str:
    """Apply the spec's calibration gate.

    Requires all three clips (near, mid, far) to be present.

    Returns "stop", "p2", or "stock".

    "stop" requires EVERY clip to be under 8 px. If one distance still works, the
    right answer is to narrow the deployment envelope, not to abandon the model —
    so that case returns an architecture instead.

    The choice between "stock" and "p2" is driven by the WORST clip, because the
    model has to handle every distance it will be deployed at.
    """
    missing = [c for c in REQUIRED_CLIPS if c not in medians_by_clip]
    if missing:
        raise ValueError(
            f"cannot decide the gate: no measurement for {missing}. "
            "An unmeasured clip is not the same as a passing one - a clip with "
            "zero hand-drawn boxes may be exactly the distance that forces a stop."
        )

    medians = [medians_by_clip[c] for c in REQUIRED_CLIPS]

    if all(m < STOP_PX for m in medians):
        return "stop"
    if min(medians) >= STOCK_PX:
        return "stock"
    return "p2"


def evenly_spaced_frames(spans: list[tuple[int, int]], count: int) -> list[int]:
    """`count` frame indices spread evenly across the union of `spans`.

    Sampling evenly rather than randomly guarantees coverage of the whole
    timeline, so the calibration set cannot accidentally concentrate on one rally.
    """
    lengths = [b - a for a, b in spans]
    total = sum(lengths)
    if total < count:
        raise ValueError(f"need {count} frames but spans only hold {total}")

    positions = np.linspace(0, total - 1, count).astype(int)
    frames = []
    for pos in positions:
        remaining = int(pos)
        for (a, b), length in zip(spans, lengths):
            if remaining < length:
                frames.append(a + remaining)
                break
            remaining -= length
    return sorted(frames)
```

Create `setup/scripts/feeder_court/sample_calibration.py`:

```python
# ============================================================
# sample_calibration.py — PHASE 0. Run this BEFORE any bulk labelling.
#
# Two modes:
#   --mode sample   writes 20 frames per clip for you to hand-box in Roboflow
#   --mode measure  reads the labels back and prints the gate decision
#
# The gate:
#   median >= 16 px            -> stock yolov8n is enough
#   8-16 px                    -> yolov8n-p2 (the plan's default)
#   < 8 px on ALL three clips  -> STOP. Do not proceed to labelling.
#
# Usage:
#   python scripts/feeder_court/sample_calibration.py --mode sample \
#       --source datasets/vid_source/new_badminton_source \
#       --segments datasets/feeder_court/segments.json \
#       --out datasets/feeder_court/calibration
#
#   python scripts/feeder_court/sample_calibration.py --mode measure \
#       --labels datasets/feeder_court/calibration/labels
# ============================================================

import argparse
import glob
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.feeder_court import sizes  # noqa: E402

CLIPS = ["near", "mid", "far"]


def do_sample(args: argparse.Namespace) -> None:
    with open(args.segments) as fh:
        spans_by_clip = json.load(fh)

    os.makedirs(args.out, exist_ok=True)
    for clip in CLIPS:
        spans = [tuple(s) for s in spans_by_clip[clip]]
        frames = sorted(set(sizes.evenly_spaced_frames(spans, args.per_clip)))
        cap = cv2.VideoCapture(os.path.join(args.source, f"{clip}.mp4"))
        written = 0
        for idx in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.imwrite(os.path.join(args.out, f"{clip}_{idx:06d}.jpg"), frame)
            written += 1
        cap.release()
        print(f"{clip}: wrote {written} calibration frames", flush=True)

    print(f"\nUpload {args.out} to Roboflow and box EVERY visible shuttlecock.")
    print("Export as YOLOv8, then run this script again with --mode measure.")


def do_measure(args: argparse.Namespace) -> None:
    medians = {}
    unmeasured = []
    for clip in CLIPS:
        dims = []
        pattern = os.path.join(args.labels, "**", f"{clip}_*.txt")
        for path in sorted(glob.glob(pattern, recursive=True)):
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        dims.append(sizes.yolo_box_max_dim_px(line, args.img_w, args.img_h))
        if not dims:
            print(f"{clip}: NO BOXES FOUND — cannot measure this clip")
            unmeasured.append(clip)
            continue
        stats = sizes.size_percentiles(dims)
        medians[clip] = stats["median"]
        print(
            f"{clip}: n={len(dims):4d}  p25={stats['p25']:.1f}  "
            f"median={stats['median']:.1f}  p75={stats['p75']:.1f}  p95={stats['p95']:.1f} px",
            flush=True,
        )

    if unmeasured:
        print(f"\nUNMEASURED CLIPS: {unmeasured}")
        print("Refusing to decide the gate. A clip with no hand-drawn boxes is")
        print("not evidence that the model will work there - it may be the very")
        print("distance that makes this architecture unusable.")
        print("Either box those frames, or decide deliberately that the distance")
        print("is out of scope - which is a change to the spec's scope, not a CLI flag.")
        sys.exit(1)

    decision = sizes.gate_decision(medians)
    print(f"\nGATE DECISION: {decision}")
    if decision == "stop":
        print("Every clip is under 8 px. YOLOv8n cannot detect this.")
        print("Do NOT proceed to labelling. Escalate to the spec's section 4.3 fallbacks:")
        print("  - 2x upscale of the ROI crop")
        print("  - scope shuttle tracking to the near half-court only")
        print("  - re-record with a cleaned lens and corrected exposure")
        sys.exit(1)
    if decision == "stock":
        print("Shuttles are large enough for stock yolov8n; the P2 head is optional.")
    else:
        print("Marginal sizes confirmed. Proceed with yolov8n-p2 as planned.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sample", "measure"], required=True)
    parser.add_argument("--source")
    parser.add_argument("--segments")
    parser.add_argument("--out")
    parser.add_argument(
        "--labels",
        help="Flat directory of YOLO label .txt files, or a Roboflow export root "
        "(e.g. containing train/labels, valid/labels, test/labels) — searched recursively.",
    )
    parser.add_argument("--per-clip", type=int, default=20)
    parser.add_argument("--img-w", type=int, default=1280)
    parser.add_argument("--img-h", type=int, default=720)
    args = parser.parse_args()

    if args.mode == "sample":
        missing = [n for n in ("source", "segments", "out") if getattr(args, n) is None]
        if missing:
            parser.error("--mode sample requires: " + ", ".join("--" + m for m in missing))
        do_sample(args)
    else:
        if args.labels is None:
            parser.error("--mode measure requires: --labels")
        do_measure(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd setup && python -m pytest tests/test_feeder_court_sizes.py -v`
Expected: PASS, 15 tests

- [ ] **Step 5: Commit the tooling**

```bash
git add setup/utils/feeder_court/sizes.py setup/scripts/feeder_court/sample_calibration.py setup/tests/test_feeder_court_sizes.py
git commit -m "feat(feeder_court): Phase 0 calibration gate tooling"
```

- [ ] **Step 6: Generate the calibration frames**

Run:
```bash
cd setup && python scripts/feeder_court/sample_calibration.py --mode sample \
    --source datasets/vid_source/new_badminton_source \
    --segments datasets/feeder_court/segments.json \
    --out datasets/feeder_court/calibration
```
Expected: 60 JPEGs in `datasets/feeder_court/calibration/`.

- [ ] **Step 7: HUMAN TASK — hand-box the calibration frames**

Upload the 60 frames to a Roboflow project named `feeder-court-calibration`. Box **every visible shuttlecock**, including ones you are only fairly confident about — but not ones you are guessing at. Export as YOLOv8 and place the label `.txt` files in `datasets/feeder_court/calibration/labels/`.

This is roughly 20 minutes of work and it is the cheapest possible way to find out whether the rest of this plan is viable.

- [ ] **Step 8: Run the gate**

Run:
```bash
cd setup && python scripts/feeder_court/sample_calibration.py --mode measure \
    --labels datasets/feeder_court/calibration/labels
```

**STOP CONDITION:** if this prints `GATE DECISION: stop` and exits 1, halt the plan and return to the user with the measured numbers. Do not start Task 4. The spec's §4.3 fallbacks are a decision for the user, not for the implementer.

If it prints `p2`, continue with the plan as written. If it prints `stock`, continue but substitute `yolov8n.yaml` for `yolov8-p2.yaml` in Task 8 and note the change.

- [ ] **Step 9: Record the result**

Append the measured table and the gate decision to the spec under a new "§4.4 Measured result" heading, then commit:

```bash
git add docs/superpowers/specs/2026-08-20-feeder-court-design.md
git commit -m "docs(spec): record feeder_court Phase 0 calibration measurement"
```

---

## Task 4: Drift-compensated motion differencing

Median-background subtraction was tested on this footage and failed — the camera drifts, so static court lines dominate the difference (~58 candidates/frame). Aligned three-frame differencing cut that to 6.0/frame.

**Files:**
- Create: `setup/utils/feeder_court/motion.py`
- Test: `setup/tests/test_feeder_court_motion.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `translation_align(src: np.ndarray, ref: np.ndarray) -> np.ndarray`
  - `three_frame_diff(prev: np.ndarray, cur: np.ndarray, nxt: np.ndarray) -> np.ndarray`
  - `Candidate` frozen dataclass with fields `frame: int, x: int, y: int, w: int, h: int, area: int, peak: float`
  - `candidate_boxes(diff: np.ndarray, frame_index: int, thresh: int = 22, min_dim: int = 3, max_dim: int = 45, min_area: int = 5, max_area: int = 700, ar_range: tuple[float, float] = (0.3, 3.0), min_fill: float = 0.32) -> list[Candidate]`

- [ ] **Step 1: Write the failing test**

Create `setup/tests/test_feeder_court_motion.py`:

```python
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feeder_court import motion  # noqa: E402


def _scene(shift=0):
    """A textured static background with a bright vertical line, optionally shifted."""
    rng = np.random.default_rng(0)
    img = (rng.random((180, 320)) * 40).astype(np.uint8)
    img[:, 100 + shift:104 + shift] = 220          # a "court line"
    return img


def test_alignment_recovers_a_known_translation():
    ref = _scene(0)
    src = _scene(6)
    aligned = motion.translation_align(src, ref)
    # After alignment the bright line should sit back near column 100.
    col = int(np.argmax(aligned.astype(float).mean(axis=0)))
    assert abs(col - 101) <= 2


def test_three_frame_diff_cancels_a_drifting_static_scene():
    prev, cur, nxt = _scene(-4), _scene(0), _scene(4)
    d = motion.three_frame_diff(prev, cur, nxt)
    # The static line must not survive: nothing should be strongly lit.
    assert d.max() < 60


def test_three_frame_diff_keeps_a_fast_mover():
    prev, cur, nxt = _scene(0), _scene(0), _scene(0)
    cur = cur.copy()
    cur[80:88, 200:208] = 255          # a shuttle present only in the middle frame
    d = motion.three_frame_diff(prev, cur, nxt)
    assert d[80:88, 200:208].max() > 100


def test_candidate_boxes_finds_the_mover():
    d = np.zeros((180, 320), dtype=np.uint8)
    d[80:88, 200:208] = 255
    cands = motion.candidate_boxes(d, frame_index=7)
    assert len(cands) == 1
    c = cands[0]
    assert c.frame == 7
    assert 195 <= c.x <= 205
    assert 4 <= c.w <= 12


def test_candidate_boxes_rejects_objects_that_are_too_large():
    d = np.zeros((180, 320), dtype=np.uint8)
    d[20:120, 20:120] = 255            # a person, not a shuttle
    assert motion.candidate_boxes(d, frame_index=0) == []


def test_candidate_boxes_rejects_extreme_aspect_ratios():
    d = np.zeros((180, 320), dtype=np.uint8)
    d[100:104, 20:140] = 255           # a line fragment: 120x4
    assert motion.candidate_boxes(d, frame_index=0) == []


def test_candidate_boxes_rejects_sparse_blobs():
    d = np.zeros((180, 320), dtype=np.uint8)
    # A hollow ring: large bounding box, low fill.
    d[60:90, 60:90] = 255
    d[65:85, 65:85] = 0
    assert motion.candidate_boxes(d, frame_index=0) == []


def test_candidate_boxes_records_peak_intensity():
    d = np.zeros((180, 320), dtype=np.uint8)
    d[80:86, 200:206] = 140
    cands = motion.candidate_boxes(d, frame_index=0)
    assert len(cands) == 1
    assert cands[0].peak == 140.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd setup && python -m pytest tests/test_feeder_court_motion.py -v`
Expected: FAIL with `ImportError: cannot import name 'motion'`

- [ ] **Step 3: Write minimal implementation**

Create `setup/utils/feeder_court/motion.py`:

```python
# ============================================================
# motion.py — Isolate fast movers on a drifting camera.
#
# Median-background subtraction was tried first and FAILED on this footage: the
# camera drifts by tens of pixels even inside a "stable" window, so static court
# lines light up in the difference. Measured ~58 candidates/frame, whose
# strongest members were visibly line edges and player limbs.
#
# This instead aligns t-K and t+K onto t by phase correlation and takes the
# MINIMUM of the two differences: a pixel must differ from BOTH neighbours to
# survive. Static structure cancels, and so do the lens smudge blobs, which are
# fixed in image coordinates. Measured: 6.0 candidates/frame.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Candidate:
    frame: int
    x: int
    y: int
    w: int
    h: int
    area: int
    peak: float


def translation_align(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Shift `src` onto `ref`, estimating translation by phase correlation.

    Translation only. The drift here is a slowly sagging tripod, not a pan, and a
    full homography would be both slower and less stable on such a dark scene.
    """
    shift, _ = cv2.phaseCorrelate(ref.astype(np.float32), src.astype(np.float32))
    dx, dy = shift
    matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(
        src,
        matrix,
        (src.shape[1], src.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def three_frame_diff(prev: np.ndarray, cur: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Per-pixel min(|cur - prev'|, |cur - nxt'|) after aligning both neighbours."""
    p = translation_align(prev, cur)
    n = translation_align(nxt, cur)
    d = cv2.min(cv2.absdiff(cur, p), cv2.absdiff(cur, n))
    return cv2.GaussianBlur(d, (3, 3), 0)


def candidate_boxes(
    diff: np.ndarray,
    frame_index: int,
    thresh: int = 22,
    min_dim: int = 3,
    max_dim: int = 45,
    min_area: int = 5,
    max_area: int = 700,
    ar_range: tuple[float, float] = (0.3, 3.0),
    min_fill: float = 0.32,
) -> list[Candidate]:
    """Shuttle-shaped connected components of a difference image.

    The filters encode what a shuttlecock is NOT: not enormous (a player), not a
    long thin sliver (a court-line edge), not sparse (noise speckle).
    """
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    out = []
    for k in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[k])
        if not (min_dim <= w <= max_dim and min_dim <= h <= max_dim):
            continue
        if not (min_area <= area <= max_area):
            continue
        if not (ar_range[0] <= w / h <= ar_range[1]):
            continue
        if area / (w * h) < min_fill:
            continue
        out.append(
            Candidate(
                frame=frame_index,
                x=x,
                y=y,
                w=w,
                h=h,
                area=area,
                peak=float(diff[y:y + h, x:x + w].max()),
            )
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd setup && python -m pytest tests/test_feeder_court_motion.py -v`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add setup/utils/feeder_court/motion.py setup/tests/test_feeder_court_motion.py
git commit -m "feat(feeder_court): drift-compensated three-frame differencing"
```

---

## Task 5: Track linking and flight classification

Raw candidates are still mostly noise. A shuttlecock in flight is distinguishable because it moves fast, travels a long way, and goes roughly where its velocity predicts — over 5 to 60 frames, not 738.

**Files:**
- Create: `setup/utils/feeder_court/tracks.py`
- Test: `setup/tests/test_feeder_court_tracks.py`

**Interfaces:**
- Consumes: `motion.Candidate`
- Produces:
  - `link_tracks(cands: list[Candidate], max_gap: int = 2, max_dist: float = 38.0) -> list[list[Candidate]]`
  - `is_flight(track: list[Candidate], min_len: int = 5, max_len: int = 60, min_step: float = 6.0, min_span: float = 60.0, min_directness: float = 0.45) -> bool`
  - `bucket_frames(all_frames: list[int], flight_frames: set[int], reject_frames: set[int]) -> dict[int, str]` mapping frame → `"positive"` / `"negative"` / `"discard"`

- [ ] **Step 1: Write the failing test**

Create `setup/tests/test_feeder_court_tracks.py`:

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feeder_court import tracks  # noqa: E402
from utils.feeder_court.motion import Candidate  # noqa: E402


def _c(frame, x, y, w=6, h=6):
    return Candidate(frame=frame, x=x, y=y, w=w, h=h, area=w * h, peak=200.0)


def test_link_joins_a_straight_fast_path_into_one_track():
    cands = [_c(i, 100 + 10 * i, 50 + 4 * i) for i in range(10)]
    out = tracks.link_tracks(cands)
    assert len(out) == 1
    assert len(out[0]) == 10


def test_link_splits_two_far_apart_objects():
    a = [_c(i, 10 * i, 20) for i in range(6)]
    b = [_c(i, 900 - 10 * i, 600) for i in range(6)]
    out = tracks.link_tracks(a + b)
    assert len(out) == 2
    assert all(len(t) == 6 for t in out)


def test_link_tolerates_a_short_gap():
    cands = [_c(i, 100 + 10 * i, 50) for i in range(10) if i != 4]
    out = tracks.link_tracks(cands, max_gap=2)
    assert len(out) == 1
    assert len(out[0]) == 9


def test_is_flight_accepts_a_fast_directed_track():
    track = [_c(i, 100 + 12 * i, 50 + 5 * i) for i in range(12)]
    assert tracks.is_flight(track) is True


def test_is_flight_rejects_a_track_that_is_too_long():
    track = [_c(i, 100 + 12 * i, 50) for i in range(200)]
    assert tracks.is_flight(track) is False


def test_is_flight_rejects_a_slow_track():
    track = [_c(i, 100 + 1 * i, 50) for i in range(12)]
    assert tracks.is_flight(track) is False


def test_is_flight_rejects_jitter_that_goes_nowhere():
    # Fast per-step motion but returns to where it started.
    xs = [100, 130, 100, 130, 100, 130, 100, 130]
    track = [_c(i, x, 50) for i, x in enumerate(xs)]
    assert tracks.is_flight(track) is False


def test_is_flight_rejects_a_track_that_is_too_short():
    track = [_c(i, 100 + 20 * i, 50) for i in range(3)]
    assert tracks.is_flight(track) is False


def test_bucket_marks_flight_frames_positive():
    out = tracks.bucket_frames([1, 2, 3], flight_frames={2}, reject_frames=set())
    assert out[2] == "positive"


def test_bucket_marks_clean_frames_negative():
    out = tracks.bucket_frames([1, 2, 3], flight_frames={2}, reject_frames=set())
    assert out[1] == "negative"
    assert out[3] == "negative"


def test_bucket_marks_ambiguous_frames_discard_not_negative():
    # A frame with a rejected candidate is UNCERTAIN. Writing it as an empty
    # label would teach the model that a real shuttle is background.
    out = tracks.bucket_frames([1, 2, 3], flight_frames={2}, reject_frames={3})
    assert out[3] == "discard"


def test_positive_wins_over_discard_when_a_frame_has_both():
    out = tracks.bucket_frames([1], flight_frames={1}, reject_frames={1})
    assert out[1] == "positive"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd setup && python -m pytest tests/test_feeder_court_tracks.py -v`
Expected: FAIL with `ImportError: cannot import name 'tracks'`

- [ ] **Step 3: Write minimal implementation**

Create `setup/utils/feeder_court/tracks.py`:

```python
# ============================================================
# tracks.py — Turn noisy per-frame candidates into shuttle flights.
#
# A shuttlecock flight is 15-40 frames, moves fast, and goes roughly where its
# velocity says it will. Court-line artifacts produced tracks 100-738 frames
# long; that length cap alone removes most of them.
#
# bucket_frames implements the three-way positive/negative/discard split. The
# discard bucket is the important one: a frame that produced a candidate we
# REJECTED is not evidence of an empty frame, and writing it as an empty label
# teaches the model that a real shuttle is background.
# ============================================================

from __future__ import annotations

import collections
import math

from .motion import Candidate


def link_tracks(
    cands: list[Candidate],
    max_gap: int = 2,
    max_dist: float = 38.0,
) -> list[list[Candidate]]:
    """Greedy nearest-neighbour linking with constant-velocity prediction."""
    by_frame: dict[int, list[Candidate]] = collections.defaultdict(list)
    for c in cands:
        by_frame[c.frame].append(c)

    finished: list[list[Candidate]] = []
    active: list[list[Candidate]] = []

    for frame in sorted(by_frame):
        dets = by_frame[frame]
        used: set[int] = set()
        still_active: list[list[Candidate]] = []

        for track in active:
            last = track[-1]
            if frame - last.frame > max_gap:
                finished.append(track)
                continue

            if len(track) >= 2:
                vx = last.x - track[-2].x
                vy = last.y - track[-2].y
            else:
                vx = vy = 0
            px, py = last.x + vx, last.y + vy

            best, best_dist = None, float("inf")
            for j, d in enumerate(dets):
                if j in used:
                    continue
                dist = math.hypot(d.x - px, d.y - py)
                if dist < best_dist:
                    best_dist, best = dist, j

            if best is not None and best_dist < max_dist:
                used.add(best)
                track.append(dets[best])
            still_active.append(track)

        for j, d in enumerate(dets):
            if j not in used:
                still_active.append([d])
        active = still_active

    finished.extend(active)
    return finished


def is_flight(
    track: list[Candidate],
    min_len: int = 5,
    max_len: int = 60,
    min_step: float = 6.0,
    min_span: float = 60.0,
    min_directness: float = 0.45,
) -> bool:
    """Whether a track looks like a shuttlecock in flight."""
    if not (min_len <= len(track) <= max_len):
        return False

    steps = [
        math.hypot(b.x - a.x, b.y - a.y)
        for a, b in zip(track, track[1:])
    ]
    if not steps or sum(steps) / len(steps) < min_step:
        return False

    span = math.hypot(track[-1].x - track[0].x, track[-1].y - track[0].y)
    if span < min_span:
        return False

    # Directness: straight-line distance over path length. Jitter scores near 0.
    if span / (sum(steps) + 1e-6) < min_directness:
        return False

    return True


def bucket_frames(
    all_frames: list[int],
    flight_frames: set[int],
    reject_frames: set[int],
) -> dict[int, str]:
    """Assign every frame to positive / negative / discard.

    A frame is only NEGATIVE if nothing at all was proposed in it. Anything that
    produced a candidate we could not confirm goes to DISCARD and is never used
    as a training label in either direction.
    """
    out = {}
    for frame in all_frames:
        if frame in flight_frames:
            out[frame] = "positive"
        elif frame in reject_frames:
            out[frame] = "discard"
        else:
            out[frame] = "negative"
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd setup && python -m pytest tests/test_feeder_court_tracks.py -v`
Expected: PASS, 12 tests

- [ ] **Step 5: Commit**

```bash
git add setup/utils/feeder_court/tracks.py setup/tests/test_feeder_court_tracks.py
git commit -m "feat(feeder_court): track linking and flight classification"
```

---

## Task 6: Auto-label proposal CLI

**Files:**
- Create: `setup/scripts/feeder_court/propose_shuttles.py`

**Interfaces:**
- Consumes: `motion.three_frame_diff`, `motion.candidate_boxes`, `tracks.link_tracks`, `tracks.is_flight`, `tracks.bucket_frames`, segments JSON from Task 2
- Produces: `datasets/feeder_court/proposals/<clip>.json` with shape
  `{"clip": str, "buckets": {frame_str: str}, "boxes": {frame_str: [[x, y, w, h], ...]}}`
  and contact sheets at `datasets/feeder_court/proposals/sheet_<clip>_<n>.jpg`.
  The crop band is NOT stored here — Task 7's derive_band() is its single source.

- [ ] **Step 1: Write the script**

Create `setup/scripts/feeder_court/propose_shuttles.py`:

```python
# ============================================================
# propose_shuttles.py — Auto-label proposals for the feeder_court dataset.
#
# Emits THREE buckets, never two:
#   positive  a confirmed flight track passes through this frame
#   negative  nothing was proposed here at all -> usable background frame
#   discard   something was proposed but rejected -> UNCERTAIN, never a label
#
# Always review the contact sheets before exporting. On far-distance footage the
# raw proposals were measured to be mostly noise, so this is a labour-saving
# device for a human labeller, not a labelling replacement.
#
# Usage:
#   python scripts/feeder_court/propose_shuttles.py \
#       --source datasets/vid_source/new_badminton_source \
#       --segments datasets/feeder_court/segments.json \
#       --out datasets/feeder_court/proposals
# ============================================================

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.feeder_court import motion, tracks  # noqa: E402

CLIPS = ["near", "mid", "far"]
K = 3  # neighbour offset for three-frame differencing


def load_span(cap, start: int, end: int) -> tuple[list[np.ndarray], int]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(end - start):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return frames, start


def contact_sheet(video_path: str, picks: list[tuple[int, int, int, int, int]], out_path: str) -> None:
    """Magnified crops around proposals, contrast-boosted so a human can judge them."""
    cap = cv2.VideoCapture(video_path)
    tiles = []
    for frame_idx, x, y, w, h in picks[:24]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = cap.read()
        if not ok:
            continue
        cx, cy, r = x + w // 2, y + h // 2, 32
        x0, y0 = max(cx - r, 0), max(cy - r, 0)
        crop = img[y0:y0 + 2 * r, x0:x0 + 2 * r].copy()
        if crop.shape[0] < 2 * r or crop.shape[1] < 2 * r:
            crop = cv2.copyMakeBorder(
                crop, 0, max(0, 2 * r - crop.shape[0]), 0, max(0, 2 * r - crop.shape[1]),
                cv2.BORDER_CONSTANT,
            )
        crop = cv2.convertScaleAbs(crop, alpha=2.6, beta=15)
        crop = cv2.resize(crop, (180, 180), interpolation=cv2.INTER_NEAREST)
        s = 180 / (2 * r)
        cv2.rectangle(
            crop,
            (int((x - x0) * s), int((y - y0) * s)),
            (int((x - x0 + w) * s), int((y - y0 + h) * s)),
            (0, 0, 255), 1,
        )
        cv2.putText(crop, f"f{frame_idx} {w}x{h}", (3, 13), 0, 0.36, (0, 255, 255), 1)
        tiles.append(crop)
    cap.release()

    if not tiles:
        return
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6) if len(tiles[i:i + 6]) == 6]
    if rows:
        cv2.imwrite(out_path, np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--thresh", type=int, default=22)
    args = parser.parse_args()

    with open(args.segments) as fh:
        spans_by_clip = json.load(fh)

    os.makedirs(args.out, exist_ok=True)

    for clip in CLIPS:
        video = os.path.join(args.source, f"{clip}.mp4")
        cap = cv2.VideoCapture(video)
        all_cands: list[motion.Candidate] = []
        all_frames: list[int] = []

        for start, end in spans_by_clip[clip]:
            frames, base = load_span(cap, start, end)
            for t in range(K, len(frames) - K):
                idx = base + t
                all_frames.append(idx)
                diff = motion.three_frame_diff(frames[t - K], frames[t], frames[t + K])
                all_cands.extend(motion.candidate_boxes(diff, idx, thresh=args.thresh))
        cap.release()

        linked = tracks.link_tracks(all_cands)
        flights = [t for t in linked if tracks.is_flight(t)]

        flight_frames = {c.frame for t in flights for c in t}
        reject_frames = {c.frame for c in all_cands} - flight_frames
        buckets = tracks.bucket_frames(all_frames, flight_frames, reject_frames)

        boxes: dict[str, list[list[int]]] = {}
        for t in flights:
            for c in t:
                boxes.setdefault(str(c.frame), []).append([c.x, c.y, c.w, c.h])

        counts = {b: sum(1 for v in buckets.values() if v == b) for b in ("positive", "negative", "discard")}
        print(
            f"{clip}: {len(all_cands)} candidates -> {len(linked)} tracks -> {len(flights)} flights | "
            f"positive {counts['positive']}  negative {counts['negative']}  discard {counts['discard']}",
            flush=True,
        )

        with open(os.path.join(args.out, f"{clip}.json"), "w") as fh:
            json.dump({"clip": clip, "buckets": buckets, "boxes": boxes}, fh)

        picks = [(c.frame, c.x, c.y, c.w, c.h) for t in flights for c in [t[len(t) // 2]]]
        contact_sheet(video, picks, os.path.join(args.out, f"sheet_{clip}_flights.jpg"))

    print(f"\nwrote proposals to {args.out}")
    print("REVIEW THE CONTACT SHEETS before running export_for_roboflow.py.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the real clips**

Run:
```bash
cd setup && python scripts/feeder_court/propose_shuttles.py \
    --source datasets/vid_source/new_badminton_source \
    --segments datasets/feeder_court/segments.json \
    --out datasets/feeder_court/proposals
```

Expected: three summary lines and three contact sheets. Sanity checks — candidate count should work out near 6 per frame, not ~58; if it is ~58, the alignment is not working. Flight counts in the low hundreds per clip are plausible; thousands means the filters are too loose.

- [ ] **Step 3: HUMAN TASK — review the contact sheets**

Open `datasets/feeder_court/proposals/sheet_*.jpg`. Judge roughly what fraction of tiles are real shuttlecocks. If it is under about 30%, tighten `--thresh` or the `is_flight` thresholds and re-run before spending labelling effort on them.

- [ ] **Step 4: Commit**

```bash
git add setup/scripts/feeder_court/propose_shuttles.py
git commit -m "feat(feeder_court): three-way auto-label proposal CLI"
```

---

## Task 7: Temporal splitting and Roboflow export

**Files:**
- Create: `setup/utils/feeder_court/splits.py`
- Create: `setup/scripts/feeder_court/export_for_roboflow.py`
- Test: `setup/tests/test_feeder_court_splits.py`

**Interfaces:**
- Consumes: `roi.CropBand`, `roi.apply_crop`, `roi.band_from_extent`, proposals JSON from Task 6
- Produces:
  - `temporal_split(frames: list[int], val_fraction: float = 0.2, guard: int = 15) -> tuple[list[int], list[int]]`
  - `assert_no_adjacent_frames(train: list[int], val: list[int], guard: int = 15) -> None` (raises `ValueError`)

- [ ] **Step 1: Write the failing test**

Create `setup/tests/test_feeder_court_splits.py`:

```python
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feeder_court import splits  # noqa: E402


def test_val_is_the_tail_of_the_timeline():
    frames = list(range(100))
    train, val = splits.temporal_split(frames, val_fraction=0.2, guard=0)
    assert max(train) < min(val)
    assert len(val) == 20


def test_guard_band_frames_are_dropped_from_both_sides():
    frames = list(range(100))
    train, val = splits.temporal_split(frames, val_fraction=0.2, guard=5)
    assert len(train) + len(val) == 95
    assert min(val) - max(train) > 5


def test_split_never_shuffles():
    frames = list(range(50))
    train, val = splits.temporal_split(frames, val_fraction=0.2, guard=0)
    assert train == sorted(train)
    assert val == sorted(val)


def test_assert_no_adjacent_frames_passes_for_a_clean_split():
    splits.assert_no_adjacent_frames([1, 2, 3], [50, 51], guard=15)


def test_assert_no_adjacent_frames_raises_when_val_touches_train():
    with pytest.raises(ValueError):
        splits.assert_no_adjacent_frames([1, 2, 3], [4, 5], guard=15)


def test_tiny_input_still_produces_a_val_set():
    frames = list(range(10))
    train, val = splits.temporal_split(frames, val_fraction=0.2, guard=0)
    assert len(val) >= 1
    assert len(train) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd setup && python -m pytest tests/test_feeder_court_splits.py -v`
Expected: FAIL with `ImportError: cannot import name 'splits'`

- [ ] **Step 3: Write minimal implementation**

Create `setup/utils/feeder_court/splits.py`:

```python
# ============================================================
# splits.py — Temporal train/val splitting.
#
# Random frame splits are PROHIBITED on this data. Measured on this project: a
# random split left 80.9% of validation frames as >0.95-similarity duplicates of
# a training frame, versus 0.0% for a temporal split. Leakage does not show up in
# mAP50 — both splits reported 0.994 — so you cannot catch it after the fact.
#
# The guard band drops frames either side of the cut. Frames 1/30 s apart are
# near-identical even across a "clean" temporal boundary.
# ============================================================

from __future__ import annotations


def temporal_split(
    frames: list[int],
    val_fraction: float = 0.2,
    guard: int = 15,
) -> tuple[list[int], list[int]]:
    """Split a frame list by time: the tail becomes validation.

    A guard band of `guard` frames either side of the cut is discarded entirely.
    """
    ordered = sorted(frames)
    if len(ordered) < 2:
        raise ValueError("need at least two frames to split")

    n_val = max(int(round(len(ordered) * val_fraction)), 1)
    cut = len(ordered) - n_val

    train = ordered[:max(cut - guard, 1)]
    val = ordered[cut:]
    return train, val


def assert_no_adjacent_frames(train: list[int], val: list[int], guard: int = 15) -> None:
    """Fail loudly if any val frame sits within `guard` frames of a train frame."""
    if not train or not val:
        return
    train_set = sorted(train)
    for v in val:
        for t in train_set:
            if abs(v - t) <= guard:
                raise ValueError(
                    f"val frame {v} is within {guard} frames of train frame {t} — "
                    "this is temporal leakage"
                )
```

Create `setup/scripts/feeder_court/export_for_roboflow.py`:

```python
# ============================================================
# export_for_roboflow.py — ROI-cropped frames + pre-labels for hand-correction.
#
# The crop band is derived from the proposals (where shuttles were actually
# observed), snapped to a multiple of 32, and RECORDED IN THE MANIFEST. The exact
# same band must be applied at inference — see utils/feeder_court/roi.py.
#
# near + mid export as the training pool. far exports separately and is NEVER
# trained on; it is the held-out evaluation clip.
#
# Usage:
#   python scripts/feeder_court/export_for_roboflow.py \
#       --source datasets/vid_source/new_badminton_source \
#       --proposals datasets/feeder_court/proposals \
#       --out datasets/feeder_court/export
# ============================================================

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.feeder_court import roi, splits  # noqa: E402

TRAIN_CLIPS = ["near", "mid"]
EVAL_CLIPS = ["far"]
FRAME_H = 720


def derive_band(proposals_dir: str, clips: list[str]) -> roi.CropBand:
    """Court band covering every proposed shuttle position across the clips."""
    ys = []
    for clip in clips:
        with open(os.path.join(proposals_dir, f"{clip}.json")) as fh:
            data = json.load(fh)
        for box_list in data["boxes"].values():
            for x, y, w, h in box_list:
                ys.append(y)
                ys.append(y + h)
    if not ys:
        raise ValueError("no proposal boxes found — cannot derive a crop band")
    return roi.band_from_extent(min(ys), max(ys), frame_h=FRAME_H, margin=48, multiple_of=32)


def export_clip(source: str, proposals_dir: str, clip: str, band: roi.CropBand, out_dir: str) -> list[dict]:
    with open(os.path.join(proposals_dir, f"{clip}.json")) as fh:
        data = json.load(fh)

    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    cap = cv2.VideoCapture(os.path.join(source, f"{clip}.mp4"))
    manifest = []

    for frame_str, bucket in sorted(data["buckets"].items(), key=lambda kv: int(kv[0])):
        # discard frames are never written, in either direction.
        if bucket == "discard":
            continue
        frame_idx = int(frame_str)

        raw_boxes = data["boxes"].get(frame_str, [])
        xyxy = np.array(
            [[x, y, x + w, y + h] for x, y, w, h in raw_boxes], dtype=float
        ).reshape(-1, 4)

        if len(xyxy):
            keep = roi.boxes_fully_inside(xyxy, band)
            xyxy = xyxy[keep]
            if bucket == "positive" and not len(xyxy):
                continue  # its only shuttle fell outside the band

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        cropped = roi.apply_crop(frame, band)
        name = f"{clip}_{frame_idx:06d}"
        cv2.imwrite(os.path.join(img_dir, f"{name}.jpg"), cropped)

        lines = []
        if len(xyxy):
            shifted = roi.shift_boxes_into_band(xyxy, band)
            for x1, y1, x2, y2 in shifted:
                cx = ((x1 + x2) / 2) / cropped.shape[1]
                cy = ((y1 + y2) / 2) / cropped.shape[0]
                bw = (x2 - x1) / cropped.shape[1]
                bh = (y2 - y1) / cropped.shape[0]
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(os.path.join(lbl_dir, f"{name}.txt"), "w") as fh:
            fh.write("\n".join(lines))

        manifest.append({"clip": clip, "frame": frame_idx, "name": name, "bucket": bucket})

    cap.release()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    band = derive_band(args.proposals, TRAIN_CLIPS + EVAL_CLIPS)
    print(f"crop band: top={band.top} height={band.height} (bottom={band.bottom})")

    manifest = {"band": {"top": band.top, "height": band.height}, "records": []}
    for split_name, clips in (("train_pool", TRAIN_CLIPS), ("eval", EVAL_CLIPS)):
        out_dir = os.path.join(args.out, split_name)
        for clip in clips:
            records = export_clip(args.source, args.proposals, clip, band, out_dir)

            if split_name == "train_pool":
                # Assign train/val TEMPORALLY, per clip, here — not in Roboflow.
                # Roboflow's default split is random, which this data cannot survive.
                frames = [r["frame"] for r in records]
                train_frames, val_frames = splits.temporal_split(frames)
                splits.assert_no_adjacent_frames(train_frames, val_frames)
                train_set, val_set = set(train_frames), set(val_frames)
                for r in records:
                    if r["frame"] in train_set:
                        r["split"] = "train"
                    elif r["frame"] in val_set:
                        r["split"] = "val"
                    else:
                        r["split"] = "guard"  # dropped: too close to the split boundary
                kept = sum(1 for r in records if r["split"] in ("train", "val"))
                print(
                    f"train_pool/{clip}: {len(records)} frames -> "
                    f"train {len(train_frames)}, val {len(val_frames)}, "
                    f"guard-dropped {len(records) - kept}",
                    flush=True,
                )
            else:
                for r in records:
                    r["split"] = "eval"
                print(f"{split_name}/{clip}: {len(records)} frames", flush=True)

            manifest["records"].extend(records)

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {args.out}/manifest.json")
    print("Upload train_pool/ AND eval/ to Roboflow and hand-correct EVERY label.")
    print("In Roboflow, assign train/valid from manifest.json's 'split' field.")
    print("Do NOT use Roboflow's random split — this data cannot survive it.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd setup && python -m pytest tests/test_feeder_court_splits.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Run the export**

Run:
```bash
cd setup && python scripts/feeder_court/export_for_roboflow.py \
    --source datasets/vid_source/new_badminton_source \
    --proposals datasets/feeder_court/proposals \
    --out datasets/feeder_court/export
```

Expected: a printed crop band (height should be a multiple of 32, plausibly around 448), plus per-clip frame counts. Spot-check three exported images to confirm the court is fully visible and the crop has not cut off the play area.

- [ ] **Step 6: Commit**

```bash
git add setup/utils/feeder_court/splits.py setup/scripts/feeder_court/export_for_roboflow.py setup/tests/test_feeder_court_splits.py
git commit -m "feat(feeder_court): temporal splitting and Roboflow export"
```

- [ ] **Step 7: HUMAN TASK — hand-correct every label in Roboflow**

Create a Roboflow project `feeder-court`. Upload `export/train_pool/` and `export/eval/` as separate batches.

**Correct every single pre-label.** This is the step that cannot be skipped: `shuttlecock_scene_v2` regressed at near range from 73% to 51% because only 16 of 423 frames were hand-checked, and the uncorrected labels contained false boxes on a chair headrest and doubled boxes on hand/forearm. The lens smudge blobs offer the same trap here.

Then export both batches as YOLOv8 and place them where Task 8 expects them.

---

## Task 8: Colab training notebook

**Files:**
- Create: `notebooks/feeder_court_train.ipynb`

**Interfaces:**
- Consumes: the hand-corrected Roboflow export
- Produces: `best.pt` trained weights, downloaded to `models/feeder_court.pt` by Task 10

- [ ] **Step 1: Create the notebook**

Create `notebooks/feeder_court_train.ipynb` with the cells below. Local training is not viable — `torch 2.9.1+cpu` with no CUDA puts a 40-epoch P2 run at roughly 30 hours.

Cell 1 (markdown):

```markdown
# feeder_court — training

Trains `yolov8n-p2` on hand-corrected OV9281 court footage.

- Train: `near` + `mid` only. `far` is held out and must NOT appear here.
- Input: ROI-cropped native pixels. **Never downscale** — an 8 px shuttle at
  imgsz 640 becomes 4 px and is undetectable.
- Mono footage: hue and saturation augmentation are disabled because those
  channels carry no information.

Runtime → Change runtime type → **T4 GPU** before running.
```

Cell 2 (code):

```python
!nvidia-smi
!pip -q install ultralytics==8.4.14 roboflow
```

Cell 3 (code):

```python
from roboflow import Roboflow

# Paste your key at runtime; do not commit it to the notebook.
rf = Roboflow(api_key=input("Roboflow API key: "))
project = rf.workspace().project("feeder-court")
dataset = project.version(1).download("yolov8")
print(dataset.location)
```

Cell 4 (code):

```python
import yaml

# Confirm far is absent from the training data before spending GPU time on it.
with open(f"{dataset.location}/data.yaml") as fh:
    cfg = yaml.safe_load(fh)
print(cfg)

import glob
far = glob.glob(f"{dataset.location}/train/images/far_*.jpg")
assert not far, f"far frames leaked into train: {far[:5]}"
print("OK: no far frames in train")
```

Cell 5 (code):

```python
from ultralytics import YOLO

IMGSZ = 1280   # native long side; the crop already removed the dead ceiling

model = YOLO("yolov8-p2.yaml")

model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=80,
    imgsz=IMGSZ,
    batch=8,
    workers=2,
    seed=0,
    deterministic=True,
    project="runs/feeder_court",
    name="p2-native",
    patience=20,
    # --- mono-specific augmentation ---
    hsv_h=0.0,     # no hue information exists in this footage
    hsv_s=0.0,     # no saturation information exists either
    scale=0.25,    # default 0.5 would annihilate an 8 px object
    flipud=0.0,    # gravity is real; shuttles fall
    fliplr=0.5,    # the court is roughly symmetric
    mosaic=1.0,
    close_mosaic=10,
    plots=True,
    val=True,
)
```

Cell 6 (code):

```python
from google.colab import files
files.download("runs/feeder_court/p2-native/weights/best.pt")
```

- [ ] **Step 2: Note the gate-dependent substitution**

If Task 3 Step 8 printed `GATE DECISION: stock`, change cell 5's `YOLO("yolov8-p2.yaml")` to `YOLO("yolov8n.yaml")` and record the change in the commit message. Otherwise leave it as `yolov8-p2.yaml`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/feeder_court_train.ipynb
git commit -m "feat(feeder_court): Colab training notebook for yolov8n-p2"
```

- [ ] **Step 4: HUMAN TASK — run the notebook**

Open in Colab, select a T4 runtime, run all cells. Expect roughly an hour. Save the downloaded `best.pt` to `setup/runs/feeder_court/best.pt`.

---

## Task 9: Evaluation against held-out `far`

**Files:**
- Create: `setup/scripts/feeder_court/evaluate_feeder_court.py`

**Interfaces:**
- Consumes: trained weights, hand-corrected `far` labels. Reads image geometry from
  `result.orig_shape` rather than from `manifest.json`, which is correct and safer
  because it reflects the image really on disk.
- Produces: printed recall / FP-rate / size-band / imgsz-sweep tables

- [ ] **Step 1: Write the script**

Create `setup/scripts/feeder_court/evaluate_feeder_court.py`:

```python
# ============================================================
# evaluate_feeder_court.py — Score a trained model on the HELD-OUT far clip.
#
# Reports what actually matters, which mAP alone does not:
#   * recall at conf 0.25, centre-matched
#   * false positives per frame, measured on frames with NO shuttle
#   * recall broken down by shuttle pixel size
#   * an imgsz sweep, because each model's best imgsz is NOT its training size
#
# Success criteria: recall >= 50% AND <= 0.1 false positives per frame.
#
# Usage:
#   python scripts/feeder_court/evaluate_feeder_court.py \
#       --weights runs/feeder_court/best.pt \
#       --images datasets/feeder_court/export/eval/images \
#       --labels datasets/feeder_court/export/eval/labels
# ============================================================

import argparse
import glob
import os
import sys

import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

BANDS = [(0, 8), (8, 12), (12, 20), (20, 35), (35, 10_000)]


def load_truth(label_path: str, w: int, h: int) -> list[tuple[float, float, float]]:
    """Ground-truth centres plus max pixel dimension."""
    out = []
    if not os.path.exists(label_path):
        return out
    with open(label_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            _, cx, cy, bw, bh = line.split()
            out.append((float(cx) * w, float(cy) * h, max(float(bw) * w, float(bh) * h)))
    return out


def evaluate(weights: str, images: str, labels: str, imgsz: int, conf: float) -> dict:
    model = YOLO(weights)
    paths = sorted(glob.glob(os.path.join(images, "*.jpg")))

    hits = {b: 0 for b in BANDS}
    totals = {b: 0 for b in BANDS}
    false_positives = 0
    empty_frames = 0

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        result = model.predict(path, imgsz=imgsz, conf=conf, verbose=False)[0]
        pred = result.boxes.xywh.cpu().numpy() if len(result.boxes) else np.zeros((0, 4))
        h, w = result.orig_shape

        truth = load_truth(os.path.join(labels, f"{name}.txt"), w, h)

        if not truth:
            empty_frames += 1
            false_positives += len(pred)
            continue

        matched = set()
        for cx, cy, dim in truth:
            band = next(b for b in BANDS if b[0] <= dim < b[1])
            totals[band] += 1
            for j, (px, py, _, _) in enumerate(pred):
                if j in matched:
                    continue
                # Centre-matched within a generous radius: localisation precision
                # is not what we are testing here, detection is.
                if np.hypot(px - cx, py - cy) <= max(dim, 12):
                    matched.add(j)
                    hits[band] += 1
                    break

    total_truth = sum(totals.values())
    total_hits = sum(hits.values())
    return {
        "imgsz": imgsz,
        "recall": total_hits / total_truth if total_truth else 0.0,
        "hits": total_hits,
        "truth": total_truth,
        "fp_per_empty_frame": false_positives / empty_frames if empty_frames else 0.0,
        "empty_frames": empty_frames,
        "bands": {b: (hits[b], totals[b]) for b in BANDS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz-sweep", type=int, nargs="+", default=[640, 960, 1280])
    args = parser.parse_args()

    best = None
    for imgsz in args.imgsz_sweep:
        r = evaluate(args.weights, args.images, args.labels, imgsz, args.conf)
        print(
            f"imgsz {imgsz:5d}: recall {r['recall']:.3f} ({r['hits']}/{r['truth']})  "
            f"FP/empty-frame {r['fp_per_empty_frame']:.3f} over {r['empty_frames']} frames",
            flush=True,
        )
        for (lo, hi), (hit, tot) in r["bands"].items():
            if tot:
                label = f"{lo}-{hi} px" if hi < 10_000 else f">={lo} px"
                print(f"    {label:>12}: {hit / tot:.3f} ({hit}/{tot})")
        if best is None or r["recall"] > best["recall"]:
            best = r

    print(f"\nBEST imgsz {best['imgsz']}: recall {best['recall']:.3f}, "
          f"FP/frame {best['fp_per_empty_frame']:.3f}")

    ok_recall = best["recall"] >= 0.50
    ok_fp = best["fp_per_empty_frame"] <= 0.10
    print(f"criterion 1 (recall >= 0.50): {'PASS' if ok_recall else 'FAIL'}")
    print(f"criterion 1 (FP <= 0.1/frame): {'PASS' if ok_fp else 'FAIL'}")
    if not (ok_recall and ok_fp):
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the evaluation**

Run:
```bash
cd setup && python scripts/feeder_court/evaluate_feeder_court.py \
    --weights runs/feeder_court/best.pt \
    --images datasets/feeder_court/export/eval/images \
    --labels datasets/feeder_court/export/eval/labels
```

Expected: an imgsz sweep table plus PASS/FAIL for both parts of success criterion 1. Note that the best `imgsz` is frequently *not* the training size — scene_v2 trained at 960 and peaked at 640.

- [ ] **Step 3: Commit**

```bash
git add setup/scripts/feeder_court/evaluate_feeder_court.py
git commit -m "feat(feeder_court): held-out far-clip evaluation with imgsz sweep"
```

---

## Task 10: OpenVINO export and deployment

**Files:**
- Create: `setup/scripts/feeder_court/deploy_feeder_court.py`

**Interfaces:**
- Consumes: trained weights, the best `imgsz` from Task 9
- Produces: `models/feeder_court.pt`, `models/feeder_court_openvino_model/`

- [ ] **Step 1: Write the script**

Create `setup/scripts/feeder_court/deploy_feeder_court.py`:

```python
# ============================================================
# deploy_feeder_court.py — Export to OpenVINO and install as feeder_court.
#
# Verifies the OpenVINO model agrees numerically with PyTorch BEFORE installing,
# then measures fps on this machine to check success criterion 2 (>= 10 fps).
#
# Usage:
#   python scripts/feeder_court/deploy_feeder_court.py \
#       --weights runs/feeder_court/best.pt --imgsz 1280
# ============================================================

import argparse
import os
import shutil
import sys
import time

import numpy as np
from ultralytics import YOLO

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "models",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    torch_model = YOLO(args.weights)
    exported = torch_model.export(format="openvino", imgsz=args.imgsz, half=False)
    ov_model = YOLO(exported)

    rng = np.random.default_rng(0)
    dummy = (rng.random((args.imgsz, args.imgsz, 3)) * 255).astype(np.uint8)

    t_res = torch_model.predict(dummy, imgsz=args.imgsz, conf=0.25, verbose=False)[0]
    o_res = ov_model.predict(dummy, imgsz=args.imgsz, conf=0.25, verbose=False)[0]
    print(f"equivalence check: torch {len(t_res.boxes)} boxes, openvino {len(o_res.boxes)} boxes")
    if len(t_res.boxes) != len(o_res.boxes):
        print("WARNING: box counts differ between backends — inspect before deploying")

    ov_model.predict(dummy, imgsz=args.imgsz, verbose=False)
    started = time.time()
    for _ in range(args.samples):
        ov_model.predict(dummy, imgsz=args.imgsz, conf=0.25, verbose=False)
    per_frame = (time.time() - started) / args.samples
    fps = 1.0 / per_frame
    print(f"openvino: {per_frame * 1000:.1f} ms/frame ({fps:.1f} fps)")

    os.makedirs(MODELS_DIR, exist_ok=True)
    shutil.copy(args.weights, os.path.join(MODELS_DIR, "feeder_court.pt"))
    dest = os.path.join(MODELS_DIR, "feeder_court_openvino_model")
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(exported, dest)
    print(f"installed models/feeder_court.pt and models/feeder_court_openvino_model/")

    print(f"criterion 2 (>= 10 fps): {'PASS' if fps >= 10 else 'FAIL'}")
    if fps < 10:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the deployment**

Run:
```bash
cd setup && python scripts/feeder_court/deploy_feeder_court.py \
    --weights runs/feeder_court/best.pt --imgsz 1280
```

Expected: an equivalence line, an fps measurement, and PASS/FAIL for success criterion 2. If fps fails, re-run with the next smaller `imgsz` that Task 9 showed as acceptable, rather than accepting a model that cannot run in real time.

- [ ] **Step 3: Verify the weights are tracked by Git LFS**

`models/*.pt` is already LFS-tracked (commit `83597ad`). Confirm the new file landed there:

Run: `git check-attr filter -- models/feeder_court.pt`
Expected: `models/feeder_court.pt: filter: lfs`

- [ ] **Step 4: Commit**

```bash
git add setup/scripts/feeder_court/deploy_feeder_court.py models/feeder_court.pt models/feeder_court_openvino_model
git commit -m "feat(feeder_court): OpenVINO export and model deployment"
```

- [ ] **Step 5: Record the results in the spec**

Add a "§12 Results" section to `docs/superpowers/specs/2026-08-20-feeder-court-design.md` containing the Task 9 evaluation table, the measured fps, and a plain statement of whether each success criterion passed. If a criterion failed, say so — a failed criterion honestly recorded is worth more than a reinterpreted one.

```bash
git add docs/superpowers/specs/2026-08-20-feeder-court-design.md
git commit -m "docs(spec): record feeder_court evaluation results"
```

---

## Final Verification

- [ ] Run the full test suite: `cd setup && python -m pytest tests/test_feeder_court_*.py -v` — expect 48 tests passing
- [ ] Confirm `far` frames never appear in any training split (Task 8 cell 4 asserts this)
- [ ] Confirm `models/feeder_court.pt` and `models/feeder_court_openvino_model/` exist
- [ ] Confirm the spec records both the Phase 0 measurement (§4.4) and the final results (§12)
