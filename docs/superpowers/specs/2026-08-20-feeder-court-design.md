# feeder_court — Shuttlecock Detector Design Spec

**Date:** 2026-08-20
**Repo:** `C:\thesis`
**Status:** Approved design — ready for implementation planning
**Scope:** Shuttlecock detection only. Line detection, net-crossing and scoring are explicitly out of scope and require their own dataset.

## 1. Goal

Train a shuttlecock detector — `feeder_court` — that works on **OV9281 footage recorded in the actual deployment hall**, replacing v1/v2/v3 and `shuttlecock_scene_v2`, none of which were trained on the target environment.

Deliverables:

- `models/feeder_court.pt`
- `models/feeder_court_openvino_model/`
- A measured, defensible evaluation on a held-out distance condition.

The single question this spec answers: **does a shuttlecock detector work at all on this camera in this hall?** Everything downstream (tracking, net-crossing, line calls, scoring) is deferred.

## 2. Source data

`setup/datasets/vid_source/new_badminton_source/` — 5 clips, all 1280×720, ~30.2 fps, **genuinely monochrome** (measured per-channel difference = 0.00).

| clip | frames | duration | role |
|---|---|---|---|
| `near.mp4` | 2242 | 74.2 s | train |
| `mid.mp4` | 1664 | 55.1 s | train |
| `far.mp4` | 1875 | 62.1 s | **held-out eval** |
| `line_1.mp4` | 1435 | 47.5 s | **excluded** — belongs to the future line-detection dataset |
| `line_2.mp4` | 1338 | 44.3 s | **excluded** — same |

`near` / `mid` / `far` differ by **player position on court**, with the camera behind the baseline at approximately net height.

### 2.1 Usable windows

Every clip has a heavy camera-motion segment at head and tail (rig being carried into and out of position — `near.mp4` ends on a close-up of a laptop keyboard). Measured approximate usable windows:

| clip | usable frames | count |
|---|---|---|
| `near` | 200–2100 | ~1900 |
| `mid` | 250–1480 | ~1230 |
| `far` | 200–1740 | ~1540 |

Total usable ≈ **4,670 frames**, not the naive 8,554.

Exact boundaries are computed by the implementation, not hard-coded from this table.

### 2.2 Conditions the design must tolerate

1. **Dirty lens.** Every frame carries large soft bright smudge/condensation blobs. These are bright, roundish and white-ish at multiple scales — the exact false-positive class that made `shuttlecock_v2` fire on a blank wall at 0.767 confidence.
2. **Severe underexposure.** The upper third of frame is near-black. A shuttle lofted high against the ceiling may carry almost no contrast.
3. **Camera drift.** The clips are *not* tripod-locked. Even within the usable windows the view drifts by tens of pixels. Measured phase-correlation confidence is 0.08–0.10 — low, because the scene is dark and low-contrast.

Consequence of (3): **median-background subtraction does not work on this footage.** It was tested; static court lines light up in the difference and dominate the output (90,354 raw candidates ≈ 58/frame on `far`, of which the strongest by contrast×length were visibly court-line edges and player limbs, with track lengths of 100–738 frames where a real shuttle flight is 15–40).

This also rules out the existing tooling as-is: `scripts/autolabel_scene_video.py` and `scripts/shuttle_motion_prototype.py` both assume a fixed camera, and the former additionally keys on HSV saturation, which does not exist in mono footage.

## 3. Key design decisions (locked)

| Decision | Choice | Why |
|---|---|---|
| Scope | Detection only | Scoring needs a line dataset that does not exist yet. |
| Train / eval split | Train `near`+`mid`, hold out `far` entirely | Directly measures distance generalisation — the variable identified as decisive. |
| Labeling | Motion auto-label → **full** hand-correction | Auto-label alone was proven to poison `shuttlecock_scene_v2`. |
| Gate before labeling | 60-frame calibration with an explicit kill criterion | v1–v3 each failed for a data reason nobody measured up front. |
| Architecture | `yolov8n-p2` (stride-4 head) | Standard fix for sub-16 px objects; the one architectural knob never tried on this problem. |
| Input | ROI crop to the court band, native resolution, **no downscale** | 2.7× cheaper than full-frame letterbox at zero cost in shuttle pixels. |
| Training venue | Google Colab free T4 | Local CPU-only puts a 40-epoch P2 run at ~30 hours. |
| Deployment | OpenVINO IR exported locally | Measured 7× speedup on the i3-1215U. |

## 4. Phase 0 — calibration gate

**Runs before any bulk labeling effort.**

Sample 20 frames per clip, evenly spaced across the usable windows (60 total). Upload to Roboflow. Every visible shuttlecock is hand-boxed by the user. The implementation then measures the box size distribution per clip.

### 4.1 Decision rule

| measured median max-dimension | action |
|---|---|
| ≥ 16 px | Stock `yolov8n` suffices; P2 head optional |
| 8–16 px | Proceed with `yolov8n-p2` as specified |
| < 8 px across all three clips | **STOP.** YOLOv8n cannot detect this. Escalate to fallbacks (§4.3) |

### 4.2 Why this gate exists

Automated probing produced a result that cannot be trusted without human ground truth. Drift-compensated three-frame differencing plus ballistic track linking gave, over 13-second windows:

| clip | flight-like tracks | p25 | median | p75 | p95 |
|---|---|---|---|---|---|
| `far` | 35 | 6 px | 8 px | 11 px | 24 px |
| `near` | 58 | 6 px | 8 px | 11 px | 20 px |

`near` and `far` returning an **identical** distribution is a red flag, not a finding. Either the detector is still measuring noise and line artifacts in both, or `near`/`mid`/`far` changes player position without much changing shuttle-to-camera distance. Phase 0 resolves this, and it is the only thing that can.

The ~8 px figure, if it survives calibration, is genuinely marginal: YOLOv8n's finest stride is 8, so an 8 px object occupies a single grid cell at native resolution — and at `imgsz 640` on a 1280-wide frame it degrades to 4 px, which is hopeless. This is the gate set by the existing project note `shuttle-longrange-root-cause`: *"if it is ~6 px, no YOLOv8n variant will work."*

### 4.3 Fallbacks if the gate fails

Chosen by the user at that point, not pre-committed here:

- 2× upscale of the ROI crop (the only option that genuinely adds pixels; costs 4× compute in the cropped region)
- Scope shuttle tracking to the near half-court only
- Re-record with a cleaned lens and corrected exposure

## 5. ROI crop

The black ceiling is dead area — no shuttle ever appears there. Cropping the court band discards only dead pixels.

- The band is **derived from the data** (union of court extent and observed shuttle positions across clips), never hand-typed from eyeballing a frame.
- It is stored in config as an explicit named constant.
- It is **applied identically at training and inference.** A train/infer crop mismatch is a silent correctness bug of exactly the kind that produced the `imgsz 640` half-scaling bug documented in `shuttle-longrange-root-cause`.
- Target shape ≈ 448×1280.

Measured cost, raw PyTorch CPU, batch 1:

| model | input | latency |
|---|---|---|
| `yolov8n` | 448×1280 | 326 ms |
| `yolov8n` | 1280×1280 | 877 ms |
| `yolov8n-p2` | 448×1280 | 615 ms |
| `yolov8n-p2` | 1280×1280 | 1670 ms |

Applying the measured 7× OpenVINO speedup: `n-p2 @ 448×1280` ≈ 88 ms (~11 fps), stock `n @ 448×1280` ≈ 47 ms (~21 fps). Both deployable.

## 6. Components

All new scripts live in `setup/scripts/feeder_court/`, keeping this pipeline separate from the v1–v3 scripts it supersedes. Each is independently runnable and independently checkable.

### 6.0 `sample_calibration.py`

**Does:** samples 20 frames per clip evenly across the usable windows and writes them for upload; after the user has boxed them, measures the size distribution per clip and prints the §4.1 decision.
**Input:** clip paths + usable segments.
**Output:** calibration frame set; then a per-clip size report.

This is the Phase 0 tool and it runs before everything below.

### 6.1 `segment_clips.py`

**Does:** motion-profiles each clip and emits per-clip usable windows.
**Input:** clip paths.
**Output:** JSON of `{clip: [(start_frame, end_frame), ...]}`.
**Depends on:** OpenCV only.

Drops setup/teardown junk, and additionally splits the usable window wherever drift is large enough to break the differencing assumption in §6.2.

### 6.2 `propose_shuttles.py`

**Does:** proposes shuttle boxes using drift-compensated three-frame differencing plus ballistic track linking.
**Input:** clip + usable segments.
**Output:** candidate boxes bucketed three ways, plus contact sheets for review.

Method, as prototyped and validated to reduce candidates from ~58/frame to 6.0/frame:

1. Align frames `t−3` and `t+3` to frame `t` by phase correlation (translation only) — this is what makes it survive the drift.
2. `min(|t − t₋₃|, |t − t₊₃|)` — an object must differ from **both** neighbours, isolating fast movers and cancelling static structure including the lens blobs, which are fixed in image coordinates.
3. Threshold, morphological open, connected components, filtered on size / aspect ratio / fill.
4. Link across frames with velocity prediction; keep tracks of 5–60 frames that are fast (mean step ≥ 6 px), travel somewhere (span ≥ 60 px) and are reasonably directed.

**Three output buckets, not two:**

- `positive` — confident shuttle
- `negative` — confident empty; becomes background training data
- `discard` — uncertain; **never written as a false negative**

The three-way pattern is carried over from the existing `scripts/autolabel_scene_video.py`, where it exists precisely so uncertain frames cannot poison labels.

### 6.3 `export_for_roboflow.py`

**Does:** writes ROI-cropped frames plus pre-labels in YOLO format, ready for upload.
**Output:** an image/label tree plus a manifest recording which clip and frame each image came from — required for the temporal split in §8.

### 6.4 Roboflow project `feeder-court`

The user hand-corrects **every** exported pre-label.

This is not optional and not a nice-to-have. `shuttlecock_scene_v2` regressed at near range from 73% to 51%, and the cause identified was that only 16 of 423 frames were hand-checked; the uncorrected pre-labels contained false boxes on a chair headrest and doubled boxes on hand/forearm. The same failure mode is available here via the lens blobs.

### 6.5 `evaluate_feeder_court.py`

**Does:** scores a trained model against the held-out `far` clip.
**Input:** weights + hand-verified `far` labels.
**Output:** the §9 numbers — recall at conf 0.25, false positives per frame measured on shuttle-free frames, a per-size-band breakdown, and an `imgsz` sweep.
**Depends on:** the ROI crop config from §5, applied identically to how training applied it.

### 6.6 `deploy_feeder_court.py`

**Does:** exports the Colab-trained weights to OpenVINO IR and installs both artefacts.
**Output:** `models/feeder_court.pt`, `models/feeder_court_openvino_model/`.

Must verify numerical equivalence between the PyTorch and OpenVINO outputs before installing, and must re-measure fps on the i3-1215U to confirm success criterion 2.

## 7. Training

`yolov8n-p2`, native resolution ROI crop, on Colab T4. `yolov8-p2.yaml` is confirmed present in the installed `ultralytics 8.4.14`.

Flow: label in Roboflow → export dataset → train on Colab → download `best.pt` → export OpenVINO locally → deploy as `models/feeder_court.pt`.

### 7.1 Mono-specific augmentation

The defaults are wrong for this footage in ways that matter:

| setting | value | why |
|---|---|---|
| `hsv_h` | 0 | No hue information exists. Default augments a nonexistent channel. |
| `hsv_s` | 0 | No saturation information exists. Same. |
| `hsv_v` | keep | Exposure genuinely varies and the hall is dark. |
| `scale` | 0.25 | Default 0.5 aggressively downscales; that annihilates an 8 px object. |
| `flipud` | 0 | Gravity is real; shuttles fall. |
| `fliplr` | 0.5 | The court is roughly symmetric. |
| `mosaic` | on, `close_mosaic=10` | Helps small objects; closed at the end for clean convergence. |

## 8. Validation

**`far` is held out entirely and is the headline number.** It is never trained on and never used for early stopping.

Within `near`+`mid`, the training-time validation set is a **temporal** split — the last 20% of each segment — so early stopping is not driven by memorised frames. Random frame splits are prohibited: they were measured to leave 80.9% of validation frames as >0.95-similarity near-duplicates of a training frame, versus 0.0% for a temporal split.

In-scene validation metrics are reported but **explicitly labelled optimistic.** Same-session holdouts on this problem have read ~0.99 regardless of real performance; that is the documented `single-clip-training-ceiling` result and it is why v1's 0.898 and its ~2% real-camera rate were never actually in conflict.

**False-positive rate is measured separately on shuttle-free frames.** The lens blobs are present in every frame and are the `shuttlecock_v2` blank-wall failure mode waiting to happen. A model that scores well on recall while firing on smudges is not shippable.

Additionally, per the `shuttle-scene-v2-distance` finding that each model's best `imgsz` is not its training size, **`imgsz` is swept at evaluation** rather than assumed.

## 9. Success criteria

1. Detects **≥ 50%** of hand-verified shuttle instances in held-out `far` at confidence 0.25, with **≤ 1 false positive per 10 frames**.
2. **≥ 10 fps** after OpenVINO export on the i3-1215U.

Both must hold. Criterion 1 without criterion 2 is a model that cannot run; criterion 2 without criterion 1 is a fast model that does not work.

## 10. Known limitations

To be stated plainly in the thesis rather than discovered by an examiner:

- **Single recording session.** The `far` holdout is a distance proxy, not cross-session validation. Genuine generalisation remains unproven, and prior work on this project established that two sessions in two rooms still did not produce a general detector.
- **Dirty lens throughout.** The model will have learned to tolerate one specific smudge pattern. Cleaning the lens changes the input distribution.
- **Severe underexposure.** A shuttle lofted against the black ceiling may be genuinely invisible — carrying no recoverable signal. No architecture fixes missing photons.
- **`near` vs `far` size anomaly.** If Phase 0 confirms shuttle size does not vary meaningfully between the three clips, then the `far` holdout is *not* a distance-generalisation test, and the validation story in §8 must be re-stated honestly rather than reinterpreted favourably.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Auto-label proposals are mostly noise at far distance (observed) | Three-way bucketing with a `discard` class; export only high-confidence tracks; full hand-correction |
| ~8 px shuttles are below the architecture's floor | Phase 0 gate with an explicit stop condition and named fallbacks |
| Camera drift shifts the ROI band between clips | Crop band defined per clip, derived from data |
| Hand-correction burden larger than hoped | Phase 0 gives a realistic per-frame cost estimate before committing |
