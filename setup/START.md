# START.md — How to run the models

Everything here runs from `C:\thesis\setup`.

## The three models

| Purpose | Weights | imgsz | conf | max_side | Trained |
|---|---|---|---|---|---|
| Flying shuttle (court) | `models/shuttle_clear_badminton_p2.pt` | 1280 | 0.25 | 60 | 2026-09-02 |
| Landed shuttle (lines) | `models/shuttle_lines_stock_n.pt` | 1280 | 0.40 | 90 | 2026-09-02 |
| Player pose | `models/yolov8n-pose.pt` (stock COCO, never fine-tuned) | 640 | 0.25 | — | — |

All three are registered in `app/pipeline.py` (the `MODELS` dict) under the keys
`shuttle`, `landed` and `pose`. The imgsz/conf above are what `pipeline.py` sets.
OpenVINO exports already exist for both shuttle models; measured on a 1280x720
clip the flying-shuttle model costs 86 ms/frame at imgsz 1280 under OpenVINO
against ~320 ms under PyTorch.

The `landed` model is set to imgsz 1280 in `pipeline.py`, but 640 is measured as
a strictly better deal: 18.3 ms against 54.9, still firing on 98% of frames
against 100%. `live_video.py` already defaults it to 640.

Pose was never trained because COCO already contains `person` with all 17
keypoints. Shuttlecock needed training because COCO has no shuttlecock.

### Court model metrics

Final epoch of `runs/clear_badminton/p2-native/results.csv` (epoch 120):

    precision 1.000   recall 0.715   mAP50 0.736   mAP50-95 0.426

## WARNING: the standalone scripts default to the OLD weights

`scripts/feeder_court/detect_video.py` and `scripts/feeder_court/live_detect.py`
both default to `runs/feeder_court/best.pt`, which is the **2026-08-21** model —
not the 2026-09-02 clear_badminton weights.

**Always pass `--weights` to those two scripts**, or you are testing the old
model and will not know it.

`pose_video.py` and `score_landings.py` already default to the correct weights
and to the clear_badminton dataset videos; they need no override.

Only `app/pipeline.py` points at the new weights by default.

The `models/` copies are the ones `app/pipeline.py` loads and the only ones
that reach the Raspberry Pi (`runs/` is gitignored; `setup/models/*.pt` is
Git LFS tracked). The `runs/` originals remain the training record and are
still what `scripts/feeder_court/*.py` default to.

## Dataset videos

    datasets/vid_source/clear_badminton_dataset_for_collab/
        dataset_court1.mp4
        dataset_court2.mp4
        lines.mp4

---

# Way 1 — `live_video.py` (start here)

One script, any of the three models, any video file, playing at the video's real
frame rate. This is the easiest way to look at a model's output.

```powershell
python scripts/feeder_court/live_video.py --model shuttle
python scripts/feeder_court/live_video.py --model landed
python scripts/feeder_court/live_video.py --model pose
python scripts/feeder_court/live_video.py --model shuttle --video path\to\your.mp4
```

Each model already knows its own weights, input size, confidence and a default
clip, so the bare commands above just work.

Keys: `q` / `ESC` quit, `SPACE` pause, `s` snapshot, `+` / `-` conf +/-0.05,
`h` hide HUD, `r` reset stats.

Useful flags: `--crop x1,y1,x2,y2` (fractions or pixels), `--imgsz`, `--conf`,
`--speed 2`, `--loop`, `--record out.mp4`, `--seconds 10`, `--no-show`,
`--backend torch`.

## Why the video stays at 30 fps

The reader thread paces itself to the clip's own fps and keeps only the newest
frame. The detector takes whatever is newest when it becomes free and discards
anything older. So display rate is the video's rate no matter what the model
costs, and the model's cost appears as the HUD's `age` — how stale the boxes are
— instead of as stutter.

Measured, 1280x720 source, OpenVINO, this i3-1215U:

| model     | imgsz | display  | model rate        |
|-----------|-------|----------|-------------------|
| `shuttle` | 1280  | 30.3 fps | 10.3 fps (97 ms)  |
| `landed`  |  640  | 30.3 fps | 53.6 fps (18.6 ms)|
| `pose`    |  640  | 30.3 fps | 39.4 fps (25.4 ms)|

## Reaching 30 fps on the MODEL, not just the display

`landed` and `pose` already clear 30 fps. `shuttle` at imgsz 1280 does not, and
**must not be sped up by lowering `--imgsz` alone** — at 640 a far-court shuttle
falls below the detector's finest stride and detections roughly halve. Measured
on 100 frames:

| config                    | ms/frame | fps  | frames with a hit |
|---------------------------|----------|------|-------------------|
| full frame @1280          |     86.1 | 11.6 | 31%               |
| full frame @960           |     55.5 | 18.0 | 26%               |
| full frame @640           |     25.3 | 39.5 | 12%  <- collapse  |
| native 640x640 crop @640  |     23.3 | 42.9 | 25%               |

Use `--crop` instead: it cuts the input area while keeping the shuttle at native
scale. The cost is field of view — anything outside the crop is invisible, not
merely missed — so place the crop on the court rather than blindly.

```powershell
python scripts/feeder_court/live_video.py --model shuttle --crop 0.15,0.05,0.85,1.0 --imgsz 896
```

## Reading the numbers honestly

The `hit%` this script prints is **not** a benchmark. Each run scores whatever
frames it happens to reach, so the figure is not comparable between runs. For
real comparisons score a fixed frame set.

`landed` reporting 0.0% at the start of `lines.mp4` is correct, not a failure:
that clip has no shuttles on the floor for its first ~300 frames.

---

# Way 2 — Older standalone scripts (one model at a time)

Each script is self-contained: it imports nothing from `app/` and reads nothing
from `config/settings.py`, so running one cannot disturb the drill system.

Add `--backend torch` to any of these if OpenVINO misbehaves.

## Flying shuttle, on a dataset video

```powershell

python scripts/feeder_court/detect_video.py `
  --weights runs/clear_badminton/p2-native/weights/best.pt `
  --videos datasets/vid_source/clear_badminton_dataset_for_collab/dataset_court1.mp4 `
  --full --show

30fps

python scripts/feeder_court/detect_video.py --playback runs/feeder_court/detect/dataset_court1_boxed.mp4 --fps 30

```

- **For just watching a model, prefer `live_video.py` above.** This script
  scores every frame, so the window advances at INFERENCE speed (~11 fps at
  imgsz 1280), which looks broken even though the model is fine. Its value is
  the per-clip summary table it prints and the annotated mp4 it writes.
- Keys: `q` / `ESC` quit, `SPACE` pause
- Without `--show` it writes an annotated mp4 to `runs/feeder_court/detect`
- `--no-save` watches without writing anything
- `--full` ignores `segments.json` and processes the whole clip
- Inference defaults to imgsz 1280 deliberately. Dropping to 640 halves the
  shuttle to sub-pixel size on far footage and the P2 head stops firing.

## Landed shuttle + line calls

Defaults are already correct for this one (`lines.mp4`, the shuttle_lines
weights), so it can be run almost bare:

```powershell
python scripts/feeder_court/score_landings.py --show `
  --lines datasets/clear_badminton/calibration/lines_f300_sketch.json `
  --events runs/landings.json
```

- `--lines` takes a `sketch_frame.py` JSON and gives in/out calls against two
  traced lines. Without it you get pixel coordinates only.
- `--court` takes a 4-point calibration JSON instead, for metre-accurate calls.
- A landing must hold still for `--confirm-frames` (default 4) before it counts,
  because test precision is 0.752 and a single-frame flicker would score a point.

## Player pose

Defaults are already correct (`dataset_court1.mp4`, `yolov8n-pose.pt`):

```powershell
python scripts/feeder_court/pose_video.py --show
```

- The default crop `200,330,1120,720` is **hand-measured for court1's camera**
  and is wrong for any other camera position. Pass `--crop` for court2.
- `--top-n 2` for singles, `4` for doubles.
- Cropping rather than downscaling is deliberate: shrinking the whole frame
  turns a 128 px player into 64 px and detection collapses. The crop keeps
  players at native scale and discards only ceiling and spectators.

## Live USB camera

```powershell
python scripts/feeder_court/live_detect.py --list-cameras

python scripts/feeder_court/live_detect.py `
  --weights runs/clear_badminton/p2-native/weights/best.pt
```

- Keys: `q` / `ESC` quit, `s` snapshot, `+` / `-` conf +/-0.05,
  `h` hide HUD, `SPACE` pause, `r` reset stats
- `--source 1` to pick another device index
- `--record live.mp4` to save the annotated view
- Capture and inference run on separate threads, so the window always shows the
  newest frame and the boxes lag behind it by the HUD's `age` value instead of
  the display stalling.
- The feed is **not** downscaled: the camera is asked for 1280x720 and the frame
  goes to the model at native size.

---

# Way 3 — The server (all three models at once)

Each of the four camera slots gets its own model, so this is how to run more
than one model simultaneously.

```powershell
python run_server.py
```

Then open <http://localhost:8000/live>.

## Driving it by API

Point a slot at a video, then start a model on it:

```powershell
curl -X POST localhost:8000/api/cameras/front/source `
  -H "Content-Type: application/json" `
  -d '{\"kind\":\"file\",\"path\":\"datasets/vid_source/clear_badminton_dataset_for_collab/dataset_court1.mp4\"}'

curl -X POST localhost:8000/api/cameras/front/start `
  -H "Content-Type: application/json" -d '{\"model\":\"shuttle\"}'
```

- Slots: `front`, `left`, `right`, `back`
- Models: `shuttle`, `landed`, `pose`, `none`
- MJPEG stream: `GET /api/cameras/front/stream`
- Stop: `POST /api/cameras/front/stop`
- List what is available: `GET /api/models`, `GET /api/sources`, `GET /api/devices`
- Pipeline status: `GET /api/pipeline`

To watch all three models on the same footage at once, point `front`, `left` and
`right` at the same file and start a different model on each. Capture and
inference are per-slot threads so they will not stall each other, but three
1280 models share one i3 CPU.

## First-run pause

Starting a model whose OpenVINO export is missing triggers an export of roughly
30 seconds, once. `pose` has no 640 export yet, so expect that pause the first
time it is started.

Note that OpenVINO compiles on first inference too. Warm up several frames
before trusting any timing, or the benchmark reads as slower than PyTorch.

## A configuration trap

The server reads `shuttleSource` from the persisted settings document in
MongoDB, and that **overrides** `config/settings.py`. Editing the file alone
will look ignored. Change it on the Settings page or via `PUT /api/settings`.
