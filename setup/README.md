# Badminton Feeder Trainer — Computer Vision System

## Full System Flow

```
┌─────────────────────────────────────────────────────────┐
│                     CAMERA (Side View)                  │
│                   OV9281 USB Grayscale                  │
└───────────────────────┬─────────────────────────────────┘
                        │ frame
          ┌─────────────┴──────────────┐
          ▼                            ▼
┌─────────────────┐         ┌──────────────────────┐
│  YOLOv8n-pose   │         │  Custom Roboflow      │
│  (Pre-trained)  │         │  Shuttlecock Model    │
│                 │         │  (train when ready)   │
│ Detects:        │         │                       │
│ - Person boxes  │         │ Detects:              │
│ - Ankle coords  │         │ - Shuttle (x, y)      │
│ - Track IDs     │         │ - Crossing net        │
└────────┬────────┘         └──────────┬────────────┘
         │                             │
         ▼                             ▼
┌─────────────────────────────────────────────────────────┐
│              COURT ZONE FILTER (hardcoded)              │
│  COURT_ZONE = (x1, y1, x2, y2)  ← set in settings.py  │
│  Only players with 60%+ overlap inside zone counted     │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                  PLAYER TRACKING                        │
│  persist=True → each player keeps same ID even         │
│  when they swap left/right positions                    │
│  Ankle position → which of 6 zones player is in        │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                 SHOT LOGIC (per shot)                   │
│                                                         │
│  1. Pick random zone from 6 court zones                 │
│  2. Find player ID standing in that zone RIGHT NOW      │
│  3. Record shot → player_scores[id][zone][shots]++      │
│  4. Fire feeder to that zone                            │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              SHUTTLE TRACKING (net crossing)            │
│                                                         │
│  shuttle_x > NET_X → feeder side                        │
│  shuttle_x < NET_X → player side                        │
│                                                         │
│  Crossing detected:                                     │
│    prev_x > NET_X AND current_x < NET_X                 │
│    → capture (x, y) → match to zone                     │
│    → confirm player ID in that zone                     │
└───────────────────────┬─────────────────────────────────┘
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
┌───────────────────┐   ┌───────────────────────────────┐
│  Return detected  │   │        Timer expires           │
│  shuttle crosses  │   │        (easy=5s, med=3s,       │
│  back to feeder   │   │         hard=1.5s)             │
│  side (2+ frames) │   │                               │
└────────┬──────────┘   └──────────────┬────────────────┘
         │                             │
         ▼                             ▼
┌─────────────────┐         ┌──────────────────────┐
│   SCORE ✅      │         │      MISS ❌          │
│                 │         │                       │
│ player_scores   │         │ No score recorded     │
│ [id][zone]      │         │ Shot counted only     │
│ [score]++       │         │                       │
└────────┬────────┘         └──────────┬────────────┘
         └─────────────┬───────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  DISPLAY OVERLAY                        │
│                                                         │
│  - Court boundary box                                   │
│  - Net line (NET_X)                                     │
│  - 6 zone grid (active zone highlighted)                │
│  - Player skeleton + ID label                           │
│  - Shuttle position + trail                             │
│  - Live scoreboard (per player per zone)                │
│  - Weak zones highlighted in red                        │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              END OF DRILL ASSESSMENT                    │
│                                                         │
│  Player 1  |  7/10 (70%)                               │
│    back_left   3/4   75%                                │
│    back_right  0/2    0%  ← weak!                       │
│    mid_center  4/4  100%                                │
│                                                         │
│  Player 2  |  5/10 (50%)                               │
│    back_mid    0/3    0%  ← weak!                       │
│    mid_right   3/4   75%                                │
└─────────────────────────────────────────────────────────┘
```

---

## File Structure

```
badminton_feeder/
├── main.py                 ← Run this to start the drill
├── calibrate.py            ← Run this FIRST to set up coordinates
├── requirements.txt        ← pip install -r requirements.txt
├── config/
│   └── settings.py         ← All hardcoded values (calibrate here)
├── utils/
│   ├── zones.py            ← Zone matching + player detection logic
│   ├── scoring.py          ← Per player per zone score tracking
│   ├── display.py          ← All OpenCV drawing/overlay functions
│   ├── roboflow_client.py  ← Roboflow serverless Workflow client (stills)
│   └── roboflow_stream.py  ← Roboflow WebRTC streaming (live video)
├── tests/
│   └── test_roboflow_client.py  ← Smoke test for the Roboflow integration
└── models/
    └── shuttlecock.pt      ← Place your local Roboflow model here later
```

---

## Setup Steps

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Calibrate your camera
```bash
python calibrate.py
```
- Click the 4 court corners → copy COURT_ZONE value to settings.py
- Click the net line → copy NET_X value to settings.py
- Adjust PLAYER_ZONES in settings.py to match your 6 zones

### 3. Run the drill
```bash
# Medium difficulty, unlimited shots
python main.py

# Hard difficulty, 20 shots
python main.py --difficulty hard --shots 20

# Easy difficulty
python main.py --difficulty easy
```

### 4. Controls while running
```
Q     = quit + show final assessment
SPACE = pause / resume
E     = switch to easy
M     = switch to medium
H     = switch to hard
```

---

## Shuttle Detection (Roboflow)

There are several ways to run your trained shuttlecock model. Pick based on
whether you need real-time speed and whether you can download weights.

> **Recommended when on the free plan:** Option D — the direct model over
> serverless. It works today, needs no paid export, and sidesteps the
> workflow's compile bug. Set `SHUTTLE_SOURCE = "serverless"` in
> `config/settings.py` and run `python main.py`.

### Option D — Direct model over serverless (FREE, works now) ⭐

Calls the trained model `shuttlecock-m9ihi-nimwo/1` directly over the free
serverless API — **no workflow, no paid export**. Detections flow straight
into the drill via `get_shuttle_position()`.

```python
# config/settings.py
SHUTTLE_SOURCE = "serverless"
```
```bash
python main.py --difficulty medium
```

Trade-off: each frame is an HTTP call (~0.1–0.4 s), so the loop runs at a few
FPS — fine for testing and slower drills, not high-speed tracking. The
underlying function is reusable directly:

```python
from utils.roboflow_client import run_shuttlecock_model, extract_shuttle_xy
det = run_shuttlecock_model("https://example.com/rally.jpg", confidence=40)
xy  = extract_shuttle_xy(det)   # (x, y) of top detection, or None
```

### Option A — Local model weights (fastest, but needs a paid plan)

1. Record footage of your feeder setup from your side-view camera
2. Upload to Roboflow → annotate shuttlecock
3. Train YOLO model
4. Export/**download weights** as YOLO PyTorch (.pt) — *requires a paid
   Roboflow plan*
5. Place them at `models/shuttlecock.pt`

Runs on-device every frame, no network. When the file exists, `main.py`
loads it automatically (`SHUTTLE_MODEL_PATH` in `config/settings.py`) and
`get_shuttle_position()` uses it — no code edits needed. Without the file,
the drill still runs with player/zone logic only.

### Option B — Real-time WebRTC streaming (`utils/roboflow_stream.py`)

Cloud-hosted GPU streaming of the **full workflow** on live video — webcam,
RTSP, or a video file. This is the real-time path when you can't download
weights (Option A) but need more than the 1–3 FPS of the serverless client
(Option C).

> **Two prerequisites, both external to this code:**
> 1. **Python 3.10–3.12** — `inference-sdk` does not support 3.13 (the repo's
>    default). Make a separate env for streaming.
> 2. **A compiling workflow** — fix the workflow's `model_id` binding in the
>    Roboflow editor first, or streaming 500s just like the REST call.

```bash
py -3.12 -m venv .venv-stream
.venv-stream\Scripts\activate
pip install -U "inference-sdk[webrtc]"

# uses ROBOFLOW_API_KEY from .env
python -m utils.roboflow_stream --source webcam
python -m utils.roboflow_stream --source rtsp  --url rtsp://host:8554/stream
python -m utils.roboflow_stream --source video --path clip.mp4

# Discover the workflow's real data-channel output keys (first frame, then exit)
python -m utils.roboflow_stream --source webcam --list-outputs
```

The window shows the annotated stream; per-frame predictions arrive on the
data channel (logged as a small, blob-free summary — hook your shuttle
tracking/scoring into `_on_data`). Use `--list-outputs` first to print the
exact output key names to wire scoring against.

### Option C — Serverless Workflow client (`utils/roboflow_client.py`)

Calls your hosted Roboflow Workflow
`shuttlecock-vshuttlecock-m9ihi-nimwo-1-yolo11s-t1-logic`
over REST (one HTTP round-trip per image). Great for testing, batch
processing, or verifying your workflow — **not** for high-FPS live video
(each frame is a network call). For real-time streaming use Roboflow's
`InferencePipeline` or the WebRTC path instead.

**Setup**

```bash
pip install -r requirements.txt

# Your Roboflow API key (https://app.roboflow.com/settings/api).
# Never hard-code it. On Windows PowerShell:
$env:ROBOFLOW_API_KEY = "rf_xxx"
# ...or put it in a .env file (already gitignored) and load it yourself.
```

**Usage**

```python
import cv2
from utils.roboflow_client import (
    run_shuttlecock_workflow, extract_shuttle_xy,
    summarize_outputs, save_image_outputs,
)

# image can be: an https URL, a local file path, a base64 string,
# or a numpy array (an OpenCV frame).
result = run_shuttlecock_workflow("https://example.com/rally.jpg")

entry = result[0]                       # one entry per input image
print(summarize_outputs(entry))         # log-safe: no blobs, no polygons
xy = extract_shuttle_xy(entry)          # (x, y) of top detection, or None
save_image_outputs(entry, out_dir=".")  # writes any annotated-image outputs
```

The API key is read from `ROBOFLOW_API_KEY`. Calls have a request timeout
plus retries with exponential backoff, and raise typed errors
(`RoboflowConfigError`, `RoboflowInferenceError`).

Besides `image`, the workflow declares these tunable inputs — pass any of
them via `parameters=`:

```python
run_shuttlecock_workflow(frame, parameters={
    "confidence": 0.4,
    "iou_threshold": 0.5,
    "class_agnostic_nms": False,
    "max_detections": 10,
})
```

**Smoke test**

```bash
# Skips cleanly if ROBOFLOW_API_KEY is unset; runs live if it is set.
python -m pytest tests/test_roboflow_client.py -s
# or without pytest:
python tests/test_roboflow_client.py
```

Until a shuttle model is wired in, player detection and court-zone logic
work fully; shuttle scoring activates once detection is ready.

---

## Moving to Raspberry Pi Later

When you get your Pi:
1. Copy this entire folder to the Pi
2. Change `CAMERA_INDEX` if needed
3. Wire feeder to GPIO → uncomment GPIO code in `fire_feeder()`
4. Run exactly the same — no other changes needed
