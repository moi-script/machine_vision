# Testing the feeder_court detector at near / mid / far distance

How to run the shuttlecock detector over the three distance clips and judge
whether the result is real. The clips exist because shuttle size in pixels is
the variable that breaks this model: `near.mp4`, `mid.mp4` and `far.mp4` are the
same court and camera at three subject distances, so comparing them isolates
distance from every other factor.

## What you need

| Thing | Path |
|---|---|
| Weights | `runs/feeder_court/best.pt` |
| Videos | `datasets/vid_source/new_badminton_source/{near,mid,far}.mp4` |
| Rally spans | `datasets/feeder_court/segments.json` |
| Script | `scripts/feeder_court/detect_video.py` |

The weights are the Colab run `p2-native`: yolov8-p2 head, trained at imgsz
1280 on 53 hand-drawn boxes. Best epoch 64 — P 0.861, R 0.583, mAP50 0.667,
mAP50-95 0.248.

Ultralytics and OpenCV must be importable:

```
python -c "import ultralytics, cv2; print(ultralytics.__version__, cv2.__version__)"
```

## Running it

From `C:\thesis\setup` (the script re-roots itself, so the working directory
does not actually matter):

```
python scripts/feeder_court/detect_video.py
```

That processes the rally span of all three clips and writes an annotated copy of
each to `runs/feeder_court/detect/<clip>_boxed.mp4`. Each frame gets the boxes
burned in with their confidence, plus a HUD reading `<clip>  f<frame>  det=<n>`
so a suspicious frame can be named and re-examined.

One clip at a time, with a lower threshold:

```
python scripts/feeder_court/detect_video.py --clips far --conf 0.15
```

Whole file instead of the rally span:

```
python scripts/feeder_court/detect_video.py --clips mid --full
```

### Flags

| Flag | Default | Why you would change it |
|---|---|---|
| `--conf` | 0.25 | Lower to trade precision for recall on far footage |
| `--imgsz` | 1280 | **Leave alone.** The P2 head was trained at 1280; at 640 the shuttle drops toward sub-pixel and detection collapses |
| `--max-side` | 60 | Drops boxes whose longest side exceeds this, in px. `0` disables — see below |
| `--max-det` | 20 | Per-frame cap before filtering |
| `--iou` | 0.5 | NMS threshold |
| `--clips` | all three | `near`, `mid`, `far` |
| `--full` | off | Ignore `segments.json` |
| `--weights` | `runs/feeder_court/best.pt` | Point at a different run |

### Why `--max-side` exists

Every hand-drawn box in `feeder_court_yolo` is between **8 and 40 px** on a
side (n=53). On `mid.mp4` the raw model emits roughly 20 boxes per frame of
300–600 px stacked over the ceiling lights, and they survive even a 0.80
confidence threshold — so thresholding alone cannot remove them. A 500 px box
is not a shuttlecock by definition, so the size filter removes them on a
geometric argument rather than a tuned one.

Run with `--max-side 0` once to see the raw behaviour. It is worth seeing: the
model **is** still producing that noise, the filter just discards it before
drawing. Do not mistake a clean output video for a clean model.

## Reading the summary

The run ends with one row per clip:

```
clip   span            frames  hit%   dets  oversized  mean_conf  ms/frame
near   (105, 2130)     2025     19.4  490   ...            0.501     317.0
mid    (195, 1530)     1335     23.3  359   ...            0.448     310.9
far    (120, 1755)     1635     21.3  442   ...            0.408     426.9
```

- `hit%` — share of frames with at least one surviving box
- `dets` — total surviving boxes
- `oversized` — boxes discarded by `--max-side`; a large number here means the
  model is degenerating on that clip even if the video looks fine
- `mean_conf` — mean confidence of surviving boxes
- `ms/frame` — CPU inference at imgsz 1280, i3-1215U

**`hit%` is not recall.** The shuttle is not airborne in every frame of a span,
and there are no hand labels over these spans to score against. A 21.3% hit rate
means "a box was drawn in 21.3% of frames" and nothing more. Do not report it as
accuracy.

## Verifying the boxes are real

The summary cannot tell a shuttlecock from a light fixture. Sample the output
video and look. This scans a boxed video for frames containing the box colour,
picks 12 spread evenly across the span, and tiles brightened crops — the raw
footage is very dark, so the crops are brightness-scaled ~2.2x:

```python
import cv2, numpy as np

CLIP = "near"
path = f"runs/feeder_court/detect/{CLIP}_boxed.mp4"

cap, hits, i = cv2.VideoCapture(path), [], 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    roi = f[35:, :]                      # skip the white HUD strip
    m = (roi[:, :, 0] < 80) & (roi[:, :, 1] > 180) & (roi[:, :, 2] > 180)
    if m.sum() > 20:
        hits.append(i)
    i += 1
cap.release()
print(f"{len(hits)}/{i} frames with a box ({100 * len(hits) / i:.1f}%)")

hits = np.array(hits)
picks = [int(x) for x in hits[np.linspace(0, len(hits) - 1, 12).astype(int)]]
cap, tiles, r = cv2.VideoCapture(path), [], 90
for fr in picks:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
    ok, f = cap.read()
    roi = f[35:, :]
    m = (roi[:, :, 0] < 80) & (roi[:, :, 1] > 180) & (roi[:, :, 2] > 180)
    ys, xs = np.nonzero(m)
    cy, cx = int(ys.mean()) + 35, int(xs.mean())
    y0, x0 = max(cy - r, 0), max(cx - r, 0)
    crop = f[y0:min(cy + r, 720), x0:min(cx + r, 1280)]
    crop = cv2.copyMakeBorder(crop, 0, 2 * r - crop.shape[0], 0, 2 * r - crop.shape[1],
                              cv2.BORDER_CONSTANT)
    crop = cv2.resize(cv2.convertScaleAbs(crop, alpha=2.2, beta=10), (240, 240))
    cv2.putText(crop, f"f{fr}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    tiles.append(crop)
cap.release()
cv2.imwrite(f"{CLIP}_sheet.png",
            np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 12, 4)]))
```

The crop centres on the mean of all box-coloured pixels in the frame, so a frame
with two boxes far apart produces a tile that looks empty. That is the sheet, not
a missing detection — open the video at that frame before concluding anything.

What a good tile looks like: a small bright blob, tight in the box, against
floor or wall. What a bad one looks like: the box sitting on a ceiling light, a
reflection on the floor, or a player's shoe.

## Measured results

Run 2026-08-24, `--conf 0.25`, `--max-side 60`, imgsz 1280, CPU.

| clip | span | frames | frames w/ box | hit% | dets | mean conf | ms/frame |
|---|---|---|---|---|---|---|---|
| near | 105–2130 | 2025 | 392 | 19.4 | 490 | 0.501 | 317.0 |
| mid | 195–1530 | 1335 | 311 | 23.3 | 359 | 0.448 | 310.9 |
| far | 120–1755 | 1635 | 348 | 21.3 | 442 | 0.408 | 426.9 |

Contact sheets for all three were inspected. near and mid: 12/12 sampled tiles
were real shuttlecocks. far: 10/12 real, 2 unreadable for the centroid reason
above. Confidence falls with distance (0.501 → 0.448 → 0.408) but hit rate does
not, so far is the weakest of the three without being a failure.

The `oversized` column is absent from that run — the summary line did not yet
print it. The filter was active throughout; only the count is missing.

### A trap worth recording

A 40-frame smoke test starting at f420 of `far.mp4` produced 21 boxes with none
above 0.60 confidence, which read as "far does not work." The full span found
442 detections, several above 0.50. The 40 frames had landed in a stretch with
no shuttle in flight. **Do not characterise a clip from a short consecutive
sample** — rallies are bursty, so sample across the whole span or not at all.

## Caveats

- 53 training labels is a very small dataset. Everything here is promising, not
  validated.
- Nothing in this test is scored against ground truth. It measures firing
  behaviour, not accuracy.
- `mid.mp4` degenerates without the size filter. Whether `near` and `far` do the
  same to a lesser degree has not been measured — run with `--max-side 0` and
  compare the `oversized` counts to find out.
- CPU inference at ~320–430 ms/frame is roughly 10x off real time. Fine for
  offline testing, not for live use as-is.
