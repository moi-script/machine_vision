# Shuttlecock dataset v2 — why v1 was replaced

## What went wrong with v1

`datasets/shuttlecock-1` (Roboflow `mois-workspace/shuttlecock-m9ihi-nimwo` v1)
reported **mAP50 = 0.898, mAP50-95 = 0.389** after a 2.1 h CPU run. Against a
real webcam the same weights detected the shuttlecock in **4 of 175 frames**,
peaking at **0.284 confidence** — below the 0.55 live threshold, so the drill
never registered a single shuttle.

Two measurements explain the gap.

**1. The whole dataset is one video clip.**

```
train  n= 392  distinct source clips=1   (backh_smashvdo_mp4)
valid  n=  20  distinct source clips=1   (backh_smashvdo_mp4)
test   n=  10  distinct source clips=1   (backh_smashvdo_mp4)
```

**2. The validation split leaked from training.** Roboflow splits by sampling
individual images. Those images are consecutive video frames, so:

```
median distance from a val frame to the nearest train frame: 0.0 frames
val frames within 2 frames of a train frame: 20/20
```

Every validation frame is 0–1 frames from a training frame — visually the same
image. So 0.898 measured how well the model memorised one clip. It was never an
estimate of generalisation, and it should not be cited as accuracy.

## What v2 changes

**Source.** `diwenne/smashspeed` (CC BY 4.0, ~13.4k images). Chosen by
inspecting sample frames rather than by image count: it spans many different
gyms, floor colours, lighting rigs and camera angles, and it is amateur hall
footage shot from a fixed side view — much closer to a feeder-mounted camera
than broadcast tournament coverage. It also includes background frames with no
shuttlecock, which suppress false positives.

Two larger-looking alternatives were rejected after inspection:
`mathieu-cartron/shuttlecock-cqzy3` (8.0k) and
`militaryvehicles-geymk/badminton-shuttlecocks` (8.0k) are both frame dumps of
the *same* Nanjing broadcast match — they would repeat v1's failure at a larger
scale.

**Splitting.** `scripts/resplit_by_clip.py` regroups frames by source clip and
assigns whole clips to train/valid/test, so a validation clip is genuinely
unseen footage. It refuses to run on a dataset with fewer than 3 distinct clips
— which is exactly what v1 was.

**Size.** Capped at ~1200 training images by default. On this CPU-only machine
the measured cost is 0.35 s per image per epoch, so the full 13.4k export would
take ~104 h. Diversity of clips matters more than raw frame count.

## Pipeline

```bash
python scripts/fetch_shuttle_v2.py      # -> datasets/shuttle-v2/
python scripts/resplit_by_clip.py       # clip-based split (run --dry-run first)
python scripts/train_shuttlecock_v2.py  # -> models/shuttlecock_v2.pt
```

v1 artefacts (`datasets/shuttlecock-1/`, `runs/shuttle/`,
`models/shuttlecock.pt`) are left untouched so the two can be compared.

## Before trusting v2

A clip-based mAP is necessary but not sufficient — smashspeed is still not
*your* court. Validate against the real camera before quoting any number, and
expect to fine-tune on footage from the mounted OV9281 for final results.
