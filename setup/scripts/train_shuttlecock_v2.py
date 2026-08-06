# ============================================================
# train_shuttlecock_v2.py — Train v2 shuttlecock weights on the new dataset.
#
# Kept separate from train_shuttlecock.py on purpose: that script is the record
# of how the v1 weights were produced, and the thesis references its numbers.
# This one reads datasets/shuttle-v2/, writes runs/shuttle_v2/, and deploys to
# models/shuttlecock_v2.pt — so v1 stays byte-for-byte intact and the two can
# be compared honestly.
#
# SIZE BUDGET — this machine is CPU-only (i3-1215U, no CUDA). Measured cost on
# the v1 run was 0.35 s per image per epoch, and dataloader workers made no
# difference (139.5 s/epoch at workers=4 vs 137.1 at workers=0), i.e. it is
# compute-bound. So:
#     1200 images x 80 epochs  ~=  9.3 h      <- the default, an overnight run
#    13400 images x 80 epochs  ~= 104 h       <- the full smashspeed export
# Hence --max-train-images, which subsamples via ultralytics' `fraction`.
# More images is NOT the goal; clip diversity is. 1200 frames from 30 clips
# beats 13000 from 3.
#
# Usage:  python scripts/train_shuttlecock_v2.py
#         python scripts/train_shuttlecock_v2.py --max-train-images 2000
#         python scripts/train_shuttlecock_v2.py --resume
#         python scripts/train_shuttlecock_v2.py --batch 4      # if RAM is tight
#
# Long run — launch it detached (PowerShell: Start-Process -WindowStyle Hidden)
# so it outlives the shell. Every epoch checkpoints to weights/last.pt.
# ============================================================

import argparse
import glob
import os
import shutil
import sys

from ultralytics import YOLO

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "datasets", "shuttle-v2", "data.yaml")
PROJECT = os.path.join(REPO, "runs", "shuttle_v2")
RUN_NAME = "yolov8n-640-v2"
DEPLOY_TO = os.path.join(REPO, "models", "shuttlecock_v2.pt")

SEC_PER_IMAGE_PER_EPOCH = 0.35  # measured on this CPU during the v1 run


def _count_train_images() -> int:
    d = os.path.join(REPO, "datasets", "shuttle-v2", "train", "images")
    return len(glob.glob(os.path.join(d, "*.*")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-train-images", type=int, default=1200)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(DATA):
        print(f"[ERROR] dataset not found at {DATA}", file=sys.stderr)
        print("        run: python scripts/fetch_shuttle_v2.py", file=sys.stderr)
        return 1

    last = os.path.join(PROJECT, RUN_NAME, "weights", "last.pt")
    if args.resume:
        if not os.path.isfile(last):
            print(f"[ERROR] --resume given but no checkpoint at {last}",
                  file=sys.stderr)
            return 1
        print(f"[RESUME] continuing from {last}")
        YOLO(last).train(resume=True)
        return _deploy()

    n = _count_train_images()
    if n == 0:
        print(f"[ERROR] no training images under datasets/shuttle-v2/train/images",
              file=sys.stderr)
        return 1
    fraction = min(1.0, args.max_train_images / n)
    used = int(n * fraction)
    hours = used * SEC_PER_IMAGE_PER_EPOCH * args.epochs / 3600.0
    print(f"[PLAN] {n} train images available, using {used} (fraction={fraction:.3f})")
    print(f"[PLAN] {args.epochs} epochs -> ~{hours:.1f} h if it runs to the end "
          f"(early stopping usually cuts this)")

    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA,
        epochs=args.epochs,
        patience=20,          # tighter than v1's 30 — the run is longer
        imgsz=640,            # do NOT lower: median shuttle box is 28x19 px
        batch=args.batch,
        fraction=fraction,
        device="cpu",
        workers=4,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
        seed=0,
        val=True,
        plots=True,
        # --- small-object augmentation profile (carried over from v1) ---
        mosaic=0.5,
        close_mosaic=15,
        scale=0.3,
        fliplr=0.5,
        flipud=0.0,
        # v2 data is COLOUR, unlike v1's grayscale export, so re-enable the
        # colour jitter that was pointless before. This also buys robustness
        # to the court/lighting differences between venues.
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
    )
    return _deploy()


def _deploy() -> int:
    best = os.path.join(PROJECT, RUN_NAME, "weights", "best.pt")
    if not os.path.isfile(best):
        print(f"[WARN] no best.pt at {best}", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(DEPLOY_TO), exist_ok=True)
    shutil.copyfile(best, DEPLOY_TO)
    print(f"[DONE] copied {best} -> {DEPLOY_TO}")
    print("[NEXT] validate against YOUR camera before trusting any mAP number:")
    print("       point SHUTTLE_MODEL_PATH at models/shuttlecock_v2.pt and "
          "re-run the shuttle diagnostic.")
    return 0


if __name__ == "__main__":  # required on Windows when workers > 0
    raise SystemExit(main())
