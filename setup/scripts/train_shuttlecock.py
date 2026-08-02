# ============================================================
# train_shuttlecock.py — Train shuttlecock weights LOCALLY.
#
# Why this exists: Roboflow's hosted serverless inference costs ~1.2-3.9 s per
# round trip (measured), of which only ~0.01 s is actual inference. That is far
# too slow to track a shuttle in flight. Downloading Roboflow's *trained* weights
# needs a paid plan — but exporting the *dataset* is free on a public project,
# so we train our own yolov8n on it here. Result: ~10 ms/frame, fully offline.
#
# Dataset: mois-workspace/shuttlecock-m9ihi-nimwo v1 (CC BY 4.0)
#   392 train / 20 valid / 10 test, already 640x640 grayscale from Roboflow.
#
# Tuning note — this is a SMALL-OBJECT problem. 86% of labelled boxes are under
# COCO's 32x32 "small" threshold (median 28x19 px). So:
#   * imgsz stays at 640; halving it would erase most of the targets.
#   * mosaic is reduced and closed early — it tiles 4 images into one, which
#     shrinks already-tiny objects further.
#   * scale jitter is reduced for the same reason.
#   * hue/saturation aug is disabled: the dataset is grayscale, so it is a no-op.
#
# Usage:  python scripts/train_shuttlecock.py            # fresh run
#         python scripts/train_shuttlecock.py --resume   # continue from last.pt
# Output: runs/shuttle/yolov8n-640/weights/best.pt
#
# A full run is ~5 h on this CPU, so it WILL outlive most shells. Launch it
# detached (PowerShell: Start-Process -WindowStyle Hidden) rather than as a
# child of an interactive session, or it gets killed mid-run. Every epoch is
# checkpointed to weights/last.pt, so --resume always picks up where it stopped.
# ============================================================

import os
import shutil
import sys

from ultralytics import YOLO

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "datasets", "shuttlecock-1", "data.yaml")
PROJECT = os.path.join(REPO, "runs", "shuttle")
RUN_NAME = "yolov8n-640"
# Where the drill engine expects to find the weights (SHUTTLE_MODEL_PATH).
DEPLOY_TO = os.path.join(REPO, "models", "shuttlecock.pt")


def main() -> int:
    if not os.path.isfile(DATA):
        print(f"[ERROR] dataset not found at {DATA}", file=sys.stderr)
        return 1

    last = os.path.join(PROJECT, RUN_NAME, "weights", "last.pt")
    if "--resume" in sys.argv:
        if not os.path.isfile(last):
            print(f"[ERROR] --resume given but no checkpoint at {last}",
                  file=sys.stderr)
            return 1
        # Ultralytics restores epoch count, optimiser state and every training
        # arg from the checkpoint itself — passing them again here would be
        # ignored at best and conflict at worst.
        print(f"[RESUME] continuing from {last}")
        YOLO(last).train(resume=True)
        return _deploy()

    model = YOLO("yolov8n.pt")  # pretrained COCO checkpoint, ~6 MB

    model.train(
        data=DATA,
        epochs=120,
        patience=30,        # early-stop once val mAP plateaus
        imgsz=640,
        batch=8,
        device="cpu",       # torch here is a CPU-only build (no CUDA)
        workers=4,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
        seed=0,             # reproducible for the thesis
        val=True,
        plots=True,
        # --- small-object augmentation profile ---
        mosaic=0.5,         # default 1.0 shrinks tiny targets too aggressively
        close_mosaic=20,    # final 20 epochs train on un-tiled images
        scale=0.3,          # default 0.5 jitters scale too widely here
        hsv_h=0.0,          # grayscale dataset — colour aug is a no-op
        hsv_s=0.0,
        fliplr=0.5,
        flipud=0.0,
    )

    return _deploy()


def _deploy() -> int:
    """Copy the best checkpoint to where the drill engine loads it from."""
    best = os.path.join(PROJECT, RUN_NAME, "weights", "best.pt")
    if os.path.isfile(best):
        os.makedirs(os.path.dirname(DEPLOY_TO), exist_ok=True)
        shutil.copyfile(best, DEPLOY_TO)
        print(f"[DONE] copied {best} -> {DEPLOY_TO}")
        print("[NEXT] set SHUTTLE_SOURCE='local' (config/settings.py and the "
              "persisted settings doc in Mongo, which overrides it).")
    else:
        print(f"[WARN] no best.pt at {best}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # required on Windows when workers > 0
    raise SystemExit(main())
