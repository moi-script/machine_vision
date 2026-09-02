# ============================================================
# train_local.py — Train the feeder_court shuttlecock detector on CPU.
#
# The Task 8 notebook targets a Colab T4. This is the local equivalent for the
# calibration-sized dataset (53 hand-boxed frames), where a CPU run is ~4 hours
# rather than the ~30 the full dataset would cost.
#
# Everything that matters for tiny objects is kept identical to the notebook:
#   - yolov8-p2.yaml (stride-4 head), per the §4.4 gate decision
#   - imgsz 1280 = native long side. NEVER downscale; a 16 px shuttle at 640
#     becomes 8 px and at 320 it is gone. That is the imgsz-640 bug all over.
#   - hue/saturation augmentation off: the OV9281 is mono.
#   - scale 0.25, not the 0.5 default, which would annihilate a 16 px object.
#
# `far` is the held-out clip and must never appear in train/images.
#
# Usage:
#   python scripts/feeder_court/train_local.py
#   python scripts/feeder_court/train_local.py --epochs 40 --name p2-short
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path

SETUP = Path(__file__).resolve().parents[2]
DATA = SETUP / "datasets" / "feeder_court_yolo" / "data.yaml"


def assert_far_held_out(data_yaml: Path) -> None:
    """Refuse to spend hours training on a leaked split."""
    train_images = data_yaml.parent / "train" / "images"
    leaked = sorted(p.name for p in train_images.glob("far_*"))
    if leaked:
        raise SystemExit(f"far frames leaked into train: {leaked[:5]}")
    val_images = data_yaml.parent / "val" / "images"
    n_train = len(list(train_images.glob("*.jpg")))
    n_val = len(list(val_images.glob("*.jpg")))
    print(f"split OK: train={n_train} (no far), val={n_val}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--model", default="yolov8-p2.yaml")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--project", type=Path, default=SETUP / "runs" / "feeder_court")
    ap.add_argument("--name", default="p2-native")
    args = ap.parse_args()

    assert_far_held_out(args.data)

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device="cpu",
        seed=0,
        deterministic=True,
        project=str(args.project),
        name=args.name,
        patience=args.patience,
        cache=False,         # 'ram' caching is flagged non-deterministic; IO is
                             # negligible next to ~4 s/image of CPU compute anyway
        # --- mono-specific augmentation ---
        hsv_h=0.0,
        hsv_s=0.0,
        scale=0.25,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        plots=True,
        val=True,
    )


if __name__ == "__main__":
    main()
