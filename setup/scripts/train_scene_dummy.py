"""Train the scene-dummy models: identical settings, only the split differs.

Any gap between the two runs' validation metrics is caused by frame leakage in
the random split, not by the model or the data.
"""

from __future__ import annotations

import argparse
import time

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["temporal", "random"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    for split in args.splits:
        started = time.time()
        print(f"\n=== training split={split} ===", flush=True)
        model = YOLO("yolov8n.pt")
        model.train(
            data=f"datasets/scene-dummy/{split}/data.yaml",
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=2,
            seed=0,
            deterministic=True,
            project="runs/scene_dummy",
            name=split,
            exist_ok=True,
            plots=False,
            val=True,
        )
        print(f"=== {split} done in {(time.time() - started) / 60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
