"""Extract frames from a clip and write YOLO pre-labels for human correction.

Pre-labels come from an existing model at a large imgsz and a LOW confidence
threshold on purpose: in a human-review workflow, deleting a wrong box is much
faster than drawing a missing one, so recall is worth more than precision here.

The pre-labels are a starting point only. The frames this model misses are the
distant ones -- exactly the cases the new footage exists to fix -- so those must
be drawn by hand. Pre-labels cannot supply them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights", default="models/shuttlecock_scene_dummy.pt")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.15)
    args = parser.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.video)
    stem = Path(args.video).stem

    idx = 0
    kept = 0
    prelabelled = 0
    boxes_total = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.stride:
            idx += 1
            continue

        name = f"{stem}_{idx:05d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), frame)

        height, width = frame.shape[:2]
        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        lines = []
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            cx, cy = ((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height
            bw, bh = (x2 - x1) / width, (y2 - y1) / height
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out / "labels" / f"{name}.txt").write_text("\n".join(lines))
        if lines:
            prelabelled += 1
            boxes_total += len(lines)
        kept += 1
        idx += 1
    cap.release()

    print(f"frames written : {kept}")
    print(f"with pre-labels: {prelabelled} ({100 * prelabelled / max(kept, 1):.1f}%)")
    print(f"boxes drawn    : {boxes_total}")
    print(f"needs drawing  : {kept - prelabelled} frames have no pre-label")
    print(f"output         : {out}")


if __name__ == "__main__":
    main()
