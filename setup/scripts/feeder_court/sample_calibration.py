# ============================================================
# sample_calibration.py — PHASE 0. Run this BEFORE any bulk labelling.
#
# Two modes:
#   --mode sample   writes 20 frames per clip for you to hand-box in Roboflow
#   --mode measure  reads the labels back and prints the gate decision
#
# The gate:
#   median >= 16 px            -> stock yolov8n is enough
#   8-16 px                    -> yolov8n-p2 (the plan's default)
#   < 8 px on ALL three clips  -> STOP. Do not proceed to labelling.
#
# Usage:
#   python scripts/feeder_court/sample_calibration.py --mode sample \
#       --source datasets/vid_source/new_badminton_source \
#       --segments datasets/feeder_court/segments.json \
#       --out datasets/feeder_court/calibration
#
#   python scripts/feeder_court/sample_calibration.py --mode measure \
#       --labels datasets/feeder_court/calibration/labels
# ============================================================

import argparse
import glob
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.feeder_court import sizes  # noqa: E402

CLIPS = ["near", "mid", "far"]


def do_sample(args: argparse.Namespace) -> None:
    with open(args.segments) as fh:
        spans_by_clip = json.load(fh)

    os.makedirs(args.out, exist_ok=True)
    for clip in CLIPS:
        spans = [tuple(s) for s in spans_by_clip[clip]]
        frames = sizes.evenly_spaced_frames(spans, args.per_clip)
        cap = cv2.VideoCapture(os.path.join(args.source, f"{clip}.mp4"))
        for idx in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.imwrite(os.path.join(args.out, f"{clip}_{idx:06d}.jpg"), frame)
        cap.release()
        print(f"{clip}: wrote {len(frames)} calibration frames", flush=True)

    print(f"\nUpload {args.out} to Roboflow and box EVERY visible shuttlecock.")
    print("Export as YOLOv8, then run this script again with --mode measure.")


def do_measure(args: argparse.Namespace) -> None:
    medians = {}
    for clip in CLIPS:
        dims = []
        for path in sorted(glob.glob(os.path.join(args.labels, f"{clip}_*.txt"))):
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        dims.append(sizes.yolo_box_max_dim_px(line, args.img_w, args.img_h))
        if not dims:
            print(f"{clip}: NO BOXES FOUND — cannot measure this clip")
            continue
        stats = sizes.size_percentiles(dims)
        medians[clip] = stats["median"]
        print(
            f"{clip}: n={len(dims):4d}  p25={stats['p25']:.1f}  "
            f"median={stats['median']:.1f}  p75={stats['p75']:.1f}  p95={stats['p95']:.1f} px",
            flush=True,
        )

    decision = sizes.gate_decision(medians)
    print(f"\nGATE DECISION: {decision}")
    if decision == "stop":
        print("Every clip is under 8 px. YOLOv8n cannot detect this.")
        print("Do NOT proceed to labelling. Escalate to the spec's section 4.3 fallbacks:")
        print("  - 2x upscale of the ROI crop")
        print("  - scope shuttle tracking to the near half-court only")
        print("  - re-record with a cleaned lens and corrected exposure")
        sys.exit(1)
    if decision == "stock":
        print("Shuttles are large enough for stock yolov8n; the P2 head is optional.")
    else:
        print("Marginal sizes confirmed. Proceed with yolov8n-p2 as planned.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sample", "measure"], required=True)
    parser.add_argument("--source")
    parser.add_argument("--segments")
    parser.add_argument("--out")
    parser.add_argument("--labels")
    parser.add_argument("--per-clip", type=int, default=20)
    parser.add_argument("--img-w", type=int, default=1280)
    parser.add_argument("--img-h", type=int, default=720)
    args = parser.parse_args()

    if args.mode == "sample":
        do_sample(args)
    else:
        do_measure(args)


if __name__ == "__main__":
    main()
