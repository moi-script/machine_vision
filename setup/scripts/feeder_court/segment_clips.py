# ============================================================
# segment_clips.py — Emit the usable window(s) of each source clip as JSON.
#
# Usage:
#   python scripts/feeder_court/segment_clips.py \
#       --source datasets/vid_source/new_badminton_source \
#       --out datasets/feeder_court/segments.json
# ============================================================

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.feeder_court import segments  # noqa: E402

# line_1/line_2 are excluded: they belong to the future line-detection dataset.
CLIPS = ["near", "mid", "far"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-step", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=15.0)
    parser.add_argument("--min-frames", type=int, default=150)
    args = parser.parse_args()

    result = {}
    for clip in CLIPS:
        path = os.path.join(args.source, f"{clip}.mp4")
        profile = segments.motion_profile(path, sample_step=args.sample_step)
        spans = segments.usable_segments(
            profile,
            sample_step=args.sample_step,
            threshold=args.threshold,
            min_frames=args.min_frames,
        )
        result[clip] = spans
        total = sum(b - a for a, b in spans)
        print(f"{clip}: {len(spans)} segment(s), {total} usable frames -> {spans}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
