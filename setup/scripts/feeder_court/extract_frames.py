# ============================================================
# extract_frames.py — Build the frame set for the clear_badminton dataset,
# with train/val/test already decided so Roboflow cannot re-introduce leakage.
#
# WHY BLOCKS AND NOT RANDOM FRAMES
# Frames 7 apart in a 30 fps clip share the same court, lighting and players;
# only the shuttle has moved. If one lands in train and its neighbour in val,
# val stops measuring generalization and starts measuring memorisation — that
# is exactly how the v1 weights scored mAP50 0.898 and then detected ~2% on a
# real camera. So the video is cut into blocks of consecutive frames and WHOLE
# BLOCKS are assigned to a split. Neighbours always share a split.
#
# WHY BOTH COURTS APPEAR IN TRAINING
# Splitting court1=train / court2=val would give the most honest number but
# would never teach the model court2. Blocks are drawn from both clips.
#
# WHY SOME FRAMES ARE THROWN AWAY
# A frame where nothing moved is a frame with no shuttle in flight: it costs
# annotation time and teaches nothing. A frame during heavy camera motion is
# blurred and its boxes would be guesses. Both tails of the motion distribution
# are dropped before sampling.
#
# Usage:
#   python scripts/feeder_court/extract_frames.py
#   python scripts/feeder_court/extract_frames.py --budget 800 --block 200
# ============================================================

import argparse
import json
import os
import random

import cv2
import numpy as np

DEFAULT_SOURCE = "datasets/vid_source/clear_badminton_dataset_for_collab"
DEFAULT_VIDEOS = ["dataset_court1", "dataset_court2"]
DEFAULT_OUT = "datasets/clear_badminton/frames"

SPLITS = ("train", "valid", "test")
DEFAULT_RATIOS = (0.70, 0.20, 0.10)


def motion_per_frame(video_path: str, size=(160, 90)) -> np.ndarray:
    """Mean absolute difference against the previous frame, for every frame.

    Decoded sequentially rather than with cap.set() per index: seeking a
    1280x720 mp4 thousands of times costs minutes, a straight read costs
    seconds. Index i is frame i; frame 0 inherits frame 1's score.
    """
    cap = cv2.VideoCapture(video_path)
    scores, prev = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size).astype(np.float32)
        scores.append(0.0 if prev is None else float(np.abs(gray - prev).mean()))
        prev = gray
    cap.release()
    if scores:
        scores[0] = scores[1] if len(scores) > 1 else 0.0
    return np.array(scores, dtype=float)


def eligible_frames(scores: np.ndarray, args) -> tuple[np.ndarray, float, float]:
    """Frame indices that survive head/tail trim and both motion cutoffs."""
    n = len(scores)
    lo_idx, hi_idx = args.head_trim, max(n - args.tail_trim, args.head_trim)
    window = np.arange(lo_idx, hi_idx)
    if len(window) == 0:
        return window, 0.0, 0.0

    inner = scores[window]
    lo = float(np.percentile(inner, args.min_motion_pct))
    hi = float(np.percentile(inner, args.max_motion_pct))
    keep = window[(inner >= lo) & (inner <= hi)]
    return keep, lo, hi


def assign_blocks(n_blocks: int, ratios: tuple, rng: random.Random) -> list[str]:
    """Shuffle block indices, then deal them out to train/valid/test.

    Shuffled rather than sliced in order, because a contiguous last-10% test
    set would be the end of the recording — often one player, one lighting
    state. Each split still gets at least one block whenever there are enough.
    """
    order = list(range(n_blocks))
    rng.shuffle(order)

    counts = [max(1, int(round(r * n_blocks))) for r in ratios]
    while sum(counts) > n_blocks:  # rounding can overshoot on small clips
        counts[counts.index(max(counts))] -= 1
    counts[0] += n_blocks - sum(counts)

    labels = [""] * n_blocks
    cursor = 0
    for split, count in zip(SPLITS, counts):
        for block in order[cursor:cursor + count]:
            labels[block] = split
        cursor += count
    return labels


def plan_video(name: str, path: str, budget: int, args, rng: random.Random) -> list[dict]:
    """Pick which frames to keep, and which split each one belongs to."""
    scores = motion_per_frame(path)
    keep, lo, hi = eligible_frames(scores, args)
    total = len(scores)
    if len(keep) == 0:
        raise SystemExit(f"{name}: no eligible frames after trimming")

    n_blocks = max(1, (total + args.block - 1) // args.block)
    labels = assign_blocks(n_blocks, args.ratios, rng)

    # Evenly spaced across the eligible frames rather than per block, so a
    # block with little usable footage does not get the same share as a busy
    # one. min_gap then drops picks that landed too close to their neighbour.
    picks = []
    step = max(len(keep) / budget, 1.0)
    last = None
    for i in range(budget):
        idx = int(keep[min(int(i * step), len(keep) - 1)])
        if last is not None and idx - last < args.min_gap:
            continue
        picks.append({
            "frame": idx,
            "video": name,
            "block": idx // args.block,
            "split": labels[idx // args.block],
            "motion": round(float(scores[idx]), 3),
        })
        last = idx

    print(f"{name}: {total} frames, {len(keep)} eligible "
          f"(motion {lo:.2f}-{hi:.2f}), {n_blocks} blocks, {len(picks)} picked", flush=True)
    return picks


def write_frames(path: str, picks: list[dict], out_root: str) -> int:
    """Decode once, writing the picked frames as they go past."""
    wanted = {p["frame"]: p for p in picks}
    cap = cv2.VideoCapture(path)
    idx = written = 0
    while wanted:
        ok, frame = cap.read()
        if not ok:
            break
        pick = wanted.pop(idx, None)
        if pick is not None:
            out_dir = os.path.join(out_root, pick["split"])
            os.makedirs(out_dir, exist_ok=True)
            name = f'{pick["video"]}_{idx:06d}.jpg'
            cv2.imwrite(os.path.join(out_dir, name), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            pick["file"] = os.path.join(pick["split"], name)
            written += 1
        idx += 1
    cap.release()
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract a leakage-safe frame set for annotation.")
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--videos", nargs="+", default=DEFAULT_VIDEOS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--budget", type=int, default=500,
                    help="total frames across all videos, split evenly between them")
    ap.add_argument("--block", type=int, default=150,
                    help="consecutive frames per split block; neighbours never cross splits")
    ap.add_argument("--min-gap", type=int, default=6, dest="min_gap",
                    help="minimum frames between two kept samples")
    ap.add_argument("--head-trim", type=int, default=30, dest="head_trim")
    ap.add_argument("--tail-trim", type=int, default=30, dest="tail_trim")
    ap.add_argument("--min-motion-pct", type=float, default=20.0, dest="min_motion_pct",
                    help="drop the quietest N%% of frames — nothing is moving, so no shuttle")
    ap.add_argument("--max-motion-pct", type=float, default=99.0, dest="max_motion_pct",
                    help="drop the most violent N%% — camera moves, motion-blurred boxes")
    ap.add_argument("--seed", type=int, default=1337, help="block-to-split assignment seed")
    args = ap.parse_args()
    args.ratios = DEFAULT_RATIOS

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(root)

    rng = random.Random(args.seed)
    per_video = args.budget // len(args.videos)

    all_picks = []
    for name in args.videos:
        path = os.path.join(args.source, f"{name}.mp4")
        if not os.path.exists(path):
            raise SystemExit(f"missing video: {path}")
        picks = plan_video(name, path, per_video, args, rng)
        write_frames(path, picks, args.out)
        all_picks.extend(picks)

    manifest = os.path.join(args.out, "manifest.json")
    os.makedirs(args.out, exist_ok=True)
    with open(manifest, "w") as fh:
        json.dump({"seed": args.seed, "block": args.block, "min_gap": args.min_gap,
                   "frames": all_picks}, fh, indent=2)

    print("\nsplit   frames  blocks  videos")
    for split in SPLITS:
        rows = [p for p in all_picks if p["split"] == split]
        blocks = len({(p["video"], p["block"]) for p in rows})
        vids = len({p["video"] for p in rows})
        print(f"{split:<7} {len(rows):<7} {blocks:<7} {vids}")
    print(f"\ntotal {len(all_picks)} frames -> {args.out}")
    print(f"manifest -> {manifest}")


if __name__ == "__main__":
    main()
