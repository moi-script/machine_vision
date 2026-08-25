# ============================================================
# resplit_by_clip.py — Re-split a Roboflow export by SOURCE CLIP, not by frame.
#
# This is the script that stops v1's mistake from repeating.
#
# Roboflow splits train/valid/test by sampling individual images. When those
# images are frames extracted from video, consecutive frames are near-identical,
# so a "validation" frame is usually 1-2 frames away from a training frame. The
# model has effectively seen it. In v1 all 20 val frames sat 0-1 frames from a
# train frame and mAP50 came out at 0.898 while real-world performance was ~2%.
#
# Fix: group frames by the clip they came from, then assign WHOLE CLIPS to
# train/valid/test. A val clip is then genuinely unseen footage, and the
# resulting mAP is a number worth putting in the thesis.
#
# Roboflow keeps the original filename as a prefix:
#     backh_smashvdo_mp4-607_jpg.rf.c25c8238c8d....jpg
#     ^--------- clip ---------^ ^-frame-^
#
# Usage:  python scripts/resplit_by_clip.py
#         python scripts/resplit_by_clip.py --dry-run     # report only
# ============================================================

import argparse
import collections
import os
import random
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "datasets", "shuttle-v2")
SPLITS = ("train", "valid", "test")
RATIOS = {"train": 0.70, "valid": 0.20, "test": 0.10}

# Strip the Roboflow suffix and the trailing frame number to get the clip name.
_FRAME_RE = re.compile(r"[-_]\d+(?=(_jpg|_png|\.rf)|$)", re.IGNORECASE)


def clip_of(filename: str) -> str:
    stem = re.split(r"\.rf\.", filename)[0]
    stem = re.sub(r"_(jpg|png|jpeg)$", "", stem, flags=re.IGNORECASE)
    stem = os.path.splitext(stem)[0]
    return _FRAME_RE.sub("", stem) or stem


def collect(root: str):
    """-> {clip: [(split, image_path, label_path_or_None)]}"""
    groups = collections.defaultdict(list)
    for sp in SPLITS:
        idir = os.path.join(root, sp, "images")
        ldir = os.path.join(root, sp, "labels")
        if not os.path.isdir(idir):
            continue
        for fn in os.listdir(idir):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            lp = os.path.join(ldir, os.path.splitext(fn)[0] + ".txt")
            groups[clip_of(fn)].append(
                (sp, os.path.join(idir, fn), lp if os.path.isfile(lp) else None))
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=ROOT,
                    help="dataset directory to re-split (default datasets/shuttle-v2)")
    args = ap.parse_args()
    root = args.root

    if not os.path.isdir(root):
        print(f"[ERROR] {root} not found — run fetch_shuttle_v2.py first",
              file=sys.stderr)
        return 1

    groups = collect(root)
    if not groups:
        print(f"[ERROR] no images under {root}", file=sys.stderr)
        return 1

    total = sum(len(v) for v in groups.values())
    print(f"[SCAN] {total} images across {len(groups)} distinct source clips")
    for c, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"         {len(v):6}  {c}")

    if len(groups) < 3:
        print("\n[WARN] fewer than 3 distinct clips — a clip-based split is not "
              "possible. Any split you make here will still leak. Collect "
              "footage from more sessions before trusting the metrics.",
              file=sys.stderr)
        if not args.dry_run:
            return 1

    # Greedy: assign the largest clips first, always to whichever split is
    # furthest below its target share. Keeps whole clips intact.
    clips = sorted(groups, key=lambda c: -len(groups[c]))
    random.Random(args.seed).shuffle(clips[len(clips) // 2:])
    assign, counts = {}, {s: 0 for s in SPLITS}
    for c in clips:
        want = min(SPLITS, key=lambda s: counts[s] - RATIOS[s] * (sum(counts.values()) or 1))
        assign[c] = want
        counts[want] += len(groups[c])

    print("\n[PLAN] clip-based split")
    for s in SPLITS:
        pct = 100.0 * counts[s] / total if total else 0
        nclips = sum(1 for c in assign if assign[c] == s)
        print(f"         {s:6} {counts[s]:6} images  ({pct:4.1f}%)  {nclips} clips")

    if args.dry_run:
        print("\n[DRY-RUN] nothing moved")
        return 0

    moved = 0
    for c, items in groups.items():
        dst = assign[c]
        for sp, ip, lp in items:
            if sp == dst:
                continue
            for src, sub in ((ip, "images"), (lp, "labels")):
                if not src:
                    continue
                out = os.path.join(root, dst, sub, os.path.basename(src))
                os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.move(src, out)
            moved += 1

    # Ultralytics caches the old split in *.cache — stale caches silently
    # reintroduce the original assignment on the next run.
    for sp in SPLITS:
        for cache in (os.path.join(root, sp, "labels.cache"),
                      os.path.join(root, sp, "images.cache")):
            if os.path.isfile(cache):
                os.remove(cache)

    print(f"\n[DONE] moved {moved} images to honour clip boundaries")
    print("[NEXT] python scripts/train_shuttlecock_v2.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
