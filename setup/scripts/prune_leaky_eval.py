# ============================================================
# prune_leaky_eval.py — Move near-duplicate eval images back into train.
#
# resplit_by_clip.py groups by FILENAME, so it cannot tell that two clips named
# differently are the same rally, the same venue, or the same video exported
# twice. On smashspeed it cut val near-duplicates from 54% to 14% — better, but
# 14% of the validation set was still visually identical to a training image,
# and every one of those inflates mAP.
#
# This does the check filenames cannot: compares actual pixels. Any eval image
# whose nearest training image exceeds --threshold cosine similarity is moved
# INTO train (not deleted — it is still fine as training data, it just cannot
# be used to measure generalisation).
#
# Run AFTER resplit_by_clip.py. Re-run audit_dataset.py afterwards to confirm.
#
# Usage:  python scripts/prune_leaky_eval.py --root datasets/smashspeed-v8
#         python scripts/prune_leaky_eval.py --root ... --dry-run
# ============================================================

import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def thumbs(paths, side=32, batch_note=""):
    rows, kept = [], []
    for i, p in enumerate(paths):
        if batch_note and i and i % 2000 == 0:
            print(f"    {batch_note} {i}/{len(paths)}", flush=True)
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        v = cv2.resize(im, (side, side)).astype(np.float32).ravel()
        v -= v.mean()
        n = np.linalg.norm(v)
        if n < 1e-6:
            continue
        rows.append(v / n)
        kept.append(p)
    return (np.array(rows, np.float32) if rows else
            np.zeros((0, side * side), np.float32)), kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--threshold", type=float, default=0.98,
                    help="cosine similarity above which an eval image counts "
                         "as a duplicate of a training image")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    train_dir = os.path.join(args.root, "train", "images")
    if not os.path.isdir(train_dir):
        print(f"[ERROR] no train/images under {args.root}", file=sys.stderr)
        return 1

    print("[SCAN] hashing train images…", flush=True)
    tr_paths = sorted(glob.glob(os.path.join(train_dir, "*.*")))
    TR, _ = thumbs(tr_paths, batch_note="train")
    print(f"[SCAN] {len(TR)} train thumbnails")

    total_moved = 0
    for split in ("valid", "test"):
        d = os.path.join(args.root, split, "images")
        if not os.path.isdir(d):
            continue
        ev_paths = sorted(glob.glob(os.path.join(d, "*.*")))
        EV, kept = thumbs(ev_paths, batch_note=split)
        if not len(EV):
            continue

        # Chunked so a big dataset doesn't allocate one huge score matrix.
        best = np.zeros(len(EV), np.float32)
        for i in range(0, len(EV), 512):
            best[i:i + 512] = (EV[i:i + 512] @ TR.T).max(axis=1)

        dupes = [kept[i] for i in np.where(best >= args.threshold)[0]]
        print(f"[{split}] {len(dupes)}/{len(kept)} images exceed "
              f"{args.threshold} similarity to a train image "
              f"({100.0 * len(dupes) / len(kept):.1f}%)")

        if args.dry_run:
            continue

        for ip in dupes:
            stem = os.path.splitext(os.path.basename(ip))[0]
            lp = os.path.join(args.root, split, "labels", stem + ".txt")
            shutil.move(ip, os.path.join(train_dir, os.path.basename(ip)))
            if os.path.isfile(lp):
                out = os.path.join(args.root, "train", "labels", stem + ".txt")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.move(lp, out)
            total_moved += 1

    # Stale ultralytics caches would otherwise resurrect the old assignment.
    if not args.dry_run:
        for sp in ("train", "valid", "test"):
            for c in (os.path.join(args.root, sp, "labels.cache"),
                      os.path.join(args.root, sp, "images.cache")):
                if os.path.isfile(c):
                    os.remove(c)
        print(f"\n[DONE] moved {total_moved} duplicate eval images into train")
    else:
        print("\n[DRY-RUN] nothing moved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
