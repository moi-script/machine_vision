# ============================================================
# audit_dataset.py — Inspect a YOLO dataset BEFORE spending hours training on it.
#
# Written after two wasted training runs, and it checks for exactly those two
# failure modes plus the things that would have caught them earlier:
#
#   v1 failed because every frame came from ONE clip and the val split sat 0-1
#       frames from a train frame, so mAP50 0.898 measured memorisation.
#   v2 failed because the export was 555 CLOSE-UP photos (median box 248 px at
#       imgsz 640) instead of court footage, so the model learned "big centered
#       blob = shuttlecock" and fires on blank walls.
#
# The single most important line of output is the box-size distribution. If the
# median box is a large fraction of the frame, the dataset is close-ups and the
# model will not detect anything at court distance, whatever its mAP says.
#
# Usage:  python scripts/audit_dataset.py datasets/smashspeed-v8
#         python scripts/audit_dataset.py datasets/shuttle-v2 --sample 400
# ============================================================

import argparse
import glob
import os
import re
import sys
from collections import Counter

import cv2
import numpy as np

# Roboflow mangles names to "<original>_jpg.rf.<32-hex>.jpg".
_RF = re.compile(r"_(jpg|png|jpeg)\.rf\.[0-9a-f]{6,}", re.I)

# A trailing frame counter, e.g. "backh_smashvdo_mp4-607" -> "backh_smashvdo_mp4".
# The separator is REQUIRED. Making it optional collapses unrelated names like
# "images-2023-05-14T162540" and "...T162551" into one key and invents leakage
# that isn't there — which this script did on its first run.
_FRAME_TAIL = re.compile(r"[-_]\d{1,6}$")


def source_key(filename: str) -> str:
    """The original photo, before Roboflow's augmentation copies.

    Exact, no heuristic: augmented copies of one image share this prefix, so
    the same key in two splits is unambiguous augmentation leakage.
    """
    return _RF.split(os.path.basename(filename))[0]


def clip_key(filename: str) -> str:
    """Best-effort source clip, for datasets built from extracted video frames.

    Heuristic: only meaningful when filenames actually carry a frame counter.
    """
    stem = _FRAME_TAIL.sub("", source_key(filename))
    return stem.strip("-_ ") or "(unnamed)"


def label_for(image_path: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(image_path)), "labels",
        os.path.splitext(os.path.basename(image_path))[0] + ".txt",
    )


def read_boxes(label_path: str):
    """Return [(w, h), ...] normalized. Missing/empty file = background frame."""
    if not os.path.isfile(label_path):
        return None
    out = []
    for line in open(label_path):
        parts = line.split()
        if len(parts) >= 5:
            out.append((float(parts[3]), float(parts[4])))
    return out


def pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * q))]


def audit_split(root: str, split: str, sample: int, imgsz: int):
    d = os.path.join(root, split, "images")
    if not os.path.isdir(d):
        return None
    imgs = sorted(glob.glob(os.path.join(d, "*.*")))
    if not imgs:
        return None

    clips = Counter(clip_key(p) for p in imgs)
    sources = set(source_key(p) for p in imgs)

    ws, hs = [], []
    backgrounds = 0
    labelled = 0
    for p in imgs:
        boxes = read_boxes(label_for(p))
        if boxes is None:
            continue
        labelled += 1
        if not boxes:
            backgrounds += 1
        for w, h in boxes:
            ws.append(w)
            hs.append(h)

    # Pixel checks are the slow part, so only sample.
    step = max(1, len(imgs) // sample)
    dims = Counter()
    gray = colour = 0
    for p in imgs[::step][:sample]:
        im = cv2.imread(p)
        if im is None:
            continue
        dims[im.shape[:2]] += 1
        b, g, r = cv2.split(im)
        if np.array_equal(b, g) and np.array_equal(g, r):
            gray += 1
        else:
            colour += 1

    wpx = sorted(w * imgsz for w in ws)
    hpx = sorted(h * imgsz for h in hs)
    frac = sorted(ws)

    return {
        "images": len(imgs), "labelled": labelled, "boxes": len(ws),
        "backgrounds": backgrounds, "clips": clips, "sources": sources,
        "dims": dims, "gray": gray, "colour": colour,
        "wpx": wpx, "hpx": hpx, "frac": frac,
    }


def _thumbs(paths, side=32):
    """Downsampled, contrast-normalized gray thumbnails as unit row vectors."""
    rows, kept = [], []
    for p in paths:
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
    return (np.array(rows, np.float32) if rows else np.zeros((0, side * side), np.float32)), kept


def similarity_leak(root: str, split: str, cap_train=4000, cap_eval=600):
    """Nearest-neighbour image similarity between a split and train.

    Filename heuristics guess at leakage; this measures it. Two frames grabbed
    a fraction of a second apart correlate ~0.99 even with different names.
    Returns (n_compared, cosine similarities to nearest train image).
    """
    tr = sorted(glob.glob(os.path.join(root, "train", "images", "*.*")))
    ev = sorted(glob.glob(os.path.join(root, split, "images", "*.*")))
    if not tr or not ev:
        return None
    tr = tr[:: max(1, len(tr) // cap_train)][:cap_train]
    ev = ev[:: max(1, len(ev) // cap_eval)][:cap_eval]

    A, _ = _thumbs(ev)
    B, _ = _thumbs(tr)
    if not len(A) or not len(B):
        return None
    best = (A @ B.T).max(axis=1)  # cosine, since rows are unit vectors
    return len(A), np.sort(best)[::-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="dataset directory containing train/ valid/ test/")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="report box sizes as pixels at this inference size")
    ap.add_argument("--sample", type=int, default=250,
                    help="images per split to open for dimension/colour checks")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"[ERROR] no such directory: {args.root}", file=sys.stderr)
        return 1

    yaml_path = os.path.join(args.root, "data.yaml")
    if os.path.isfile(yaml_path):
        print("=== data.yaml (the authority on provenance) ===")
        for line in open(yaml_path):
            if line.strip():
                print("   ", line.rstrip())
        print()

    all_clips = {}
    all_sources = {}
    verdicts = []

    for split in ("train", "valid", "val", "test"):
        r = audit_split(args.root, split, args.sample, args.imgsz)
        if r is None:
            continue
        all_clips[split] = set(r["clips"])
        all_sources[split] = r["sources"]

        print(f"=== {split} ===")
        print(f"  images {r['images']}   labelled {r['labelled']}   "
              f"boxes {r['boxes']}   background frames {r['backgrounds']}")

        n_clips = len(r["clips"])
        top = ", ".join(f"{k}({v})" for k, v in r["clips"].most_common(4))
        print(f"  distinct source clips: {n_clips}   top: {top}")
        if n_clips < 3:
            verdicts.append(f"{split}: only {n_clips} distinct clip(s) - this is "
                            f"how v1 failed")

        if r["colour"] or r["gray"]:
            tag = ("GRAYSCALE" if r["colour"] == 0 else
                   "colour" if r["gray"] == 0 else
                   f"MIXED ({r['gray']} gray / {r['colour']} colour)")
            print(f"  pixels: {tag}")
            if r["colour"] == 0:
                verdicts.append(f"{split}: grayscale export - a colour webcam "
                                f"feed is out of distribution")

        print(f"  dims: {', '.join(f'{h}x{w} x{n}' for (h, w), n in r['dims'].most_common(3))}")

        if r["wpx"]:
            print(f"  box width  px @{args.imgsz}: "
                  f"p05={pct(r['wpx'], .05):6.1f}  median={pct(r['wpx'], .5):6.1f}  "
                  f"p95={pct(r['wpx'], .95):6.1f}")
            print(f"  box height px @{args.imgsz}: "
                  f"p05={pct(r['hpx'], .05):6.1f}  median={pct(r['hpx'], .5):6.1f}  "
                  f"p95={pct(r['hpx'], .95):6.1f}")
            med_frac = pct(r["frac"], .5) * 100
            print(f"  median box is {med_frac:.1f}% of frame width")
            if split.startswith(("train",)) and med_frac > 10:
                verdicts.append(
                    f"{split}: median box is {med_frac:.1f}% of frame width - "
                    f"these are CLOSE-UPS, not court-distance shots")
        print()

    # Two different leaks, reported separately because they carry very
    # different confidence. Augmentation overlap is exact; clip overlap rests
    # on a filename heuristic and is only meaningful for video-frame datasets.
    tr_src = all_sources.get("train", set())
    tr_clip = all_clips.get("train", set())
    for split in ("valid", "val", "test"):
        ev_src, ev_clip = all_sources.get(split), all_clips.get(split)
        if not ev_src:
            continue
        print(f"=== leakage: train vs {split} ===")

        shared_src = tr_src & ev_src
        print(f"  same source photo in both (exact): {len(shared_src)}")
        if shared_src:
            verdicts.append(
                f"{split}: {len(shared_src)} source photo(s) have augmented "
                f"copies in BOTH train and {split} - the split leaks")

        shared_clip = tr_clip & ev_clip
        print(f"  same source clip in both (heuristic): {len(shared_clip)} "
              f"of {len(ev_clip)}")

        # The measurement that settles it, independent of any naming scheme.
        sim = similarity_leak(args.root, split)
        if sim:
            n, best = sim
            near = int((best >= 0.98).sum())
            close = int((best >= 0.90).sum())
            print(f"  image similarity to nearest train image (n={n}): "
                  f"median {np.median(best):.3f}  p90 {np.percentile(best, 90):.3f}  "
                  f"max {best.max():.3f}")
            print(f"    >=0.98 (near-duplicate): {near}/{n} "
                  f"({100.0 * near / n:.1f}%)   >=0.90: {close}/{n} "
                  f"({100.0 * close / n:.1f}%)")
            if near > 0.05 * n:
                verdicts.append(
                    f"{split}: {near}/{n} images ({100.0 * near / n:.0f}%) are "
                    f"near-duplicates (cosine >=0.98) of a train image - the "
                    f"split leaks and its mAP is inflated")
        print()

    print("=" * 62)
    if verdicts:
        print("  PROBLEMS FOUND - do not start a long training run yet:")
        for v in verdicts:
            print(f"   - {v}")
    else:
        print("  No structural problems found. Box sizes, colour, clip count")
        print("  and split independence all look sane for court-distance use.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
