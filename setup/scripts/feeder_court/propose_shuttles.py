# ============================================================
# propose_shuttles.py — Auto-label proposals for the feeder_court dataset.
#
# Emits THREE buckets, never two:
#   positive  a confirmed flight track passes through this frame
#   negative  nothing was proposed here at all -> usable background frame
#   discard   something was proposed but rejected -> UNCERTAIN, never a label
#
# Always review the contact sheets before exporting. On far-distance footage the
# raw proposals were measured to be mostly noise, so this is a labour-saving
# device for a human labeller, not a labelling replacement.
#
# Usage:
#   python scripts/feeder_court/propose_shuttles.py \
#       --source datasets/vid_source/new_badminton_source \
#       --segments datasets/feeder_court/segments.json \
#       --out datasets/feeder_court/proposals
# ============================================================

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.feeder_court import motion, tracks  # noqa: E402

CLIPS = ["near", "mid", "far"]
K = 3  # neighbour offset for three-frame differencing


def load_span(cap, start: int, end: int) -> tuple[list[np.ndarray], int]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(end - start):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return frames, start


def contact_sheet(video_path: str, picks: list[tuple[int, int, int, int, int]], out_path: str) -> None:
    """Magnified crops around proposals, contrast-boosted so a human can judge them."""
    cap = cv2.VideoCapture(video_path)
    tiles = []
    for frame_idx, x, y, w, h in picks[:24]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = cap.read()
        if not ok:
            continue
        cx, cy, r = x + w // 2, y + h // 2, 32
        x0, y0 = max(cx - r, 0), max(cy - r, 0)
        crop = img[y0:y0 + 2 * r, x0:x0 + 2 * r].copy()
        if crop.shape[0] < 2 * r or crop.shape[1] < 2 * r:
            crop = cv2.copyMakeBorder(
                crop, 0, max(0, 2 * r - crop.shape[0]), 0, max(0, 2 * r - crop.shape[1]),
                cv2.BORDER_CONSTANT,
            )
        crop = cv2.convertScaleAbs(crop, alpha=2.6, beta=15)
        crop = cv2.resize(crop, (180, 180), interpolation=cv2.INTER_NEAREST)
        s = 180 / (2 * r)
        cv2.rectangle(
            crop,
            (int((x - x0) * s), int((y - y0) * s)),
            (int((x - x0 + w) * s), int((y - y0 + h) * s)),
            (0, 0, 255), 1,
        )
        cv2.putText(crop, f"f{frame_idx} {w}x{h}", (3, 13), 0, 0.36, (0, 255, 255), 1)
        tiles.append(crop)
    cap.release()

    if not tiles:
        return
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6) if len(tiles[i:i + 6]) == 6]
    if rows:
        cv2.imwrite(out_path, np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--thresh", type=int, default=22)
    args = parser.parse_args()

    with open(args.segments) as fh:
        spans_by_clip = json.load(fh)

    os.makedirs(args.out, exist_ok=True)

    for clip in CLIPS:
        video = os.path.join(args.source, f"{clip}.mp4")
        cap = cv2.VideoCapture(video)
        all_cands: list[motion.Candidate] = []
        all_frames: list[int] = []

        for start, end in spans_by_clip[clip]:
            frames, base = load_span(cap, start, end)
            for t in range(K, len(frames) - K):
                idx = base + t
                all_frames.append(idx)
                diff = motion.three_frame_diff(frames[t - K], frames[t], frames[t + K])
                all_cands.extend(motion.candidate_boxes(diff, idx, thresh=args.thresh))
        cap.release()

        linked = tracks.link_tracks(all_cands)
        flights = [t for t in linked if tracks.is_flight(t)]

        flight_frames = {c.frame for t in flights for c in t}
        reject_frames = {c.frame for c in all_cands} - flight_frames
        buckets = tracks.bucket_frames(all_frames, flight_frames, reject_frames)

        boxes: dict[str, list[list[int]]] = {}
        for t in flights:
            for c in t:
                boxes.setdefault(str(c.frame), []).append([c.x, c.y, c.w, c.h])

        counts = {b: sum(1 for v in buckets.values() if v == b) for b in ("positive", "negative", "discard")}
        print(
            f"{clip}: {len(all_cands)} candidates -> {len(linked)} tracks -> {len(flights)} flights | "
            f"positive {counts['positive']}  negative {counts['negative']}  discard {counts['discard']}",
            flush=True,
        )

        with open(os.path.join(args.out, f"{clip}.json"), "w") as fh:
            json.dump({"clip": clip, "buckets": buckets, "boxes": boxes}, fh)

        # Sample the sheet EVENLY across the timeline. Taking the first 24 in
        # list order covered 1.5% of the clip, all bunched at the start, which
        # is not a sample a human can judge the proposals from.
        picks = [(c.frame, c.x, c.y, c.w, c.h) for t in flights for c in [t[len(t) // 2]]]
        picks.sort(key=lambda p: p[0])
        step = max(1, len(picks) // 24)
        picks = picks[::step][:24]
        contact_sheet(video, picks, os.path.join(args.out, f"sheet_{clip}_flights.jpg"))

    print(f"\nwrote proposals to {args.out}")
    print("REVIEW THE CONTACT SHEETS before running export_for_roboflow.py.")


if __name__ == "__main__":
    main()
