"""Scene-specific auto-labeller for a single fixed-camera clip.

Written for runs/shuttlecock_testing_camera_live.mp4, where the shuttlecock is
the only large, bright, DESATURATED object in frame. Skin and the wooden door
are warmer (higher saturation); the wall is desaturated but darker than the
shuttle's white feathers.

This is deliberately NOT a general shuttlecock detector. It exploits one scene
so that clip can be labelled without hand-boxing every frame. Always review the
contact sheet it writes before training on the output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def static_bright_mask(video: str, samples: int = 64) -> np.ndarray:
    """Bright, desaturated regions that never move — i.e. scene, not shuttle.

    Only valid because the camera is fixed. The shuttle moves throughout the
    clip, so it averages out of the median and is not masked.
    """
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // samples, 1)
    stack = []
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            stack.append(frame)
    cap.release()

    background = np.median(np.stack(stack), axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > 175) & (hsv[:, :, 1] < 60)).astype(np.uint8) * 255
    # Dilate so the halo around each static highlight is covered too.
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))


def median_background(video: str, samples: int = 64) -> np.ndarray:
    """Per-pixel median frame. Valid only for a fixed camera."""
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // samples, 1)
    stack = []
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            stack.append(frame)
    cap.release()
    return np.median(np.stack(stack), axis=0).astype(np.uint8)


def detect(
    frame: np.ndarray,
    min_area: int,
    max_area: int,
    static_mask: np.ndarray | None = None,
    background: np.ndarray | None = None,
    motion_thresh: int = 22,
) -> tuple[str, tuple[int, int, int, int] | None]:
    """Classify a frame as positive / negative / discard.

    Returns ("positive", box), ("negative", None) or ("discard", None).

    The three-way outcome matters: a frame holding a shuttle we merely failed to
    segment must NOT be written as an empty label, or training is taught that a
    visible shuttle is background. Anything ambiguous is dropped instead.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]

    # White feathers: bright and near-greyscale. The saturation threshold is what
    # separates the shuttle from the (warmer, more saturated) hand holding it.
    if background is not None:
        # Motion gate. Against an overexposed white wall the shuttle has almost
        # no brightness contrast, so "bright + desaturated" alone finds nothing
        # (or finds the wall). What still separates it is that it MOVES and the
        # wall does not. Brightness is then only used to rank candidates.
        diff = cv2.absdiff(frame, background)
        moving = (diff.max(axis=2) > motion_thresh).astype(np.uint8) * 255
        moving = cv2.dilate(moving, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        mask = ((val > 140) & (sat < 85)).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, moving)
    else:
        mask = ((val > 175) & (sat < 60)).astype(np.uint8) * 255
    if static_mask is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(static_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # A shadow line separates the feather skirt from the cork base. Close hard
    # enough to bridge it, or every box clips the cork off.
    bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, bridge, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = frame.shape[:2]
    best: tuple[int, int, int, int] | None = None
    best_score = 0.0
    ambiguous = False

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area * 0.35:
            continue  # genuine speckle noise, ignore entirely
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            continue
        aspect = w / h
        fill = area / (w * h)
        touches_edge = (
            x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2
        )

        accepted = (
            min_area <= area <= max_area
            and 0.35 <= aspect <= 2.6
            and fill >= 0.45
            and not touches_edge
        )
        if accepted:
            score = area * fill
            if score > best_score:
                best_score = score
                best = (x, y, w, h)
        elif touches_edge and area > max_area:
            # The grey wall itself: a huge low-saturation region running off the
            # frame. Not a shuttle candidate, so it must not block a negative.
            continue
        else:
            # A sizeable white blob we could not confidently accept. Could be a
            # part-occluded or edge-clipped shuttle; refuse to call it empty.
            ambiguous = True

    if best is not None:
        return "positive", best
    if ambiguous:
        return "discard", None
    return "negative", None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True, help="output dataset root")
    parser.add_argument("--min-area", type=int, default=6000)
    parser.add_argument("--max-area", type=int, default=180000)
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    parser.add_argument(
        "--motion",
        action="store_true",
        help="gate candidates on movement vs a median background; needed when the "
        "background is bright enough that brightness alone cannot isolate the shuttle",
    )
    args = parser.parse_args()

    out_root = Path(args.out)
    img_dir = out_root / "images"
    lbl_dir = out_root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    static_mask = static_bright_mask(args.video)
    cv2.imwrite(str(out_root / "static_mask.jpg"), static_mask)
    background = median_background(args.video) if args.motion else None
    if background is not None:
        cv2.imwrite(str(out_root / "background.jpg"), background)
        # With a motion gate the static mask is redundant and only costs recall:
        # a shuttle passing over a static highlight would be erased.
        static_mask = None

    cap = cv2.VideoCapture(args.video)
    idx = 0
    kept = 0
    labelled = 0
    discarded = 0
    tiles = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.stride:
            idx += 1
            continue

        height, width = frame.shape[:2]
        status, box = detect(
            frame, args.min_area, args.max_area, static_mask, background
        )

        stem = f"frame_{idx:05d}"
        if status != "discard":
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame)
            if box is None:
                # Empty label file: a legitimate background/negative example.
                (lbl_dir / f"{stem}.txt").write_text("")
            else:
                x, y, w, h = box
                cx, cy = (x + w / 2) / width, (y + h / 2) / height
                nw, nh = w / width, h / height
                (lbl_dir / f"{stem}.txt").write_text(
                    f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"
                )
                labelled += 1
            kept += 1
        else:
            discarded += 1

        if idx % 24 == 0 and len(tiles) < 24:
            preview = frame.copy()
            if box is not None:
                x, y, w, h = box
                cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)
            else:
                colour = (0, 165, 255) if status == "discard" else (0, 0, 255)
                cv2.putText(preview, status.upper(), (12, 64), 0, 1.4, colour, 3)
            cv2.putText(preview, f"#{idx}", (12, 28), 0, 1.0, (0, 0, 255), 2)
            tiles.append(cv2.resize(preview, (426, 240)))
        idx += 1
    cap.release()

    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[r * 4 : r * 4 + 4]) for r in range(len(tiles) // 4)]
        cv2.imwrite(str(out_root / "contact_sheet.jpg"), np.vstack(rows))

    print(f"frames kept   : {kept}")
    print(f"with shuttle  : {labelled} ({100 * labelled / max(kept, 1):.1f}%)")
    print(f"negatives     : {kept - labelled}")
    print(f"discarded     : {discarded} (ambiguous, deliberately unlabelled)")
    print(f"contact sheet : {out_root / 'contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
