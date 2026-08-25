# ============================================================
# detect_video.py — Run the feeder_court shuttlecock detector over a video and
# burn the boxes into an annotated copy.
#
# The weights come from the Colab run (yolov8-p2 head, trained at imgsz 1280),
# so inference defaults to 1280 too: dropping to 640 halves the shuttle to
# sub-pixel size on far footage and the P2 head stops firing.
#
# Usage:
#   python scripts/feeder_court/detect_video.py                 # near, mid, far rally spans
#   python scripts/feeder_court/detect_video.py --clips far --full --conf 0.15
# ============================================================

import argparse
import json
import os
import sys
import time

import cv2

DEFAULT_SOURCE = "datasets/vid_source/new_badminton_source"
DEFAULT_WEIGHTS = "runs/feeder_court/best.pt"
DEFAULT_SEGMENTS = "datasets/feeder_court/segments.json"
CLIPS = ["near", "mid", "far"]

BOX_COLOR = (0, 255, 255)
HUD_COLOR = (255, 255, 255)

# Every hand-drawn shuttle box in feeder_court_yolo is between 8 and 40 px on a
# side. The model happily emits 500 px boxes over the ceiling lights on mid.mp4,
# so anything well outside that range is a false positive by definition.
MAX_SIDE_PX = 60


def span_for(clip: str, segments: dict, total: int, full: bool) -> tuple[int, int]:
    """Rally span from segments.json, or the whole clip when --full."""
    if full or clip not in segments or not segments[clip]:
        return 0, total
    starts = [s for s, _ in segments[clip]]
    ends = [e for _, e in segments[clip]]
    return max(min(starts), 0), min(max(ends), total)


def draw(frame, boxes, frame_idx: int, clip: str) -> None:
    for x1, y1, x2, y2, conf in boxes:
        # The shuttle is a few pixels wide; an inflated box keeps it visible.
        pad = max(6 - (x2 - x1) // 2, 0)
        cv2.rectangle(frame, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), BOX_COLOR, 2)
        cv2.putText(frame, f"{conf:.2f}", (x1 - pad, max(y1 - pad - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1, cv2.LINE_AA)
    hud = f"{clip}  f{frame_idx}  det={len(boxes)}"
    cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HUD_COLOR, 2, cv2.LINE_AA)


def run_clip(model, clip: str, args, segments: dict) -> dict:
    video_path = os.path.join(args.source, f"{clip}.mp4")
    if not os.path.exists(video_path):
        raise SystemExit(f"missing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start, end = span_for(clip, segments, total, args.full)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{clip}_boxed.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = hits = dets = oversized = 0
    conf_sum = 0.0
    t0 = time.perf_counter()

    for frame_idx in range(start, end):
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                               iou=args.iou, max_det=args.max_det, verbose=False)[0]

        boxes = []
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            if args.max_side and max(x2 - x1, y2 - y1) > args.max_side:
                oversized += 1
                continue
            conf = float(box.conf[0])
            boxes.append((x1, y1, x2, y2, conf))
            conf_sum += conf
        dets += len(boxes)
        hits += 1 if boxes else 0
        frames += 1

        draw(frame, boxes, frame_idx, clip)
        writer.write(frame)

        if frames % 100 == 0:
            print(f"  {clip}: {frames}/{end - start} frames, {dets} detections", flush=True)

    elapsed = time.perf_counter() - t0
    cap.release()
    writer.release()

    return {
        "clip": clip,
        "out": out_path,
        "span": (start, end),
        "frames": frames,
        "frames_with_det": hits,
        "detections": dets,
        "oversized_dropped": oversized,
        "mean_conf": conf_sum / dets if dets else 0.0,
        "ms_per_frame": 1000 * elapsed / frames if frames else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Box shuttlecocks in the feeder_court source videos.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--segments", default=DEFAULT_SEGMENTS)
    parser.add_argument("--out", default="runs/feeder_court/detect")
    parser.add_argument("--clips", nargs="+", default=CLIPS, choices=CLIPS)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=20, dest="max_det")
    parser.add_argument("--max-side", type=int, default=MAX_SIDE_PX, dest="max_side",
                        help="drop boxes whose longest side exceeds this (px); 0 disables")
    parser.add_argument("--full", action="store_true", help="ignore segments.json, process the whole clip")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(root)

    if not os.path.exists(args.weights):
        raise SystemExit(f"missing weights: {args.weights}")

    segments = {}
    if os.path.exists(args.segments):
        with open(args.segments) as fh:
            segments = json.load(fh)

    from ultralytics import YOLO  # imported late so --help stays instant
    model = YOLO(args.weights)
    print(f"loaded {args.weights}  classes={model.names}  imgsz={args.imgsz}  conf={args.conf}\n")

    rows = [run_clip(model, clip, args, segments) for clip in args.clips]

    print("\nclip   span            frames  hit%   dets  oversized  mean_conf  ms/frame")
    for r in rows:
        hit = 100 * r["frames_with_det"] / r["frames"] if r["frames"] else 0
        print(f'{r["clip"]:<6} {str(r["span"]):<15} {r["frames"]:<7} {hit:5.1f}  {r["detections"]:<5} '
              f'{r["oversized_dropped"]:<9} {r["mean_conf"]:9.3f}  {r["ms_per_frame"]:8.1f}')
    for r in rows:
        print(f'  -> {r["out"]}')


if __name__ == "__main__":
    main()
