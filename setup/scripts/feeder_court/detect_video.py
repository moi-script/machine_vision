# ============================================================
# detect_video.py — Run the feeder_court shuttlecock detector over a video and
# burn the boxes into an annotated copy, optionally watching it live.
#
# The weights come from the Colab run (yolov8-p2 head, trained at imgsz 1280),
# so inference defaults to 1280 too: dropping to 640 halves the shuttle to
# sub-pixel size on far footage and the P2 head stops firing.
#
# Usage:
#   python scripts/feeder_court/detect_video.py                 # near, mid, far rally spans
#   python scripts/feeder_court/detect_video.py --clips far --full --conf 0.15
#   python scripts/feeder_court/detect_video.py --show --videos <path>/*.mp4
#
# Window keys (--show):  q / ESC  quit          SPACE  pause
# ============================================================

import argparse
import glob
import json
import os
import time

import cv2

DEFAULT_SOURCE = "datasets/vid_source/new_badminton_source"
DEFAULT_WEIGHTS = "runs/feeder_court/best.pt"
DEFAULT_SEGMENTS = "datasets/feeder_court/segments.json"
CLIPS = ["near", "mid", "far"]

BOX_COLOR = (0, 255, 255)
HUD_COLOR = (255, 255, 255)
WINDOW = "feeder_court detect"

# Every hand-drawn shuttle box in feeder_court_yolo is between 8 and 40 px on a
# side. The model happily emits 500 px boxes over the ceiling lights on mid.mp4,
# so anything well outside that range is a false positive by definition.
MAX_SIDE_PX = 60


def load_model(YOLO, weights: str, backend: str, imgsz: int):
    """Load weights, exporting to OpenVINO on first use for that imgsz.

    OpenVINO bakes the input size into the compiled model, so each imgsz needs
    its own export directory — reusing a 640 export at 1280 silently runs at
    640. Measured ~7x faster than PyTorch on this i3-1215U, which is the
    difference between watching a clip and waiting for it. Same directory
    naming as live_detect.py, so the two scripts share one export.
    """
    if backend == "torch":
        return YOLO(weights)

    stem = os.path.splitext(weights)[0]
    ov_dir = f"{stem}_{imgsz}_openvino_model"
    if not os.path.isdir(ov_dir):
        print(f"[EXPORT] building OpenVINO model for imgsz={imgsz} (one time, ~1 min)...")
        produced = YOLO(weights).export(format="openvino", imgsz=imgsz, half=False)
        os.rename(str(produced), ov_dir)
        print(f"[EXPORT] {ov_dir}")
    return YOLO(ov_dir, task="detect")


def span_for(label: str, segments: dict, total: int, full: bool) -> tuple[int, int]:
    """Rally span from segments.json, or the whole clip when --full."""
    if full or label not in segments or not segments[label]:
        return 0, total
    starts = [s for s, _ in segments[label]]
    ends = [e for _, e in segments[label]]
    return max(min(starts), 0), min(max(ends), total)


def draw(frame, boxes, frame_idx: int, label: str, min_marker: int = 26) -> None:
    for x1, y1, x2, y2, conf in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

        # A shuttle at court distance is ~20 px wide; a rectangle that small is
        # invisible on screen, which makes a working model look broken.
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if (x2 - x1) < min_marker or (y2 - y1) < min_marker:
            half = min_marker // 2
            cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half),
                          BOX_COLOR, 1)
            cv2.line(frame, (cx - half - 6, cy), (cx - half, cy), BOX_COLOR, 1)
            cv2.line(frame, (cx + half, cy), (cx + half + 6, cy), BOX_COLOR, 1)

        cv2.putText(frame, f"{conf:.2f}", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1, cv2.LINE_AA)
    hud = f"{label}  f{frame_idx}  det={len(boxes)}"
    cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HUD_COLOR, 2, cv2.LINE_AA)


def run_video(model, video_path: str, label: str, args, segments: dict) -> dict:
    if not os.path.exists(video_path):
        raise SystemExit(f"missing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start, end = span_for(label, segments, total, args.full)

    writer = None
    out_path = None
    if not args.no_save:
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, f"{label}_boxed.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = hits = dets = oversized = 0
    conf_sum = 0.0
    quit_early = False
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

        draw(frame, boxes, frame_idx, label)
        if writer is not None:
            writer.write(frame)

        if args.show:
            cv2.imshow(WINDOW, frame)
            # Every frame is scored, so the window advances at inference speed
            # rather than at the clip's fps — this shows what the model sees,
            # not a real-time playback of it.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                quit_early = True
                break
            if key == ord(" "):
                while True:
                    k2 = cv2.waitKey(50) & 0xFF
                    if k2 in (ord(" "), ord("q"), 27):
                        quit_early = k2 in (ord("q"), 27)
                        break
                if quit_early:
                    break

        if frames % 100 == 0:
            print(f"  {label}: {frames}/{end - start} frames, {dets} detections", flush=True)

    elapsed = time.perf_counter() - t0
    cap.release()
    if writer is not None:
        writer.release()

    return {
        "clip": label,
        "out": out_path,
        "span": (start, end),
        "frames": frames,
        "frames_with_det": hits,
        "detections": dets,
        "oversized_dropped": oversized,
        "mean_conf": conf_sum / dets if dets else 0.0,
        "ms_per_frame": 1000 * elapsed / frames if frames else 0.0,
        "quit_early": quit_early,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Box shuttlecocks in badminton source videos.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--segments", default=DEFAULT_SEGMENTS)
    parser.add_argument("--out", default="runs/feeder_court/detect")
    parser.add_argument("--clips", nargs="+", default=CLIPS, choices=CLIPS)
    parser.add_argument("--videos", nargs="+", default=None,
                        help="explicit video paths or globs; overrides --source/--clips")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=20, dest="max_det")
    parser.add_argument("--max-side", type=int, default=MAX_SIDE_PX, dest="max_side",
                        help="drop boxes whose longest side exceeds this (px); 0 disables")
    parser.add_argument("--backend", choices=["torch", "openvino"], default="openvino",
                        help="openvino is ~7x faster on this CPU and numerically equivalent")
    parser.add_argument("--show", action="store_true", help="live window with the boxes drawn")
    parser.add_argument("--no-save", action="store_true", dest="no_save",
                        help="skip writing the annotated mp4 (pair with --show)")
    parser.add_argument("--full", action="store_true", help="ignore segments.json, process the whole clip")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(root)

    if not os.path.exists(args.weights):
        raise SystemExit(f"missing weights: {args.weights}")

    if args.videos:
        paths = []
        for pattern in args.videos:
            # PowerShell does not expand globs the way a POSIX shell does, so
            # expand here too and fall through for literal paths.
            matched = sorted(glob.glob(pattern))
            paths.extend(matched or [pattern])
    else:
        paths = [os.path.join(args.source, f"{clip}.mp4") for clip in args.clips]
    if not paths:
        raise SystemExit("no videos matched")

    segments = {}
    if os.path.exists(args.segments):
        with open(args.segments) as fh:
            segments = json.load(fh)

    from ultralytics import YOLO  # imported late so --help stays instant
    model = load_model(YOLO, args.weights, args.backend, args.imgsz)
    print(f"loaded {args.weights} [{args.backend}]  imgsz={args.imgsz}  conf={args.conf}\n")

    rows = []
    for path in paths:
        label = os.path.splitext(os.path.basename(path))[0]
        rows.append(run_video(model, path, label, args, segments))
        if rows[-1]["quit_early"]:
            print("quit requested - stopping")
            break

    if args.show:
        cv2.destroyAllWindows()

    print("\nclip             span            frames  hit%   dets  oversized  mean_conf  ms/frame")
    for r in rows:
        hit = 100 * r["frames_with_det"] / r["frames"] if r["frames"] else 0
        print(f'{r["clip"]:<16} {str(r["span"]):<15} {r["frames"]:<7} {hit:5.1f}  {r["detections"]:<5} '
              f'{r["oversized_dropped"]:<9} {r["mean_conf"]:9.3f}  {r["ms_per_frame"]:8.1f}')
    for r in rows:
        if r["out"]:
            print(f'  -> {r["out"]}')


if __name__ == "__main__":
    main()
