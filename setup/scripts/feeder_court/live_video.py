# ============================================================
# live_video.py — Play a video file at its real frame rate with one of the
# trained models drawing boxes on it. STANDALONE: imports nothing from app/ and
# reads nothing from config/settings.py, so running it cannot affect the drill
# system.
#
# WHY THIS EXISTS ALONGSIDE detect_video.py AND live_detect.py
#   detect_video.py  scores every frame, so the window advances at INFERENCE
#                    speed — 86 ms/frame at imgsz 1280 is ~11 fps, which looks
#                    broken even though the model is fine.
#   live_detect.py   has the right threading but only reads a USB camera.
#   live_video.py    is live_detect.py's architecture pointed at a FILE.
#
# HOW IT STAYS AT 30 FPS
# The reader thread paces itself to the video's own fps and keeps only the
# newest frame. The detector thread takes whatever the newest frame is when it
# becomes free and throws away anything older. So the display rate is the
# video's rate no matter what the model costs; the model's cost shows up as the
# HUD's "age" — how stale the boxes are — instead of as stutter.
#
# MEASURED ON THIS i3-1215U (100 frames, 1280x720 source, OpenVINO)
#     model    imgsz  ms/frame   fps   frames-with-a-hit
#     shuttle   1280      86.1  11.6   31%
#     shuttle    640      25.3  39.5   12%   <- downscaling halves the shuttle
#     landed    1280      54.9  18.2   100%
#     landed     640      18.3  54.7   98%   <- safe here, floor shuttles stay big
# So `landed` defaults to 640 and `shuttle` to 1280. Do NOT "speed up" the
# shuttle model by dropping imgsz: at 640 a far-court shuttle falls below the
# detector's finest stride and it stops firing. Use --crop instead, which cuts
# the input area while keeping the shuttle at native scale.
#
# Usage:
#   python scripts/feeder_court/live_video.py --model shuttle
#   python scripts/feeder_court/live_video.py --model landed
#   python scripts/feeder_court/live_video.py --model shuttle --video path/to.mp4
#   python scripts/feeder_court/live_video.py --model shuttle --crop 0.2,0.1,0.8,1.0
#   python scripts/feeder_court/live_video.py --model landed --record out.mp4
#
# Keys:  q / ESC  quit        SPACE  pause         s  snapshot
#        + / -    conf +-0.05     h  hide the HUD  r  reset stats
# ============================================================

from __future__ import annotations

import argparse
import os
import threading
import time
from collections import deque

import cv2

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VID_DIR = os.path.join(REPO, "datasets", "vid_source", "clear_badminton_dataset_for_collab")
SNAPSHOT_DIR = os.path.join(REPO, "scratch", "live_video")

BOX_COLOR = (0, 255, 255)
POSE_COLOR = (120, 255, 120)
HUD_COLOR = (255, 255, 255)
WARN_COLOR = (60, 120, 255)

# Same registry as app/pipeline.py, duplicated on purpose: this script must stay
# runnable when the server, its Mongo settings and its camera slots are not.
MODELS: dict[str, dict] = {
    "shuttle": {
        "weights": "runs/clear_badminton/p2-native/weights/best.pt",
        "imgsz": 1280, "conf": 0.25, "max_side": 60, "task": "detect",
        "video": os.path.join(VID_DIR, "dataset_court1.mp4"),
        "label": "flying shuttle",
    },
    "landed": {
        "weights": "runs/shuttle_lines/stock-n/weights/best.pt",
        "imgsz": 640, "conf": 0.40, "max_side": 90, "task": "detect",
        "video": os.path.join(VID_DIR, "lines.mp4"),
        "label": "landed shuttle",
    },
    "pose": {
        "weights": "yolov8n-pose.pt",
        "imgsz": 640, "conf": 0.25, "max_side": 0, "task": "pose",
        "video": os.path.join(VID_DIR, "dataset_court1.mp4"),
        "label": "player pose",
    },
}

# COCO 17-keypoint skeleton, used only to draw.
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]


# -- threads --------------------------------------------------


class _VideoSource(threading.Thread):
    """Reader that paces itself to the video's fps and keeps only the newest frame.

    A file reads as fast as the disk allows, so without pacing the clip would
    blast past in a few seconds. Keeping only the newest frame means a slow
    consumer never falls progressively further behind — it skips instead.
    """

    def __init__(self, path: str, speed: float = 1.0, loop: bool = False):
        super().__init__(daemon=True)
        self._path = path
        self._speed = max(speed, 0.01)
        self._loop = loop
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise SystemExit(f"cannot open video: {path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._lock = threading.Lock()
        self._frame = None
        self._index = 0
        self._ended = False
        self._paused = threading.Event()
        self._stop = threading.Event()

    def run(self) -> None:
        interval = 1.0 / (self.fps * self._speed)
        next_due = time.perf_counter()
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.01)
                next_due = time.perf_counter()
                continue
            ok, frame = self._cap.read()
            if not ok:
                if self._loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    with self._lock:
                        self._index = 0
                    continue
                with self._lock:
                    self._ended = True
                return
            with self._lock:
                self._frame = frame
                self._index += 1
            next_due += interval
            slack = next_due - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                # Reading fell behind the clock; resync rather than accumulate debt.
                next_due = time.perf_counter()

    def read(self):
        with self._lock:
            return self._frame, self._index, self._ended

    def toggle_pause(self) -> bool:
        if self._paused.is_set():
            self._paused.clear()
        else:
            self._paused.set()
        return self._paused.is_set()

    def stop(self) -> None:
        self._stop.set()


class _Detector(threading.Thread):
    """Predicts on the newest submitted frame and drops anything older.

    Stats count one sample per INFERENCE, not per displayed frame, so the hit
    rate stays honest when the display outruns the model.
    """

    def __init__(self, model, cfg: dict, imgsz: int, conf: float, crop=None):
        super().__init__(daemon=True)
        self._model = model
        self._cfg = cfg
        self._imgsz = imgsz
        self._conf = conf
        self._crop = crop           # (x1, y1, x2, y2) in pixels, or None
        self._lock = threading.Lock()
        self._pending = None
        self._result = None         # (seq, boxes, kpts, oversized, ms, submitted_at)
        self._seq = 0
        self._times = deque(maxlen=30)
        self._hits = 0
        self._runs = 0
        self._stop = threading.Event()

    def submit(self, frame, conf: float) -> None:
        with self._lock:
            self._pending = (frame, time.perf_counter())
            self._conf = conf

    def latest(self):
        with self._lock:
            return self._result

    def stats(self):
        with self._lock:
            avg = sum(self._times) / len(self._times) if self._times else 0.0
            return avg, self._hits, self._runs

    def reset_stats(self) -> None:
        with self._lock:
            self._times.clear()
            self._hits = self._runs = 0

    def run(self) -> None:
        is_pose = self._cfg["task"] == "pose"
        max_side = self._cfg["max_side"]
        while not self._stop.is_set():
            with self._lock:
                pending, conf = self._pending, self._conf
                self._pending = None
            if pending is None:
                time.sleep(0.001)
                continue

            frame, submitted_at = pending
            ox = oy = 0
            if self._crop is not None:
                x1, y1, x2, y2 = self._crop
                frame = frame[y1:y2, x1:x2]
                ox, oy = x1, y1

            t0 = time.perf_counter()
            res = self._model.predict(frame, imgsz=self._imgsz, conf=conf,
                                      classes=[0] if is_pose else None,
                                      max_det=100, verbose=False)[0]
            ms = (time.perf_counter() - t0) * 1000.0

            boxes, oversized = [], 0
            for box in res.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                if max_side and max(x2 - x1, y2 - y1) > max_side:
                    oversized += 1
                    continue
                boxes.append((x1 + ox, y1 + oy, x2 + ox, y2 + oy, float(box.conf[0])))

            kpts = []
            if is_pose and res.keypoints is not None:
                for person in res.keypoints.data.tolist():
                    kpts.append([(int(x) + ox, int(y) + oy, c) for x, y, c in person])

            with self._lock:
                self._seq += 1
                self._times.append(ms)
                self._runs += 1
                if boxes:
                    self._hits += 1
                self._result = (self._seq, boxes, kpts, oversized, ms, submitted_at)

    def stop(self) -> None:
        self._stop.set()


# -- helpers --------------------------------------------------


def _load(YOLO, weights: str, backend: str, imgsz: int, task: str):
    """Load weights, exporting to OpenVINO on first use for that imgsz.

    OpenVINO bakes the input size into the compiled model, so each imgsz needs
    its own export directory — reusing a 640 export at 1280 silently runs at
    640. The size tag goes BEFORE the "_openvino_model" suffix because
    Ultralytics detects the format from that suffix.
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
    return YOLO(ov_dir, task=task)


def _parse_crop(spec: str | None, width: int, height: int):
    """Accept fractions (0.2,0.1,0.8,1.0) or pixels (200,330,1120,720)."""
    if not spec:
        return None
    parts = [float(v) for v in spec.split(",")]
    if len(parts) != 4:
        raise SystemExit("--crop needs four comma-separated numbers")
    if all(v <= 1.0 for v in parts):
        x1, y1, x2, y2 = (parts[0] * width, parts[1] * height,
                          parts[2] * width, parts[3] * height)
    else:
        x1, y1, x2, y2 = parts
    box = (max(int(x1), 0), max(int(y1), 0), min(int(x2), width), min(int(y2), height))
    if box[2] - box[0] < 32 or box[3] - box[1] < 32:
        raise SystemExit(f"--crop is degenerate: {box}")
    return box


def _draw_detections(frame, boxes, min_marker=26):
    """Draw boxes, enlarging the marker for tiny ones so they stay visible.

    A shuttle at court distance is ~20 px wide; a rectangle that small is
    invisible on screen, which makes a working model look broken.
    """
    best = 0.0
    for x1, y1, x2, y2, conf in boxes:
        best = max(best, conf)
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if (x2 - x1) < min_marker or (y2 - y1) < min_marker:
            half = min_marker // 2
            cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half),
                          BOX_COLOR, 1)
            cv2.line(frame, (cx - half - 6, cy), (cx - half, cy), BOX_COLOR, 1)
            cv2.line(frame, (cx + half, cy), (cx + half + 6, cy), BOX_COLOR, 1)
        cv2.putText(frame, f"{conf:.2f}", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, BOX_COLOR, 1, cv2.LINE_AA)
    return best


def _draw_pose(frame, people, min_conf=0.5):
    for kp in people:
        for a, b in SKELETON:
            if a < len(kp) and b < len(kp) and kp[a][2] > min_conf and kp[b][2] > min_conf:
                cv2.line(frame, kp[a][:2], kp[b][:2], POSE_COLOR, 2, cv2.LINE_AA)
        for x, y, c in kp:
            if c > min_conf:
                cv2.circle(frame, (x, y), 3, POSE_COLOR, -1, cv2.LINE_AA)


def _draw_hud(frame, lines):
    x, y = 10, 22
    pad = 6
    width = max(len(t) for t, _ in lines) * 9 + pad * 2
    height = len(lines) * 20 + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - pad, y - 16 - pad), (x + width, y + height - 20),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for text, color in lines:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                    1, cv2.LINE_AA)
        y += 20


# -- main -----------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Play a video at its real frame rate with a trained model drawing on it.")
    ap.add_argument("--model", choices=sorted(MODELS), default="shuttle",
                    help="which trained model to run (default: shuttle)")
    ap.add_argument("--video", default=None,
                    help="video file; defaults to the clip that suits the model")
    ap.add_argument("--weights", default=None, help="override the model's weights")
    ap.add_argument("--imgsz", type=int, default=None, help="override the model's input size")
    ap.add_argument("--conf", type=float, default=None, help="override the model's confidence")
    ap.add_argument("--crop", default=None,
                    help="x1,y1,x2,y2 as fractions or pixels; cuts inference cost "
                         "WITHOUT shrinking the shuttle")
    ap.add_argument("--backend", choices=["torch", "openvino"], default="openvino")
    ap.add_argument("--speed", type=float, default=1.0, help="playback rate multiplier")
    ap.add_argument("--loop", action="store_true", help="restart the clip at the end")
    ap.add_argument("--record", default=None, help="write the annotated view to this mp4")
    ap.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    ap.add_argument("--no-show", action="store_true", dest="no_show",
                    help="run headless; useful with --record")
    args = ap.parse_args()

    os.chdir(REPO)
    cfg = MODELS[args.model]
    weights = args.weights or cfg["weights"]
    imgsz = args.imgsz or cfg["imgsz"]
    conf = cfg["conf"] if args.conf is None else args.conf
    video = args.video or cfg["video"]

    if not os.path.exists(weights):
        raise SystemExit(f"missing weights: {weights}")
    if not os.path.exists(video):
        raise SystemExit(f"missing video: {video}")

    source = _VideoSource(video, args.speed, args.loop)
    crop = _parse_crop(args.crop, source.width, source.height)

    from ultralytics import YOLO  # imported late so --help stays instant
    import numpy as np

    model = _load(YOLO, weights, args.backend, imgsz, cfg["task"])

    # OpenVINO compiles on FIRST inference, which costs 300 ms to several
    # seconds. Without this the HUD's "model fps" reads absurdly low for the
    # opening seconds and the model looks broken when it is merely cold.
    #
    # Warm up on the SHAPE THE DETECTOR WILL ACTUALLY SEE, crop included. An
    # OpenVINO export has a static input shape, so handing it a differently
    # shaped image makes it re-shape — and if the warm-up shape and the real
    # shape disagree, that cost is paid on every single frame rather than once.
    # Measured: --crop with --imgsz 896 ran 434 ms/frame warmed on a full frame
    # against 51 ms uncropped, an 8x penalty that looked like a slow model.
    print("warming up...", end="", flush=True)
    if crop is not None:
        wh, ww = crop[3] - crop[1], crop[2] - crop[0]
    else:
        wh, ww = source.height, source.width
    blank = np.zeros((wh, ww, 3), dtype="uint8")
    warm = []
    for _ in range(4):
        t0 = time.perf_counter()
        model.predict(blank, imgsz=imgsz, conf=conf, verbose=False)
        warm.append((time.perf_counter() - t0) * 1000.0)
    print(f" ready ({warm[-1]:.0f} ms/frame warm)\n")

    print(f"model    {args.model}  ({cfg['label']})")
    print(f"weights  {weights}")
    print(f"video    {video}  {source.width}x{source.height}  "
          f"{source.total} frames @ {source.fps:.1f} fps")
    print(f"imgsz    {imgsz}   conf {conf}   backend {args.backend}"
          + (f"   crop {crop}" if crop else ""))
    print("keys     q quit   SPACE pause   s snapshot   +/- conf   h HUD   r reset\n")

    detector = _Detector(model, cfg, imgsz, conf, crop)
    detector.start()
    source.start()

    writer = None
    if args.record:
        os.makedirs(os.path.dirname(os.path.abspath(args.record)) or ".", exist_ok=True)
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                 source.fps, (source.width, source.height))

    window = f"live_video [{args.model}]"
    show_hud = True
    shown = 0
    last_index = -1
    started = time.perf_counter()
    display_times = deque(maxlen=30)
    last_shown_at = time.perf_counter()

    try:
        while True:
            frame, index, ended = source.read()
            if frame is None:
                if ended:
                    break
                time.sleep(0.005)
                continue
            if ended and not args.loop:
                break
            if args.seconds and (time.perf_counter() - started) > args.seconds:
                break

            # Render ONLY when the reader has produced a new frame. Without this
            # the loop spins on whatever is in the buffer and "displays" the same
            # frame hundreds of times a second, which inflates the fps readout and
            # writes a --record file many times longer than the clip.
            if index == last_index:
                if args.no_show:
                    time.sleep(0.001)
                    continue
                # Still pump the window so keys keep working while paused.
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    source.toggle_pause()
                elif key == ord("h"):
                    show_hud = not show_hud
                continue
            last_index = index

            canvas = frame.copy()
            detector.submit(frame, conf)

            result = detector.latest()
            age_ms = 0.0
            boxes, kpts, oversized = [], [], 0
            if result is not None:
                _seq, boxes, kpts, oversized, _ms, submitted_at = result
                age_ms = (time.perf_counter() - submitted_at) * 1000.0

            best = _draw_detections(canvas, boxes)
            if kpts:
                _draw_pose(canvas, kpts)
            if crop is not None:
                cv2.rectangle(canvas, (crop[0], crop[1]), (crop[2], crop[3]),
                              (90, 90, 90), 1)

            now = time.perf_counter()
            display_times.append(now - last_shown_at)
            last_shown_at = now
            disp_fps = len(display_times) / sum(display_times) if sum(display_times) else 0.0
            avg_ms, hits, runs = detector.stats()

            if show_hud:
                age_color = HUD_COLOR if age_ms < 200 else WARN_COLOR
                _draw_hud(canvas, [
                    (f"{args.model}  imgsz {imgsz}  conf {conf:.2f}", HUD_COLOR),
                    (f"display {disp_fps:5.1f} fps   frame {index}/{source.total}", HUD_COLOR),
                    (f"model   {1000.0 / avg_ms if avg_ms else 0:5.1f} fps  "
                     f"({avg_ms:5.1f} ms)", HUD_COLOR),
                    (f"age     {age_ms:5.0f} ms", age_color),
                    (f"boxes   {len(boxes)}  best {best:.2f}  dropped {oversized}", HUD_COLOR),
                    (f"hit     {100.0 * hits / runs if runs else 0:5.1f}% of {runs} runs", HUD_COLOR),
                ])

            if writer is not None:
                writer.write(canvas)
            shown += 1

            if args.no_show:
                continue

            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                source.toggle_pause()
            elif key == ord("h"):
                show_hud = not show_hud
            elif key == ord("r"):
                detector.reset_stats()
            elif key in (ord("+"), ord("=")):
                conf = min(conf + 0.05, 0.95)
            elif key == ord("-"):
                conf = max(conf - 0.05, 0.05)
            elif key == ord("s"):
                os.makedirs(SNAPSHOT_DIR, exist_ok=True)
                path = os.path.join(SNAPSHOT_DIR, f"{args.model}_{int(time.time())}.png")
                cv2.imwrite(path, canvas)
                print(f"[SNAPSHOT] {path}")
    finally:
        source.stop()
        detector.stop()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    avg_ms, hits, runs = detector.stats()
    elapsed = time.perf_counter() - started
    print(f"\nshown {shown} frames in {elapsed:.1f}s  "
          f"({shown / elapsed if elapsed else 0:.1f} fps displayed)")
    print(f"model ran {runs} times, {avg_ms:.1f} ms each "
          f"({1000.0 / avg_ms if avg_ms else 0:.1f} fps), "
          f"hit on {100.0 * hits / runs if runs else 0:.1f}%")
    if args.record:
        print(f"wrote {args.record}")


if __name__ == "__main__":
    main()
