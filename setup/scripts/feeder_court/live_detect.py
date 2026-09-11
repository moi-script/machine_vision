# ============================================================
# live_detect.py — Live USB-camera shuttlecock detection with the feeder_court
# weights. STANDALONE: imports nothing from app/ and reads nothing from
# config/settings.py, so running it cannot affect the drill system.
#
# WHY THIS IS SEPARATE FROM detect_video.py
# The offline script processes a whole clip at ~320 ms/frame, which is fine for
# a file and useless for a camera. Here the capture and the inference each run
# on their own thread, so the window always shows the newest frame and the
# boxes lag behind it by the HUD's "age" instead of the display stalling.
#
# THE FEED MUST NOT BE DOWNSCALED
# A shuttle at court distance is ~8-40 px. Halving the feed puts it near the
# sub-pixel floor and the P2 head stops firing — this has bitten this project
# before. The camera is asked for 1280x720 (what the model was trained on) and
# the frame is handed to the model at native size.
#
# Usage:  python scripts/feeder_court/live_detect.py
#         python scripts/feeder_court/live_detect.py --source 1
#         python scripts/feeder_court/live_detect.py --conf 0.15
#         python scripts/feeder_court/live_detect.py --backend torch
#         python scripts/feeder_court/live_detect.py --record live.mp4
#
# Keys:   q / ESC  quit          s  save a snapshot
#         + / -    conf +-0.05   h  hide/show the HUD
#         SPACE    pause         r  reset the running stats
# ============================================================

import argparse
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_WEIGHTS = os.path.join(REPO, "runs", "feeder_court", "best.pt")
SNAPSHOT_DIR = os.path.join(REPO, "scratch", "feeder_court_live")

BOX_COLOR = (0, 255, 255)
HUD_COLOR = (255, 255, 255)
WARN_COLOR = (60, 120, 255)

# Every hand-drawn box in feeder_court_yolo is 8-40 px on a side (n=53). The
# model emits 300-600 px boxes over ceiling lights that survive a 0.80
# confidence threshold, so thresholding alone cannot remove them. Same filter
# and same reasoning as detect_video.py.
MAX_SIDE_PX = 60


def _resolve_source(source: str):
    """A bare integer means a camera index; anything else is a file path."""
    try:
        return int(source)
    except ValueError:
        return source


def _open_capture(source, width: int, height: int):
    if isinstance(source, int) and sys.platform == "win32":
        # DirectShow: MSMF takes ~10 s to open the built-in cam on this laptop,
        # and it enumerates USB devices in a different order than DSHOW, so an
        # index that works under one backend can point elsewhere under the other.
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


class _Camera(threading.Thread):
    """Reader thread that keeps only the newest frame.

    VideoCapture buffers internally, so a consumer slower than the camera is
    handed progressively staler frames until the window is seconds behind
    reality. Draining here and keeping just the last frame means the display
    always shows now, whatever the model costs.
    """

    def __init__(self, cap):
        super().__init__(daemon=True)
        self._cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._ended = False
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                with self._lock:
                    self._ended = True
                return
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            return self._frame, self._ended

    def stop(self) -> None:
        self._stop.set()


class _Detector(threading.Thread):
    """Runs prediction on the newest submitted frame, dropping anything older.

    This keeps the displayed frame rate independent of inference cost. Raising
    --imgsz buys accuracy at the cost of box staleness (the HUD's "age"), not
    at the cost of fps. Stats count one sample per inference, not per displayed
    frame, so the hit rate stays honest.
    """

    def __init__(self, model, predict_kw: dict, conf: float, max_side: int):
        super().__init__(daemon=True)
        self._model = model
        self._predict_kw = predict_kw
        self._conf = conf
        self._max_side = max_side
        self._lock = threading.Lock()
        self._pending = None
        self._result = None  # (seq, boxes, oversized, ms, submitted_at)
        self._seq = 0
        self._stop = threading.Event()

    def submit(self, frame, conf: float) -> None:
        with self._lock:
            self._pending = (frame, time.perf_counter())
            self._conf = conf

    def latest(self):
        with self._lock:
            return self._result

    def run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                pending, conf = self._pending, self._conf
                self._pending = None
            if pending is None:
                time.sleep(0.001)
                continue

            frame, submitted_at = pending
            t0 = time.perf_counter()
            raw = self._model.predict(frame, conf=conf, **self._predict_kw)[0].boxes
            ms = (time.perf_counter() - t0) * 1000.0

            boxes, oversized = [], 0
            for box in raw:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                if self._max_side and max(x2 - x1, y2 - y1) > self._max_side:
                    oversized += 1
                    continue
                boxes.append((x1, y1, x2, y2, float(box.conf[0])))

            with self._lock:
                self._seq += 1
                self._result = (self._seq, boxes, oversized, ms, submitted_at)

    def stop(self) -> None:
        self._stop.set()


def _load(YOLO, weights: str, backend: str, imgsz: int):
    """Load weights, exporting to OpenVINO on first use for that imgsz.

    OpenVINO bakes the input size into the compiled model, so each imgsz needs
    its own export directory — reusing a 640 export at 1280 silently runs at
    640. Measured ~7x faster than PyTorch on this i3-1215U.
    """
    if backend == "torch":
        return YOLO(weights)

    # Ultralytics detects the format from the "_openvino_model" suffix, so the
    # size tag has to go before it, not after.
    stem = os.path.splitext(weights)[0]
    ov_dir = f"{stem}_{imgsz}_openvino_model"
    if not os.path.isdir(ov_dir):
        print(f"[EXPORT] building OpenVINO model for imgsz={imgsz} (one time, ~1 min)…")
        produced = YOLO(weights).export(format="openvino", imgsz=imgsz, half=False)
        os.rename(str(produced), ov_dir)
        print(f"[EXPORT] {ov_dir}")
    return YOLO(ov_dir, task="detect")


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Live shuttlecock detection on a USB camera.")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="default: runs/feeder_court/best.pt (the Colab p2-native run)")
    ap.add_argument("--source", default="0",
                    help="camera index (0 = built-in, 1 = external USB) or a video path")
    ap.add_argument("--width", type=int, default=1280, help="requested capture width")
    ap.add_argument("--height", type=int, default=720, help="requested capture height")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="LEAVE AT 1280. The P2 head was trained there; at 640 the "
                         "shuttle approaches sub-pixel and detection collapses")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence floor, adjustable live with + / -")
    ap.add_argument("--max-side", type=int, default=MAX_SIDE_PX, dest="max_side",
                    help="drop boxes whose longest side exceeds this (px); 0 disables")
    ap.add_argument("--max-det", type=int, default=20, dest="max_det")
    ap.add_argument("--backend", choices=["torch", "openvino"], default="openvino",
                    help="openvino is ~7x faster on this Intel CPU")
    ap.add_argument("--record", default=None, help="write the annotated view to this mp4")
    ap.add_argument("--seconds", type=float, default=None,
                    help="exit cleanly after N seconds. Without this the loop only ends "
                         "on a keypress, so an unattended run killed by a timeout leaves "
                         "the --record file unfinalised and unreadable")
    ap.add_argument("--list-cameras", action="store_true",
                    help="probe indices 0-3, report what opens, and exit")
    args = ap.parse_args()

    if args.list_cameras:
        for i in range(4):
            cap = _open_capture(i, args.width, args.height)
            if cap is None:
                print(f"  index {i}: not opened")
                continue
            ok, f = cap.read()
            print(f"  index {i}: {f.shape[1]}x{f.shape[0]}" if ok
                  else f"  index {i}: opened but no frame")
            cap.release()
        return 0

    if not os.path.exists(args.weights):
        print(f"missing weights: {args.weights}", file=sys.stderr)
        return 1

    source = _resolve_source(args.source)
    cap = _open_capture(source, args.width, args.height)
    if cap is None:
        print(f"could not open source {args.source!r}. Try --list-cameras.", file=sys.stderr)
        return 1

    feed_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    feed_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[FEED] {feed_w}x{feed_h}")
    if isinstance(source, int) and (feed_w, feed_h) != (args.width, args.height):
        print(f"[FEED] camera refused {args.width}x{args.height}. A smaller feed makes "
              f"the shuttle smaller in pixels, which is exactly what this model is "
              f"worst at — expect fewer detections than the offline clips.", flush=True)

    from ultralytics import YOLO
    model = _load(YOLO, args.weights, args.backend, args.imgsz)
    print(f"[MODEL] {args.weights}  {args.backend}  imgsz={args.imgsz}  conf={args.conf}")

    predict_kw = {"imgsz": args.imgsz, "max_det": args.max_det, "verbose": False}
    camera = _Camera(cap)
    detector = _Detector(model, predict_kw, args.conf, args.max_side)
    camera.start()
    detector.start()

    writer = None
    if args.record:
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                 20.0, (feed_w, feed_h))

    conf = args.conf
    show_hud = True
    paused = False
    last_seq = 0
    infer_n = infer_hits = oversized_total = 0
    best_seen = 0.0
    frame_times = deque(maxlen=30)
    boxes, age_ms, infer_ms = [], 0.0, 0.0
    t_prev = time.perf_counter()
    started = time.perf_counter()
    status = ""

    try:
        while True:
            if args.seconds and time.perf_counter() - started >= args.seconds:
                print(f"[EXIT] --seconds {args.seconds} reached")
                break

            frame, ended = camera.read()
            if ended:
                print("[FEED] source ended")
                break
            if frame is None:
                time.sleep(0.005)
                continue

            if not paused:
                detector.submit(frame, conf)

            latest = detector.latest()
            if latest is not None:
                seq, new_boxes, oversized, ms, submitted_at = latest
                boxes, infer_ms = new_boxes, ms
                age_ms = (time.perf_counter() - submitted_at) * 1000.0
                if seq != last_seq:
                    last_seq = seq
                    infer_n += 1
                    oversized_total += oversized
                    if new_boxes:
                        infer_hits += 1
                        best_seen = max(best_seen, max(b[4] for b in new_boxes))

            view = frame.copy()
            best = _draw_detections(view, boxes)

            now = time.perf_counter()
            frame_times.append(now - t_prev)
            t_prev = now
            fps = len(frame_times) / sum(frame_times) if sum(frame_times) else 0.0

            if show_hud:
                hit = 100 * infer_hits / infer_n if infer_n else 0.0
                lines = [
                    (f"{fps:5.1f} fps   infer {infer_ms:6.1f} ms   age {age_ms:5.0f} ms", HUD_COLOR),
                    (f"det {len(boxes)}   best {best:.2f}   peak {best_seen:.2f}", BOX_COLOR),
                    (f"hit {hit:5.1f}% of {infer_n} inferences   conf {conf:.2f}", HUD_COLOR),
                ]
                if oversized_total:
                    # Not cosmetic: a climbing count means the model is
                    # degenerating on this scene the way it does on mid.mp4.
                    lines.append((f"oversized dropped {oversized_total}", WARN_COLOR))
                if paused:
                    lines.append(("PAUSED", WARN_COLOR))
                if status:
                    lines.append((status, HUD_COLOR))
                _draw_hud(view, lines)

            if writer is not None:
                writer.write(view)

            cv2.imshow("feeder_court live", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                os.makedirs(SNAPSHOT_DIR, exist_ok=True)
                path = os.path.join(SNAPSHOT_DIR, f"snap_{int(time.time())}.png")
                cv2.imwrite(path, view)
                status = f"saved {os.path.basename(path)}"
            elif key in (ord("+"), ord("=")):
                conf = min(0.95, conf + 0.05)
            elif key in (ord("-"), ord("_")):
                conf = max(0.05, conf - 0.05)
            elif key == ord("h"):
                show_hud = not show_hud
            elif key == ord(" "):
                paused = not paused
            elif key == ord("r"):
                infer_n = infer_hits = oversized_total = 0
                best_seen = 0.0
                status = "stats reset"
    finally:
        detector.stop()
        camera.stop()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if infer_n:
        print(f"\n{infer_n} inferences, {infer_hits} with a detection "
              f"({100 * infer_hits / infer_n:.1f}%), peak conf {best_seen:.2f}, "
              f"{oversized_total} oversized boxes dropped")
        print("Hit rate counts frames where a box was drawn, including frames with no "
              "shuttle in view. It is not recall.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
