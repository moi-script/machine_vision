# ============================================================
# test_shuttle_v2_webcam.py — Live webcam smoke test for the v2 shuttlecock
# weights. STANDALONE: imports nothing from app/ and reads nothing from
# config/settings.py, so running it cannot affect the drill system.
#
# WHY THIS EXISTS
# A validation mAP is measured on frames from the same dataset the model was
# trained on. v1 scored mAP50 0.898 that way and then detected almost nothing
# on a real camera, because its train/val split shared frames from the same
# clips. The only honest test is a camera the model has never seen — yours.
#
# WHAT TO WATCH
# The HUD's "hit rate" (share of frames with at least one detection while a
# shuttle is actually in view) tells you more than any single box. A model that
# fires on 3% of frames is not usable for tracking no matter how good the
# occasional detection looks.
#
# Usage:  python scripts/test_shuttle_v2_webcam.py
#         python scripts/test_shuttle_v2_webcam.py --conf 0.15
#         python scripts/test_shuttle_v2_webcam.py --compare models/shuttlecock.pt
#         python scripts/test_shuttle_v2_webcam.py --source clips/rally.mp4
#         python scripts/test_shuttle_v2_webcam.py --record out.mp4
#
# Keys:   q / ESC  quit          s  save a snapshot to scratch/
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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(REPO, "models", "shuttlecock_v3.pt")
SNAPSHOT_DIR = os.path.join(REPO, "scratch", "shuttle_v2_test")

TEST_COLOR = (0, 235, 160)   # green  — whatever --weights points at
BASE_COLOR = (60, 120, 255)  # orange — whatever --compare points at


def _resolve_source(source: str):
    """A bare integer means a camera index; anything else is a file path."""
    try:
        return int(source)
    except ValueError:
        return source


def _open_capture(source, width: int, height: int):
    if isinstance(source, int) and sys.platform == "win32":
        # DirectShow: MSMF takes ~10s to open the built-in cam on this laptop.
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

    VideoCapture buffers internally, so a consumer slower than the camera gets
    handed progressively staler frames until the window is seconds behind
    reality. Draining the buffer here and keeping just the last frame means the
    display always shows now, whatever the model costs.
    """

    def __init__(self, cap, crop: int):
        super().__init__(daemon=True)
        self._cap = cap
        self._crop = crop
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
            frame = _center_crop(frame, self._crop)
            with self._lock:
                self._frame = frame

    def read(self):
        """Newest frame, or None until the first one lands. (frame, ended)"""
        with self._lock:
            return self._frame, self._ended

    def stop(self) -> None:
        self._stop.set()


class _Detector(threading.Thread):
    """Runs prediction on the newest submitted frame, dropping anything older.

    This is what keeps the frame rate flat across --imgsz. The display loop no
    longer waits on inference; it draws the most recent boxes available. Raising
    imgsz therefore buys accuracy at the cost of box STALENESS (the HUD's "age"
    field), not at the cost of fps. The stats below still count one sample per
    inference, not per displayed frame, so the hit rate stays honest.
    """

    def __init__(self, models, predict_kw: dict, conf: float):
        super().__init__(daemon=True)
        self._models = models  # [(tag, model, color), ...]
        self._predict_kw = predict_kw
        self._conf = conf
        self._lock = threading.Lock()
        self._pending = None
        self._result = None  # (seq, [(tag, color, boxes)], ms, submitted_at)
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
            drawn = [(tag, color,
                      model.predict(frame, conf=conf, **self._predict_kw)[0].boxes)
                     for tag, model, color in self._models]
            ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self._seq += 1
                self._result = (self._seq, drawn, ms, submitted_at)

    def stop(self) -> None:
        self._stop.set()


def _load(YOLO, weights: str, backend: str, imgsz: int):
    """Load weights, exporting to OpenVINO on first use for that imgsz.

    OpenVINO bakes the input size into the compiled model, so each imgsz needs
    its own export directory — reusing a 640 export at 960 silently runs at 640.
    Measured on this i3-1215U: 222 ms -> 29 ms at 640, with detections matching
    the PyTorch model on 24/25 val images (max confidence delta 0.018).
    """
    if backend == "torch":
        return YOLO(weights)

    # Ultralytics detects the format from the "_openvino_model" suffix, so the
    # size tag has to go before it, not after.
    stem = os.path.splitext(weights)[0]
    ov_dir = f"{stem}_{imgsz}_openvino_model"
    if not os.path.isdir(ov_dir):
        print(f"[EXPORT] building OpenVINO model for imgsz={imgsz} (one time)…")
        produced = YOLO(weights).export(format="openvino", imgsz=imgsz, half=False)
        os.rename(str(produced), ov_dir)
        print(f"[EXPORT] {ov_dir}")
    return YOLO(ov_dir, task="detect")


def _draw_detections(frame, boxes, color, label_prefix, min_marker=26):
    """Draw boxes, enlarging the marker for tiny ones so they stay visible.

    A shuttle at court distance is ~20px wide; a 1px rectangle that small is
    invisible on screen, which makes a working model look broken.
    """
    best = 0.0
    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        conf = float(box.conf[0])
        best = max(best, conf)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if (x2 - x1) < min_marker or (y2 - y1) < min_marker:
            half = min_marker // 2
            cv2.rectangle(frame, (cx - half, cy - half), (cx + half, cy + half),
                          color, 1)
            cv2.line(frame, (cx - half - 6, cy), (cx - half, cy), color, 1)
            cv2.line(frame, (cx + half, cy), (cx + half + 6, cy), color, 1)

        text = f"{label_prefix} {conf:.2f}"
        ty = max(y1 - 6, 14)
        cv2.putText(frame, text, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)
    return best


def _draw_hud(frame, lines, show_bar=None):
    x, y = 10, 22
    pad = 6
    width = max(len(t) for t, _ in lines) * 9 + pad * 2
    height = len(lines) * 20 + pad * 2 + (10 if show_bar is not None else 0)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - pad, y - 16 - pad), (x + width, y + height - 20),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for text, color in lines:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                    1, cv2.LINE_AA)
        y += 20

    if show_bar is not None:
        bar_w = width - pad * 2
        filled = int(bar_w * max(0.0, min(1.0, show_bar)))
        cv2.rectangle(frame, (x, y - 8), (x + bar_w, y), (70, 70, 70), -1)
        if filled:
            cv2.rectangle(frame, (x, y - 8), (x + filled, y), TEST_COLOR, -1)


def _center_crop(frame, side: int):
    """Center NxN window of native pixels, or the frame itself if it is smaller.

    Returned contiguous: OpenVINO wants a contiguous buffer, and a numpy slice
    is a view.
    """
    if side <= 0:
        return frame
    h, w = frame.shape[:2]
    side = min(side, h, w)
    x = (w - side) // 2
    y = (h - side) // 2
    import numpy as np
    return np.ascontiguousarray(frame[y:y + side, x:x + side])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="model under test (default: models/shuttlecock_v3.pt)")
    ap.add_argument("--compare", default=None,
                    help="second weights file to run alongside, e.g. the older "
                         "weights, drawn in orange for a side-by-side read")
    ap.add_argument("--source", default="0",
                    help="camera index (default 0) or a video file path")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence floor. Deliberately below the system's "
                         "0.4 so you can see what it is ALMOST detecting")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="inference size — must match training (640)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--crop", type=int, default=0, metavar="N",
                    help="center-crop an NxN window of NATIVE camera pixels and "
                         "feed that instead of the whole frame. 0 = off. "
                         "Without this, a 1280x720 frame is letterboxed down to "
                         "640x360 and every shuttle is HALVED before the model "
                         "sees it. --crop 640 makes the mapping 1:1, doubling "
                         "apparent shuttle size at the cost of field of view — "
                         "that is what puts 2-3 m into v3's trained 20-35 px "
                         "band. Trades FOV for range; it invents no detail, it "
                         "just stops discarding pixels the sensor already gave.")
    ap.add_argument("--threaded", choices=["auto", "on", "off"], default="auto",
                    help="run capture and inference on their own threads so the "
                         "window redraws at CAMERA speed regardless of how slow "
                         "the model is — raising --imgsz then costs box "
                         "freshness, not fps. auto = on for a camera, off for a "
                         "video file (where dropping frames would skip content)")
    ap.add_argument("--record", default=None, help="write the annotated feed to this mp4")
    ap.add_argument("--backend", choices=["torch", "openvino"], default="openvino",
                    help="openvino is ~7x faster on this Intel CPU (29ms vs 222ms "
                         "at 640) and numerically equivalent; torch is the fallback")
    args = ap.parse_args()

    if not os.path.isfile(args.weights):
        print(f"[ERROR] weights not found: {args.weights}", file=sys.stderr)
        print("        train them first: python scripts/train_shuttlecock_v2.py",
              file=sys.stderr)
        return 1
    if args.compare and not os.path.isfile(args.compare):
        print(f"[ERROR] --compare weights not found: {args.compare}", file=sys.stderr)
        return 1

    from ultralytics import YOLO  # imported late: it takes a few seconds

    print(f"[LOAD] {args.weights}  backend={args.backend}")
    model = _load(YOLO, args.weights, args.backend, args.imgsz)
    print(f"[LOAD] classes: {model.names}")

    baseline = None
    if args.compare:
        print(f"[LOAD] baseline {args.compare}")
        baseline = _load(YOLO, args.compare, args.backend, args.imgsz)

    # OpenVINO rejects an explicit torch device; torch wants one pinned.
    test_tag = os.path.splitext(os.path.basename(args.weights))[0]
    base_tag = (os.path.splitext(os.path.basename(args.compare))[0]
                if args.compare else "")

    predict_kw = {"imgsz": args.imgsz, "verbose": False}
    if args.backend == "torch":
        predict_kw["device"] = "cpu"

    source = _resolve_source(args.source)
    cap = _open_capture(source, args.width, args.height)
    if cap is None:
        print(f"[ERROR] could not open source {args.source!r}", file=sys.stderr)
        if isinstance(source, int):
            print("        try --source 1 (external cam), and close any other "
                  "app using the webcam (Teams, Zoom, Camera).", file=sys.stderr)
        return 1

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CAM ] {args.source} opened at {actual_w}x{actual_h}")

    # What the model actually receives, after cropping and after ultralytics
    # letterboxes to --imgsz. The scale is the whole ballgame for a small
    # object: at 0.5 a 40 px shuttle arrives as 20 px.
    feed_w = min(args.crop, actual_w) if args.crop > 0 else actual_w
    feed_h = min(args.crop, actual_h) if args.crop > 0 else actual_h
    scale = args.imgsz / float(max(feed_w, feed_h))
    print(f"[FEED] {feed_w}x{feed_h} -> imgsz {args.imgsz}   "
          f"scale {scale:.2f}x  (camera px x {scale:.2f} = px the model sees)")
    if scale < 0.9 and args.crop <= 0:
        print(f"[WARN] the frame is being shrunk {1/scale:.1f}x before inference, "
              f"so every shuttle is {1/scale:.1f}x smaller than the camera "
              f"resolved it. Try --crop {args.imgsz} for a 1:1 mapping.")

    # OpenVINO compiles the graph on the first inference (~300ms+). Burn that
    # here so it doesn't land in the ms/frame average and read as "slow".
    ok, warm = cap.read()
    if ok:
        warm = _center_crop(warm, args.crop)
        for _ in range(3):
            model.predict(warm, conf=args.conf, **predict_kw)
        print("[WARM] backend compiled")

    if args.crop > 0:
        print(f"[HINT] cropped 1:1, so v3's trained 20-35 px band now sits at "
              f"roughly 2-4 m. Field of view is the {feed_w}x{feed_h} centre of "
              f"the frame — keep the shuttle near the middle.")
    else:
        print("[HINT] v3 was trained on court-distance footage, so hold the "
              "shuttle ~1-3 m away, NOT close to the lens. Measured on held-out "
              "clips: ~67% at 20-35px, ~29% below 20px. A close-up is out of "
              "distribution and may detect worse than a distant one.")
    print("[KEYS] q quit | s snapshot | +/- conf | h hud | SPACE pause | r reset")

    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, 15.0, (feed_w, feed_h))
        print(f"[REC ] writing to {args.record}")

    threaded = (args.threaded == "on"
                or (args.threaded == "auto" and isinstance(source, int)))
    models = [(test_tag, model, TEST_COLOR)]
    if baseline is not None:
        models.append((base_tag, baseline, BASE_COLOR))

    camera = detector = None
    if threaded:
        camera = _Camera(cap, args.crop)
        detector = _Detector(models, predict_kw, args.conf)
        camera.start()
        detector.start()
        print("[MODE] threaded: display runs at camera speed, inference runs as "
              "fast as it can. fps is now independent of --imgsz.")
    else:
        print("[MODE] synchronous: every frame is inferred, so fps falls as "
              "--imgsz rises. Correct for files; use --threaded on for a camera.")

    conf = args.conf
    show_hud = True
    paused = False
    frames = 0          # inferences, not displayed frames — the honest denominator
    hits = 0
    best_conf_seen = 0.0
    infer_ms = deque(maxlen=30)
    display_ts = deque(maxlen=30)
    recent_hits = deque(maxlen=60)  # rolling window ~ last 60 inferences
    last_frame = None
    last_seq = 0
    n = 0
    best = 0.0
    age_ms = 0.0

    try:
        while True:
            if threaded:
                frame, ended = camera.read()
                if ended and frame is None:
                    print("[END ] source produced no more frames")
                    break
                if frame is None:
                    time.sleep(0.005)   # camera has not delivered frame 1 yet
                    continue
                if not paused:
                    last_frame = frame
                    detector.submit(frame, conf)
                frame = last_frame.copy()
            else:
                if not paused:
                    ok, frame = cap.read()
                    if not ok:
                        print("[END ] source produced no more frames")
                        break
                    # Crop before anything else so detections, HUD and the
                    # recording all share one coordinate space.
                    frame = _center_crop(frame, args.crop)
                    last_frame = frame
                else:
                    if last_frame is None:
                        break
                    frame = last_frame.copy()

                t0 = time.perf_counter()
                detections = [(tag, color,
                               m.predict(frame, conf=conf, **predict_kw)[0].boxes)
                              for tag, m, color in models]
                detector_result = (last_seq + 1, detections,
                                   (time.perf_counter() - t0) * 1000.0, t0)

            if threaded:
                detector_result = detector.latest()

            display_ts.append(time.perf_counter())

            if detector_result is not None:
                seq, detections, ms, submitted_at = detector_result
                age_ms = (time.perf_counter() - submitted_at) * 1000.0

                # Boxes are drawn every displayed frame, but a result is only
                # SCORED once — otherwise a slow model's single detection would
                # be counted by every frame that reused it.
                fresh = seq != last_seq
                last_seq = seq

                best = 0.0
                for i, (tag, color, boxes) in enumerate(detections):
                    hit_conf = _draw_detections(frame, boxes, color, tag)
                    if i == 0:
                        best = hit_conf
                        n = len(boxes)

                if fresh and not paused:
                    infer_ms.append(ms)
                    frames += 1
                    recent_hits.append(1 if n else 0)
                    if n:
                        hits += 1
                    best_conf_seen = max(best_conf_seen, best)

            if show_hud:
                mean_ms = sum(infer_ms) / len(infer_ms) if infer_ms else 0.0
                rolling = (sum(recent_hits) / len(recent_hits)) if recent_hits else 0.0
                overall = (hits / frames) if frames else 0.0
                span = (display_ts[-1] - display_ts[0]) if len(display_ts) > 1 else 0.0
                display_fps = (len(display_ts) - 1) / span if span > 0 else 0.0
                lines = [
                    (f"{os.path.basename(args.weights)}   conf>={conf:.2f}   "
                     f"imgsz={args.imgsz}   {args.backend}", (255, 255, 255)),
                    (f"display {display_fps:5.1f} fps   |   infer "
                     f"{mean_ms:6.1f} ms ({1000.0 / mean_ms:4.1f} fps)"
                     if mean_ms else f"display {display_fps:5.1f} fps   |   infer --",
                     (200, 200, 200)),
                    (f"box age {age_ms:5.0f} ms" if threaded else "sync mode",
                     (200, 200, 200)),
                    (f"detections this frame: {n}   best {best:.2f}",
                     TEST_COLOR if n else (140, 140, 140)),
                    (f"hit rate  last60 {rolling * 100:5.1f}%   "
                     f"overall {overall * 100:5.1f}%  ({hits}/{frames})",
                     (255, 255, 255)),
                    (f"best conf seen: {best_conf_seen:.2f}", (200, 200, 200)),
                ]
                if baseline is not None:
                    lines.append((f"green = {test_tag}   orange = {base_tag}",
                                  BASE_COLOR))
                if paused:
                    lines.append(("PAUSED", (0, 200, 255)))
                _draw_hud(frame, lines, show_bar=rolling)

            if writer is not None:
                writer.write(frame)

            cv2.imshow(f"shuttlecock live test - {test_tag}", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                os.makedirs(SNAPSHOT_DIR, exist_ok=True)
                path = os.path.join(SNAPSHOT_DIR, f"snap_{int(time.time())}.jpg")
                cv2.imwrite(path, frame)
                print(f"[SNAP] {path}")
            elif key in (ord("+"), ord("=")):
                conf = min(0.95, conf + 0.05)
                print(f"[CONF] {conf:.2f}")
            elif key in (ord("-"), ord("_")):
                conf = max(0.05, conf - 0.05)
                print(f"[CONF] {conf:.2f}")
            elif key == ord("h"):
                show_hud = not show_hud
            elif key == ord(" "):
                paused = not paused
            elif key == ord("r"):
                frames = hits = 0
                best_conf_seen = 0.0
                infer_ms.clear()
                recent_hits.clear()
                print("[STAT] reset")
    finally:
        if detector is not None:
            detector.stop()
        if camera is not None:
            camera.stop()
            camera.join(timeout=1.0)  # it may be blocked in read(); daemon covers that
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print()
    print("=" * 58)
    print(f"  weights        {args.weights}")
    print(f"  inferences     {frames}")
    if frames:
        print(f"  with a det     {hits}  ({hits / frames * 100:.1f}%)")
    if infer_ms:
        mean_ms = sum(infer_ms) / len(infer_ms)
        print(f"  speed          {mean_ms:.1f} ms/frame ({1000.0 / mean_ms:.1f} fps)")
    print(f"  best conf      {best_conf_seen:.2f}")
    print("=" * 58)
    print("  Read this against the drill system's SHUTTLE_CONFIDENCE = 0.4:")
    print("  detections below that would be discarded in the real pipeline.")
    print("  A low hit rate here is the honest signal, not the val mAP.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
