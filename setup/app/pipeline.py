"""One worker per camera slot: capture, detect, annotate, publish a JPEG.

WHY CAPTURE AND INFERENCE RUN ON SEPARATE THREADS
Locked together, the display can never be faster than the model: the flying
shuttle at imgsz 1280 costs ~106 ms, so the feed was stuck at ~9 fps alone and
~1.2 fps with two cameras competing. Split, the capture thread publishes every
frame at the source's own rate while the inference thread works on the newest
frame it can get and overwrites its result when done. Display rate becomes the
video rate; the model rate only decides how stale the boxes are, which the HUD
reports as `age`. This is the same design live_detect.py uses for the camera.

WHY OPENVINO, AND WHY A CROP
Measured on this i3-1215U:
    pose   torch full frame 1280   172 ms   3.5 people/frame
    pose   torch full frame  640    83 ms   1.6   <- downscaling loses people
    pose   openvino + court crop    30 ms   1.9   <- same pixels, third the area
    shuttle torch  1280            320 ms
    shuttle openvino 1280          106 ms
Downscaling the whole frame destroys small-object detection - a 128 px player
becomes 64 px and the detector stops finding them. Cropping keeps them at native
scale and throws away only ceiling and spectators, which is also why it fixes
"too many people": the extras were spectators and adjacent courts.

WHY MODELS ARE PER WORKER
Ultralytics predict is not documented as thread-safe, and a shared model behind
two threads fails as occasional wrong boxes rather than as a crash. These nets
are 3-6 MB.

WHY MODELS ARE FIXED PER SLOT
A camera's job is decided by where it is mounted, not picked in the UI: the
front camera watches the rally (flying shuttle + player pose), the three
sideline/baseline cameras only need to call where the shuttle landed, and the
face camera is raw because its frames feed face enrollment rather than a
detector. See `SLOT_MODELS`.
"""
from __future__ import annotations

import os
import platform
import threading
import time

import cv2
import numpy as np

from app import sources
from app.landings import LandingCounter
from config import settings

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# model key -> config. `crop` is a fraction of the frame (x1,y1,x2,y2) rather
# than pixels, so one setting survives a change of camera resolution.
MODELS: dict[str, dict] = {
    "none": {},
    "shuttle": {
        "weights": "models/shuttle_clear_badminton_p2.pt",
        "imgsz": 1280, "task": "detect", "conf": 0.25, "max_side": 60,
        "label": "flying shuttle",
    },
    "landed": {
        "weights": "models/shuttle_lines_stock_n.pt",
        "imgsz": 1280, "task": "detect", "conf": 0.40, "max_side": 90,
        "label": "landed shuttle",
    },
    "pose": {
        "weights": "models/yolov8n-pose.pt",
        "imgsz": 640, "task": "pose", "conf": 0.25,
        # Lower-middle of the frame: the near court. Excludes the ceiling and
        # the spectator tables along the top, which is where the extra people
        # were coming from.
        "crop": (0.10, 0.35, 0.92, 1.00),
        "top_n": 2,              # a singles rally; raise to 4 for doubles
        "label": "player pose",
    },
}

# Rig v2: what each slot runs is fixed by where the camera is, not picked in
# the UI. front watches the rally (flying shuttle + the player's pose); the
# sideline/baseline views only call where the shuttle landed. face is raw: its
# frames feed face enrollment, not a detector.
SLOT_MODELS: dict[str, list[str]] = {
    "front": ["shuttle", "pose"],
    "left": ["landed"],
    "right": ["landed"],
    "back": ["landed"],
    "face": [],
}


def models_for(camera_id: str, raw: bool = False) -> list[str]:
    if camera_id not in SLOT_MODELS:
        raise ValueError(f"unknown camera {camera_id!r}")
    return [] if raw else list(SLOT_MODELS[camera_id])


def heads_for_tick(n_heads: int, tick: int, alternate: bool) -> list[int]:
    """Which heads run on this inference tick. Alternating halves the per-frame
    cost on the Pi at the price of each model seeing every other frame."""
    if n_heads <= 1 or not alternate:
        return list(range(n_heads))
    return [tick % n_heads]


BOX_COLOR = (0, 255, 255)
HUD_COLOR = (255, 255, 255)


def _resolve(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(_ROOT, rel)


def model_catalog(backend: str | None = None) -> list[dict]:
    """Models available to run, with the imgsz that will actually be used.

    `backend` defaults to DEFAULT_BACKEND (the backend resolved for this
    machine) and applies the exact same PI_IMGSZ override condition
    Worker.__init__ uses, so the reported imgsz cannot drift from what
    actually runs.
    """
    backend = backend or DEFAULT_BACKEND
    out = []
    for key, cfg in MODELS.items():
        if key == "none":
            out.append({"key": "none", "label": "no model", "available": True})
            continue
        imgsz = cfg["imgsz"]
        if backend == "ncnn" and key in settings.PI_IMGSZ:
            imgsz = settings.PI_IMGSZ[key]
        out.append({"key": key, "label": cfg["label"], "imgsz": imgsz,
                    "available": os.path.exists(_resolve(cfg["weights"]))})
    return out


# Inference backend by CPU architecture. OpenVINO is Intel-oriented and has no
# useful Pi story; NCNN is the ultralytics-recommended aarch64 export. Unknown
# architectures fall toward NCNN because it is the portable one.
_X86 = ("AMD64", "x86_64")


def resolve_backend(machine: str) -> str:
    return "openvino" if machine in _X86 else "ncnn"


DEFAULT_BACKEND = resolve_backend(platform.machine())


def _export_dir(path: str, imgsz: int, backend: str) -> str:
    """Per-(size, backend) export cache path.

    Both backends bake the input size in, so each imgsz needs its own
    directory - reusing a 640 export at 1280 silently runs at 640. The
    backend is in the name too so an OpenVINO tree is never loaded as NCNN.
    """
    return f"{os.path.splitext(path)[0]}_{imgsz}_{backend}_model"


def _load(weights: str, imgsz: int, task: str, backend: str):
    """Load, exporting for `backend` on first use for that imgsz."""
    from ultralytics import YOLO
    path = _resolve(weights)
    if not os.path.exists(path):
        raise FileNotFoundError(f"weights missing: {weights}")
    if backend == "torch":
        return YOLO(path)
    out_dir = _export_dir(path, imgsz, backend)
    if not os.path.isdir(out_dir):
        produced = YOLO(path).export(format=backend, imgsz=imgsz, half=False)
        os.rename(str(produced), out_dir)
    return YOLO(out_dir, task=task)


class _Head:
    """One model on one camera: its config, weights, tracker state, last result."""

    def __init__(self, key: str, backend: str, calib: dict | None):
        self.key = key
        self.cfg = dict(MODELS[key])
        if backend == "ncnn" and key in settings.PI_IMGSZ:
            self.cfg["imgsz"] = settings.PI_IMGSZ[key]
        self.counter = (LandingCounter(calib["lines"])
                        if key == "landed" and calib
                        and calib.get("mode") == "lines" else None)
        self.model = None
        self.result = None            # (res, box, produced_at)

    def crop_box(self, w: int, h: int):
        c = self.cfg.get("crop")
        if not c:
            return None
        return (int(c[0] * w), int(c[1] * h), int(c[2] * w), int(c[3] * h))


class Worker:
    """Capture thread + inference thread + the newest annotated JPEG."""

    def __init__(self, camera_id: str, model_keys: list[str],
                 backend: str | None = None, target_fps: float = 30.0):
        self.camera_id = camera_id
        self.backend = backend or DEFAULT_BACKEND
        self.target_fps = target_fps
        # Calibration is read once at start: it changes when an operator saves
        # it, and restarting the worker is how that takes effect.
        self.calib = _load_calibration(camera_id)
        self.heads = [_Head(k, self.backend, self.calib) for k in model_keys]
        self.model_key = "+".join(model_keys) or "none"
        self.label = " + ".join(h.cfg["label"] for h in self.heads) or "raw"

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame = None            # newest raw frame, for inference + freezing
        self._frame_at = 0.0
        self._jpeg: bytes | None = None
        self._stats = {"frames": 0, "detections": 0, "fps": 0.0,
                       "infer_fps": 0.0, "infer_ms": 0.0, "age_ms": 0.0,
                       "error": None, "loading": bool(self.heads),
                       "calibrated": bool(self.calib),
                       "landings": 0, "counts": {}, "last_call": None,
                       "active": None, "seeding": False}
        self._threads: list[threading.Thread] = []

    # ── lifecycle ────────────────────────────────────────────
    def start(self):
        self._threads = [
            threading.Thread(target=self._capture_loop, daemon=True,
                             name=f"cap-{self.camera_id}"),
            threading.Thread(target=self._infer_loop, daemon=True,
                             name=f"inf-{self.camera_id}"),
        ]
        for t in self._threads:
            t.start()

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3.0)

    @property
    def alive(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    @property
    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats, camera_id=self.camera_id,
                        model=self.model_key, backend=self.backend,
                        running=self.alive)

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def latest_frame(self):
        """Newest raw frame (a copy). Calibration freezes and face enrollment
        read this so nothing opens a device this worker already holds."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    # ── inference thread ─────────────────────────────────────
    def _infer_loop(self):
        if not self.heads:
            with self._lock:
                self._stats["loading"] = False
            return
        try:
            for h in self.heads:
                h.model = _load(h.cfg["weights"], h.cfg["imgsz"],
                                h.cfg.get("task", "detect"), self.backend)
        except Exception as exc:
            with self._lock:
                self._stats["error"] = str(exc)
                self._stats["loading"] = False
            return
        with self._lock:
            self._stats["loading"] = False

        window: list[float] = []
        tick = 0
        alternate = self.camera_id == "front" and settings.FRONT_ALTERNATE
        while not self._stop.is_set():
            with self._lock:
                frame = None if self._frame is None else self._frame.copy()
            if frame is None:
                time.sleep(0.01)
                continue

            t0 = time.perf_counter()
            for i in heads_for_tick(len(self.heads), tick, alternate):
                h = self.heads[i]
                box = h.crop_box(frame.shape[1], frame.shape[0])
                src = frame[box[1]:box[3], box[0]:box[2]] if box else frame
                try:
                    if h.counter is not None:
                        # Track, not predict: stable ids are what make "is this
                        # a new shuttle" answerable.
                        res = h.model.track(src, imgsz=h.cfg["imgsz"],
                                            conf=h.cfg.get("conf", 0.25),
                                            persist=True, tracker="bytetrack.yaml",
                                            verbose=False)[0]
                    else:
                        res = h.model.predict(src, imgsz=h.cfg["imgsz"],
                                              conf=h.cfg.get("conf", 0.25),
                                              classes=[0] if h.cfg.get("task") == "pose" else None,
                                              verbose=False)[0]
                except Exception as exc:
                    with self._lock:
                        self._stats["error"] = str(exc)
                    return
                with self._lock:
                    h.result = (res, box, time.perf_counter())
            tick += 1
            ms = (time.perf_counter() - t0) * 1000.0
            window.append(ms)
            if len(window) > 20:
                window.pop(0)
            with self._lock:
                self._stats["infer_ms"] = round(ms, 1)
                self._stats["infer_fps"] = round(1000.0 / (sum(window) / len(window)), 1)

    # ── capture thread ───────────────────────────────────────
    def _draw(self, frame, head, res, box):
        """Draw the last inference onto the current frame."""
        if res is None:
            return 0
        ox, oy = (box[0], box[1]) if box else (0, 0)

        order = sorted(range(len(res.boxes)),
                       key=lambda i: -float((res.boxes.xyxy[i][2] - res.boxes.xyxy[i][0]) *
                                            (res.boxes.xyxy[i][3] - res.boxes.xyxy[i][1])))
        top_n = head.cfg.get("top_n")
        if top_n:
            order = order[:top_n]

        if head.cfg.get("task") == "pose":
            # Skeleton lines from COCO's 17-keypoint topology.
            links = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
                     (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
            kp = res.keypoints
            for i in order:
                x1, y1, x2, y2 = (int(v) for v in res.boxes.xyxy[i].tolist())
                cv2.rectangle(frame, (x1 + ox, y1 + oy), (x2 + ox, y2 + oy), (255, 128, 0), 2)
                if kp is None:
                    continue
                pts = kp.xy[i].tolist()
                cf = kp.conf[i].tolist() if kp.conf is not None else [1.0] * len(pts)
                for a, b in links:
                    if a < len(pts) and b < len(pts) and cf[a] > 0.3 and cf[b] > 0.3:
                        cv2.line(frame, (int(pts[a][0]) + ox, int(pts[a][1]) + oy),
                                 (int(pts[b][0]) + ox, int(pts[b][1]) + oy), (0, 255, 120), 2)
                for (px, py), c in zip(pts, cf):
                    if c > 0.3:
                        cv2.circle(frame, (int(px) + ox, int(py) + oy), 3, (0, 220, 255), -1)
            return len(order)

        n = 0
        max_side = head.cfg.get("max_side")
        for i in order:
            x1, y1, x2, y2 = (int(v) for v in res.boxes.xyxy[i].tolist())
            if max_side and max(x2 - x1, y2 - y1) > max_side:
                continue
            n += 1
            cv2.rectangle(frame, (x1 + ox, y1 + oy), (x2 + ox, y2 + oy), BOX_COLOR, 2)
            cx, cy = (x1 + x2) // 2 + ox, (y1 + y2) // 2 + oy
            if (x2 - x1) < 26 or (y2 - y1) < 26:
                # A 13 px box is invisible on screen; the marker keeps a working
                # model from looking broken.
                cv2.rectangle(frame, (cx - 13, cy - 13), (cx + 13, cy + 13), BOX_COLOR, 1)
            cv2.putText(frame, f"{float(res.boxes.conf[i]):.2f}",
                        (x1 + ox, max(y1 + oy - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, BOX_COLOR, 1, cv2.LINE_AA)
        return n

    def _capture_loop(self):
        cap = None
        frames = 0
        last_started: float | None = None
        window: list[float] = []
        min_dt = 1.0 / self.target_fps if self.target_fps else 0.0

        try:
            while not self._stop.is_set():
                started = time.perf_counter()
                if cap is None or not cap.isOpened():
                    try:
                        cap = sources.open_capture(self.camera_id)
                    except Exception as exc:
                        with self._lock:
                            self._stats["error"] = str(exc)
                        return
                    if not cap.isOpened():
                        with self._lock:
                            self._stats["error"] = "could not open source"
                        return

                ok, frame = cap.read()
                if not ok:
                    if sources.is_file(self.camera_id):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop demo footage
                        continue
                    with self._lock:
                        self._stats["error"] = "capture returned no frame"
                    return

                with self._lock:
                    self._frame = frame          # hand the newest to inference
                    self._frame_at = started
                    results = [(h, h.result) for h in self.heads]

                view = frame.copy()
                age_ms = 0.0
                n = 0
                for h, res in results:
                    if res is None:
                        continue
                    age_ms = max(age_ms, (time.perf_counter() - res[2]) * 1000.0)
                    if h.counter is not None:
                        # Update first, then draw: the state of each shuttle
                        # (active / counted / pending) is what gets drawn.
                        tracks = _landing_tracks(res[0], res[1], h.cfg.get("max_side"))
                        for ev in h.counter.update(tracks, frames):
                            with self._lock:
                                self._stats["last_call"] = ev
                        n += _draw_landings(view, tracks, h.counter)
                    else:
                        n += self._draw(view, h, res[0], res[1])

                for h in self.heads:
                    box = h.crop_box(view.shape[1], view.shape[0])
                    if box:
                        cv2.rectangle(view, (box[0], box[1]), (box[2], box[3]), (90, 90, 90), 1)

                # Raw workers (calibration, face) stay clean: no overlay at all.
                if self.heads:
                    _draw_calibration(view, self.calib)
                counter = next((h.counter for h in self.heads if h.counter), None)
                if counter is not None:
                    _draw_counts(view, counter.counts, counter.active, counter.seeding)

                frames += 1
                # Measured across the WHOLE cycle including the pacing sleep
                # below, via the previous iteration's start. Timing only the
                # work reports the rate the loop could run at, not the rate it
                # actually publishes, which overstated this by 4x.
                if last_started is not None:
                    window.append(started - last_started)
                    if len(window) > 30:
                        window.pop(0)
                last_started = started
                fps = len(window) / sum(window) if sum(window) else 0.0

                with self._lock:
                    infer_ms = self._stats["infer_ms"]
                if self.heads:
                    cv2.putText(view,
                                f"{self.camera_id}  {self.label}  "
                                f"{fps:4.1f} fps  det={n}  age={age_ms:.0f}ms  inf={infer_ms:.0f}ms",
                                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, HUD_COLOR, 2, cv2.LINE_AA)

                ok_enc, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok_enc:
                    with self._lock:
                        self._jpeg = buf.tobytes()
                        self._stats.update(frames=frames, detections=n,
                                           fps=round(fps, 1),
                                           age_ms=round(age_ms, 1), error=None)
                        if counter is not None:
                            self._stats["landings"] = counter.total
                            self._stats["counts"] = dict(counter.counts)
                            self._stats["active"] = counter.active
                            self._stats["seeding"] = counter.seeding

                slack = min_dt - (time.perf_counter() - started)
                if slack > 0:
                    time.sleep(slack)
        finally:
            if cap is not None:
                cap.release()



def _load_calibration(camera_id: str) -> dict | None:
    """This camera's saved calibration, or None. Mongo being down is not fatal:
    the feed is still worth watching without in/out calls."""
    try:
        from app import db
        return db.calibrations().find_one({"_id": camera_id})
    except Exception:
        return None


def _draw_calibration(frame, calib) -> None:
    if not calib:
        return
    h, w = frame.shape[:2]
    if calib.get("mode") == "lines":
        colors = {"inside": (120, 196, 53), "outside": (59, 59, 255)}  # BGR
        for role, fit in (calib.get("lines") or {}).items():
            a, b = fit["a"], fit["b"]
            cv2.line(frame, (0, int(b)), (w - 1, int(a * (w - 1) + b)),
                     colors.get(role, (200, 200, 200)), 2, cv2.LINE_AA)
            cv2.putText(frame, role, (8, max(int(b) - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colors.get(role, (200, 200, 200)), 1, cv2.LINE_AA)
    elif calib.get("corners"):
        pts = np.asarray(calib["corners"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, (0, 122, 255), 2, cv2.LINE_AA)


def _landing_tracks(res, box, max_side):
    """[(track_id, (x, y))] in full-frame coordinates, one per surviving box.

    The point is the box's bottom-centre, not its centroid: the centre floats
    half a shuttle above the floor, which at this scale is centimetres of
    line-call error.
    """
    ox, oy = (box[0], box[1]) if box else (0, 0)
    ids = res.boxes.id
    out = []
    for i in range(len(res.boxes)):
        x1, y1, x2, y2 = res.boxes.xyxy[i].tolist()
        if max_side and max(x2 - x1, y2 - y1) > max_side:
            continue
        tid = int(ids[i]) if ids is not None else None
        out.append((tid, ((x1 + x2) / 2.0 + ox, y2 + oy),
                    (int(x1 + ox), int(y1 + oy), int(x2 + ox), int(y2 + oy))))
    return out


ACTIVE_COLOR = (0, 255, 0)       # the shuttle that just landed
PENDING_COLOR = (0, 200, 255)    # a new track, not yet counted


def _draw_landings(frame, tracks, counter) -> int:
    """Box the shuttle that just landed. Counted ones are not drawn at all.

    Marking every shuttle on the floor made the display unreadable once dozens
    had accumulated, and none of them can score again anyway. The only mark that
    carries information is the current one.
    """
    a = counter.active or {}
    for tid, _point, box in tracks:
        state = counter.state_of(tid)
        if state in ("counted", "active"):
            continue          # counted ones vanish; the active one is drawn below
        if not counter.seeding:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), PENDING_COLOR, 1)

    # The active box comes from the registry, not from this frame. The detector
    # misses shuttles between inferences (det=0 is common), and a box that
    # blinks out is useless for judging a call.
    if not a or not a.get("box"):
        return 0
    x1, y1, x2, y2 = a["box"]
    pad = 10
    cv2.rectangle(frame, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), ACTIVE_COLOR, 2)
    call = "  ".join(f"{c}:{a[c]}" for c in ("green", "red") if c in a)
    cv2.putText(frame, f"#{a.get('id','')}  {call}", (x1 - pad, max(y1 - pad - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACTIVE_COLOR, 2, cv2.LINE_AA)
    return 1


def _draw_counts(frame, counts, active=None, seeding=False) -> None:
    rows = [("green in", counts.get("green_inside", 0), (120, 196, 53)),
            ("green out", counts.get("green_outside", 0), (120, 196, 53)),
            ("red in", counts.get("red_inside", 0), (59, 59, 255)),
            ("red out", counts.get("red_outside", 0), (59, 59, 255))]
    y = 56
    for label, n, color in rows:
        cv2.putText(frame, f"{label}: {n}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 24
    if seeding:
        cv2.putText(frame, "seeding: shuttles already on court", (10, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    elif active:
        call = "  ".join(f"{c}:{active[c]}" for c in ("green", "red") if c in active)
        cv2.putText(frame, f"active #{active['id']}  {call}", (10, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACTIVE_COLOR, 2, cv2.LINE_AA)


_workers: dict[str, Worker] = {}
_wlock = threading.Lock()


def start(camera_id: str, raw: bool = False, backend: str | None = None,
          target_fps: float = 30.0) -> dict:
    keys = models_for(camera_id, raw)
    stop(camera_id)
    w = Worker(camera_id, keys, backend, target_fps)
    with _wlock:
        _workers[camera_id] = w
    w.start()
    # Long enough to surface an immediate failure (missing weights, dead device)
    # but not long enough to block the request on a first-time model export.
    time.sleep(0.5)
    return w.stats


def stop(camera_id: str) -> None:
    with _wlock:
        w = _workers.pop(camera_id, None)
    if w:
        w.stop()


def stop_all() -> None:
    for cid in list(_workers):
        stop(cid)


def get(camera_id: str) -> Worker | None:
    with _wlock:
        return _workers.get(camera_id)


def all_stats() -> list[dict]:
    with _wlock:
        ws = list(_workers.values())
    return [w.stats for w in ws]
