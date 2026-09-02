"""One worker thread per camera slot: read, detect, annotate, publish a JPEG.

WHY A WORKER RATHER THAN DETECTING ON REQUEST
Two cameras must run at once now and four eventually, and inference is the
expensive part. A worker per slot lets each camera advance at whatever rate its
model allows while the browser pulls the newest frame whenever it likes, so a
slow camera cannot stall a fast one or the page.

THE SPEED BUDGET IS REAL
Measured on this i3-1215U, per frame: flying-shuttle p2 at imgsz 1280 costs
~106 ms, landed-shuttle ~84 ms, pose cropped to the court with OpenVINO ~30 ms.
Two cameras on shuttle models is therefore ~5 fps each, not 30. `infer_every`
exists for that: run the model on every Nth frame and keep drawing the last
boxes in between, which holds the displayed frame rate up. Players barely move
between frames, so this costs almost nothing for pose; a smashed shuttle moves
a long way, so raising it there loses real detections.

MODELS ARE LOADED PER WORKER, NOT SHARED
Ultralytics predict is not documented as thread-safe, and a shared model behind
two threads is the kind of bug that shows up as occasional wrong boxes rather
than as a crash. Memory is cheap enough here: these are 3-6 MB nets.
"""
from __future__ import annotations

import os
import threading
import time

import cv2

from app import sources

# model key -> (weights, imgsz, task)
MODELS: dict[str, dict] = {
    "none": {},
    "shuttle": {
        "weights": "runs/clear_badminton/p2-native/weights/best.pt",
        "imgsz": 1280, "task": "detect", "conf": 0.25, "max_side": 60,
        "label": "flying shuttle",
    },
    "landed": {
        "weights": "runs/shuttle_lines/stock-n/weights/best.pt",
        "imgsz": 1280, "task": "detect", "conf": 0.40, "max_side": 90,
        "label": "landed shuttle",
    },
    "pose": {
        "weights": "yolov8n-pose.pt",
        "imgsz": 640, "task": "pose", "conf": 0.25,
        "label": "player pose",
    },
}

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOX_COLOR = (0, 255, 255)
HUD_COLOR = (255, 255, 255)


def _resolve(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(_ROOT, rel)


def model_catalog() -> list[dict]:
    out = []
    for key, cfg in MODELS.items():
        if key == "none":
            out.append({"key": "none", "label": "no model", "available": True})
            continue
        out.append({"key": key, "label": cfg["label"], "imgsz": cfg["imgsz"],
                    "available": os.path.exists(_resolve(cfg["weights"]))})
    return out


class Worker(threading.Thread):
    """Reads one slot, annotates, and keeps only the newest JPEG."""

    def __init__(self, camera_id: str, model_key: str = "none",
                 infer_every: int = 1, target_fps: float = 15.0):
        super().__init__(daemon=True, name=f"cam-{camera_id}")
        self.camera_id = camera_id
        self.model_key = model_key
        self.infer_every = max(1, int(infer_every))
        self.target_fps = target_fps
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._stats = {"frames": 0, "detections": 0, "ms": 0.0,
                       "fps": 0.0, "error": None}

    # ── lifecycle ────────────────────────────────────────────
    def stop(self):
        self._stop.set()

    @property
    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats, camera_id=self.camera_id,
                        model=self.model_key, running=self.is_alive())

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    # ── internals ────────────────────────────────────────────
    def _load_model(self):
        cfg = MODELS.get(self.model_key) or {}
        if not cfg:
            return None, {}
        path = _resolve(cfg["weights"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"weights missing: {cfg['weights']}")
        from ultralytics import YOLO  # imported late so the API starts fast
        return YOLO(path), cfg

    def _annotate(self, frame, result, cfg):
        """Draw and count. Pose gets ultralytics' skeleton; detect gets boxes."""
        if result is None:
            return frame, 0
        if cfg.get("task") == "pose":
            return result.plot(), len(result.boxes)

        n = 0
        max_side = cfg.get("max_side")
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            if max_side and max(x2 - x1, y2 - y1) > max_side:
                continue  # the large-box false positives, same filter as the CLI
            n += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if (x2 - x1) < 26 or (y2 - y1) < 26:
                # A 13 px box is invisible on screen, which makes a working
                # model look broken.
                cv2.rectangle(frame, (cx - 13, cy - 13), (cx + 13, cy + 13),
                              BOX_COLOR, 1)
            cv2.putText(frame, f"{float(box.conf[0]):.2f}", (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, BOX_COLOR, 1, cv2.LINE_AA)
        return frame, n

    def run(self):
        try:
            model, cfg = self._load_model()
        except Exception as exc:
            with self._lock:
                self._stats["error"] = str(exc)
            return

        cap = None
        result = None
        frame_no = 0
        window = []
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
                    # A file ran out: loop it, so a demo source keeps playing.
                    if sources.is_file(self.camera_id):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    with self._lock:
                        self._stats["error"] = "capture returned no frame"
                    return

                if model is not None and frame_no % self.infer_every == 0:
                    result = model.predict(frame, imgsz=cfg["imgsz"],
                                           conf=cfg.get("conf", 0.25),
                                           verbose=False)[0]
                view, n = self._annotate(frame, result, cfg)

                frame_no += 1
                dt = time.perf_counter() - started
                window.append(dt)
                if len(window) > 30:
                    window.pop(0)
                fps = len(window) / sum(window) if sum(window) else 0.0

                cv2.putText(view, f"{self.camera_id}  {cfg.get('label','raw')}  "
                                  f"{fps:4.1f} fps  det={n}",
                            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            HUD_COLOR, 2, cv2.LINE_AA)

                ok_enc, buf = cv2.imencode(".jpg", view,
                                           [cv2.IMWRITE_JPEG_QUALITY, 72])
                if ok_enc:
                    with self._lock:
                        self._jpeg = buf.tobytes()
                        self._stats.update(frames=frame_no, detections=n,
                                           ms=round(dt * 1000, 1),
                                           fps=round(fps, 1), error=None)

                # Never outrun the target: a file source would otherwise burn
                # CPU racing through frames nobody is looking at.
                slack = min_dt - (time.perf_counter() - started)
                if slack > 0:
                    time.sleep(slack)
        finally:
            if cap is not None:
                cap.release()


_workers: dict[str, Worker] = {}
_wlock = threading.Lock()


def start(camera_id: str, model_key: str = "none", infer_every: int = 1,
          target_fps: float = 15.0) -> dict:
    if model_key not in MODELS:
        raise ValueError(f"unknown model {model_key!r}")
    stop(camera_id)
    w = Worker(camera_id, model_key, infer_every, target_fps)
    with _wlock:
        _workers[camera_id] = w
    w.start()
    # Give the worker a moment so an immediate status call reports a real error
    # (missing weights, dead device) rather than an empty "starting".
    time.sleep(0.4)
    return w.stats


def stop(camera_id: str) -> None:
    with _wlock:
        w = _workers.pop(camera_id, None)
    if w:
        w.stop()
        w.join(timeout=3.0)


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
