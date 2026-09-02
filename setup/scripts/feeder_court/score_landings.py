# ============================================================
# score_landings.py — Count NEW shuttle landings, ignoring the ones already on
# the floor.
#
# THE PROBLEM THIS SOLVES
# Measured over lines.mp4: 1 detection at frame 300, 210 by frame 500, 5140 by
# frame 1140. A feeder session buries the court in shuttles, so "how many
# shuttles does this frame contain" answers nothing about scoring. What matters
# is which shuttle just arrived.
#
# HOW A LANDING IS DECIDED
# Every detection is matched against a registry of shuttles already seen. A
# match inside MATCH_RADIUS is old and is ignored forever after. An unmatched
# detection is only a candidate: it must hold still in roughly the same place
# for --confirm-frames consecutive frames before it counts. That gate exists
# because the detector's test precision is 0.752 — about one detection in four
# on unseen frames is spurious, and a single-frame flicker would otherwise
# score a point.
#
# WHY THIS IS THE WEAKER OF TWO SIGNALS
# The authoritative landing point is where the AIRBORNE track ends, not where a
# floor blob appears: a trajectory ending at a coordinate is unambiguous even
# with 500 shuttles down. This script is the confirmation half of that pair,
# and works standalone until the tracker exists.
#
# CALIBRATION
# Without --court this reports pixels only. With a calibration JSON it maps each
# landing through the court homography and calls in or out. Use the bottom-centre
# of the box, not the centroid: the box centre floats above the floor by half the
# shuttle's height, which at this scale is centimetres of error on a line call.
#
# Usage:
#   python scripts/feeder_court/score_landings.py --show
#   python scripts/feeder_court/score_landings.py --show --court datasets/clear_badminton/court_calib.json
#
# Keys:  q / ESC  quit     SPACE  pause     r  clear the registry (new rally)
# ============================================================

import argparse
import json
import os
import time

import cv2
import numpy as np

DEFAULT_VIDEO = "datasets/vid_source/clear_badminton_dataset_for_collab/lines.mp4"
DEFAULT_WEIGHTS = "runs/shuttle_lines/stock-n/weights/best.pt"
WINDOW = "feeder_court landings"

NEW_COLOR = (0, 255, 0)       # a landing scored on this frame
PENDING_COLOR = (0, 200, 255)  # seen, not yet confirmed
OLD_COLOR = (110, 110, 110)   # retired: already counted, never counted again
HUD_COLOR = (255, 255, 255)
OUT_COLOR = (60, 60, 255)

# Boxes ran 17-67 px in the training data; 90 leaves headroom without admitting
# the large false positives that plagued the flying-shuttle model.
MAX_SIDE_PX = 90

# Official court, in metres, measured from the left baseline corner.
COURT_LEN = 13.40
DOUBLES_W = 6.10
SINGLES_W = 5.18


class Registry:
    """Shuttles already on the floor, plus candidates not yet confirmed.

    Retired entries are kept, not deleted: forgetting one would let the same
    shuttle score a second time the moment a detection flickered.
    """

    def __init__(self, match_radius: float, confirm_frames: int):
        self.match_radius = match_radius
        self.confirm_frames = confirm_frames
        self.retired: list[dict] = []   # confirmed landings, counted once
        self.pending: list[dict] = []   # candidates building up evidence
        self.next_id = 1

    def _nearest(self, entries, point):
        best, best_d = None, self.match_radius
        for e in entries:
            d = float(np.hypot(e["point"][0] - point[0], e["point"][1] - point[1]))
            if d <= best_d:
                best, best_d = e, d
        return best

    def update(self, points: list[tuple[float, float]], frame_idx: int) -> list[dict]:
        """Classify this frame's detections; return newly confirmed landings."""
        confirmed = []
        seen_pending = set()

        for point in points:
            if self._nearest(self.retired, point) is not None:
                continue  # already counted; nothing to do

            cand = self._nearest(self.pending, point)
            if cand is None:
                self.pending.append({"point": point, "hits": 1, "last": frame_idx})
                continue

            cand["hits"] += 1
            cand["last"] = frame_idx
            # Track the latest position: a settling shuttle drifts a little.
            cand["point"] = point
            seen_pending.add(id(cand))

            if cand["hits"] >= self.confirm_frames:
                entry = {"id": self.next_id, "point": point, "frame": frame_idx}
                self.next_id += 1
                self.retired.append(entry)
                self.pending.remove(cand)
                confirmed.append(entry)

        # A candidate that stops being detected was noise, not a landing.
        self.pending = [c for c in self.pending
                        if frame_idx - c["last"] <= self.confirm_frames]
        return confirmed

    def clear(self) -> None:
        self.retired.clear()
        self.pending.clear()


def load_bands(path: str):
    """Fit one straight line per sketch colour; the pair forms an in/out band.

    This is the weaker half of calibration and deliberately so. Two parallel
    traced lines fix the court's across direction but carry no scale and no
    origin, so "between the lines" is answerable and "3 cm past the baseline"
    is not. It needs two traced lines instead of four known points, which is
    what actually exists right now.

    Every stroke of a colour is pooled before fitting rather than fitted
    separately: retracing the same line twice is a hand-jitter sample, and a
    least-squares fit through all of it beats either trace alone.
    """
    with open(path) as fh:
        data = json.load(fh)
    pooled: dict[str, list] = {}
    for s in data.get("strokes", []):
        if s.get("mode") != "line" or len(s.get("points", [])) < 2:
            continue  # stray single clicks are not a line
        pooled.setdefault(s["color"], []).extend(s["points"])

    fits = {}
    for color, pts in pooled.items():
        arr = np.asarray(pts, dtype=float)
        if len(arr) < 2 or np.ptp(arr[:, 0]) < 5:
            continue  # near-vertical or too short to fit as y = ax + b
        a, b = np.polyfit(arr[:, 0], arr[:, 1], 1)
        fits[color] = (float(a), float(b), len(arr))
    if len(fits) < 2:
        raise SystemExit(f"--lines needs at least 2 traced lines, found {sorted(fits)}")
    return fits


def band_call(fits, point) -> str:
    """IN when the landing sits between the outermost fitted lines."""
    x, y = point
    ys = sorted(a * x + b for a, b, _ in fits.values())
    return "IN" if ys[0] <= y <= ys[-1] else "OUT"


def draw_bands(frame, fits, width: int) -> None:
    palette = {"red": (0, 0, 255), "green": (0, 255, 0),
               "blue": (255, 128, 0), "yellow": (0, 255, 255)}
    for color, (a, b, _) in fits.items():
        p1 = (0, int(round(b)))
        p2 = (width - 1, int(round(a * (width - 1) + b)))
        cv2.line(frame, p1, p2, palette.get(color, (200, 200, 200)), 1, cv2.LINE_AA)


def load_court(path: str):
    """Homography from image pixels to court metres, plus the in/out bounds."""
    with open(path) as fh:
        cfg = json.load(fh)
    src = np.array(cfg["image_points"], dtype=np.float32)
    dst = np.array(cfg["court_points"], dtype=np.float32)
    if len(src) < 4 or len(src) != len(dst):
        raise SystemExit("--court needs at least 4 matching image_points and court_points")
    H, _ = cv2.findHomography(src, dst)
    if H is None:
        raise SystemExit("could not solve the homography — are the points collinear?")
    return H, cfg.get("singles", False)


def to_court(H, point) -> tuple[float, float]:
    p = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    x, y = cv2.perspectiveTransform(p, H)[0][0]
    return float(x), float(y)


def call_in_out(xy, singles: bool) -> str:
    half_w = (SINGLES_W if singles else DOUBLES_W) / 2.0
    x, y = xy
    return "IN" if (0.0 <= y <= COURT_LEN and abs(x) <= half_w) else "OUT"


def main() -> None:
    ap = argparse.ArgumentParser(description="Count new shuttle landings, ignoring old ones.")
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.40,
                    help="higher than the 0.25 default: mean confidence is 0.582 and "
                         "a weak false positive here becomes a wrong point")
    ap.add_argument("--max-det", type=int, default=100, dest="max_det")
    ap.add_argument("--max-side", type=int, default=MAX_SIDE_PX, dest="max_side")
    ap.add_argument("--match-radius", type=float, default=25.0, dest="match_radius",
                    help="px: a detection this close to a known shuttle is that shuttle")
    ap.add_argument("--confirm-frames", type=int, default=4, dest="confirm_frames",
                    help="consecutive frames a new blob must persist before it scores")
    ap.add_argument("--court", default=None, help="4-point calibration JSON; enables metre-accurate calls")
    ap.add_argument("--lines", default=None,
                    help="sketch_frame.py JSON; two traced lines give a between-the-lines "
                         "in/out call without scale")
    ap.add_argument("--backend", choices=["torch", "openvino"], default="openvino")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--out", default=None, help="write the annotated video here")
    ap.add_argument("--events", default=None, help="write the landing log to this JSON")
    ap.add_argument("--fps", type=float, default=None)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(root)
    if not os.path.exists(args.video):
        raise SystemExit(f"missing video: {args.video}")

    fits = None
    if args.lines:
        fits = load_bands(args.lines)
        for c, (a, b, n) in fits.items():
            print(f"  line {c:<7} y = {a:+.5f}x {b:+.2f}   ({n} traced points)")

    H, singles = (None, False)
    if args.court:
        H, singles = load_court(args.court)
        print(f"calibrated from {args.court}  ({'singles' if singles else 'doubles'} lines)")
    elif fits is None:
        print("no --court or --lines given: reporting pixels only, no in/out calls")

    from ultralytics import YOLO  # imported late so --help stays instant
    if args.backend == "openvino":
        stem = os.path.splitext(args.weights)[0]
        ov = f"{stem}_{args.imgsz}_openvino_model"
        if not os.path.isdir(ov):
            print(f"[EXPORT] OpenVINO for imgsz={args.imgsz}...")
            produced = YOLO(args.weights).export(format="openvino", imgsz=args.imgsz, half=False)
            os.rename(str(produced), ov)
        model = YOLO(ov, task="detect")
    else:
        model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))

    reg = Registry(args.match_radius, args.confirm_frames)
    events: list[dict] = []
    flash: dict[int, int] = {}   # entry id -> frame it was confirmed, for the highlight
    frames = 0
    t0 = time.perf_counter()

    while True:
        started = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                               max_det=args.max_det, verbose=False)[0]

        points, boxes = [], []
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            if args.max_side and max(x2 - x1, y2 - y1) > args.max_side:
                continue
            # Bottom-centre: where the shuttle meets the floor. The box centre
            # sits half a shuttle above it, which is centimetres of line-call error.
            points.append(((x1 + x2) / 2.0, float(y2)))
            boxes.append((x1, y1, x2, y2))

        confirmed = reg.update(points, frames)
        for entry in confirmed:
            flash[entry["id"]] = frames
            record = {"id": entry["id"], "frame": frames,
                      "px": [round(entry["point"][0], 1), round(entry["point"][1], 1)]}
            if H is not None:
                xy = to_court(H, entry["point"])
                record["court_m"] = [round(xy[0], 3), round(xy[1], 3)]
                record["call"] = call_in_out(xy, singles)
            elif fits is not None:
                record["call"] = band_call(fits, entry["point"])
            events.append(record)
            tail = f"  {record.get('call','')} {record.get('court_m','')}".rstrip() if record.get("call") else ""
            print(f"  LANDING #{entry['id']} at frame {frames} px={record['px']}{tail}", flush=True)

        if fits is not None:
            draw_bands(frame, fits, w)

        # Draw: retired grey, pending amber, just-scored green for half a second.
        for (x1, y1, x2, y2), point in zip(boxes, points):
            known = reg._nearest(reg.retired, point)
            color = OLD_COLOR
            label = ""
            if known is not None:
                recent = flash.get(known["id"])
                if recent is not None and frames - recent < 15:
                    color, label = NEW_COLOR, f"#{known['id']}"
            else:
                color = PENDING_COLOR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if label:
                cv2.putText(frame, label, (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        frames += 1
        last = events[-1] if events else None
        hud2 = f"last: #{last['id']} {last.get('call','')}" if last else "last: -"
        cv2.putText(frame, f"f{frames}/{total}  landings={len(reg.retired)}  "
                           f"pending={len(reg.pending)}  det={len(boxes)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HUD_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, hud2, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    OUT_COLOR if last and last.get("call") == "OUT" else HUD_COLOR,
                    2, cv2.LINE_AA)

        if writer is not None:
            writer.write(frame)

        if args.show:
            cv2.imshow(WINDOW, frame)
            wait_ms = 1
            if args.fps:
                spent = (time.perf_counter() - started) * 1000.0
                wait_ms = max(int(1000.0 / args.fps - spent), 1)
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                reg.clear()
                print(f"  registry cleared at frame {frames}", flush=True)
            if key == ord(" "):
                while True:
                    k2 = cv2.waitKey(50) & 0xFF
                    if k2 in (ord(" "), ord("q"), 27):
                        break
                if k2 in (ord("q"), 27):
                    break

    elapsed = time.perf_counter() - t0
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\nframes {frames}  landings {len(reg.retired)}  "
          f"{1000 * elapsed / frames if frames else 0:.1f} ms/frame")
    if events and any("call" in e for e in events):
        calls = {}
        for e in events:
            calls[e["call"]] = calls.get(e["call"], 0) + 1
        print(f"calls: {calls}")
    if args.events:
        os.makedirs(os.path.dirname(args.events) or ".", exist_ok=True)
        with open(args.events, "w") as fh:
            json.dump(events, fh, indent=2)
        print(f"  -> {args.events}")
    if args.out:
        print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
