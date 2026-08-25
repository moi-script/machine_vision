# ============================================================
# shuttle_motion_prototype.py — Fixed-camera shuttle detection by MOTION,
# not by appearance.
#
# WHY THIS EXISTS
# Every YOLO attempt (v1/v2/v3) failed because a single frame does not contain
# enough information to identify a shuttlecock. Measured on held-out data:
# v3 finds a far-field shuttle 8% of the time even when handed the crop
# location, and finds a sharp close-up shuttle 1 time in 221. The published
# ceiling for YOLO on this task is ~53% (TrackNetV3 paper); TrackNet-family
# models reach ~97% by consuming 8 consecutive frames and using MOTION.
#
# TrackNetV3 itself is not an option here: 25 FPS on a dedicated GPU, and this
# machine is a CPU-only i3-1215U where yolov8n-pose already eats ~30 ms of a
# 33 ms frame budget.
#
# But TrackNet is built for BROADCAST video, which cuts, pans and zooms. Our
# camera is bolted down. That makes classical background subtraction both cheap
# and unusually effective — we can use the same motion cue for single-digit
# milliseconds instead of a GPU.
#
# THE PIPELINE
#   1. MOG2 background subtraction        -> foreground mask
#   2. contour filter (area + aspect)     -> shuttle-sized moving blobs
#   3. player-box rejection (optional)    -> drop limbs/rackets
#   4. Kalman constant-velocity filter    -> pick the candidate on the
#                                            trajectory, coast through misses
#
# KNOWN LIMITS — be honest about these in the writeup:
#   * needs the shuttle to be MOVING (fine for a drill; a shuttle resting on
#     the floor is invisible to this method by construction)
#   * degrades when the shuttle crosses a player (that is what the Kalman
#     coast is for)
#   * needs a FIXED camera; any pan/zoom/bump invalidates the background model
#     and the mask floods until MOG2 relearns
#
# USAGE
#   # validate on a recorded drill clip (what you want first)
#   python scripts/shuttle_motion_prototype.py --source drill.mp4
#
#   # same, but reject blobs inside player boxes (slower, more precise)
#   python scripts/shuttle_motion_prototype.py --source drill.mp4 --exclude-players
#
#   # live from the camera
#   python scripts/shuttle_motion_prototype.py --source 0
#
#   # write per-frame results for offline scoring / plots
#   python scripts/shuttle_motion_prototype.py --source drill.mp4 --csv out.csv
#
#   # tuning: show the raw mask and every rejected blob
#   python scripts/shuttle_motion_prototype.py --source drill.mp4 --debug
#
# KEYS  q quit | SPACE pause | m cycle view (normal/mask/both) | s snapshot
# ============================================================

import argparse
import csv
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COL_DET = (0, 235, 160)    # green  — accepted detection
COL_PRED = (0, 200, 255)   # amber  — Kalman coast (no measurement this frame)
COL_REJ = (80, 80, 200)    # red    — rejected candidate (debug view)
COL_PLAYER = (200, 120, 60)


# ---------------------------------------------------------------- Kalman ----

class TrajectoryFilter:
    """Constant-velocity Kalman filter over (x, y).

    A shuttle in flight is close to ballistic over the few frames we care
    about, so constant velocity is enough to (a) gate which blob is plausible
    and (b) coast through frames where the shuttle is occluded or merged into
    a player. Acceleration is absorbed by the process noise.
    """

    def __init__(self, process_noise=8.0, meas_noise=6.0):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_noise
        self.initialised = False
        self.misses = 0

    def predict(self):
        """Predicted (x, y) for this frame, or None if not yet locked on."""
        if not self.initialised:
            return None
        p = self.kf.predict()          # 4x1 column vector
        return float(p[0, 0]), float(p[1, 0])

    def correct(self, x, y):
        if not self.initialised:
            self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.initialised = True
            self.misses = 0
            return
        self.kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))
        self.misses = 0

    def miss(self, max_coast):
        """Count a frame with no measurement; drop the lock if we coast too long."""
        self.misses += 1
        if self.misses > max_coast:
            self.initialised = False
        return self.initialised


# -------------------------------------------------------------- detector ----

class ShuttleMotionDetector:
    def __init__(self, min_area=12, max_area=1200, max_aspect=6.0,
                 gate=140.0, max_coast=8, history=350, var_threshold=28,
                 player_pad=14, scale=0.5, confirm=4, tent_gate=55.0,
                 min_speed=6.0):
        # MOG2 relearns the background continuously, so slow lighting drift is
        # absorbed.
        #
        # detectShadows=False: measured 26.6 -> 23.0 ms at full res on this CPU,
        # and the hard-foreground threshold below discards the shadow label
        # anyway, so we were paying for a result we then threw away.
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False)
        # Subtraction runs on a downscaled frame; everything the caller sees is
        # in FULL-RES coordinates. Measured on this i3-1215U over 400 frames:
        #     full res            23.0 ms
        #     half res (0.5)      ~10.5 ms   <- default
        #     quarter res (0.25)  10.4 ms    (no better: the resize dominates)
        # INTER_AREA beat INTER_NEAREST and a strided [::2,::2] slice, which is
        # cache-hostile despite copying less.
        self.scale = scale
        self.min_area = min_area
        self.max_area = max_area
        self.max_aspect = max_aspect
        self.gate = gate               # px radius around the Kalman prediction
        self.max_coast = max_coast
        self.player_pad = player_pad
        self.track = TrajectoryFilter()
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # --- lock acquisition -------------------------------------------
        # Measured on real footage: ~31 blobs/frame, median area 40 px^2,
        # i.e. camera noise. Picking "the smallest blob in range" therefore
        # locks onto noise almost immediately. A shuttle in flight is not
        # distinguished from noise by SIZE — it is distinguished by tracing a
        # coherent path. So no lock is granted until a candidate has been
        # followed for `confirm` frames along a roughly constant-velocity
        # track that actually goes somewhere (`min_speed`). Noise blobs
        # appear and vanish at random and almost never survive that.
        self.confirm = confirm
        self.tent_gate = tent_gate     # px, tighter than the locked-on gate
        self.min_speed = min_speed     # px/frame; a hovering blob is not a shuttle
        self.tentative = []            # [{"pts": [(x, y)], "last": frame_i}]
        self.frame_i = 0

    def _candidates(self, mask, player_boxes):
        """Shuttle-sized blobs surviving the area / aspect / player filters."""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        kept, rejected = [], []
        inv = 1.0 / self.scale
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            # Contours are found on the downscaled mask. Convert to full-res
            # NOW so every threshold, the player boxes, the gate and the Kalman
            # state all live in one coordinate space — the frame's. Area scales
            # with the square of the linear factor.
            x, y, w, h = int(x * inv), int(y * inv), int(w * inv), int(h * inv)
            area *= inv * inv
            cx, cy = x + w / 2.0, y + h / 2.0
            if not (self.min_area <= area <= self.max_area):
                rejected.append((x, y, w, h, "area"))
                continue
            # Motion blur stretches the shuttle into a streak, so allow a
            # generous aspect ratio — but a very long thin blob is usually an
            # arm, a racket edge or a line artefact.
            ar = max(w, h) / float(max(1, min(w, h)))
            if ar > self.max_aspect:
                rejected.append((x, y, w, h, "aspect"))
                continue
            if self._in_player(cx, cy, player_boxes):
                rejected.append((x, y, w, h, "player"))
                continue
            kept.append((cx, cy, area, (x, y, w, h)))
        return kept, rejected

    def _in_player(self, cx, cy, boxes):
        p = self.player_pad
        for (x1, y1, x2, y2) in boxes:
            if x1 - p <= cx <= x2 + p and y1 - p <= cy <= y2 + p:
                return True
        return False

    def _acquire(self, cands):
        """Grow tentative tracks; return a candidate once one is confirmed.

        A tentative track is extended by whichever candidate best matches its
        constant-velocity prediction. Once a track has `confirm` points and has
        travelled at least `min_speed` px/frame on average, it is promoted and
        its newest point becomes the lock.
        """
        i = self.frame_i
        for t in self.tentative:
            t["matched"] = False

        for c in cands:
            cx, cy = c[0], c[1]
            best_t, best_d = None, None
            for t in self.tentative:
                if t["matched"] or t["last"] != i - 1:
                    continue
                pts = t["pts"]
                if len(pts) >= 2:      # extrapolate constant velocity
                    px = 2 * pts[-1][0] - pts[-2][0]
                    py = 2 * pts[-1][1] - pts[-2][1]
                else:
                    px, py = pts[-1]
                d = np.hypot(cx - px, cy - py)
                if d <= self.tent_gate and (best_d is None or d < best_d):
                    best_t, best_d = t, d
            if best_t is not None:
                best_t["pts"].append((cx, cy))
                best_t["last"] = i
                best_t["matched"] = True
                best_t["cand"] = c
            else:
                self.tentative.append(
                    {"pts": [(cx, cy)], "last": i, "matched": True, "cand": c})

        # Drop anything not extended this frame — a real shuttle is present in
        # consecutive frames, noise is not.
        self.tentative = [t for t in self.tentative if t["last"] >= i - 1]

        for t in self.tentative:
            pts = t["pts"]
            if len(pts) < self.confirm:
                continue
            span = np.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
            if span / max(1, len(pts) - 1) < self.min_speed:
                continue           # drifting or stationary: not a shuttle
            self.tentative = []
            return t["cand"]
        # Bound memory if a frame is full of noise.
        if len(self.tentative) > 120:
            self.tentative = self.tentative[-120:]
        return None

    def update(self, frame, player_boxes=()):
        """-> (result, mask, candidates, rejected)

        result is (x, y, state) where state is "det" (measured this frame),
        "coast" (Kalman prediction only) or None (no lock).
        """
        self.frame_i += 1
        small = (frame if self.scale == 1.0 else
                 cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                            interpolation=cv2.INTER_AREA))
        mask = self.bg.apply(small)
        # Keep only hard foreground (MOG2 would mark shadows 127 if enabled).
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_open)
        # Close small gaps so a blurred streak stays one contour.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close)

        cands, rejected = self._candidates(mask, player_boxes)
        pred = self.track.predict()

        best = None
        if cands:
            if pred is not None:
                # Locked on: take the candidate closest to where the shuttle
                # should be, inside the gate. This is what rejects players'
                # incidental motion once a trajectory exists.
                inside = [(np.hypot(cx - pred[0], cy - pred[1]), c)
                          for c in cands for cx, cy in [(c[0], c[1])]]
                inside = [t for t in inside if t[0] <= self.gate]
                if inside:
                    best = min(inside, key=lambda t: t[0])[1]
            else:
                # Cold start: grow tentative tracks and only lock on once one
                # of them has proved itself over several frames.
                best = self._acquire(cands)

        if best is not None:
            cx, cy, _, bbox = best
            self.track.correct(cx, cy)
            return (cx, cy, "det", bbox), mask, cands, rejected

        if pred is not None and self.track.miss(self.max_coast):
            return (pred[0], pred[1], "coast", None), mask, cands, rejected

        self.track.miss(self.max_coast)
        return None, mask, cands, rejected


# ------------------------------------------------------------ player boxes --

class PlayerBoxes:
    """Cached yolov8n-pose boxes, refreshed every N frames.

    Players move far slower than the shuttle, so re-running pose every frame
    buys nothing and costs the whole frame budget. Refreshing every N frames
    and reusing the boxes in between keeps the exclusion useful for a few ms
    amortised.
    """

    def __init__(self, every=5, imgsz=640):
        from ultralytics import YOLO
        w = os.path.join(REPO, "yolov8n-pose_openvino_model")
        if not os.path.isdir(w):
            w = os.path.join(REPO, "yolov8n-pose.pt")
        self.model = YOLO(w, task="pose")
        self.every = every
        self.imgsz = imgsz
        self.i = 0
        self.boxes = []

    def get(self, frame):
        if self.i % self.every == 0:
            r = self.model.predict(frame, imgsz=self.imgsz, conf=0.35,
                                   verbose=False)[0]
            self.boxes = ([tuple(b) for b in r.boxes.xyxy.tolist()]
                          if r.boxes is not None else [])
        self.i += 1
        return self.boxes


# ------------------------------------------------------------------ main ----

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="video file, or camera index")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--min-area", type=float, default=12)
    ap.add_argument("--max-area", type=float, default=1200)
    ap.add_argument("--max-aspect", type=float, default=6.0)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="run background subtraction at this fraction of the "
                         "frame. 0.5 measured ~10.5ms vs ~23ms at 1.0 on this "
                         "CPU. Detections are always returned in FULL-RES "
                         "coords, and --min/max-area are full-res too, so "
                         "changing this does not change the other thresholds. "
                         "Lower than 0.5 does not help: the resize dominates.")
    ap.add_argument("--gate", type=float, default=140.0,
                    help="px radius around the Kalman prediction to accept a blob")
    ap.add_argument("--max-coast", type=int, default=8,
                    help="frames to coast on prediction alone before dropping lock")
    ap.add_argument("--confirm", type=int, default=4,
                    help="consecutive frames a candidate must trace a coherent "
                         "path before it is allowed to become the lock. Raise "
                         "it if the tracker latches onto noise; lower it if it "
                         "is slow to pick up a rally.")
    ap.add_argument("--tent-gate", type=float, default=55.0,
                    help="px gate while a track is still unconfirmed")
    ap.add_argument("--min-speed", type=float, default=6.0,
                    help="px/frame a confirmed track must average. A shuttle in "
                         "flight moves; noise and slow drift do not.")
    ap.add_argument("--warmup", type=int, default=30,
                    help="frames to let MOG2 learn the background before scoring")
    ap.add_argument("--exclude-players", action="store_true",
                    help="reject blobs inside yolov8n-pose boxes (costs ~30ms/N frames)")
    ap.add_argument("--pose-every", type=int, default=5)
    ap.add_argument("--csv", default=None, help="write per-frame results here")
    ap.add_argument("--record", default=None,
                    help="write the annotated feed to this mp4 so the tracking "
                         "can be reviewed frame by frame")
    ap.add_argument("--debug", action="store_true", help="draw rejected blobs")
    ap.add_argument("--no-display", action="store_true", help="headless benchmark")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = (cv2.VideoCapture(src, cv2.CAP_DSHOW) if isinstance(src, int)
           else cv2.VideoCapture(src))
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"[ERROR] could not open source {args.source!r}", file=sys.stderr)
        return 1
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[SRC ] {args.source} at {W}x{H}")

    det = ShuttleMotionDetector(
        min_area=args.min_area, max_area=args.max_area,
        max_aspect=args.max_aspect, gate=args.gate, max_coast=args.max_coast,
        scale=args.scale, confirm=args.confirm, tent_gate=args.tent_gate,
        min_speed=args.min_speed)

    players = None
    if args.exclude_players:
        print("[POSE] loading yolov8n-pose for player exclusion…")
        players = PlayerBoxes(every=args.pose_every)

    vid = None
    if args.record:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        vid = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                              src_fps, (W, H))
        print(f"[REC ] {args.record} at {src_fps:.2f} fps")

    writer = None
    if args.csv:
        writer = csv.writer(open(args.csv, "w", newline=""))
        writer.writerow(["frame", "x", "y", "state", "n_candidates", "ms"])

    print("[KEYS] q quit | SPACE pause | m view | s snapshot")
    view = 0
    paused = False
    n = det_n = coast_n = 0
    ms = deque(maxlen=60)
    ncand = deque(maxlen=60)
    trail = deque(maxlen=25)   # recent path, so a trajectory is visible at a glance
    t_start = time.time()

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    print("[END ] source exhausted")
                    break

                boxes = players.get(frame) if players else ()
                t0 = time.perf_counter()
                res, mask, cands, rej = det.update(frame, boxes)
                dt = (time.perf_counter() - t0) * 1000.0

                n += 1
                if n > args.warmup:
                    ms.append(dt)
                    ncand.append(len(cands))
                    if res and res[2] == "det":
                        det_n += 1
                    elif res and res[2] == "coast":
                        coast_n += 1
                if writer:
                    writer.writerow([n,
                                     f"{res[0]:.1f}" if res else "",
                                     f"{res[1]:.1f}" if res else "",
                                     res[2] if res else "none",
                                     len(cands), f"{dt:.2f}"])

            if args.no_display and vid is None:
                continue

            vis = frame.copy()
            for (x1, y1, x2, y2) in (boxes or ()):
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                              COL_PLAYER, 1)
            if args.debug:
                for (x, y, w, h, why) in rej:
                    cv2.rectangle(vis, (x, y), (x + w, y + h), COL_REJ, 1)
            for cx, cy, area, (x, y, w, h) in cands:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (160, 160, 160), 1)
            if res:
                c = COL_DET if res[2] == "det" else COL_PRED
                rx, ry = int(res[0]), int(res[1])
                if res[3] is not None:
                    # Measured: box the actual moving blob, padded so a small
                    # shuttle is still visible when scrubbing the video.
                    bx, by, bw, bh = res[3]
                    p = 6
                    cv2.rectangle(vis, (bx - p, by - p), (bx + bw + p, by + bh + p),
                                  c, 2)
                    cv2.putText(vis, f"shuttle {bw}x{bh}", (bx - p, by - p - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
                else:
                    # Coasting: no blob this frame, so box the Kalman estimate
                    # dashed-style (thin) to distinguish it from a real measurement.
                    cv2.rectangle(vis, (rx - 22, ry - 22), (rx + 22, ry + 22), c, 1)
                    cv2.putText(vis, "coast", (rx - 22, ry - 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
                trail.append((rx, ry, res[2]))
            for i in range(1, len(trail)):
                if trail[i - 1] is None or trail[i] is None:
                    continue
                cv2.line(vis, trail[i - 1][:2], trail[i][:2],
                         COL_DET if trail[i][2] == "det" else COL_PRED, 1)

            scored = max(1, n - args.warmup)
            hud = (f"{det_n/scored:5.1%} det  {coast_n/scored:4.1%} coast   "
                   f"{np.mean(ms) if ms else 0:5.1f} ms   "
                   f"cand {np.mean(ncand) if ncand else 0:4.1f}")
            cv2.rectangle(vis, (0, 0), (W, 26), (0, 0, 0), -1)
            cv2.putText(vis, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1)

            if view in (1, 2):
                # The mask lives at --scale; bring it back to frame size so it
                # can sit beside the annotated view.
                mvis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                if mvis.shape[:2] != vis.shape[:2]:
                    mvis = cv2.resize(mvis, (vis.shape[1], vis.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            show = mvis if view == 1 else (
                np.hstack([vis, mvis]) if view == 2 else vis)

            if vid is not None:
                vid.write(vis)          # always the annotated normal view
            if args.no_display:
                continue
            cv2.imshow("shuttle motion prototype", show)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                paused = not paused
            if k == ord("m"):
                view = (view + 1) % 3
            if k == ord("s"):
                os.makedirs(os.path.join(REPO, "runs"), exist_ok=True)
                p = os.path.join(REPO, "runs", f"motion_{n:06d}.png")
                cv2.imwrite(p, show)
                print(f"[SNAP] {p}")
    finally:
        cap.release()
        if vid is not None:
            vid.release()
            print(f"[REC ] wrote {args.record}")
        cv2.destroyAllWindows()

    scored = max(1, n - args.warmup)
    wall = time.time() - t_start
    print("\n" + "=" * 58)
    print(f"  frames scored     {scored}  (first {args.warmup} = MOG2 warmup)")
    print(f"  measured          {det_n}  ({det_n/scored:.1%})")
    print(f"  coasted (Kalman)  {coast_n}  ({coast_n/scored:.1%})")
    print(f"  either            {(det_n+coast_n)/scored:.1%}   <- trajectory coverage")
    print(f"  detector cost     {np.mean(ms) if ms else 0:.2f} ms/frame "
          f"(budget is 33 ms at 30 fps; yolov8n-pose already uses ~30)")
    print(f"  mean candidates   {np.mean(ncand) if ncand else 0:.1f} per frame")
    print(f"  wall clock        {wall:.1f}s for {n} frames "
          f"({n/max(wall,1e-6):.1f} fps end to end)")
    print("=" * 58)
    print("  'either' is the number that matters: the drill needs a continuous")
    print("  trajectory, not a hit on every single frame. Coasting through a")
    print("  few frames is fine; long coasts mean the gate or area filter is")
    print("  wrong. Re-run with --debug to see what is being rejected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
