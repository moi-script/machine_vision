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
sys.path.insert(0, REPO)

# The detector itself lives in utils/ so the drill engine runs the same code;
# this script is the tuning and validation harness around it.
from utils.shuttle_motion import ShuttleMotionDetector  # noqa: E402

COL_DET = (0, 235, 160)    # green  — accepted detection
COL_PRED = (0, 200, 255)   # amber  — Kalman coast (no measurement this frame)
COL_REJ = (80, 80, 200)    # red    — rejected candidate (debug view)
COL_PLAYER = (200, 120, 60)


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
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed for a video file in the window: 1.0 is "
                         "real time, 0.5 slow motion, 0 as fast as the CPU goes")
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
    # Pace a video file to its own fps in the window; the detector runs in a
    # few ms, so unpaced playback is far too fast to follow the shuttle.
    frame_s = 0.0
    if not isinstance(src, int) and args.speed > 0:
        frame_s = 1.0 / ((cap.get(cv2.CAP_PROP_FPS) or 30.0) * args.speed)
    t_next = time.perf_counter()

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

            wait_ms = 1
            if frame_s and not paused:
                t_next += frame_s
                wait_ms = max(1, int((t_next - time.perf_counter()) * 1000))
            else:
                t_next = time.perf_counter()   # no catch-up burst after a pause
            k = cv2.waitKey(wait_ms) & 0xFF
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
