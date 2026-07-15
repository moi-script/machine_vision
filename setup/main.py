# ============================================================
# main.py — Badminton Feeder Trainer — Full System
#
# Setup:
#   pip install ultralytics opencv-python
#
# Run:
#   python main.py --difficulty medium
#   python main.py --difficulty hard --shots 20
#
# Controls (while running):
#   Q     = quit
#   E     = easy difficulty
#   M     = medium difficulty
#   H     = hard difficulty
#   SPACE = pause / resume
# ============================================================

import cv2
import time
import random
import argparse
import numpy as np
from collections import deque
from ultralytics import YOLO

from config.settings import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    PLAYER_ZONES, DIFFICULTY,
    PERSON_CONFIDENCE, SHUTTLE_CONFIDENCE,
    RETURN_CONFIRM_FRAMES, GRAYSCALE,
    COURT_W, COURT_L, VIDEO_SOURCE,
)

# Draw+log every YOLO-detected person and why it's kept/dropped. Flip to True
# when a person isn't getting a box and you need to see why.
DEBUG_DETECT = False

# Video/camera source: file path or webcam index. Overridden by --source.
video_source = VIDEO_SOURCE if VIDEO_SOURCE is not None else CAMERA_INDEX
from utils.zones import (
    get_ankle_position, get_zone_from_position,
    get_player_in_zone, get_shuttle_side,
    to_court, in_court_bounds, crossed_net, build_homography,
)
from utils.scoring  import PlayerScores
from utils.shuttle_worker import ShuttleWorker
from utils.display  import (
    draw_court_zone, draw_net, draw_zones,
    draw_player, draw_shuttle, draw_scoreboard, draw_status, draw_fps
)


# ── Models ──────────────────────────────────────────────────
import os
from config.settings import SHUTTLE_MODEL_PATH, SHUTTLE_SOURCE

# Pre-trained: detects players (no Roboflow needed)
player_model  = YOLO("yolov8n-pose.pt")

# Shuttlecock detection source (see SHUTTLE_SOURCE in config/settings.py):
#   "local"      → local Roboflow-trained YOLO11s weights (models/shuttlecock.pt)
#   "serverless" → Roboflow direct model over HTTP (free, adds latency)
#   "off"        → disabled
shuttle_model       = None
SHUTTLE_MODEL_READY = False
_serverless_warned  = False

if SHUTTLE_SOURCE == "local":
    if os.path.isfile(SHUTTLE_MODEL_PATH):
        try:
            shuttle_model = YOLO(SHUTTLE_MODEL_PATH)
            SHUTTLE_MODEL_READY = True
            print(f"[MODEL] Shuttlecock model loaded: {SHUTTLE_MODEL_PATH}")
        except Exception as exc:  # noqa: BLE001 - keep the drill running
            print(f"[MODEL] Failed to load {SHUTTLE_MODEL_PATH}: {exc}")
    else:
        print(f"[MODEL] SHUTTLE_SOURCE='local' but no file at {SHUTTLE_MODEL_PATH} "
              "— shuttle detection disabled.")
elif SHUTTLE_SOURCE == "serverless":
    from utils.roboflow_client import run_shuttlecock_model, extract_shuttle_xy
    from config.settings import SHUTTLE_CONFIDENCE
    print("[MODEL] Shuttle detection via Roboflow serverless model "
          "(free; runs on a background thread so the drill loop stays smooth, "
          "but shuttle position lags ~1s behind live).")
else:
    print("[MODEL] Shuttle detection off (player/zone logic only).")


# ── State ────────────────────────────────────────────────────
scores              = PlayerScores()
prev_shuttle_cy     = None
shuttle_crossed     = False
return_side_count   = 0        # consecutive frames shuttle seen on return side
active_target_id    = None     # player ID currently being scored
active_target_zone  = None     # zone shuttle entered
shot_start_time     = None
drill_paused        = False
current_difficulty  = "medium"
total_shots_fired   = 0
max_shots           = 0        # 0 = unlimited


# ── Camera setup ─────────────────────────────────────────────
def open_camera():
    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # OV9281 — force grayscale mode if needed (webcam only; ignored for files)
    if GRAYSCALE and isinstance(video_source, int):
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video source {video_source!r}. "
            "For a webcam try changing CAMERA_INDEX; for a file check the path "
            "in VIDEO_SOURCE (config/settings.py) or --source."
        )
    src_kind = "webcam index" if isinstance(video_source, int) else "video file"
    print(f"[SOURCE] Using {src_kind}: {video_source}")
    return cap


# ── Shuttle detection ────────────────────────────────────────
def get_shuttle_position(frame, results_cache=None):
    """
    Detect the shuttlecock in a frame and return (cx, cy) of the most
    confident detection, or None if detection is off / nothing is found.

    Source is chosen by SHUTTLE_SOURCE in config/settings.py:
      "local"      → local YOLO11s weights (fast, offline)
      "serverless" → Roboflow direct model over HTTP (free, adds latency)
    """
    global _serverless_warned

    if SHUTTLE_SOURCE == "serverless":
        try:
            result = run_shuttlecock_model(
                frame, confidence=int(SHUTTLE_CONFIDENCE * 100))
        except Exception as exc:  # noqa: BLE001 - never kill the drill loop
            if not _serverless_warned:
                print(f"[MODEL] Serverless shuttle detection error: {exc}")
                _serverless_warned = True
            return None
        return extract_shuttle_xy(result)  # (x, y) centre, or None

    if SHUTTLE_MODEL_READY:  # local weights
        shuttle_results = shuttle_model(frame, conf=SHUTTLE_CONFIDENCE, verbose=False)
        boxes = shuttle_results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        best_i = int(boxes.conf.argmax())
        x1, y1, x2, y2 = boxes.xyxy[best_i].tolist()
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    return None  # "off" or no local weights


# ── Player detection ─────────────────────────────────────────
def detect_players(frame):
    """
    Run YOLOv8 pose model on frame.

    Returns (player_positions, detections):
      player_positions : { track_id: (court_x, court_y) } — only players with a
                         confident ankle that maps inside the court. These are
                         the ones eligible for zone matching and scoring.
      detections       : list of every detected person for drawing, each tagged
                         "in_court": True/False. People without a court ankle
                         (e.g. feet out of frame) are still drawn, just dimmed.
    """
    results          = player_model.track(frame, persist=True,
                                          classes=[0],
                                          conf=PERSON_CONFIDENCE,
                                          verbose=False)
    player_positions = {}
    detections       = []   # for drawing

    if results[0].boxes.id is None:
        return player_positions, detections

    track_ids      = results[0].boxes.id.int().tolist()
    boxes          = results[0].boxes.xyxy
    keypoints_data = results[0].keypoints.data

    for i, tid in enumerate(track_ids):
        box = boxes[i].tolist()

        kps   = keypoints_data[i].tolist()
        ankle = get_ankle_position(keypoints_data[i])

        # A player is scorable only when we have a confident ankle that projects
        # onto the trainee's half-court. Everyone else is still drawn (dimmed),
        # so an upper-body-only detection stays visible — it just isn't tracked.
        in_court = False
        court_xy = None
        if ankle:
            court_xy = to_court(ankle)
            if in_court_bounds(*court_xy):
                in_court = True
                scores.init_player(tid)
                player_positions[tid] = court_xy

        if DEBUG_DETECT:
            if in_court:
                print(f"[DEBUG] P{tid}: SCORABLE — court "
                      f"({court_xy[0]:.0f},{court_xy[1]:.0f})")
            elif ankle is None:
                print(f"[DEBUG] P{tid}: seen, not scorable — no confident ankle keypoint")
            else:
                print(f"[DEBUG] P{tid}: seen, not scorable — ankle "
                      f"{tuple(round(a) for a in ankle)} -> court "
                      f"({court_xy[0]:.0f},{court_xy[1]:.0f}) OUTSIDE "
                      f"[0..{COURT_W:.0f}]x[0..{COURT_L:.0f}]")

        detections.append({
            "id"       : tid,
            "box"      : box,
            "keypoints": kps,
            "in_court" : in_court,
        })

    return player_positions, detections


# ── Net crossing detection ────────────────────────────────────
def check_net_crossing(shuttle_cy):
    """
    Detects when the shuttle crosses from feeder side -> player side in court
    space. Returns True on the frame the crossing is first confirmed.
    """
    global prev_shuttle_cy, shuttle_crossed

    if prev_shuttle_cy is None:
        prev_shuttle_cy = shuttle_cy
        return False

    just_crossed = crossed_net(prev_shuttle_cy, shuttle_cy)
    prev_shuttle_cy = shuttle_cy

    if just_crossed and not shuttle_crossed:
        shuttle_crossed = True
        return True
    return False


# ── Return detection ─────────────────────────────────────────
def check_return(shuttle_cy):
    """
    Detects when the shuttle crosses back player side -> feeder side. Requires
    RETURN_CONFIRM_FRAMES consecutive feeder-side frames to avoid flicker.
    """
    global return_side_count

    if get_shuttle_side(shuttle_cy) == "feeder_side":
        return_side_count += 1
    else:
        return_side_count = 0

    return return_side_count >= RETURN_CONFIRM_FRAMES


# ── Fire feeder ──────────────────────────────────────────────
def fire_feeder(zone_name):
    """
    Trigger your physical feeder machine here.
    Currently just prints — wire this to GPIO or serial
    when you move to Raspberry Pi.
    """
    print(f"[FEEDER] Firing to zone: {zone_name}")
    # GPIO example for Raspberry Pi later:
    # import RPi.GPIO as GPIO
    # GPIO.output(FEEDER_PIN, GPIO.HIGH)
    # time.sleep(0.1)
    # GPIO.output(FEEDER_PIN, GPIO.LOW)


# ── Shot logic ───────────────────────────────────────────────
def start_new_shot(player_positions):
    """
    Pick a random zone, find who is there, fire feeder.
    Returns (target_player_id, target_zone) or (None, None) if no player found.
    """
    global active_target_id, active_target_zone
    global shot_start_time, shuttle_crossed, return_side_count
    global prev_shuttle_cy, total_shots_fired

    # Pick a random zone
    zone_name = random.choice(list(PLAYER_ZONES.keys()))

    # Find who is standing there RIGHT NOW
    target_id = get_player_in_zone(zone_name, player_positions)

    if target_id is None:
        print(f"[SHOT] Zone {zone_name} is empty — skipping")
        return None, None

    # Record the shot
    scores.record_shot(target_id, zone_name)
    total_shots_fired  += 1

    # Reset crossing state
    shuttle_crossed     = False
    return_side_count   = 0
    prev_shuttle_cy     = None
    shot_start_time     = time.time()
    active_target_id    = target_id
    active_target_zone  = zone_name

    fire_feeder(zone_name)
    print(f"[SHOT] Fired to {zone_name} → Player {target_id} is target")

    return target_id, zone_name


def reset_shot():
    """Reset all shot state after a shot completes (hit or miss)"""
    global active_target_id, active_target_zone
    global shot_start_time, shuttle_crossed, return_side_count

    active_target_id   = None
    active_target_zone = None
    shot_start_time    = None
    shuttle_crossed    = False
    return_side_count  = 0


# ── Main loop ────────────────────────────────────────────────
def run(difficulty="medium", shots=0):
    global current_difficulty, max_shots, drill_paused

    current_difficulty = difficulty
    max_shots          = shots
    interval           = DIFFICULTY[difficulty]["interval"]

    try:
        build_homography()
    except ValueError as exc:
        print(f"[CALIBRATION] {exc}")
        return

    cap = open_camera()

    # Shuttle detection is slow (serverless ~1s/frame); run it off the main
    # loop so rendering + player tracking stay smooth. shuttle_pos read from
    # the worker may lag detection latency behind the live frame.
    shuttle_worker = ShuttleWorker(get_shuttle_position)
    shuttle_worker.start()

    print(f"\n[START] Badminton Feeder Trainer")
    print(f"        Difficulty : {difficulty.upper()}")
    print(f"        Interval   : {interval}s per shot")
    print(f"        Max shots  : {'unlimited' if shots == 0 else shots}")
    print(f"        Press Q to quit, SPACE to pause\n")

    last_shot_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed")
            break

        # OV9281 is grayscale — convert to BGR so YOLO and OpenCV work correctly
        if GRAYSCALE and len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Smoothed FPS overlay (top-right, above the scoreboard)
        draw_fps(frame)

        # ── Handle keyboard input ────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            drill_paused = not drill_paused
            print("[PAUSED]" if drill_paused else "[RESUMED]")
        elif key == ord('e'):
            current_difficulty = "easy"
            interval = DIFFICULTY["easy"]["interval"]
            print("[DIFFICULTY] Easy")
        elif key == ord('m'):
            current_difficulty = "medium"
            interval = DIFFICULTY["medium"]["interval"]
            print("[DIFFICULTY] Medium")
        elif key == ord('h'):
            current_difficulty = "hard"
            interval = DIFFICULTY["hard"]["interval"]
            print("[DIFFICULTY] Hard")

        if drill_paused:
            draw_status(frame, "PAUSED — press SPACE to resume", (0, 200, 255))
            cv2.imshow("Badminton Feeder Trainer", frame)
            continue

        # ── Detect players ───────────────────────────────────
        player_positions, detections = detect_players(frame)

        # ── Detect shuttle (async) ───────────────────────────
        # Hand the newest frame to the worker and read its latest result;
        # neither call blocks on the ~1s detection.
        shuttle_worker.submit(frame.copy())
        shuttle_pos = shuttle_worker.get()

        # ── Draw base overlays ───────────────────────────────
        draw_court_zone(frame)
        draw_net(frame)

        weak_zones_all = []
        for pid in scores.scores:
            weak_zones_all.extend(scores.get_weak_zones(pid))

        draw_zones(frame,
                   active_zone=active_target_zone,
                   weak_zones=weak_zones_all)

        for d in detections:
            draw_player(frame, d["id"], d["box"], d["keypoints"],
                        scorable=d["in_court"])

        if shuttle_pos:
            draw_shuttle(frame, shuttle_pos[0], shuttle_pos[1])

        draw_scoreboard(frame, scores, current_difficulty, active_target_id)

        # ── Shot state machine ───────────────────────────────

        # No active shot — fire a new one after interval
        if active_target_id is None:
            elapsed = time.time() - last_shot_time

            # Check shot limit
            if max_shots > 0 and total_shots_fired >= max_shots:
                draw_status(frame, "DRILL COMPLETE!", (0, 255, 100))
                cv2.imshow("Badminton Feeder Trainer", frame)
                cv2.waitKey(2000)
                break

            remaining_wait = interval - elapsed
            if remaining_wait > 0:
                draw_status(frame,
                    f"Next shot in {remaining_wait:.1f}s | "
                    f"Players in court: {len(player_positions)}")
            else:
                if player_positions:
                    start_new_shot(player_positions)
                    last_shot_time = time.time()
                else:
                    draw_status(frame,
                        "Waiting for player to enter court...",
                        (0, 200, 255))

        # Active shot in progress
        else:
            elapsed  = time.time() - shot_start_time
            time_left = interval - elapsed

            # Process shuttle position
            if shuttle_pos:
                sx, sy = shuttle_pos
                scx, scy = to_court((sx, sy))

                # Phase 1: wait for shuttle to cross net into player zone
                if not shuttle_crossed:
                    if check_net_crossing(scy):
                        zone = get_zone_from_position(scx, scy)
                        print(f"[CROSSING] Shuttle entered zone: {zone}")
                        draw_status(frame,
                            f"Shuttle in {zone} → P{active_target_id} returning... "
                            f"({time_left:.1f}s)",
                            (0, 255, 255))
                    else:
                        draw_status(frame,
                            f"Shuttle in flight → P{active_target_id} | "
                            f"{time_left:.1f}s left")

                # Phase 2: shuttle crossed — wait for return
                else:
                    if check_return(scy):
                        # Successful return!
                        scores.record_score(active_target_id, active_target_zone)
                        print(f"[SCORE] Player {active_target_id} returned "
                              f"from {active_target_zone}! ✅")
                        draw_status(frame,
                            f"Player {active_target_id} scored! ✅",
                            (0, 255, 100))
                        cv2.imshow("Badminton Feeder Trainer", frame)
                        cv2.waitKey(800)
                        reset_shot()
                        last_shot_time = time.time()
                        continue
                    else:
                        draw_status(frame,
                            f"Waiting for return → P{active_target_id} | "
                            f"{time_left:.1f}s left")

            else:
                # Shuttle not detected — show waiting status
                draw_status(frame,
                    f"Tracking shuttle... P{active_target_id} | "
                    f"{time_left:.1f}s left",
                    (200, 200, 0))

            # Timer expired — player missed
            if elapsed >= interval:
                print(f"[MISS] Player {active_target_id} missed "
                      f"zone {active_target_zone} ❌")
                draw_status(frame,
                    f"Player {active_target_id} missed ❌",
                    (0, 0, 255))
                cv2.imshow("Badminton Feeder Trainer", frame)
                cv2.waitKey(800)
                reset_shot()
                last_shot_time = time.time()

        cv2.imshow("Badminton Feeder Trainer", frame)

    # ── End of drill ─────────────────────────────────────────
    shuttle_worker.stop()
    cap.release()
    cv2.destroyAllWindows()
    scores.print_assessment()


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Badminton Feeder Trainer")
    parser.add_argument("--difficulty", default="medium",
                        choices=["easy", "medium", "hard"],
                        help="Drill difficulty")
    parser.add_argument("--shots", type=int, default=0,
                        help="Number of shots (0 = unlimited)")
    parser.add_argument("--source", default=None,
                        help="Video file path to run on instead of the webcam "
                             "(overrides VIDEO_SOURCE / CAMERA_INDEX)")
    args = parser.parse_args()

    if args.source is not None:
        video_source = args.source

    run(difficulty=args.difficulty, shots=args.shots)
