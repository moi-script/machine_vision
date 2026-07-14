"""DrillEngine — headless, controllable version of main.py's run() loop.

Runs the capture -> detect -> score loop on a background thread, publishing
annotated JPEG frames to app.streamer.buffer and structured events to
app.events.hub. Control is via method calls (no cv2 window / keyboard)."""
from __future__ import annotations
import os
import threading
import time
import random
from datetime import datetime, timezone

import cv2
import numpy as np
from ultralytics import YOLO

from config.settings import (
    PLAYER_ZONES, DIFFICULTY, PERSON_CONFIDENCE, RETURN_CONFIRM_FRAMES,
    GRAYSCALE, FRAME_WIDTH, FRAME_HEIGHT, CAMERA_INDEX,
    SHUTTLE_SOURCE, SHUTTLE_MODEL_PATH, SHUTTLE_CONFIDENCE,
)
from utils.zones import (
    get_ankle_position, get_zone_from_position, get_player_in_zone,
    get_shuttle_side, to_court, in_court_bounds, crossed_net, build_homography,
)
from utils.scoring import PlayerScores
from utils.shuttle_worker import ShuttleWorker
from utils.display import (
    draw_court_zone, draw_net, draw_zones, draw_player, draw_shuttle,
    draw_scoreboard, draw_status, draw_fps,
)

from app.streamer import buffer as frame_buffer
from app.events import hub


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DrillEngine:
    def __init__(self):
        self._state = "idle"           # idle | running | paused
        self._difficulty = "medium"
        self._interval = DIFFICULTY["medium"]["interval"]
        self._shots = 0
        self._session_id: str | None = None
        self._thread: threading.Thread | None = None
        self._stop_flag = False
        self._camera = "unknown"
        self._fps = 0.0
        self._needs_restart = False
        self._player_model: YOLO | None = None
        self._scores = PlayerScores()  # kept for D4 to read
        # Live-tunable thresholds (bound on the instance so reload_settings()
        # actually reaches the loop closures — module-level imports won't).
        self._person_conf = PERSON_CONFIDENCE
        self._shuttle_conf = SHUTTLE_CONFIDENCE
        # Shuttle detection state (ported from main.py module-level state)
        self._shuttle_loaded = False   # guard so _load_shuttle_model runs once
        self._shuttle_model: YOLO | None = None
        self._shuttle_ready = False    # local weights loaded and usable
        self._shuttle_serverless = False
        self._serverless_warned = False
        # start() -> _run() handshake so start() can raise synchronously.
        self._started_evt = threading.Event()
        self._start_error: str | None = None

    # ---- public state ----
    @property
    def state(self) -> str:
        return self._state

    def status(self) -> dict:
        return {
            "type": "engine_status", "state": self._state,
            "difficulty": self._difficulty, "camera": self._camera,
            "fps": round(self._fps, 1), "sessionId": self._session_id,
        }

    def _emit_status(self) -> None:
        hub.broadcast(self.status())

    # ---- Mongo persistence (Task D4) ----
    @staticmethod
    def _zone_to_frontend(zone: str) -> str:
        return zone.replace("_", "-")

    def _persist_shot(self, player_id: str, zone: str) -> None:
        self._persist(player_id, zone, "shots", "totalShots")

    def _persist_score(self, player_id: str, zone: str) -> None:
        self._persist(player_id, zone, "scores", "totalScores")

    def _persist(self, player_id, zone, field, total_field) -> None:
        fz = self._zone_to_frontend(zone)
        from app import db
        # player cumulative
        db.players().update_one(
            {"_id": player_id},
            {"$inc": {f"stats.{total_field}": 1,
                      f"stats.zones.{fz}.{field}": 1}})
        # session liveData (upsert the player's entry)
        if not self._session_id:
            return
        sess = db.sessions()
        matched = sess.update_one(
            {"_id": self._session_id, "liveData.playerId": player_id},
            {"$inc": {f"liveData.$.{field}": 1,
                      f"liveData.$.zones.{fz}.{field}": 1}})
        if matched.matched_count == 0:
            empty = {z: {"shots": 0, "scores": 0} for z in
                     ["front-left", "front-center", "front-right",
                      "back-left", "back-center", "back-right"]}
            empty[fz][field] = 1
            sess.update_one({"_id": self._session_id}, {"$push": {"liveData": {
                "playerId": player_id, "shots": int(field == "shots"),
                "scores": int(field == "scores"), "zones": empty}}})

    def _live_players_payload(self) -> list[dict]:
        payload = []
        for player_id, data in self._scores.scores.items():
            zones = {}
            for zone, zdata in data.items():
                if zone in ("total_score", "total_shots"):
                    continue
                zones[self._zone_to_frontend(zone)] = {
                    "shots": zdata["shots"], "scores": zdata["score"],
                }
            payload.append({
                "playerId": str(player_id),
                "shots": data.get("total_shots", 0),
                "scores": data.get("total_score", 0),
                "zones": zones,
            })
        return payload

    # ---- control ----
    def set_difficulty(self, difficulty: str) -> None:
        if difficulty in DIFFICULTY:
            self._difficulty = difficulty
            self._interval = DIFFICULTY[difficulty]["interval"]
            self._emit_status()

    def pause(self) -> None:
        if self._state == "running":
            self._state = "paused"
            self._emit_status()

    def resume(self) -> None:
        if self._state == "paused":
            self._state = "running"
            self._emit_status()

    def reload_settings(self) -> None:
        from app.routers.settings import load_settings
        s = load_settings()
        self._interval = s.drill.intervals.get(self._difficulty, self._interval)
        # Live-apply thresholds via instance attrs (the loop closures read these).
        self._person_conf = s.detection.personConf
        self._shuttle_conf = s.detection.shuttleConf
        # ZONE_WEAK_THRESHOLD was bound at import into utils.scoring — patch it
        # on the module that actually holds it so weak-zone logic updates.
        import utils.scoring as scoring_mod
        scoring_mod.ZONE_WEAK_THRESHOLD = s.drill.weakZoneThreshold
        self._needs_restart = True  # structural (camera) changes need a restart

    def start(self, session_id: str | None, difficulty: str, shots: int) -> None:
        if self._state != "idle":
            raise RuntimeError("engine already running")
        self.set_difficulty(difficulty)
        self._shots = shots
        self._session_id = session_id
        self._stop_flag = False
        self._state = "running"
        self._start_error = None
        self._started_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="DrillEngine")
        self._thread.start()
        # Wait for _run() to report camera/calibration outcome so we can raise
        # synchronously. _run() sets the event before the heavy model load.
        if not self._started_evt.wait(timeout=10.0) or self._start_error is not None:
            self._state = "idle"
            msg = self._start_error or "engine failed to start (timeout)"
            raise RuntimeError(msg)
        self._emit_status()

    def stop(self) -> None:
        self._stop_flag = True
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._state = "idle"
        self._emit_status()

    # ---- helpers ----
    def _publish(self, frame) -> None:
        ok, enc = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            frame_buffer.publish(enc.tobytes())

    def capture_frame(self) -> bytes:
        latest = frame_buffer.latest()
        if latest is not None and self._state != "idle":
            return latest
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("camera capture failed")
        return cv2.imencode(".jpg", frame)[1].tobytes()

    # ---- feeder (ported from main.py fire_feeder) ----
    def _fire_feeder(self, zone_name: str) -> None:
        """Trigger the physical feeder machine. Currently just broadcasts the
        event; wire this to GPIO / serial when moving to the Raspberry Pi."""
        hub.broadcast({"type": "feeder", "zone": zone_name, "at": _iso()})
        # GPIO example for Raspberry Pi later:
        #   GPIO.output(FEEDER_PIN, GPIO.HIGH); time.sleep(0.1)
        #   GPIO.output(FEEDER_PIN, GPIO.LOW)

    # ---- shuttle detection (ported from main.py:54-157) ----
    def _load_shuttle_model(self) -> None:
        """Load the shuttle-detection source once. Mirrors main.py's model
        loading block. Never raises — on any problem detection stays off."""
        if self._shuttle_loaded:
            return
        self._shuttle_loaded = True

        if SHUTTLE_SOURCE == "local":
            if os.path.isfile(SHUTTLE_MODEL_PATH):
                try:
                    self._shuttle_model = YOLO(SHUTTLE_MODEL_PATH)
                    self._shuttle_ready = True
                except Exception as exc:  # noqa: BLE001 - keep the drill running
                    hub.broadcast({"type": "error",
                                    "message": f"[MODEL] Failed to load "
                                               f"{SHUTTLE_MODEL_PATH}: {exc}"})
            else:
                hub.broadcast({"type": "error",
                                "message": f"[MODEL] SHUTTLE_SOURCE='local' but no "
                                           f"file at {SHUTTLE_MODEL_PATH} — shuttle "
                                           f"detection disabled."})
        elif SHUTTLE_SOURCE == "serverless":
            self._shuttle_serverless = True
        # else "off" — nothing to load.

    def _detect_shuttle(self, frame, results_cache=None):
        """Detect the shuttlecock in a frame -> (cx, cy) or None.

        Ported from main.py's get_shuttle_position. All exceptions are
        swallowed (a serverless network blip must never kill the loop)."""
        if self._shuttle_serverless:
            try:
                from utils.roboflow_client import (
                    run_shuttlecock_model, extract_shuttle_xy,
                )
                result = run_shuttlecock_model(
                    frame, confidence=int(self._shuttle_conf * 100))
            except Exception as exc:  # noqa: BLE001 - never kill the drill loop
                if not self._serverless_warned:
                    hub.broadcast({"type": "error",
                                    "message": f"[MODEL] Serverless shuttle "
                                               f"detection error: {exc}"})
                    self._serverless_warned = True
                return None
            try:
                return extract_shuttle_xy(result)  # (x, y) centre, or None
            except Exception:  # noqa: BLE001
                return None

        if self._shuttle_ready:  # local weights
            try:
                shuttle_results = self._shuttle_model(
                    frame, conf=self._shuttle_conf, verbose=False)
                boxes = shuttle_results[0].boxes
                if boxes is None or len(boxes) == 0:
                    return None
                best_i = int(boxes.conf.argmax())
                x1, y1, x2, y2 = boxes.xyxy[best_i].tolist()
                return (x1 + x2) / 2.0, (y1 + y2) / 2.0
            except Exception:  # noqa: BLE001
                return None

        return None  # "off" or no local weights

    # ---- the loop (adapted from main.py run()) ----
    def _run(self) -> None:
        try:
            build_homography()
        except ValueError as exc:
            self._start_error = f"[CALIBRATION] {exc}"
            self._camera = "unavailable"
            self._state = "idle"
            self._started_evt.set()
            hub.broadcast({"type": "error", "message": f"[CALIBRATION] {exc}"})
            return

        # ── Camera setup (inline open_camera(), substitution 1) ──────
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if GRAYSCALE and isinstance(CAMERA_INDEX, int):
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        if not cap.isOpened():
            self._start_error = "camera unavailable"
            self._camera = "unavailable"
            self._state = "idle"
            self._started_evt.set()
            return
        self._camera = "ok"
        # Success — let start() return promptly; heavy model load follows.
        self._started_evt.set()

        if self._player_model is None:
            self._player_model = YOLO("yolov8n-pose.pt")
        player_model = self._player_model

        # Load shuttle detection source (local weights / serverless / off).
        self._load_shuttle_model()

        shuttle_worker = ShuttleWorker(self._detect_shuttle)
        shuttle_worker.start()

        scores = self._scores
        prev_shuttle_cy = None
        shuttle_crossed = False
        return_side_count = 0
        active_target_id = None
        active_target_zone = None
        shot_start_time = None
        total_shots_fired = 0
        max_shots = self._shots
        interval = self._interval
        last_shot_time = time.time()
        last_stats = 0.0

        def detect_players(frame):
            results = player_model.track(frame, persist=True,
                                          classes=[0],
                                          conf=self._person_conf,
                                          verbose=False)
            player_positions = {}
            detections = []

            if results[0].boxes.id is None:
                return player_positions, detections

            track_ids = results[0].boxes.id.int().tolist()
            boxes = results[0].boxes.xyxy
            keypoints_data = results[0].keypoints.data

            for i, tid in enumerate(track_ids):
                box = boxes[i].tolist()
                kps = keypoints_data[i].tolist()
                ankle = get_ankle_position(keypoints_data[i])

                in_court = False
                court_xy = None
                if ankle:
                    court_xy = to_court(ankle)
                    if in_court_bounds(*court_xy):
                        in_court = True
                        scores.init_player(tid)
                        player_positions[tid] = court_xy

                detections.append({
                    "id": tid,
                    "box": box,
                    "keypoints": kps,
                    "in_court": in_court,
                })

            return player_positions, detections

        def check_net_crossing(shuttle_cy):
            nonlocal prev_shuttle_cy, shuttle_crossed
            if prev_shuttle_cy is None:
                prev_shuttle_cy = shuttle_cy
                return False
            just_crossed = crossed_net(prev_shuttle_cy, shuttle_cy)
            prev_shuttle_cy = shuttle_cy
            if just_crossed and not shuttle_crossed:
                shuttle_crossed = True
                return True
            return False

        def check_return(shuttle_cy):
            nonlocal return_side_count
            if get_shuttle_side(shuttle_cy) == "feeder_side":
                return_side_count += 1
            else:
                return_side_count = 0
            return return_side_count >= RETURN_CONFIRM_FRAMES

        def start_new_shot(player_positions):
            nonlocal active_target_id, active_target_zone
            nonlocal shot_start_time, shuttle_crossed, return_side_count
            nonlocal prev_shuttle_cy, total_shots_fired

            zone_name = random.choice(list(PLAYER_ZONES.keys()))
            target_id = get_player_in_zone(zone_name, player_positions)
            if target_id is None:
                return None, None

            scores.record_shot(target_id, zone_name)
            self._persist_shot(str(target_id), zone_name)
            total_shots_fired += 1

            shuttle_crossed = False
            return_side_count = 0
            prev_shuttle_cy = None
            shot_start_time = time.time()
            active_target_id = target_id
            active_target_zone = zone_name

            hub.broadcast({"type": "shot", "zone": zone_name,
                            "targetPlayerId": str(target_id), "at": _iso()})
            self._fire_feeder(zone_name)

            return target_id, zone_name

        def reset_shot():
            nonlocal active_target_id, active_target_zone
            nonlocal shot_start_time, shuttle_crossed, return_side_count
            active_target_id = None
            active_target_zone = None
            shot_start_time = None
            shuttle_crossed = False
            return_side_count = 0

        fps_times = []

        while True:
            if self._stop_flag:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if GRAYSCALE and len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Smoothed FPS (substitution 9)
            now = time.time()
            fps_times.append(now)
            if len(fps_times) > 30:
                fps_times.pop(0)
            if len(fps_times) >= 2:
                span = fps_times[-1] - fps_times[0]
                self._fps = (len(fps_times) - 1) / span if span > 0 else 0.0
            draw_fps(frame)

            # Difficulty may have changed live via set_difficulty()
            interval = self._interval

            if self._state == "paused":
                self._publish(frame)
                time.sleep(0.03)
                continue

            # ── Detect players ───────────────────────────────────
            player_positions, detections = detect_players(frame)

            # ── Detect shuttle (async) ───────────────────────────
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

            draw_scoreboard(frame, scores, self._difficulty, active_target_id)

            # ── live_stats throttle (~2/s, substitution 8) ────────
            if now - last_stats > 0.5:
                hub.broadcast({"type": "live_stats",
                                "sessionId": self._session_id,
                                "players": self._live_players_payload()})
                last_stats = now

            # ── Shot state machine ───────────────────────────────
            if active_target_id is None:
                elapsed = time.time() - last_shot_time

                if max_shots > 0 and total_shots_fired >= max_shots:
                    draw_status(frame, "DRILL COMPLETE!", (0, 255, 100))
                    self._publish(frame)
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

            else:
                elapsed = time.time() - shot_start_time
                time_left = interval - elapsed

                if shuttle_pos:
                    sx, sy = shuttle_pos
                    scx, scy = to_court((sx, sy))

                    if not shuttle_crossed:
                        if check_net_crossing(scy):
                            zone = get_zone_from_position(scx, scy)
                            hub.broadcast({"type": "crossing", "zone": zone,
                                            "at": _iso()})
                            draw_status(frame,
                                f"Shuttle in {zone} → P{active_target_id} returning... "
                                f"({time_left:.1f}s)",
                                (0, 255, 255))
                        else:
                            draw_status(frame,
                                f"Shuttle in flight → P{active_target_id} | "
                                f"{time_left:.1f}s left")
                    else:
                        if check_return(scy):
                            scores.record_score(active_target_id, active_target_zone)
                            self._persist_score(str(active_target_id), active_target_zone)
                            hub.broadcast({"type": "score",
                                            "playerId": str(active_target_id),
                                            "zone": active_target_zone,
                                            "at": _iso()})
                            draw_status(frame,
                                f"Player {active_target_id} scored! ✅",
                                (0, 255, 100))
                            self._publish(frame)
                            time.sleep(0.8)
                            reset_shot()
                            last_shot_time = time.time()
                            continue
                        else:
                            draw_status(frame,
                                f"Waiting for return → P{active_target_id} | "
                                f"{time_left:.1f}s left")
                else:
                    draw_status(frame,
                        f"Tracking shuttle... P{active_target_id} | "
                        f"{time_left:.1f}s left",
                        (200, 200, 0))

                if elapsed >= interval:
                    hub.broadcast({"type": "miss",
                                    "playerId": str(active_target_id),
                                    "zone": active_target_zone,
                                    "at": _iso()})
                    draw_status(frame,
                        f"Player {active_target_id} missed ❌",
                        (0, 0, 255))
                    self._publish(frame)
                    time.sleep(0.8)
                    reset_shot()
                    last_shot_time = time.time()

            self._publish(frame)

        # ── End of drill ─────────────────────────────────────────
        shuttle_worker.stop()
        cap.release()
        self._state = "idle"
        self._emit_status()


_engine: DrillEngine | None = None


def get_engine() -> DrillEngine:
    global _engine
    if _engine is None:
        _engine = DrillEngine()
    return _engine
