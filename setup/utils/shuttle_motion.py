# ============================================================
# shuttle_motion.py — Flying-shuttle detection by MOTION, for a fixed camera.
#
# Moved here from scripts/shuttle_motion_prototype.py so the drill engine and
# the prototype run the same code. The prototype stays the tuning tool.
#
# WHY MOTION, NOT YOLO
# A single frame does not carry enough information to find a far shuttle: our
# YOLO models found it at court distance a few percent of the time. The front
# camera is bolted down, so background subtraction isolates anything that
# moves for ~10 ms/frame on the i3 and no neural network at all — which is
# what makes it affordable on the Raspberry Pi next to pose.
#
# THE PIPELINE
#   1. MOG2 background subtraction        -> foreground mask
#   2. contour filter (area + aspect)     -> shuttle-sized moving blobs
#   3. player-box rejection               -> drop limbs/rackets
#   4. Kalman constant-velocity filter    -> pick the candidate on the
#                                            trajectory, coast through misses
#
# KNOWN LIMITS
#   * the shuttle must be MOVING; one resting on the floor is invisible here
#     (the side cameras and app/landings.py handle landed shuttles)
#   * the camera must not move; a bump floods the mask until MOG2 relearns
#   * thresholds are in FRAMES (confirm, max_coast, min_speed px/frame), so
#     they assume the loop keeps a steady frame rate
# ============================================================

import cv2
import numpy as np


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


class ShuttleMotionDetector:
    def __init__(self, min_area=12, max_area=1200, max_aspect=6.0,
                 gate=140.0, max_coast=8, history=350, var_threshold=28,
                 player_pad=14, scale=0.5, confirm=4, tent_gate=55.0,
                 min_speed=6.0):
        # MOG2 relearns the background continuously, so slow lighting drift is
        # absorbed.
        #
        # detectShadows=False: measured 26.6 -> 23.0 ms at full res on the i3,
        # and the hard-foreground threshold below discards the shadow label
        # anyway, so we were paying for a result we then threw away.
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False)
        # Subtraction runs on a downscaled frame; everything the caller sees is
        # in FULL-RES coordinates. Measured on the i3-1215U over 400 frames:
        #     full res            23.0 ms
        #     half res (0.5)      ~10.5 ms   <- default
        #     quarter res (0.25)  10.4 ms    (no better: the resize dominates)
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
        # i.e. camera noise. A shuttle in flight is not distinguished from
        # noise by SIZE — it is distinguished by tracing a coherent path. So no
        # lock is granted until a candidate has been followed for `confirm`
        # frames along a roughly constant-velocity track that actually goes
        # somewhere (`min_speed`). Noise blobs appear and vanish at random and
        # almost never survive that.
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

        result is (x, y, state, bbox) where state is "det" (measured this
        frame) or "coast" (Kalman prediction only, bbox None); result is None
        when there is no lock.
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
