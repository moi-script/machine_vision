# ============================================================
# settings.py — All hardcoded values for your setup
# Calibrate these values by capturing a still frame first
# and measuring pixel positions manually
# ============================================================

# --- Camera / video source ---
CAMERA_INDEX    = 0        # USB camera index (0 = built-in laptop cam, 1 = external USB cam)
# Set to a video file path (e.g. "clips/rally.mp4") to run on recorded footage
# instead of the live webcam. None = use the live CAMERA_INDEX webcam.
# Both calibrate.py and main.py honor this, and a --source CLI flag overrides it.
VIDEO_SOURCE    = None
FRAME_WIDTH     = 1280     # OV9281 native width
FRAME_HEIGHT    = 800      # OV9281 native height
FPS_TARGET      = 10       # target FPS for Raspberry Pi later

# OV9281 is grayscale — convert for YOLO compatibility.
# Set True for the OV9281; False for a normal color webcam (e.g. laptop cam).
GRAYSCALE       = False

# --- Difficulty settings (seconds between shots) ---
DIFFICULTY = {
    "easy"  : {"interval": 5.0},
    "medium": {"interval": 3.0},
    "hard"  : {"interval": 1.5},
}

# --- Shuttlecock detection source ---
# Where shuttle detections come from:
#   "local"      → models/shuttlecock.pt, ~99 ms/frame on this CPU, fully offline
#   "serverless" → Roboflow direct model over HTTP (FREE, but 1.2–3.9 s/frame
#                  measured round trip, so the drill loop runs at well under 1 FPS)
#   "motion"     → MOG2 background subtraction + Kalman tracking of the one
#                  shuttle in flight (utils/shuttle_motion.py). No neural
#                  network: ~10 ms/frame on the i3, and the only option cheap
#                  enough for the front camera on the Raspberry Pi next to pose.
#                  Needs the front camera fixed in place.
#   "off"        → no shuttle detection (player/zone logic only)
#
# "local" no longer needs a paid Roboflow export: scripts/train_shuttlecock.py
# trains our own yolov8n on the free dataset export. Current weights are epoch 55
# of that run — mAP50 0.898, mAP50-95 0.389 on a 20-image val split.
SHUTTLE_SOURCE        = "motion"

# Path used when SHUTTLE_SOURCE = "local". Relative to the repo root, so run
# main.py / uvicorn from C:\thesis\setup or this will not resolve.
SHUTTLE_MODEL_PATH    = "models/shuttlecock.pt"

# --- Detection thresholds ---
PERSON_CONFIDENCE     = 0.5   # min confidence to count a person
ANKLE_CONFIDENCE      = 0.5   # min confidence to use ankle keypoint
SHUTTLE_CONFIDENCE    = 0.4   # min confidence to count shuttle

# --- Face recognition ---
# SFace cosine similarity gate: a live face must score at least this against an
# enrolled embedding to count as a match. 0.363 is OpenCV's published default;
# grayscale feeds carry less signal, so you may need to lower it (~0.30) once
# you test on the real camera.
FACE_MATCH_THRESHOLD  = 0.363

# --- Scoring ---
ZONE_WEAK_THRESHOLD   = 50.0  # below this % accuracy = weak zone
RETURN_CONFIRM_FRAMES = 2     # frames shuttle must be on return side

# --- Display ---
SHOW_SKELETON         = True
SHOW_ZONES            = True
SHOW_SHUTTLE_TRAIL    = True
TRAIL_LENGTH          = 10    # number of trail points to show

# --- Colors (BGR for OpenCV) ---
COLOR_COURT_ZONE   = (0,  255, 100)   # green
COLOR_NET          = (255, 255, 255)  # white
COLOR_PLAYER_1     = (0,  200, 255)   # yellow
COLOR_PLAYER_2     = (255, 100,  0)   # blue
COLOR_SHUTTLE      = (0,  100, 255)   # red
COLOR_ZONE_ACTIVE  = (0,  255, 255)   # cyan when shuttle lands
COLOR_SCORE_TEXT   = (255, 255, 255)  # white
COLOR_WEAK_ZONE    = (0,    0, 255)   # red for weak zones

# ============================================================
# Court-space geometry (top-down homography model)
# ============================================================

# 4 pixel corners of the TRAINEE's far half-court, in this order:
#   net_left, net_right, baseline_right, baseline_left
# Filled in by calibrate.py. None until calibrated (build_homography errors).
COURT_CORNERS = [(441, 36), (842, 45), (1134, 668), (197, 678)]

# Court-space dimensions (arbitrary units; only ratios matter for zones).
COURT_W = 518.0        # width  (net_left -> net_right)
COURT_L = 670.0        # length (net -> far baseline)

# Dead-band (court units) around the net line y=0 to debounce side/crossing.
NET_DEADBAND = 15.0

# --- 6 target zones in COURT SPACE (x1, y1, x2, y2); 3 cols x 2 rows ---
# "front" = nearer the net (small y); "back" = nearer the far baseline.
_CW3 = COURT_W / 3.0
_CL2 = COURT_L / 2.0
PLAYER_ZONES = {
    "front_left":   (0.0,       0.0,   _CW3,      _CL2),
    "front_center": (_CW3,      0.0,   2 * _CW3,  _CL2),
    "front_right":  (2 * _CW3,  0.0,   COURT_W,   _CL2),
    "back_left":    (0.0,       _CL2,  _CW3,      COURT_L),
    "back_center":  (_CW3,      _CL2,  2 * _CW3,  COURT_L),
    "back_right":   (2 * _CW3,  _CL2,  COURT_W,   COURT_L),
}


# ============================================================
# Player skill ranking (rule-based rubric) — all tunable
# See docs/superpowers/specs/2026-07-22-player-skill-ranking-design.md
# ============================================================

# --- Sampling gates ---
SKILL_KP_CONF    = 0.5    # min keypoint confidence to use a fine (posture/stroke) point
SKILL_BBOX_MIN_H = 200.0  # min person bbox height in px to sample posture/stroke (distance gate)
SKILL_REACT_DIST = 20.0   # court-space ankle displacement that counts as "started moving"
SKILL_SWING_SPEED = 3.0   # shoulder-relative wrist speed (units/sec) rising-edge = a swing
SKILL_MIN_SHOTS  = 20     # min accumulated shots before any tier is assigned (else Unranked)

# --- Per-metric normalization references ---
# type "monotonic": linear lo->hi mapped to 0->100 (invert=True flips it)
# type "target":    full score inside [target +/- tol], decaying linearly outside
# type "consistency": std mapped hi->0 / 0->100 (lower std = higher score)
SKILL_REFS = {
    # movement family
    "move_speed": {"type": "monotonic", "lo": 0.5, "hi": 6.0},
    "coverage":   {"type": "monotonic", "lo": 1.0, "hi": 6.0},
    "reaction":   {"type": "monotonic", "lo": 0.3, "hi": 1.5, "invert": True},
    # accuracy family
    "accuracy":   {"type": "monotonic", "lo": 20.0, "hi": 90.0},
    # stroke family
    "swing_consistency": {"type": "consistency", "hi": 4.0},
    # posture family
    "knee":       {"type": "target", "target": 150.0, "tol": 40.0},
    "stance":     {"type": "target", "target": 1.4, "tol": 0.9},
    "posture_consistency": {"type": "consistency", "hi": 25.0},
}

# --- Family weights (scaled by data sufficiency at eval time) ---
SKILL_FAMILY_WEIGHTS = {"move": 0.30, "accuracy": 0.30, "stroke": 0.20, "posture": 0.20}

# --- Min reliable samples for a family to count toward the composite ---
SKILL_FAMILY_MIN_SAMPLES = {"move": 30, "accuracy": 20, "stroke": 5, "posture": 20}

# --- Tier bands: (min_composite_inclusive, name), highest first ---
SKILL_TIERS = [
    (80.0, "Expert"),
    (60.0, "Advanced"),
    (40.0, "Intermediate"),
    (20.0, "Novice"),
    (0.0,  "Beginner"),
]




# --- Rig v2: per-slot capture size ---
# The OV9281 front runs at its native size. The ESP32-S3 boards advertise one
# fixed UVC mode each (firmware/esp32s3_uvc_cam/sdkconfig.role.*), and asking
# for anything else just gets refused, so these must match the firmware.
SLOT_FRAME_SIZE = {
    "front": (FRAME_WIDTH, FRAME_HEIGHT),
    "left": (640, 480),
    "right": (640, 480),
    "back": (640, 480),
    "face": (800, 600),
}

# Front runs flying-shuttle AND pose. If the Pi can't hold both per frame,
# True runs them on alternate frames instead (decided at bring-up).
FRONT_ALTERNATE = False

# --- Servo aim (Arduino over USB serial, udev name from 99-aerosense.rules) ---
import sys
SERVO_PORT = "/dev/aero-servo" if sys.platform != "win32" else "COM5"
SERVO_BAUD = 115200

# --- Raspberry Pi inference sizes ---
# On aarch64 the models run on CPU under NCNN, several times slower than
# OpenVINO on x86. START.md measures `landed` at 640 as strictly the better
# deal (18.3 ms vs 54.9 ms at 1280, still firing on 98% of frames), and the
# same argument applies to `shuttle`. app/pipeline.py applies these only when
# resolve_backend() picks ncnn.
PI_IMGSZ = {"shuttle": 640, "landed": 640, "pose": 640}

# CPU threads per NCNN model. NCNN defaults to every core, so each started
# camera ran its models on all 4 at once - a current spike that browned out a
# marginal supply and reset the Pi. 2 is also faster here (landed @ 640:
# 91 ms vs 120 ms at 4) because 4-thread nets fight each other for cores.
PI_NCNN_THREADS = 2

# --- Registration face camera ---
# Two boards can serve /api/face-cam, and the UI does not care which one does:
#   "usb"  -> the AERO-FACE ESP32-S3 board (udev /dev/aero-face, the "face" slot)
#   "wifi" -> the ESP32-CAM station on the LAN at ESP32_CAM_IP
#   "auto" -> USB when its device is present, else the ESP32-CAM if it answers
FACE_CAM_SOURCE = "auto"
# None turns the Wi-Fi camera off. Give the board a static IP / DHCP reservation.
ESP32_CAM_IP = "10.200.33.50"
ESP32_CAM_TIMEOUT_S = 5.0
# The board's MJPEG preview runs on its own server; must match the sketch.
ESP32_CAM_STREAM_PORT = 81
