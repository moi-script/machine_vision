# ============================================================
# settings.py — All hardcoded values for your setup
# Calibrate these values by capturing a still frame first
# and measuring pixel positions manually
# ============================================================

# --- Camera / video source ---
CAMERA_INDEX    = 0        # USB camera index (try 0 or 1)
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
#   "local"      → models/shuttlecock.pt (fast, offline; needs a paid export)
#   "serverless" → Roboflow direct model over HTTP (FREE, but ~0.1–0.4s/frame
#                  network latency, so the drill loop runs at only a few FPS)
#   "off"        → no shuttle detection (player/zone logic only)
SHUTTLE_SOURCE        = "serverless"

# Path used when SHUTTLE_SOURCE = "local".
SHUTTLE_MODEL_PATH    = "models/shuttlecock.pt"

# --- Detection thresholds ---
PERSON_CONFIDENCE     = 0.5   # min confidence to count a person
ANKLE_CONFIDENCE      = 0.5   # min confidence to use ankle keypoint
SHUTTLE_CONFIDENCE    = 0.4   # min confidence to count shuttle

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
