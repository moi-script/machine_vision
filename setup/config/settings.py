# ============================================================
# settings.py — All hardcoded values for your setup
# Calibrate these values by capturing a still frame first
# and measuring pixel positions manually
# ============================================================

# --- Camera ---
CAMERA_INDEX    = 0        # USB camera index (try 0 or 1)
FRAME_WIDTH     = 1280     # OV9281 native width
FRAME_HEIGHT    = 800      # OV9281 native height
FPS_TARGET      = 10       # target FPS for Raspberry Pi later

# OV9281 is grayscale — convert for YOLO compatibility.
# Set True for the OV9281; False for a normal color webcam (e.g. laptop cam).
GRAYSCALE       = False

# --- Court boundary (hardcode after calibrating your camera) ---
# This is the full court area visible in your camera frame
# Format: (x1, y1, x2, y2)
COURT_ZONE      = (100, 50, 1180, 750)

# --- Net position ---
# Vertical X pixel position of the net in your side-view camera
NET_X           = 640      # adjust after calibrating

# --- Player side vs feeder side ---
# In side view: player is LEFT of net, feeder is RIGHT
# Change this if your camera is flipped
PLAYER_SIDE     = "left"   # shuttle_x < NET_X
FEEDER_SIDE     = "right"  # shuttle_x > NET_X

# --- 6 Court zones on PLAYER side (x1, y1, x2, y2) ---
# Divide player side into 3 columns x 2 rows
# Calibrate these after setting up your camera
PLAYER_ZONES = {
    "back_left"  : (100,  50,  380, 300),
    "back_mid"   : (380,  50,  640, 300),
    "back_right" : (380,  50,  640, 300),  # adjust per your frame
    "mid_left"   : (100,  300, 380, 550),
    "mid_center" : (380,  300, 640, 550),
    "mid_right"  : (640,  300, 920, 550),
}

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
PLAYER_OVERLAP_RATIO  = 0.6   # % of player box inside court zone

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
