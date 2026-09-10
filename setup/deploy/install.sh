#!/usr/bin/env bash
# Provision a Raspberry Pi 5 to run AeroSense as a kiosk appliance.
#
#   sudo ./deploy/install.sh
#
# Idempotent: safe to re-run after a git pull.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
RUN_USER="${RUN_USER:-aerosense}"

log() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run with sudo"
[ "$(uname -m)" = "aarch64" ] || die "expected aarch64, got $(uname -m) - use Pi OS 64-bit"

# ── the committed UI bundle ─────────────────────────────────
log "checking the UI bundle"
[ -f "$ROOT/app/static/ui/index.html" ] || die \
  "app/static/ui/index.html is missing. The bundle is built on the Windows dev
   machine with scripts/build_ui.ps1 and committed. Run it, commit, and git pull."

# ── LFS-tracked model weights ───────────────────────────────
log "pulling LFS objects (model weights, face ONNX)"
command -v git-lfs >/dev/null || apt-get install -y git-lfs
sudo -u "$RUN_USER" git -C "$ROOT/.." lfs pull
for w in models/shuttle_clear_badminton_p2.pt models/shuttle_lines_stock_n.pt \
         models/yolov8n-pose.pt models/face_detection_yunet_2023mar.onnx \
         models/face_recognition_sface_2021dec.onnx; do
  [ -s "$ROOT/$w" ] || die "$w missing or empty after git lfs pull"
  # An unfetched LFS pointer is a ~130-byte text file, not a model.
  [ "$(stat -c%s "$ROOT/$w")" -gt 100000 ] || die "$w is an unfetched LFS pointer"
done

# ── system packages ─────────────────────────────────────────
log "installing system packages"
apt-get update
# python3-opencv from apt, not pip: a pip opencv-python build on the Pi is slow
# and frequently fails. The venv is created --system-site-packages so it is
# visible.
apt-get install -y python3-venv python3-pip python3-opencv chromium-browser \
                   v4l-utils curl gnupg

# ── kiosk browser binary ────────────────────────────────────
# Raspberry Pi OS images differ on the binary name: chromium-browser on some,
# chromium on others. kiosk.service hardcodes chromium-browser as its
# committed default; get this wrong and the symptom is a black screen with a
# perfectly healthy backend - one of the worst combinations to debug.
log "checking the kiosk browser"
if   [ -x /usr/bin/chromium-browser ]; then KIOSK_BIN=/usr/bin/chromium-browser
elif [ -x /usr/bin/chromium ];         then KIOSK_BIN=/usr/bin/chromium
else die "no chromium binary found at /usr/bin/chromium-browser or /usr/bin/chromium"
fi
echo "    using $KIOSK_BIN"

# ── MongoDB 7.0 (arm64) ─────────────────────────────────────
if ! command -v mongod >/dev/null; then
  log "installing MongoDB 7.0"
  # The Pi 5's Cortex-A76 is ARMv8.2-A, which Mongo 6+ requires. A Pi 4's
  # Cortex-A72 (ARMv8.0) cannot run this and would need Mongo 4.4.
  curl -fsSL https://pgp.mongodb.com/server-7.0.asc \
    | gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg
  echo "deb [arch=arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg] \
https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" \
    > /etc/apt/sources.list.d/mongodb-org-7.0.list
  apt-get update
  apt-get install -y mongodb-org
fi
systemctl enable --now mongod
mongosh --quiet --eval 'db.runCommand({ping:1}).ok' | grep -q 1 \
  || die "mongod is not answering"

# ── Python environment ──────────────────────────────────────
log "creating the venv"
[ -d "$VENV" ] || sudo -u "$RUN_USER" python3 -m venv --system-site-packages "$VENV"
sudo -u "$RUN_USER" "$VENV/bin/pip" install --upgrade pip
# opencv-python comes from apt (above); dropping it here avoids a source build.
sudo -u "$RUN_USER" grep -v '^opencv-python' "$ROOT/requirements.txt" \
  > /tmp/req-pi.txt
sudo -u "$RUN_USER" "$VENV/bin/pip" install -r /tmp/req-pi.txt
sudo -u "$RUN_USER" "$VENV/bin/pip" install ncnn
sudo -u "$RUN_USER" "$VENV/bin/python" -c "import cv2, httpx, ultralytics; print('deps ok', cv2.__version__)"

# ── .env ────────────────────────────────────────────────────
if [ ! -f "$ROOT/.env" ]; then
  log "writing .env"
  cat > "$ROOT/.env" <<'ENV'
ROBOFLOW_API_KEY=
MONGO_URL=mongodb://localhost:27017/
MONGO_DB=aerosense
ENV
  chown "$RUN_USER" "$ROOT/.env"
  cat >&2 <<'WARN'

    ###########################################################
    #  ACTION REQUIRED                                        #
    #                                                          #
    #  .env was written with an EMPTY ROBOFLOW_API_KEY.        #
    #  Shuttle detection will fail - and it will look like a   #
    #  model problem, not a missing-key problem - until you    #
    #  edit /opt/aerosense/setup/.env by hand and add it.      #
    #  .env is gitignored: install.sh can never commit a real  #
    #  key for you.                                            #
    ###########################################################

WARN
fi

# ── warm the NCNN exports ───────────────────────────────────
# First use triggers an export that takes minutes. Doing it here means the
# first drill does not stall.
log "warming NCNN exports (several minutes)"
cd "$ROOT"
sudo -u "$RUN_USER" "$VENV/bin/python" - <<'PY'
from app import pipeline
from config import settings
print("backend:", pipeline.DEFAULT_BACKEND)
assert pipeline.DEFAULT_BACKEND == "ncnn", pipeline.DEFAULT_BACKEND
for key in ("shuttle", "landed", "pose"):
    cfg = pipeline.MODELS[key]
    imgsz = settings.PI_IMGSZ[key]
    print(f"  {key} @ {imgsz} ...", flush=True)
    pipeline._load(cfg["weights"], imgsz, cfg["task"], "ncnn")
print("exports ready")
PY

# ── systemd units ───────────────────────────────────────────
log "installing systemd units"
install -m644 "$ROOT/deploy/aerosense.service" /etc/systemd/system/
# kiosk.service in git keeps chromium-browser as its documented default;
# substitute in whichever binary this image actually has.
sed "s|^ExecStart=/usr/bin/chromium-browser|ExecStart=$KIOSK_BIN|" \
  "$ROOT/deploy/kiosk.service" > /etc/systemd/system/kiosk.service
chmod 644 /etc/systemd/system/kiosk.service
systemctl daemon-reload
systemctl enable aerosense.service kiosk.service
systemctl restart aerosense.service

log "waiting for the API"
for _ in $(seq 30); do
  if curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS http://127.0.0.1:8000/api/health || die "the API never came up: journalctl -u aerosense -n 50"
echo

log "done. Reboot to bring up the kiosk: sudo reboot"
echo "    Cameras still need assigning - see deploy/PI-SETUP.md."
