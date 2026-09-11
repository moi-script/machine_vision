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
id "$RUN_USER" >/dev/null 2>&1 || die \
  "user $RUN_USER does not exist - create it (see PI-SETUP.md) or re-run with RUN_USER=<user>"

# ── the committed UI bundle ─────────────────────────────────
log "checking the UI bundle"
[ -f "$ROOT/app/static/ui/index.html" ] || die \
  "app/static/ui/index.html is missing. The bundle is built on the Windows dev
   machine with scripts/build_ui.ps1 and committed. Run it, commit, and git pull."

# ── LFS-tracked model weights ───────────────────────────────
log "pulling LFS objects (model weights, face ONNX)"
# apt lists may be empty/stale on a freshly flashed image - update before the
# very first apt-get install, not just before the "system packages" section
# below, or this dies on "Unable to locate package git-lfs".
command -v git-lfs >/dev/null || { apt-get update; apt-get install -y git-lfs; }
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
apt-get install -y python3-venv python3-pip python3-opencv \
                   v4l-utils curl gnupg
# Raspberry Pi OS images differ on the browser package name too - installed
# in its own guarded step so a mismatch here doesn't abort the whole section
# before the binary check below ever runs.
apt-get install -y chromium-browser || apt-get install -y chromium || true

# ── kiosk browser binary ────────────────────────────────────
# ...and they differ on the installed *binary* path the same way. kiosk is
# launched via an XDG autostart entry that hardcodes chromium-browser as its
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
# mongodb-org normally pulls in mongodb-mongosh, but check separately from
# "is mongod answering" below - a missing mongosh (exit 127) and a mongod
# that truly isn't listening are two different problems, and the combined
# check used to blame the database for a client that's simply not there.
command -v mongosh >/dev/null || die "mongosh missing: apt-get install -y mongodb-mongosh"
systemctl enable --now mongod

log "waiting for mongod"
mongo_ok() { [ "$(mongosh --quiet --eval 'db.runCommand({ping:1}).ok' 2>/dev/null)" = "1" ]; }
for _ in $(seq 30); do
  mongo_ok && break
  sleep 1
done
# Run under `set -e` directly (not piped through grep) - under pipefail a
# `mongosh | grep -q` pipe can hand mongosh a SIGPIPE the instant grep finds
# its match, and pipefail turns that into a failure even when Mongo is fine.
mongo_ok || die "mongod is not answering: journalctl -u mongod -n 50"

# ── Python environment ──────────────────────────────────────
log "creating the venv"
# Guard on the interpreter, not just the directory: a venv creation
# interrupted partway (Ctrl-C, power loss, an earlier die) leaves the
# directory in place with no bin/pip, and re-running with `[ -d "$VENV" ]`
# would then die confusingly on a missing pip instead of just recreating it.
[ -x "$VENV/bin/python" ] || sudo -u "$RUN_USER" python3 -m venv --system-site-packages "$VENV"
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
  # Group, not just owner, and no world/group read - this file can end up
  # holding a real API key.
  chown "$RUN_USER:$RUN_USER" "$ROOT/.env"
  chmod 600 "$ROOT/.env"
  cat >&2 <<'WARN'

    ###########################################################
    #  NOTE                                                    #
    #                                                          #
    #  .env was written with an EMPTY ROBOFLOW_API_KEY. The    #
    #  shipped config uses SHUTTLE_SOURCE=local (a committed   #
    #  weights file), so shuttle detection works offline with  #
    #  no key at all. Only fill this in if you deliberately    #
    #  switch SHUTTLE_SOURCE to "serverless" for the Roboflow  #
    #  cloud workflow - see config/settings.py.                #
    #  .env is gitignored: install.sh can never commit a real  #
    #  key for you.                                            #
    ###########################################################

WARN
fi

# ── warm the NCNN exports ───────────────────────────────────
# First use of the Cameras / live-detection page triggers an export that
# takes minutes. Doing it here means that page does not stall the first time
# it is opened. (The drill engine loads its models separately, in plain
# torch, straight from committed weights - it needs no warm-up.)
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

# ── aerosense backend systemd unit ──────────────────────────
log "installing the aerosense systemd unit"
# aerosense.service in git hardcodes /opt/aerosense/setup, both in
# WorkingDirectory and in ExecStart's venv path, as its documented default
# layout. Substitute the actual checkout root so a relocated clone (PI-SETUP
# says relocation is fine - just edit this unit) gets a unit that matches
# where it actually lives, instead of one pointing at a path that doesn't
# exist.
sed "s|/opt/aerosense/setup|$ROOT|g" "$ROOT/deploy/aerosense.service" \
  > /etc/systemd/system/aerosense.service
chmod 644 /etc/systemd/system/aerosense.service
systemctl daemon-reload
systemctl enable aerosense.service
systemctl restart aerosense.service

log "waiting for the API"
for _ in $(seq 30); do
  if curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS http://127.0.0.1:8000/api/health || die "the API never came up: journalctl -u aerosense -n 50"
echo

# ── kiosk autostart (user-session XDG entry, not a system unit) ─
# graphical-session.target is a systemd *user*-manager target - the system
# manager never activates it, so a system-level kiosk.service enabled here
# would silently never start. Ship it the conventional Pi way instead: an
# XDG autostart .desktop entry that the desktop session runs directly once
# LightDM's autologin brings up X11, where DISPLAY/XAUTHORITY are already
# real and no target games are needed.
log "installing the kiosk autostart entry"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
[ -n "$RUN_HOME" ] && [ -d "$RUN_HOME" ] || die "no home directory for $RUN_USER"
chmod 755 "$ROOT/deploy/kiosk-launch.sh"
sudo -u "$RUN_USER" mkdir -p "$RUN_HOME/.config/autostart"
# kiosk.desktop in git keeps /opt/aerosense/setup and chromium-browser as its
# documented defaults. Rewrite the whole Exec= line from $ROOT and
# $KIOSK_BIN rather than matching its current value - matching the old value
# only works when the checkout is still at /opt/aerosense; a relocated
# checkout (PI-SETUP says relocation is fine) would then pass through
# unmatched and point Exec= at a kiosk-launch.sh that doesn't exist, failing
# silently with no diagnostic at all.
sed "s|^Exec=.*|Exec=$ROOT/deploy/kiosk-launch.sh $KIOSK_BIN|" \
  "$ROOT/deploy/kiosk.desktop" > "$RUN_HOME/.config/autostart/kiosk.desktop"
chown "$RUN_USER:$RUN_USER" "$RUN_HOME/.config/autostart/kiosk.desktop"
chmod 644 "$RUN_HOME/.config/autostart/kiosk.desktop"

log "done. Reboot to bring up the kiosk: sudo reboot"
echo "    Cameras still need assigning - see deploy/PI-SETUP.md."
