#!/usr/bin/env bash
# Launched by the aerosense user's XDG autostart entry (kiosk.desktop) once
# the X11 desktop session comes up after LightDM autologin. Loops relaunching
# the browser if it ever exits, since an autostart .desktop entry has no
# systemd Restart= to fall back on.
#
#   kiosk-launch.sh /usr/bin/chromium-browser
#
set -u

BIN="${1:?usage: kiosk-launch.sh <chromium binary>}"

# Autostart entries have no ordering against aerosense.service - the desktop
# session can come up well before uvicorn finishes importing ultralytics and
# connecting to Mongo. Without this wait, Chromium renders "This site can't
# be reached" for a perfectly healthy backend and just sits there: the
# process is alive, so the restart loop below never fires. Log to a file
# because an autostart entry has no attached terminal to see stdout on.
LOG="${XDG_CACHE_HOME:-$HOME/.cache}/aerosense-kiosk-wait.log"
mkdir -p "$(dirname "$LOG")" 2>/dev/null || true

WAIT_CEILING=90   # seconds - generous for a cold Pi with a first-boot Mongo
log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$LOG" 2>/dev/null; }

log "waiting up to ${WAIT_CEILING}s for http://127.0.0.1:8000/api/health"
start=$(date +%s)
next_progress=10
# --connect-timeout/--max-time bound each individual poll: without them a
# single curl can block far longer than the loop's own bookkeeping expects
# (a cold Mongo makes /api/health's db.ping() hang up to pymongo's default
# 30s serverSelectionTimeoutMS, so a couple of unbounded polls alone could
# blow past the ceiling below). Elapsed time is read from the wall clock,
# not counted in sleep-sized increments, so the ceiling means what it says
# regardless of how long any one poll took.
until curl -fsS --connect-timeout 2 --max-time 5 http://127.0.0.1:8000/api/health >/dev/null 2>&1; do
  elapsed=$(( $(date +%s) - start ))
  if [ "$elapsed" -ge "$WAIT_CEILING" ]; then
    log "API never answered after ${WAIT_CEILING}s - launching the browser anyway (it will show an error page; the restart loop and a reload will recover once the backend comes up)"
    break
  fi
  # Progress roughly every 10s, driven by elapsed wall-clock time so it
  # can't fire more than once for the same window even if a poll took
  # several seconds.
  if [ "$elapsed" -ge "$next_progress" ]; then
    log "still waiting (${elapsed}s elapsed)"
    next_progress=$((next_progress + 10))
  fi
  sleep 1
done
elapsed=$(( $(date +%s) - start ))
[ "$elapsed" -lt "$WAIT_CEILING" ] && log "API answered after ${elapsed}s - launching the browser"

while true; do
  "$BIN" \
    --app=http://127.0.0.1:8000 \
    --start-fullscreen \
    --kiosk \
    --noerrdialogs \
    --disable-infobars \
    --disable-session-crashed-bubble \
    --check-for-update-interval=31536000 \
    --autoplay-policy=no-user-gesture-required
  sleep 3
done
