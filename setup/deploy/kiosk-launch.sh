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
elapsed=0
until curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; do
  if [ "$elapsed" -ge "$WAIT_CEILING" ]; then
    log "API never answered after ${WAIT_CEILING}s - launching the browser anyway (it will show an error page; the restart loop and a reload will recover once the backend comes up)"
    break
  fi
  sleep 1
  elapsed=$((elapsed + 1))
  # Progress line every 10s so a hang is diagnosable, not mute.
  [ $((elapsed % 10)) -eq 0 ] && log "still waiting (${elapsed}s elapsed)"
done
[ "$elapsed" -lt "$WAIT_CEILING" ] && log "API answered after ${elapsed}s - launching the browser"

while true; do
  "$BIN" \
    --app=http://localhost:8000 \
    --start-fullscreen \
    --kiosk \
    --noerrdialogs \
    --disable-infobars \
    --disable-session-crashed-bubble \
    --check-for-update-interval=31536000 \
    --autoplay-policy=no-user-gesture-required
  sleep 3
done
