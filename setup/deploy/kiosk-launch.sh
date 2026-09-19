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

# ── two screens ──────────────────────────────────────────────
# Touchscreen = the full app; the non-touch monitor = the view-only scoreboard.
APP_OUTPUT=""; SCOREBOARD_OUTPUT=""; TOUCH_DEVICE=""
[ -r /etc/aerosense/displays.conf ] && . /etc/aerosense/displays.conf

# "NAME X" per monitor, in xrandr's order. A line looks like:
#  0: +*HDMI-A-1 1920/527x1080/296+0+0  HDMI-A-1
monitors() {
  xrandr --listmonitors 2>/dev/null | awk 'NR>1 {split($3,g,"+"); print $4, g[2]}'
}
offset_of() { monitors | awk -v n="$1" '$1==n {print $2}'; }

MONS="$(monitors)"
[ -z "$APP_OUTPUT" ] && APP_OUTPUT="$(printf '%s\n' "$MONS" | awk 'NR==1{print $1}')"
[ -z "$SCOREBOARD_OUTPUT" ] && SCOREBOARD_OUTPUT="$(printf '%s\n' "$MONS" | awk -v a="$APP_OUTPUT" '$1!=a{print $1; exit}')"
APP_X="$(offset_of "$APP_OUTPUT")"; APP_X="${APP_X:-0}"
log "monitors: $(printf '%s' "$MONS" | tr '\n' ';') app=$APP_OUTPUT@$APP_X scoreboard=${SCOREBOARD_OUTPUT:-none}"

# With two monitors X spans touch over both; pin it to the app screen.
[ -z "$TOUCH_DEVICE" ] && TOUCH_DEVICE="$(xinput list --name-only 2>/dev/null | grep -i -m1 touch || true)"
if [ -n "$TOUCH_DEVICE" ] && [ -n "$APP_OUTPUT" ]; then
  xinput map-to-output "$TOUCH_DEVICE" "$APP_OUTPUT" && log "touch '$TOUCH_DEVICE' -> $APP_OUTPUT"
fi

# Each window needs its own profile dir, or Chromium hands the second URL to
# the first process and ignores its position/kiosk flags.
kiosk_loop() {   # $1 url  $2 x-offset  $3 profile-name
  while true; do
    "$BIN" \
      --app="$1" \
      --user-data-dir="${XDG_CONFIG_HOME:-$HOME/.config}/aerosense-$3" \
      --window-position="$2,0" \
      --start-fullscreen \
      --kiosk \
      --noerrdialogs \
      --disable-infobars \
      --disable-session-crashed-bubble \
      --check-for-update-interval=31536000 \
      --autoplay-policy=no-user-gesture-required
    sleep 3
  done
}

if [ -n "$SCOREBOARD_OUTPUT" ]; then
  SB_X="$(offset_of "$SCOREBOARD_OUTPUT")"
  kiosk_loop "http://127.0.0.1:8000/#/scoreboard" "${SB_X:-0}" scoreboard &
fi
kiosk_loop "http://127.0.0.1:8000" "$APP_X" app
