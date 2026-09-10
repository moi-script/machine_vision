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
