# 2026-09-24 - Wi-Fi ESP32-CAM back as a second face camera

## What changed
- `/api/face-cam/*` serves either the USB `aero-face` board or the Wi-Fi
  ESP32-CAM. `FACE_CAM_SOURCE` = `"auto"` (default: USB first, then Wi-Fi),
  `"usb"` or `"wifi"`. No UI rebuild; `/health` now also returns `mode`.
- New `app/esp32_camera_client.py`; `ESP32_CAM_IP` / `_TIMEOUT_S` /
  `_STREAM_PORT` are back in `config/settings.py`.
- USB behaviour is unchanged.

## Commands
    cd /opt/aerosense && git pull && sudo setup/deploy/install.sh
    # set ESP32_CAM_IP in setup/config/settings.py to the board's address, then
    sudo systemctl restart aerosense

## Checks
- [ ] `curl -s http://<esp-ip>/health` -> 200 from the Pi
- [ ] USB board unplugged: `curl -s localhost:8000/api/face-cam/health` shows `"mode":"wifi"`
- [ ] USB board plugged in: same call shows `"mode":"usb"`
- [ ] Register a player from each board; both enroll
