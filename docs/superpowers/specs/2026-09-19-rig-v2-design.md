# Rig v2 — fixed USB camera rig, face camera, second-screen scoreboard, servo aiming

Date: 2026-09-19
Branch: `feat/rig-v2` in both `thesis_setup` (this repo) and `thesis_ui`
Rollback point: tag `pi-kiosk-v1` in both repos (the working Pi 5 kiosk before this work)

## Goal

Turn the Pi 5 appliance from "one camera + one screen + a stub feeder" into the real
court rig:

| Role | Device | Link to the Pi 5 |
|---|---|---|
| Front court camera | OV9281 USB module | USB |
| Left / Right / Back court cameras | Freenove ESP32-S3-WROOM CAM + OV5640, flashed as USB UVC webcams | USB via a **powered** hub |
| Face-registration camera | same board, role `AERO-FACE` | USB via the hub |
| Operator screen | touchscreen, full app | HDMI 0 |
| Scoreboard | non-touch monitor, view-only | HDMI 1 |
| Feeder aim | 2 servos (X pan, Y tilt) on an Arduino Nano/Uno | USB serial |

The cameras are fixed in place, so a calibration drawn once stays aligned.

## Non-goals (this round)

- Side-camera landed-shuttle detections do **not** feed scoring. They are shown on the
  live view only. Scoring stays on the front camera as today. Multi-view landing fusion
  comes after the landed model is retrained on OV5640 footage.
- No retraining. The OV5640 cameras run the existing landed model until a new dataset exists.
- No feeder launch motor. Only aiming.

## 1. Repo layout, backup and updates

- `pi-kiosk-v1` is an annotated tag on the pre-change `main` of both repos, pushed to
  `machine_vision` (the remote the Pi clones) and to `thesis_ui` origin.
- New files in this repo:

```
firmware/
  esp32s3_uvc_cam/          ESP-IDF project (usb_device_uvc + esp32-camera)
    main/                   one codebase; Kconfig AERO_ROLE picks the USB serial string
    sdkconfig.defaults      board = Freenove ESP32-S3-WROOM CAM, OV5640, MJPEG
    FLASHING.md             web flasher / esptool steps, per-role .bin names
    build/                  prebuilt aero-left.bin, aero-right.bin, aero-back.bin, aero-face.bin
                            (only if the ESP-IDF toolchain builds on the dev box)
  servo_aim/servo_aim.ino   Arduino sketch
setup/deploy/99-aerosense.rules   udev: stable /dev names
setup/deploy/updates/<delivery-date>-rig-v2.md   the Pi update note (dated the day it ships)
```

- **Update note** (one per change, in `setup/deploy/updates/`) covers: what changed and
  the commits in both repos; the exact Pi commands; any extra steps (pip packages, udev
  rules, systemd/kiosk files, settings); how to verify; how to roll back to `pi-kiosk-v1`.
- **Mongo changes are additive only**: a new `face` source doc and new keys in the
  settings doc. The `pi-kiosk-v1` code must still run against the same database after a
  rollback.

### Stable device names (udev)

Each ESP32-S3 is flashed with a unique USB serial string. `99-aerosense.rules` maps:

| USB serial | Symlink |
|---|---|
| `AERO-LEFT` | `/dev/aero-left` |
| `AERO-RIGHT` | `/dev/aero-right` |
| `AERO-BACK` | `/dev/aero-back` |
| `AERO-FACE` | `/dev/aero-face` |
| Arduino (by VID:PID, CH340 or FTDI or ATmega16U2) | `/dev/aero-servo` |

The rule matches `SUBSYSTEM=="video4linux"`, `ATTR{index}=="0"` (UVC exposes a
metadata node too; only the capture node gets the link). The OV9281 keeps its
`/dev/v4l/by-id/...` path. `install.sh` copies the rules file and runs
`udevadm control --reload && udevadm trigger`. The file is pinned `eol=lf`.

## 2. Camera rig

### Firmware (`firmware/esp32s3_uvc_cam`)

- Based on Espressif's `usb_device_uvc` component and `esp32-camera`.
- Camera pins: the Freenove / ESP32-S3-EYE layout: XCLK 15, SIOD 4, SIOC 5, VSYNC 6,
  HREF 7, PCLK 13, D0..D7 = 11, 9, 8, 10, 12, 18, 17, 16, PWDN/RESET unused. The pin
  map is a Kconfig choice with Freenove as the default.
- Sensor outputs JPEG. Frames go straight into UVC MJPEG. Full-speed USB (12 Mbit/s)
  sets the ceiling. Default mode: 640×480 at 15 fps for court roles and 800×600 at 10 fps
  for `AERO-FACE`. Both are Kconfig values.
- Must be connected by the board's **native USB** port, not the UART port.
- USB descriptor: product `AeroSense Cam`, serial = role string.

### Backend (`thesis_setup/setup/app`)

- `sources.CAMERA_IDS` becomes `("front", "left", "right", "back")` for the grid, plus
  `FACE_ID = "face"` accepted by `get`/`set_source`/`open_capture`/`describe`. `face`
  has no bundled default. Until assigned it defaults to device path `/dev/aero-face` on
  Linux and reports `available: false` if missing.
- `_configure_v4l2` stops forcing 1280×800 on every device. It requests MJPG and a
  per-slot size: 1280×800 for `front`, the firmware mode for the ESP32 slots. The
  sizes come from a new `settings.SLOT_FRAME_SIZE` dict.
- **Fixed model map** in `pipeline.py`:
  `SLOT_MODELS = {"front": ["shuttle", "pose"], "left": ["landed"], "right": ["landed"], "back": ["landed"]}`.
  `Worker` takes a list of model keys and runs each on the frame, drawing all results.
  `POST /api/cameras/{id}/start` ignores any `model` field and uses the map. If the front
  worker's measured fps is too low on the Pi, `settings.FRONT_ALTERNATE = True` makes it
  run shuttle and pose on alternate frames instead (default `False`, decided at bring-up).
- **Front-camera ownership.** `engine.py` opens the `front` slot through
  `sources.open_capture("front")` and stops using `CAMERA_INDEX`. While a session runs,
  the pipeline's front worker does not open the device. It serves the engine's latest
  annotated frame from a shared buffer (`app/frame_bus.py`: last JPEG + timestamp per
  slot). When the session ends, the front worker opens the device again.

### UI (`thesis_ui`)

- **Calibration page:** each of the 4 tiles shows the live MJPEG stream for its slot.
  **Freeze** grabs the current frame (existing `get_frame` endpoint), and line drawing and
  commit work on the frozen frame exactly as today. **Unfreeze** returns to live.
- **Live cameras page:** the model dropdown is removed. Each tile shows a fixed label
  (`flying shuttle + pose` / `landed shuttle`) and start/stop.
- Source picker: device list shows the udev names and the `face` slot is excluded
  from the 4-tile grid (assigned in Settings instead).

## 3. Face registration

- New endpoints (in a new router `app/routers/face_cam.py`, replacing `esp32_enroll.py`):
  - `GET /api/face-cam/health` → `{available: bool, source: …}`
  - `GET /api/face-cam/stream` → MJPEG from the `face` slot
  - `POST /api/players/{pid}/enroll` → grabs 3 frames from the `face` slot, same
    averaging and response shape as today
- The `face` slot is opened by one backend owner (a small ref-counted reader) so preview and
  capture never fight over the device.
- `RegisterModal.tsx`: the ESP32 source becomes "Face camera" and points at the new
  stream URL. **Upload photo** and **browser webcam** are unchanged.
- `esp32_camera_client.py`, `ESP32_CAM_IP` and the Wi-Fi sketch dependency are removed on this
  branch (still in `pi-kiosk-v1`).
- **Check-in** stays on the front OV9281 through the existing identity acquisition,
  now opened via the `front` slot. `FACE_MATCH_THRESHOLD` will likely drop to ~0.30
  for the grayscale OV9281. This gets tuned at bring-up.

## 4. Scoreboard on the second monitor

- Route `/#/scoreboard` renders outside the normal app shell: full screen, view-only,
  no navigation. It shows:
  - player name
  - score / shots
  - session timer
  - difficulty
  - last zone fed, on a mini court
  - countdown ring to the next shot, driven by the difficulty interval and reset on each `feeder` event

  Between sessions it shows an idle screen.
- The data comes only from the existing websocket (`shot`, `feeder`, score and session events) plus
  a one-time `GET` of the active session on connect. No new backend state.
- `kiosk-launch.sh` launches two Chromium loops, each with its own `--user-data-dir`:
  app at `http://127.0.0.1:8000` and scoreboard at `http://127.0.0.1:8000/#/scoreboard`,
  each `--kiosk --window-position=X,0`. X offsets come from `xrandr --listmonitors`.
  `/etc/aerosense/displays.conf` can set `APP_OUTPUT` / `SCOREBOARD_OUTPUT` to swap
  them. With only one monitor attached, only the app window starts.

## 5. Servo aiming

### Arduino (`firmware/servo_aim/servo_aim.ino`)

- 115200 baud, line protocol:
  - `A <x> <y>\n` → clamp to `[MIN, MAX]`, move, reply `OK <x> <y>`
  - `C\n` → center, reply `OK <cx> <cy>`
  - `P\n` → reply `PONG`
  - boot → center, print `READY`
  - no command for 60 s → center
- Pins: X servo D9, Y servo D10 (Nano/Uno). `#ifdef ESP32` branch uses `ESP32Servo`
  on GPIO 18/19.
- Servos powered from a separate 5–6 V supply, grounds tied to the Arduino. Never from the
  Arduino 5 V pin.

### Backend (`app/aim.py`)

- `Aimer` opens `/dev/aero-servo` with pyserial (added to `requirements.txt`), waits
  for `READY`, and reconnects lazily. `aim(zone)`:
  - looks up `zone_angles[zone]` from the settings doc
  - adds `uniform(-jitter, +jitter)` to each axis
  - clamps to `[angle_min, angle_max]`
  - sends `A x y`, returns the angles actually sent
- Serial failure → log a warning and return `None`. The drill never stops because of the servo.
- `engine._fire_feeder(zone)` calls `aimer.aim(zone)` and broadcasts
  `{"type": "feeder", "zone", "x", "y", "at"}`. Zone choice stays the engine's existing
  `random.choice(PLAYER_ZONES)`. The interval stays `DIFFICULTY[...]["interval"]`.
- Settings doc keys (additive): `aim.zone_angles` (6 zones → `{x, y}`, default all 90/90),
  `aim.jitter_deg` (default 4), `aim.angle_min` (30), `aim.angle_max` (150).
- Endpoints: `GET/PUT /api/aim/config`, `POST /api/aim/move {x, y}` (live nudge),
  `POST /api/aim/center`, `GET /api/aim/health`.

### UI

- Settings → **Aim calibration** panel:
  - zone picker (the 6 zones on a mini court)
  - X/Y readouts with ±1 and ±5 buttons that move the servos live
  - Save for the zone
  - jitter slider, min/max angle fields
  - servo connection status

## Error handling

- Missing camera device → the tile shows "unavailable (/dev/aero-left missing)". Other
  slots keep running.
- Missing face camera → the register modal disables "Face camera" with the reason, and upload
  still works.
- Missing servo → the drill runs and the Settings panel shows disconnected.
- Second monitor missing → only the app window launches.

## Testing

All backend tests run as `MONGO_DB=aerosense_test python -m pytest`, because the plain run wipes live data.

- `aim.py`: seeded jitter within bounds, clamping, protocol lines, fake serial port,
  serial failure returns `None` without raising.
- `sources`: `face` slot get/set/describe, per-slot frame size.
- `pipeline`: slot→models map. `start` ignores a supplied model. Multi-model worker runs
  each model (models stubbed).
- `frame_bus` / front ownership: while the engine holds `front`, the worker reads the bus
  and does not open a capture.
- `face_cam` router: health/stream/enroll with a stubbed capture.
- Firmware: build check only on the dev box. Behaviour checked on hardware.
- UI: `npm run build` + lint, plus a manual `/scoreboard` check against the simulator.
- On the Pi, the update note carries a hardware checklist: the 5 udev names exist, 4 live
  calibration tiles, face preview + enroll, servo `READY` and a zone move, both screens.

## Delivery order

1. Firmware (UVC project + servo sketch) and udev rules
2. Camera rig backend + calibration/live UI
3. Face camera
4. Scoreboard + dual kiosk
5. Servo aiming backend + Settings panel
6. Build UI bundle, update note, merge
