# Running AeroSense on a Raspberry Pi 5

## What you need

- Raspberry Pi 5, 8 GB. The 4 GB model has not been tried and mongod plus four
  camera pipelines plus YOLO is not a small footprint.
- Raspberry Pi OS **64-bit** (bookworm). 32-bit will not work: MongoDB needs
  arm64, and NCNN wheels are aarch64.
- Active cooling. Sustained CPU inference on all four cores will thermally
  throttle a bare Pi 5, and throttling looks exactly like "the app got slow".
- The four court USB cameras, plus the ESP32-CAM on the same LAN.
- A display on HDMI0.

## 1. Base image

Flash Pi OS 64-bit with Raspberry Pi Imager. In the Imager's advanced options
set the hostname, enable SSH, and create the user **`aerosense`** — the backend
systemd unit runs as that user, and the kiosk browser is autostarted from
that user's desktop session. If you use a different name, set `RUN_USER` when
running `install.sh` and edit `aerosense.service` too.

Then, on the Pi:

    sudo raspi-config
      System Options  -> Boot / Auto Login -> Desktop Autologin
      Advanced        -> Wayland           -> X11

X11 rather than Wayland: the kiosk autostart entry runs `chromium` under the
X11 session that LightDM's autologin brings up. (`DISPLAY`/`XAUTHORITY` come
from that session automatically — nothing in this repo has to guess them.)

## 2. Clone and install

    sudo apt-get install -y git git-lfs
    sudo mkdir -p /opt/aerosense && sudo chown aerosense /opt/aerosense
    git clone <your remote> /opt/aerosense
    cd /opt/aerosense/setup
    sudo ./deploy/install.sh

`/opt/aerosense` is baked into `aerosense.service`'s `WorkingDirectory`. To put
it elsewhere, edit that unit file.

`install.sh` also detects whether this image's browser binary is
`/usr/bin/chromium-browser` or `/usr/bin/chromium` and writes the kiosk
autostart entry with the right one — see the troubleshooting table below if
that ever looks wrong.

## 3. Roboflow API key (optional — not needed by default)

`install.sh` writes `/opt/aerosense/setup/.env` with an **empty**
`ROBOFLOW_API_KEY`. The shipped configuration
(`config/settings.py: SHUTTLE_SOURCE = "local"`) detects the shuttle from the
committed local weights file and needs no key at all — this is what keeps the
appliance working with no internet on court.

Only fill this in if you deliberately switch `SHUTTLE_SOURCE` to
`"serverless"` to use the Roboflow cloud workflow instead:

    sudo -u aerosense nano /opt/aerosense/setup/.env
    # set ROBOFLOW_API_KEY=<your key>

`.env` is gitignored, so a real key can never be committed — `install.sh` has
no way to fill it in for you, which is why it is left empty.

## 4. Assign the four cameras

Unlike Windows, capture indices are not stable across reboots here, so sources
are stored as `/dev/v4l/by-id/...` paths instead. List what is attached:

    ls -l /dev/v4l/by-id/
    v4l2-ctl --list-devices

Then in the UI (Cameras page), probe for devices and assign one to each of
`front`, `left`, `right`, `back`. The assignment is stored in Mongo and
survives reboots. Record the mapping here for your own build:

| Slot  | by-id path | Physical camera |
|-------|-----------|-----------------|
| front |           |                 |
| left  |           |                 |
| right |           |                 |
| back  |           |                 |

Then calibrate each camera's court corners as usual.

## 5. ESP32-CAM

`config/settings.py` pins `ESP32_CAM_IP = "10.200.33.50"`. The Pi must be on
that same subnet. Check with:

    curl -o /dev/null -w '%{http_code}\n' http://10.200.33.50/capture

## 6. Reboot

    sudo reboot

The Pi should come up into the fullscreen UI with no interaction.

## Operating it

The backend is a systemd service; the kiosk browser is a per-user XDG
autostart entry (not a systemd unit — `graphical-session.target` only exists
in the systemd *user* manager, so a system-level kiosk unit enabled against it
would silently never start; the fullscreen browser is launched by the desktop
session itself instead).

    systemctl status aerosense mongod    # backend + database
    journalctl -u aerosense -f           # backend logs
    pgrep -af chromium                   # is the kiosk browser actually running?

    # reload just the display, without a full reboot
    sudo -u aerosense pkill chromium     # kiosk-launch.sh relaunches it automatically

    # ship an update (UI bundle is built on the Windows box)
    cd /opt/aerosense && git pull && git lfs pull
    sudo systemctl restart aerosense

There is no `systemctl status kiosk` — that would be checking a unit that
does not exist under this mechanism. Use `pgrep -af chromium` or look at the
desktop directly.

## Troubleshooting

| Symptom | Check |
|---|---|
| Black screen, no UI | Is autologin to desktop on? Is it X11, not Wayland? Then check `cat /home/aerosense/.config/autostart/kiosk.desktop` exists and `pgrep -af chromium` |
| Black screen, backend is healthy | Wrong chromium binary. `ls -l /usr/bin/chromium-browser /usr/bin/chromium` and compare against the `Exec=` line in `/home/aerosense/.config/autostart/kiosk.desktop` — re-run `install.sh` to regenerate it |
| UI loads, no data | `curl localhost:8000/api/health`; `journalctl -u aerosense -n 50` |
| `"mongo": false` in health | `systemctl status mongod`; on a Pi 4 Mongo 7 cannot run at all |
| Shuttle detection never fires and you switched to serverless | `ROBOFLOW_API_KEY` in `/opt/aerosense/setup/.env` is still empty — see step 3 |
| Only 1-2 cameras give frames | USB bandwidth. Confirm MJPG: `v4l2-ctl -d <dev> --list-formats`. Spread cameras across both USB3 and USB2 ports |
| Cameras swapped after reboot | A slot is stored as an `index`, not a by-id `path`. Reassign it in the UI |
| Everything is slow, then slower | Thermal throttling: `vcgencmd measure_temp`, `vcgencmd get_throttled` (nonzero = throttled) |
| Live-detection view (Cameras page) stalls for minutes the first time | Its NCNN export was not warmed. Re-run `install.sh` |
