# AeroSense firmware

| Folder | Board | Flash with |
|---|---|---|
| `esp32s3_uvc_cam/` | Freenove ESP32-S3-WROOM CAM + OV5640 (×4) | the prebuilt `.bin` (below) or ESP-IDF |
| `servo_aim/` | Arduino Nano/Uno (or an ESP32 with `ESP32Servo`) | Arduino IDE |

## Camera boards (USB webcam firmware)

One codebase; the **role** only sets the USB serial string the Pi uses to tell
boards apart, plus the frame size.

| Role | `.bin` | Where it goes | Mode |
|---|---|---|---|
| left | `aero-left.bin` | left sideline camera | 640×480 @ 15 fps |
| right | `aero-right.bin` | right sideline camera | 640×480 @ 15 fps |
| back | `aero-back.bin` | baseline camera | 640×480 @ 15 fps |
| face | `aero-face.bin` | registration desk | 800×600 @ 10 fps |

**Label each board with its role after flashing.** Two boards with the same
role make udev give both of them the same `/dev/aero-*` name.

### Get the binaries
GitHub → *Actions* → **firmware** → latest green run → download `aero-uvc-bins`.
Or from a terminal: `gh run download -n aero-uvc-bins -R moi-script/machine_vision`.

### Flash (no toolchain needed)
1. Connect the board's **COM/UART** USB-C port to a PC (flashing goes through the UART chip).
2. Open <https://espressif.github.io/esptool-js/> in Chrome, **Connect**, pick the port.
3. Add the role's `.bin` at address **0x0**, click **Program**.
4. Unplug. From now on use the board's **native USB** port (the one that is *not* COM/UART). That port is the webcam.

With esptool instead: `esptool.py --chip esp32s3 write_flash 0x0 aero-left.bin`.

### Check it on the Pi
```bash
v4l2-ctl --list-devices          # shows "AeroSense Cam"
ls -l /dev/aero-*                # after the udev rules are installed
```

### Build it yourself (optional)
ESP-IDF 5.3:
```bash
cd firmware/esp32s3_uvc_cam
idf.py -B build_left -D SDKCONFIG=build_left/sdkconfig \
  -D SDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.role.left" build flash
```

## Servo board (Arduino)
1. Arduino IDE → open `servo_aim/servo_aim.ino` → board **Arduino Nano** (or Uno) → Upload.
2. Wiring: X (pan) servo signal → **D9**, Y (tilt) servo signal → **D10**. Servo power from a separate 5-6 V supply, with its GND tied to the Arduino GND.
3. Test in Serial Monitor at 115200, line ending *Newline*: you should see `READY`. Type `A 60 120` and expect `OK 60 120`.
4. Plug it into the Pi. It appears as `/dev/aero-servo`.
