# AeroSense as a Raspberry Pi 5 desktop app — design

Date: 2026-09-11
Status: approved, ready for implementation planning

## Goal

Run the whole AeroSense stack — React UI, FastAPI backend, MongoDB, and the
four-camera YOLO pipeline — self-contained on a Raspberry Pi 5. The user
switches the Pi on and the app is on screen fullscreen, with no terminal, no
browser chrome, and no second machine involved.

Today the two halves run separately on Windows: the Vite dev server on
`localhost:5173` and `uvicorn app.server:app` on `localhost:8000`, wired by
`VITE_API_URL` and a permissive localhost CORS regex.

## Non-goals

- Real-time (30 FPS) multi-camera inference. See "Throughput" below.
- Replacing MongoDB. Mongo 7.0 has aarch64 packages and the Pi 5's Cortex-A76
  is ARMv8.2-A, which satisfies Mongo 6+ (a Pi 4's A72 would not).
- A windowed native app (Tauri/Electron). Rejected: Electron's bundled Chromium
  costs 150-300 MB RSS on a box that also runs four camera pipelines plus YOLO;
  Tauri adds a Rust cross-compile toolchain for no user-visible gain over kiosk.
- LAN or multi-user access. This is a single-screen appliance.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| App shell | Chromium `--kiosk` + systemd | Lightest on Pi RAM; UI code needs no shell-specific changes |
| Serving | FastAPI serves the built React bundle | One process, one port, one origin — CORS and the API-base indirection both stop mattering in production |
| UI build location | On Windows; `dist/` committed into the backend repo | Pi needs no Node and no slow Vite build; update is `git pull` + restart |
| Inference backend | NCNN on aarch64, OpenVINO on x86 | OpenVINO is Intel-oriented; NCNN is the ultralytics-recommended Pi export |
| Database | MongoDB 7.0 arm64 on the Pi | Zero application code change |

## Runtime topology

```
boot
 |- mongod.service       MongoDB 7.0 arm64, 127.0.0.1:27017, db "aerosense"
 |- aerosense.service    uvicorn app.server:app --host 127.0.0.1 --port 8000
 |     GET /api/*, /ws   existing routers, unchanged
 |     GET /*            app/static/ui/ (built React bundle)
 |- kiosk.service        chromium --kiosk --app=http://localhost:8000
```

`aerosense.service` binds `127.0.0.1`, narrowing today's `0.0.0.0`. This is
safe: the ESP32-CAM integration is entirely outbound — `app/esp32_camera_client.py`
issues `httpx` GETs to `http://{ESP32_CAM_IP}/capture`, and nothing POSTs
inward. `stream_url()` hands the browser `http://{ip}:81/stream`, which the
Chromium instance on the Pi fetches directly over the LAN; that is unaffected
by our bind address.

systemd ordering: `aerosense.service` gets `After=mongod.service` and
`Wants=mongod.service`; `kiosk.service` gets `After=aerosense.service` plus
`graphical-session.target`. `aerosense.service` uses `Restart=on-failure`.

## Component changes

### 1. `app/server.py` — serve the UI

Mount the bundle after every `include_router` call, so `/api/*` and `/ws` keep
priority and only unmatched paths fall through to static assets:

```python
UI_DIR = Path(__file__).parent / "static" / "ui"
if UI_DIR.is_dir():
    app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
```

The `is_dir()` guard keeps the Windows dev flow (Vite on 5173, no bundle
present) working unchanged.

No SPA fallback is required. `src/App.tsx` switches on `state.ui.activePage`
and the project has no react-router dependency, so the app lives at exactly one
URL and `html=True` is sufficient. **If client-side routing is ever added, this
mount must grow a 404-to-`index.html` handler.**

### 2. UI — same-origin API base

- Add `.env.production` with `VITE_API_URL=` (empty). `src/lib/api.ts` reads
  `import.meta.env.VITE_API_URL ?? "http://localhost:8000"`; `??` is nullish
  coalescing, so an empty string survives and every fetch becomes a relative
  same-origin path.
- `src/lib/socket.ts` builds `API_URL.replace(/^http/, "ws") + "/ws"`, which
  with an empty base degrades to the bare `"/ws"`. Change it to derive the
  origin explicitly when `API_URL` is empty, from `location.protocol` and
  `location.host`. Do not depend on relative-WebSocket-URL resolution.

### 3. `app/pipeline.py` — platform-aware inference backend

Four call sites currently default to `backend="openvino"` (including `start()`
and the `backend` query parameter in `app/routers/cameras.py`). Replace the
literal with a resolved constant:

```python
DEFAULT_BACKEND = "openvino" if platform.machine() in ("AMD64", "x86_64") else "ncnn"
```

Generalise `_load()`'s export-cache path from the OpenVINO-specific
`f"{stem}_{imgsz}_openvino_model"` to `f"{stem}_{imgsz}_{backend}_model"`, so
NCNN exports cache on first use the same way OpenVINO ones do. The committed
`yolov8n-pose_{448,512,640}_openvino_model` directories stay valid on Windows.

Per-model settings on the Pi, following `START.md`'s own measurements:
`landed` at imgsz 640 (START.md records 18.3 ms vs 54.9 ms at 1280, still
firing on 98% of frames), `shuttle` dropped from 1280 to 640, `pose` unchanged
at 640. `config/settings.py` already carries `FPS_TARGET = 10` with the comment
"target FPS for Raspberry Pi later".

### 4. `app/sources.py` — Linux camera handling

`open_capture()` already branches on `sys.platform == "win32"` for `CAP_DSHOW`
and falls through to the V4L2 default, so no restructuring is needed. Two
additions on the non-Windows path:

- **MJPG + explicit resolution.** Four uncompressed 1280x800 streams exceed the
  Pi 5's shared USB3 bandwidth. Set `CAP_PROP_FOURCC` to `MJPG` and the frame
  width/height from `settings` before first read.
- **Stable device identity.** `list_devices()` probes integer indices 0..3.
  udev reorders those across reboots — the same class of bug already recorded
  for Windows in the camera-selection notes. On Linux, enumerate
  `/dev/v4l/by-id/*` symlinks and store that path as the source `path` rather
  than a volatile `index`. `set_source()` already accepts a `path` kind, so the
  Mongo schema absorbs this without migration.

### 5. `requirements.txt` — close the httpx gap

`app/esp32_camera_client.py` imports `httpx`, which is absent from
`requirements.txt` and currently resolves only transitively. A clean Pi venv
would fail at ESP32 enrollment. Add it explicitly.

Also prefer Debian's `python3-opencv` over a pip `opencv-python` build on the
Pi, with a pip fallback — a source build on the Pi is slow and error-prone.
`install.sh` owns this choice; `requirements.txt` keeps `opencv-python` for
Windows.

### 6. `deploy/` — provisioning

New directory in `setup/`:

- `aerosense.service`, `kiosk.service` — unit files (mongod's ships with Mongo)
- `install.sh` — create venv, install deps (apt-first for OpenCV), run
  `python fetch_face_models.py` for the gitignored YuNet/SFace weights, warm the
  NCNN exports so first launch is not a multi-minute stall, install and enable
  the units
- `PI-SETUP.md` — Pi OS 64-bit install, Mongo apt repo, Wi-Fi onto the same LAN
  as the ESP32-CAM, and the per-camera `by-id` mapping for front/left/right/back

### 7. `scripts/build_ui.ps1` — release step

Runs `npm run build` in `C:\thesis_ui\badminton`, then wipes and repopulates
`C:\thesis\setup\app\static\ui`. Update cycle:

```
build_ui.ps1  ->  git commit  ->  (Pi) git pull  ->  systemctl restart aerosense
```

The committed bundle is a build artifact in git. `build_ui.ps1` is the only
sanctioned way to update it; hand-editing `app/static/ui` is a mistake.

## Data flow

Unchanged from today. The Pi swap touches transport and process supervision
only, not application logic:

```
USB cameras (v4l2) -> sources.open_capture -> virtual_camera -> pipeline (NCNN)
                                                                    |
                                        engine -> events.hub -> /ws -> UI
                                        engine -> Mongo (players, sessions,
                                                  calibrations, settings)
ESP32-CAM  <-- httpx GET /capture --  esp32_enroll -> face.py -> Mongo
```

## Error handling

- **Bundle missing.** The `is_dir()` guard means `/` 404s while `/api` still
  works; `install.sh` verifies `app/static/ui/index.html` exists and fails loudly.
- **Mongo down at boot.** `db.ping()` already backs `/api/health`;
  `Restart=on-failure` plus `After=mongod.service` covers the ordering race.
- **NCNN export missing or failed.** `_load()` raises on first use; `install.sh`
  warms exports so this surfaces at install time, not mid-drill.
- **Camera absent.** `open_capture` already raises and
  `/api/cameras/{id}/frame` already returns 503; the UI already renders that.
- **ESP32 unreachable.** `ESP32CaptureError` already handled.

## Testing

1. **Windows regression first.** After each change, the existing Windows dev
   flow (Vite 5173 + uvicorn 8000) must still work. This is the guard against
   the platform-switch changes breaking the machine you develop on.
2. **Static mount, Windows.** Build the bundle, hit `http://localhost:8000/`,
   confirm the UI loads and `/api/health` still answers.
3. **Pi bring-up.** `install.sh` on a fresh Pi OS 64-bit image; `/api/health`
   reports `mongo: true`, engine state, and `face: true`.
4. **Per-camera smoke test.** `/api/cameras/{id}/frame` for all four IDs.
5. **Throughput measurement.** Record actual ms/frame per model per camera on
   the Pi. This is a deliverable, not a pass/fail gate — it is the input to the
   throughput decision below.
6. **Cold boot.** Power-cycle the Pi; the UI must reach the screen fullscreen
   with no interaction.

## Throughput — known open risk

Packaging does not make the Pi fast. Expect roughly 120-180 ms/frame for pose
and 150-250 ms/frame for shuttle at imgsz 640 on NCNN/CPU, i.e. a few FPS per
camera against the ~86 ms/frame the x86 laptop achieves for shuttle at 1280
under OpenVINO. Levers, in order of cost:

1. `pipeline.py`'s existing court crop — START.md already measures pose at
   30 ms on a cropped frame versus roughly twice that on full frames
2. `FPS_TARGET = 10`, and staggering the four cameras rather than running them
   concurrently
3. Raspberry Pi AI HAT+ (Hailo-8). Separate workstream: requires recompiling
   every weight to `.hef` through Hailo's x86 Docker toolchain and adding a
   third pipeline backend. Out of scope here.

The design's job is to make this measurable on real hardware. Decide about the
HAT with numbers from step 5, not before.
