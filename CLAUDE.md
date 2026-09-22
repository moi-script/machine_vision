# AeroSense — orientation for Claude Code

Badminton training system: five USB cameras feed a YOLO pipeline (shuttle,
pose, court lines, face recognition), FastAPI stores drills in MongoDB, and a
React UI drives it all. It runs as a kiosk appliance on a Raspberry Pi 5 and as
a normal app on the Windows dev box.

The rig is **v2** (shipped 2026-09-22): an OV9281 `front` camera plus four
ESP32-S3 USB webcams on udev-stable names — `/dev/aero-left`, `-right`,
`-back`, `-face` — and servo aiming through an Arduino at `/dev/aero-servo`.
Each device has exactly one reader. Models are fixed per slot: `front` =
shuttle + pose, `left`/`right`/`back` = landed shuttle, `face` = raw. The old
Wi-Fi ESP32-CAM enrollment is gone; registration uses the USB face camera. A
second monitor shows `/#/scoreboard`.

## Where things are

| Path | What |
|---|---|
| `setup/` | The whole backend. **Run every Python command from here** — `config/settings.py` resolves relative weight paths from this directory and nowhere else. |
| `setup/app/` | FastAPI app, routers, websockets, `ui_static.py` (serves the UI) |
| `setup/pipeline.py`, `setup/models/` | Inference. Weights are Git LFS. |
| `setup/deploy/` | Raspberry Pi: `install.sh`, `aerosense.service`, kiosk entry, **`PI-SETUP.md`** |
| `setup/scripts/` | `build_ui.ps1` (UI bundle) and the Windows packaging scripts |
| `setup/docs/RUN-WINDOWS.md` | Running it on Windows |
| `setup/app/static/ui/` | **Built** React bundle, committed. Never hand-edit. |
| `firmware/` | ESP32-S3 UVC camera firmware (ESP-IDF, built by CI) and the Arduino servo sketch — see `firmware/README.md` |
| `setup/deploy/updates/` | One note per Pi upgrade: what changed, the commands, the checks |
| `docs/superpowers/plans/` | How things were built — historical record, **not instructions**. Check the top of a plan for a STATUS banner before acting on it; a shipped plan is not a to-do list, whatever its "for agentic workers" header says. Unticked boxes do not mean unfinished — verify against the source. |

## The frontend is a separate repo

Source: **https://github.com/moi-script/thesis_ui** — the React app is in that
repo's `badminton/` subdirectory; on the Windows dev box it is checked out at
`C:\thesis_ui`.

What lives in *this* repo is the compiled output at `setup/app/static/ui/`,
which FastAPI mounts at `/`. That is deliberate: **one clone of this repo gets
both halves of the app**, so the Raspberry Pi runs the UI with no Node, no npm
and no Vite installed. Clone `thesis_ui` only to *change* the UI, and only on a
machine with Node.

The round trip for a UI change, all of it on the Windows dev box:

    cd C:\thesis_ui\badminton && npm run dev          # develop against :5173
    cd C:\thesis\setup && pwsh scripts/build_ui.ps1   # rebuild -> app/static/ui
    git add setup/app/static/ui && git commit && git push

A UI change that skips `build_ui.ps1` is not shipped, however green the tests
are. On the Pi, `git pull` picks up the new bundle; nothing is built there.

## If you are working on the Raspberry Pi

Follow **`setup/deploy/PI-SETUP.md`** top to bottom — it is the procedure, not
a suggestion. `deploy/install.sh` is idempotent, so re-running it after a
`git pull` is the normal repair move.

Do not `npm install` or try to build the UI on the Pi. If
`app/static/ui/index.html` is missing, `install.sh` stops on purpose; the fix
is on the Windows box (rebuild, commit, push), then `git pull` here.

## Dev flow on Windows

    cd C:\thesis\setup
    python run_server.py     # API + the committed bundle on :8000
    # and, when working on the UI:
    cd C:\thesis_ui\badminton && npm run dev    # :5173, talks to :8000

See `setup/docs/RUN-WINDOWS.md` for the packaged-app route.

## Conventions

- MongoDB is pinned to `mongodb://localhost:27017/`, database `aerosense`, in
  `setup/.env`. Drift here has previously made registered players vanish.
- `setup/.env` is gitignored; it holds the Roboflow key (optional).
- Model weights are Git LFS. A ~130-byte `.pt` is an unfetched pointer, not a
  corrupt model — run `git lfs pull`.
- Deploy scripts and systemd units are forced to LF in `.gitattributes`. A CRLF
  in them breaks bash and systemd on the Pi.
- `main` is the deployable branch; it is what the Pi clones.
