# Running AeroSense on Windows

Three ways, from least to most setup.

## 1. The packaged app (`window_v` release)

Download `AeroSense-window_v-win64.zip` from the repository's Releases page,
unzip it anywhere, and run `AeroSense.exe`. A console window opens with the
status, and your browser opens on the app once the API answers.

**You must install MongoDB first.** It is not inside the bundle: it would add
another ~50 MB and carries its own licence. Get MongoDB Community Edition
from <https://www.mongodb.com/try/download/community>, install it with the
"run as a service" option so it starts with Windows, and then launch
AeroSense. If Mongo is not reachable the app says so and exits rather than
starting up broken — every player, session, calibration and camera assignment
lives in that database.

Two other things the console will tell you plainly if they happen:

- **Port 8000 is in use.** Something else is already serving there. Find it
  with `netstat -ano | findstr :8000`, then `tasklist /fi "pid eq <PID>"`.
- **The browser did not open.** Go to <http://localhost:8000> yourself.

The bundle is around 1.5 GB unzipped, most of it PyTorch and OpenVINO.

## 2. From a source checkout, with a desktop shortcut

If you have the repository and Python, this is lighter than the packaged
build and is what to use while developing.

    powershell -ExecutionPolicy Bypass -File scripts\install-windows-shortcut.ps1

That creates Desktop and Start-menu shortcuts running
`scripts\start-aerosense.bat`, which starts the backend if it is not already
answering, waits for `/api/health`, and opens the app window.

You can also install it as a Chrome app: with the server running, open
<http://localhost:8000> and use the install button in the address bar. That
gives a standalone window, a Start-menu entry and a taskbar icon. It cannot
start the backend itself, which is what the shortcut above is for.

## 3. Plain development

    python run_server.py            # API on :8000, serving the built UI at /
    npm run dev                     # in the UI repo: Vite on :5173

Use the Vite server while changing frontend code. The bundle committed at
`app/static/ui` is a build artifact — after a UI change run
`scripts\build_ui.ps1`, or `:8000` will keep serving the previous UI.

## Rebuilding the packaged app

    powershell -ExecutionPolicy Bypass -File scripts\build_ui.ps1          # UI first
    powershell -ExecutionPolicy Bypass -File scripts\build_windows_exe.ps1 # then the bundle

The bundle embeds `app/static/ui`, so building the UI first is not optional —
skip it and you ship yesterday's interface inside the exe with no way to tell
from the outside. Output lands in `dist\` (git-ignored: it is far over
GitHub's 100 MB per-file limit and is published as a Release asset).

`scripts\aerosense.spec` is the recipe. It is `onedir` rather than `onefile`
on purpose: a onefile build unpacks its whole payload to a temp directory on
every launch, which for 1.5 GB means minutes of waiting each time.

## Build prerequisites and a memory warning

The packaged build needs PyInstaller (`pip install pyinstaller`) and, more
importantly, **several GB of free RAM**. Bundling PyTorch is memory-hungry:
the first attempt at this build was killed by Windows partway through torch's
analysis on a machine with 16 GB installed but only 0.8 GB free, with editors,
browsers and WSL open. If the build dies with no error of its own, that is
almost certainly why — close what you can spare (`wsl --shutdown` alone
recovers a few hundred MB) and run it again.
