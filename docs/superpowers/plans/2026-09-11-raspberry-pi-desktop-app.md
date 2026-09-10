# AeroSense Raspberry Pi 5 Desktop App — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the whole AeroSense stack (React UI, FastAPI, MongoDB, four-camera YOLO pipeline) self-contained on a Raspberry Pi 5, launching fullscreen on boot with no terminal and no second machine.

**Architecture:** FastAPI serves the pre-built React bundle from `app/static/ui`, so the app is one process on one origin at `http://localhost:8000`. Three systemd units (mongod, uvicorn, Chromium kiosk) bring it up on boot. Inference switches from OpenVINO to NCNN on aarch64 via a platform check. The UI is built on Windows and committed; the Pi only pulls and restarts.

**Tech Stack:** Python 3.11 (Pi OS bookworm) / 3.13 (Windows dev), FastAPI, uvicorn, pymongo, MongoDB 7.0 arm64, OpenCV (V4L2), ultralytics + NCNN, React 19 + Vite 8, systemd, Chromium.

**Spec:** `docs/superpowers/specs/2026-09-11-raspberry-pi-desktop-app-design.md`

## Global Constraints

- **The Windows dev flow must keep working after every task.** Vite on `localhost:5173` plus `python run_server.py` on `localhost:8000` is the machine this is developed on. Every task's verification includes it.
- Run all Python commands from `C:\thesis\setup` (on the Pi: the repo's `setup/` directory). `config/settings.py:41` documents that relative weight paths only resolve from there.
- Repo root for git operations is `C:\thesis`; the backend lives in `setup/`. The UI is a **separate** repo at `C:\thesis_ui\badminton`.
- Target platform: Raspberry Pi OS 64-bit (Debian bookworm), aarch64, Pi 5 (Cortex-A76, ARMv8.2-A).
- MongoDB 7.0 arm64, `mongodb://localhost:27017/`, database `aerosense`. These values are pinned in `setup/.env` and must not drift — drift previously caused registered players to vanish after restart.
- `aerosense.service` binds `127.0.0.1:8000`, not `0.0.0.0`.
- Pi inference sizes: `shuttle` imgsz 640 (down from 1280), `landed` imgsz 640, `pose` imgsz 640. `FPS_TARGET = 10`.
- Git LFS tracks `setup/models/*.pt` and `setup/models/*.onnx` (`.gitattributes`). Anything that must reach the Pi through git has to live under `setup/models/`.
- Tests are pytest, run from `C:\thesis\setup` as `python -m pytest`. Test files begin with the `sys.path.insert` preamble used by every existing test in `tests/`.

---

### Task 1: Deliver the production weights through Git LFS

The three weights the pipeline actually loads are untracked, so a `git pull` on the Pi delivers no models. `.gitignore:16` ignores `runs/`, and `*.pt` is ignored with only `!models/*.pt` re-included. Moving them under `setup/models/` puts them on the LFS path already used by `models/shuttlecock.pt` and the face ONNX models.

**Files:**
- Create: `setup/models/shuttle_clear_badminton_p2.pt` (moved, 6.5 MB)
- Create: `setup/models/shuttle_lines_stock_n.pt` (moved, 6.3 MB)
- Create: `setup/models/yolov8n-pose.pt` (moved, 6.8 MB)
- Modify: `setup/app/pipeline.py` — the `MODELS` dict `weights` values (lines ~44-58)
- Modify: `setup/START.md` — the weights table and the "standalone scripts default to OLD weights" warning
- Test: `setup/tests/test_pipeline_models.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `MODELS["shuttle"]["weights"] == "models/shuttle_clear_badminton_p2.pt"`, `MODELS["landed"]["weights"] == "models/shuttle_lines_stock_n.pt"`, `MODELS["pose"]["weights"] == "models/yolov8n-pose.pt"`. Task 8's `install.sh` relies on these being LFS-delivered.

- [ ] **Step 1: Write the failing test**

Create `setup/tests/test_pipeline_models.py`:

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import pipeline  # noqa: E402


def test_every_model_weight_lives_under_models_dir():
    """Weights must be under models/ to be LFS-tracked and reach the Pi.

    runs/ is gitignored, so a weight referenced there exists only on the
    machine that trained it.
    """
    for key, cfg in pipeline.MODELS.items():
        if key == "none":
            continue
        assert cfg["weights"].startswith("models/"), \
            f"{key} weights {cfg['weights']!r} are outside models/ and will not reach the Pi"


def test_every_model_weight_file_exists():
    for key, cfg in pipeline.MODELS.items():
        if key == "none":
            continue
        path = pipeline._resolve(cfg["weights"])
        assert os.path.exists(path), f"{key} weights missing at {path}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline_models.py -v`
Expected: `test_every_model_weight_lives_under_models_dir` FAILS with "shuttle weights 'runs/clear_badminton/p2-native/weights/best.pt' are outside models/".

- [ ] **Step 3: Move the weight files**

```bash
cd /c/thesis/setup
cp runs/clear_badminton/p2-native/weights/best.pt models/shuttle_clear_badminton_p2.pt
cp runs/shuttle_lines/stock-n/weights/best.pt     models/shuttle_lines_stock_n.pt
cp yolov8n-pose.pt                                 models/yolov8n-pose.pt
```

Copy, do not move: the `runs/` originals stay as the training record, and
`scripts/feeder_court/*.py` still reference them.

- [ ] **Step 4: Update the MODELS dict**

In `setup/app/pipeline.py`, change the three `weights` values:

```python
    "shuttle": {
        "weights": "models/shuttle_clear_badminton_p2.pt",
        "imgsz": 1280, "task": "detect", "conf": 0.25, "max_side": 60,
        "label": "flying shuttle",
    },
    "landed": {
        "weights": "models/shuttle_lines_stock_n.pt",
        "imgsz": 1280, "task": "detect", "conf": 0.40, "max_side": 90,
        "label": "landed shuttle",
    },
    "pose": {
        "weights": "models/yolov8n-pose.pt",
```

Leave `imgsz` alone here — Task 2 handles the Pi sizes.

- [ ] **Step 5: Confirm LFS picked the files up**

Run: `cd /c/thesis && git add setup/models/*.pt && git lfs status`
Expected: the three new `.pt` files listed as "Git LFS objects to be committed". If they appear as plain files, `.gitattributes` did not apply — stop and re-check it before committing 19 MB of binary into git history.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /c/thesis/setup && python -m pytest tests/test_pipeline_models.py -v`
Expected: both PASS.

- [ ] **Step 7: Update START.md**

In the "three models" table, replace the `runs/...` and `yolov8n-pose.pt` paths in the **Weights** column with the three new `models/` paths. In the WARNING section, add after the existing paragraph:

```markdown
The `models/` copies are the ones `app/pipeline.py` loads and the only ones
that reach the Raspberry Pi (`runs/` is gitignored; `setup/models/*.pt` is
Git LFS tracked). The `runs/` originals remain the training record and are
still what `scripts/feeder_court/*.py` default to.
```

- [ ] **Step 8: Verify the Windows dev flow still works**

Run: `python run_server.py`, then `curl http://localhost:8000/api/health`
Expected: JSON with `"mongo": true`. Then `curl http://localhost:8000/api/models` and confirm `shuttle`, `landed`, and `pose` all report `"available": true`.

- [ ] **Step 9: Commit**

```bash
cd /c/thesis
git add setup/models/*.pt setup/app/pipeline.py setup/START.md setup/tests/test_pipeline_models.py
git commit -m "feat(models): ship production weights via LFS so they reach the Pi"
```

---

### Task 2: Platform-aware inference backend

`app/pipeline.py` hardcodes `backend="openvino"` in four places and bakes the string "openvino" into the export cache directory name. OpenVINO is Intel-oriented; NCNN is the ultralytics-recommended aarch64 export.

**Files:**
- Modify: `setup/app/pipeline.py` — add `resolve_backend()` and `_export_dir()`, rewrite `_load()` (lines ~88-104), `Worker.__init__` default (line ~111), `start()` default (line ~500)
- Modify: `setup/app/routers/cameras.py:294` — the `backend` query-parameter default
- Modify: `setup/config/settings.py` — add the Pi imgsz overrides
- Test: `setup/tests/test_pipeline_backend.py`

**Interfaces:**
- Consumes: Task 1's `MODELS` weight paths.
- Produces:
  - `pipeline.resolve_backend(machine: str) -> str` — returns `"openvino"` for `"AMD64"`/`"x86_64"`, else `"ncnn"`.
  - `pipeline.DEFAULT_BACKEND: str` — `resolve_backend(platform.machine())`, evaluated at import.
  - `pipeline._export_dir(path: str, imgsz: int, backend: str) -> str` — e.g. `models/yolov8n-pose_640_ncnn_model`.
  - Task 8's `install.sh` calls `_load()` to warm exports.

- [ ] **Step 1: Write the failing test**

Create `setup/tests/test_pipeline_backend.py`:

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import pipeline  # noqa: E402


# ── backend selection ────────────────────────────────────────

def test_x86_windows_gets_openvino():
    assert pipeline.resolve_backend("AMD64") == "openvino"


def test_x86_linux_gets_openvino():
    assert pipeline.resolve_backend("x86_64") == "openvino"


def test_pi_gets_ncnn():
    assert pipeline.resolve_backend("aarch64") == "ncnn"


def test_unknown_arch_gets_ncnn():
    """Fail toward the portable backend, not the Intel-only one."""
    assert pipeline.resolve_backend("armv7l") == "ncnn"


# ── export cache directory ───────────────────────────────────

def test_export_dir_includes_backend_and_size():
    got = pipeline._export_dir("models/yolov8n-pose.pt", 640, "ncnn")
    assert got == os.path.join("models", "yolov8n-pose_640_ncnn_model")


def test_export_dir_separates_backends():
    """An OpenVINO export must never be mistaken for an NCNN one."""
    ov = pipeline._export_dir("models/pose.pt", 640, "openvino")
    nc = pipeline._export_dir("models/pose.pt", 640, "ncnn")
    assert ov != nc


def test_export_dir_separates_sizes():
    """Reusing a 640 export at 1280 silently runs at 640."""
    assert pipeline._export_dir("m.pt", 640, "ncnn") != \
           pipeline._export_dir("m.pt", 1280, "ncnn")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline_backend.py -v`
Expected: all FAIL with `AttributeError: module 'app.pipeline' has no attribute 'resolve_backend'`.

- [ ] **Step 3: Implement the backend resolution**

In `setup/app/pipeline.py`, add `import platform` to the imports, then above `_load()`:

```python
# Inference backend by CPU architecture. OpenVINO is Intel-oriented and has no
# useful Pi story; NCNN is the ultralytics-recommended aarch64 export. Unknown
# architectures fall toward NCNN because it is the portable one.
_X86 = ("AMD64", "x86_64")


def resolve_backend(machine: str) -> str:
    return "openvino" if machine in _X86 else "ncnn"


DEFAULT_BACKEND = resolve_backend(platform.machine())


def _export_dir(path: str, imgsz: int, backend: str) -> str:
    """Per-(size, backend) export cache path.

    Both backends bake the input size in, so each imgsz needs its own
    directory - reusing a 640 export at 1280 silently runs at 640. The
    backend is in the name too so an OpenVINO tree is never loaded as NCNN.
    """
    return f"{os.path.splitext(path)[0]}_{imgsz}_{backend}_model"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline_backend.py -v`
Expected: all 7 PASS.

- [ ] **Step 5: Rewrite `_load()` to use them**

Replace the body of `_load()` (currently lines ~88-104) with:

```python
def _load(weights: str, imgsz: int, task: str, backend: str):
    """Load, exporting for `backend` on first use for that imgsz."""
    from ultralytics import YOLO
    path = _resolve(weights)
    if not os.path.exists(path):
        raise FileNotFoundError(f"weights missing: {weights}")
    if backend == "torch":
        return YOLO(path)
    out_dir = _export_dir(path, imgsz, backend)
    if not os.path.isdir(out_dir):
        produced = YOLO(path).export(format=backend, imgsz=imgsz, half=False)
        os.rename(str(produced), out_dir)
    return YOLO(out_dir, task=task)
```

`format=backend` works because ultralytics' export format strings are exactly `"openvino"` and `"ncnn"`.

- [ ] **Step 6: Replace the three hardcoded defaults**

- `Worker.__init__` (line ~111): `backend: str = "openvino"` becomes `backend: str | None = None`, and add as the first line of the body: `self.backend = backend or DEFAULT_BACKEND` (replacing the existing `self.backend = backend`).
- `start()` (line ~500): `backend: str = "openvino"` becomes `backend: str | None = None`.
- `app/routers/cameras.py:294`: `backend: str = "openvino"` becomes `backend: str | None = None`.

An explicit `?backend=torch` from the UI still overrides; `None` means "pick for this machine".

- [ ] **Step 7: Update the stale comment in `start()`**

In `start()`, the `time.sleep(0.5)` comment says "a first-time OpenVINO export". Change "OpenVINO" to "model".

- [ ] **Step 8: Add the Pi inference sizes to settings**

Append to `setup/config/settings.py`:

```python
# --- Raspberry Pi inference sizes ---
# On aarch64 the models run on CPU under NCNN, several times slower than
# OpenVINO on x86. START.md measures `landed` at 640 as strictly the better
# deal (18.3 ms vs 54.9 ms at 1280, still firing on 98% of frames), and the
# same argument applies to `shuttle`. app/pipeline.py applies these only when
# resolve_backend() picks ncnn.
PI_IMGSZ = {"shuttle": 640, "landed": 640, "pose": 640}
```

Then in `Worker.__init__`, after `self.cfg = dict(MODELS.get(model_key) or {})`:

```python
        # Pi: override the x86 input sizes (see settings.PI_IMGSZ).
        if self.backend == "ncnn" and model_key in settings.PI_IMGSZ:
            self.cfg["imgsz"] = settings.PI_IMGSZ[model_key]
```

Add `from config import settings` to `app/pipeline.py`'s imports if it is not already there. Verify first with `grep -n "config import settings" app/pipeline.py`.

- [ ] **Step 9: Run the full test suite**

Run: `python -m pytest -q`
Expected: no new failures against the pre-task baseline. Record the baseline first with `git stash && python -m pytest -q; git stash pop` if you did not already.

- [ ] **Step 10: Verify the Windows dev flow still works**

Run `python run_server.py`, open the UI, go to the Cameras page, and start a camera on the `pose` model. Expected: it loads from the existing `yolov8n-pose_640_openvino_model` directory (a `models/`-relative copy of it may need creating — if the export re-runs, that is correct behavior and takes about a minute, not a failure).

- [ ] **Step 11: Commit**

```bash
cd /c/thesis
git add setup/app/pipeline.py setup/app/routers/cameras.py setup/config/settings.py setup/tests/test_pipeline_backend.py
git commit -m "feat(pipeline): pick inference backend by CPU architecture"
```

---

### Task 3: FastAPI serves the built UI

**Files:**
- Create: `setup/app/ui_static.py`
- Modify: `setup/app/server.py` — mount after the last `include_router`
- Test: `setup/tests/test_ui_static.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ui_static.mount_ui(app: FastAPI, ui_dir: Path) -> bool` — mounts `ui_dir` at `/` and returns `True`, or returns `False` without mounting when the directory is absent. `ui_static.UI_DIR: Path` is the canonical bundle location (`app/static/ui`), which Task 7's `build_ui.ps1` writes to and Task 8's `install.sh` verifies.

The mount is factored into its own module so it is testable against a throwaway
`FastAPI()` instance — mounting happens at import time in `server.py`, which is
not testable in place.

- [ ] **Step 1: Write the failing test**

Create `setup/tests/test_ui_static.py`:

```python
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import ui_static  # noqa: E402


def _bundle(tmp_path: Path) -> Path:
    """Minimal stand-in for a Vite build output."""
    ui = tmp_path / "ui"
    (ui / "assets").mkdir(parents=True)
    (ui / "index.html").write_text("<html><body>AeroSense</body></html>", encoding="utf-8")
    (ui / "assets" / "app.js").write_text("console.log('hi')", encoding="utf-8")
    return ui


def test_mount_returns_false_when_bundle_absent(tmp_path):
    """The Windows dev flow has no bundle; the API must still boot."""
    app = FastAPI()
    assert ui_static.mount_ui(app, tmp_path / "nope") is False


def test_api_still_works_when_bundle_absent(tmp_path):
    app = FastAPI()

    @app.get("/api/health")
    def health():
        return {"ok": True}

    ui_static.mount_ui(app, tmp_path / "nope")
    client = TestClient(app)
    assert client.get("/api/health").json() == {"ok": True}


def test_mount_serves_index_at_root(tmp_path):
    app = FastAPI()
    assert ui_static.mount_ui(app, _bundle(tmp_path)) is True
    res = TestClient(app).get("/")
    assert res.status_code == 200
    assert "AeroSense" in res.text


def test_mount_serves_assets(tmp_path):
    app = FastAPI()
    ui_static.mount_ui(app, _bundle(tmp_path))
    res = TestClient(app).get("/assets/app.js")
    assert res.status_code == 200
    assert "console.log" in res.text


def test_api_routes_win_over_the_static_mount(tmp_path):
    """The whole design rests on /api never being shadowed by the / mount."""
    app = FastAPI()

    @app.get("/api/health")
    def health():
        return {"ok": True}

    ui_static.mount_ui(app, _bundle(tmp_path))
    client = TestClient(app)
    assert client.get("/api/health").json() == {"ok": True}
    assert "AeroSense" in client.get("/").text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ui_static.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'app.ui_static'`.

- [ ] **Step 3: Write the implementation**

Create `setup/app/ui_static.py`:

```python
"""Serve the pre-built React bundle so the app is one process on one origin.

The bundle is built on the Windows dev machine (scripts/build_ui.ps1) and
committed to app/static/ui, so the Raspberry Pi needs no Node toolchain.

WHY THE DIRECTORY IS OPTIONAL
During development the UI is served by Vite on :5173 and no bundle exists.
mount_ui() is a no-op then, and the API behaves exactly as it always has.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

UI_DIR = Path(__file__).parent / "static" / "ui"


def mount_ui(app: FastAPI, ui_dir: Path = UI_DIR) -> bool:
    """Mount `ui_dir` at / if it exists. Returns whether it was mounted.

    MUST be called after every include_router(): a mount at / matches any
    unclaimed path, so routes registered afterwards would be shadowed.

    No SPA fallback is needed - App.tsx switches on state.ui.activePage and
    the UI has no react-router dependency, so the app lives at exactly one
    URL. Adding client-side routing means adding a 404-to-index.html handler
    here.
    """
    if not ui_dir.is_dir():
        return False
    app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ui_static.py -v`
Expected: all 5 PASS.

- [ ] **Step 5: Wire it into the server**

In `setup/app/server.py`, after the last `app.include_router(...)` line (currently `app.include_router(esp32_enroll.router)`), add:

```python
# Serve the built UI last: a mount at / claims every unclaimed path, so this
# must come after all routers. No-op when the bundle is absent (dev flow).
from app.ui_static import mount_ui  # noqa: E402

_ui_mounted = mount_ui(app)
print(f"[UI] bundle {'mounted at /' if _ui_mounted else 'absent - API only'}", flush=True)
```

The `mount_ui` import sits with the other post-declaration imports in this file, matching the existing `from app.routers import players, ...` placement below the `add_middleware` call.

- [ ] **Step 6: Verify the dev flow is unchanged**

Run: `python run_server.py`
Expected: the log line `[UI] bundle absent - API only`, and `curl http://localhost:8000/api/health` still returns the health JSON.

- [ ] **Step 7: Verify the mounted path end to end**

```bash
mkdir -p app/static/ui
printf '<html><body>bundle smoke test</body></html>' > app/static/ui/index.html
python run_server.py
```

Expected: the log says `[UI] bundle mounted at /`; `curl http://localhost:8000/` returns the smoke-test HTML; `curl http://localhost:8000/api/health` still returns health JSON. Then remove the smoke-test file: `rm -rf app/static/ui`.

- [ ] **Step 8: Run the full suite**

Run: `python -m pytest -q`
Expected: no new failures.

- [ ] **Step 9: Commit**

```bash
cd /c/thesis
git add setup/app/ui_static.py setup/app/server.py setup/tests/test_ui_static.py
git commit -m "feat(server): serve the built React bundle from app/static/ui"
```

---

### Task 4: Address capture devices by path, not index

On Linux, udev reorders `/dev/video*` across reboots, so a stored index can point at a different camera after a power cycle — the same class of bug already recorded for Windows in the camera-selection notes. `/dev/v4l/by-id/*` symlinks are stable per physical device.

**Files:**
- Modify: `setup/app/sources.py` — `set_source()` (lines ~72-96), `describe()` (lines ~98-103), `open_capture()` (lines ~106-118)
- Test: `setup/tests/test_sources_device_path.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: a `kind: "device"` source may now carry **either** `index: int` **or** `path: str`. `describe()` returns `name` as the path's basename for a path-device and `f"device {index}"` for an index-device. Task 5 populates `path` from `list_devices()`; Task 6 sends it from the UI.

- [ ] **Step 1: Write the failing test**

Create `setup/tests/test_sources_device_path.py`:

```python
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import sources  # noqa: E402

BY_ID = "/dev/v4l/by-id/usb-HBVCAM-Camera-video-index0"


@pytest.fixture(autouse=True)
def _restore():
    """Put every slot back to whatever it was before the test."""
    before = {cid: sources.get(cid) for cid in sources.CAMERA_IDS}
    yield
    for cid, src in before.items():
        sources.set_source(cid, **{k: v for k, v in src.items() if k != "kind"},
                           kind=src["kind"])


def test_device_source_accepts_a_path():
    src = sources.set_source("front", "device", path=BY_ID)
    assert src["kind"] == "device"
    assert src["path"] == BY_ID
    assert "index" not in src


def test_device_source_still_accepts_an_index():
    src = sources.set_source("front", "device", index=1)
    assert src == {"kind": "device", "index": 1}


def test_device_source_needs_index_or_path():
    with pytest.raises(ValueError, match="index or path"):
        sources.set_source("front", "device")


def test_describe_names_a_path_device_by_basename():
    sources.set_source("front", "device", path=BY_ID)
    d = sources.describe("front")
    assert d["name"] == "usb-HBVCAM-Camera-video-index0"
    assert d["available"] is True


def test_describe_names_an_index_device_by_index():
    sources.set_source("front", "device", index=2)
    assert sources.describe("front")["name"] == "device 2"


def test_a_path_device_is_not_a_file():
    """is_file() gates frame-stepping in the UI; a camera must never be one."""
    sources.set_source("front", "device", path=BY_ID)
    assert sources.is_file("front") is False


def test_open_capture_passes_the_path_to_opencv(monkeypatch):
    seen = {}

    class FakeCap:
        def set(self, *a):
            return True

    def fake_videocapture(arg, *rest):
        seen["arg"] = arg
        return FakeCap()

    monkeypatch.setattr(sources.cv2, "VideoCapture", fake_videocapture)
    sources.set_source("front", "device", path=BY_ID)
    sources.open_capture("front")
    assert seen["arg"] == BY_ID
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sources_device_path.py -v`
Expected: `test_device_source_accepts_a_path` FAILS with `ValueError: device source needs an index`.

- [ ] **Step 3: Accept a path in `set_source()`**

In `setup/app/sources.py`, replace the `device` branch of `set_source()` (the `if index is None: raise ValueError("device source needs an index")` block and the line after it) with:

```python
        # A device is addressed either by capture index (Windows) or by a
        # stable /dev/v4l/by-id path (Linux, where udev reorders indices
        # across reboots).
        if index is None and not path:
            raise ValueError("device source needs an index or path")
        src = ({"kind": "device", "path": path} if path
               else {"kind": "device", "index": int(index)})
```

- [ ] **Step 4: Name a path-device in `describe()`**

Replace `describe()`'s final line (`return {**src, "name": f"device {src['index']}", "available": True}`) with:

```python
    if "path" in src:
        return {**src, "name": os.path.basename(src["path"]),
                "available": os.path.exists(src["path"])}
    return {**src, "name": f"device {src['index']}", "available": True}
```

- [ ] **Step 5: Open a path-device in `open_capture()`**

In the `if src["kind"] == "device":` branch, insert before the `sys.platform` check:

```python
        if "path" in src:
            return cv2.VideoCapture(src["path"])
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_sources_device_path.py -v`
Expected: all 7 PASS.

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest -q`
Expected: no new failures. Pay attention to `tests/test_calibration_api.py` and any test touching `/api/cameras/*/source`.

- [ ] **Step 8: Verify the Windows dev flow still works**

Run `python run_server.py`, open the Cameras page, pick a real USB device by index, and confirm a frame arrives. The index path must be untouched by this task.

- [ ] **Step 9: Commit**

```bash
cd /c/thesis
git add setup/app/sources.py setup/tests/test_sources_device_path.py
git commit -m "feat(sources): address capture devices by stable path as well as index"
```

---

### Task 5: V4L2 capture tuning and by-id enumeration

Four uncompressed 1280x800 streams exceed the Pi 5's shared USB3 bandwidth, so the V4L2 path must request MJPG. And `list_devices()` probes integer indices, which is the wrong identity to hand the UI on Linux.

**Files:**
- Modify: `setup/app/sources.py` — `open_capture()`, `list_devices()` (lines ~125-139)
- Test: `setup/tests/test_sources_v4l2.py`

**Interfaces:**
- Consumes: Task 4's path-device support.
- Produces: `list_devices()` returns dicts of `{"index": int, "width": int, "height": int}` on Windows and `{"path": str, "name": str, "width": int, "height": int}` on Linux. Task 6's UI reads both shapes.

- [ ] **Step 1: Write the failing test**

Create `setup/tests/test_sources_v4l2.py`:

```python
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import sources  # noqa: E402
from config import settings  # noqa: E402

BY_ID_DIR = "/dev/v4l/by-id"


class FakeCap:
    """Records every .set() so we can assert on capture configuration."""

    def __init__(self):
        self.props = {}

    def set(self, prop, val):
        self.props[prop] = val
        return True

    def isOpened(self):
        return True

    def read(self):
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        pass


@pytest.fixture(autouse=True)
def _restore():
    before = {cid: sources.get(cid) for cid in sources.CAMERA_IDS}
    yield
    for cid, src in before.items():
        sources.set_source(cid, **{k: v for k, v in src.items() if k != "kind"},
                           kind=src["kind"])


# ── capture configuration ───────────────────────────────────

def test_v4l2_capture_requests_mjpg_and_resolution(monkeypatch):
    """Four raw 1280x800 streams exceed the Pi's shared USB3 bandwidth."""
    cap = FakeCap()
    monkeypatch.setattr(sources.sys, "platform", "linux")
    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda *a: cap)
    sources.set_source("front", "device", index=0)
    sources.open_capture("front")

    assert cap.props[sources.cv2.CAP_PROP_FOURCC] == \
        sources.cv2.VideoWriter_fourcc(*"MJPG")
    assert cap.props[sources.cv2.CAP_PROP_FRAME_WIDTH] == settings.FRAME_WIDTH
    assert cap.props[sources.cv2.CAP_PROP_FRAME_HEIGHT] == settings.FRAME_HEIGHT


def test_windows_capture_is_left_alone(monkeypatch):
    """DSHOW already works on the dev machine; do not touch it."""
    cap = FakeCap()
    monkeypatch.setattr(sources.sys, "platform", "win32")
    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda *a: cap)
    sources.set_source("front", "device", index=0)
    sources.open_capture("front")
    assert cap.props == {}


def test_file_sources_are_not_reconfigured(monkeypatch, tmp_path):
    """Forcing MJPG on a video file would corrupt decoding."""
    clip = tmp_path / "rally.mp4"
    clip.write_bytes(b"\x00")
    cap = FakeCap()
    monkeypatch.setattr(sources.sys, "platform", "linux")
    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda *a: cap)
    sources.set_source("front", "file", path=str(clip))
    sources.open_capture("front")
    assert cap.props == {}


# ── enumeration ─────────────────────────────────────────────

def test_linux_enumeration_returns_stable_paths(monkeypatch):
    names = ["usb-HBVCAM-Camera-video-index0", "usb-OV9281-video-index0"]
    monkeypatch.setattr(sources.sys, "platform", "linux")
    monkeypatch.setattr(sources.os.path, "isdir", lambda p: p == BY_ID_DIR)
    monkeypatch.setattr(sources.os, "listdir", lambda p: names)
    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda *a: FakeCap())

    found = sources.list_devices()
    assert [d["path"] for d in found] == [f"{BY_ID_DIR}/{n}" for n in names]
    assert found[0]["name"] == names[0]
    assert found[0]["width"] == 640 and found[0]["height"] == 480
    assert all("index" not in d for d in found)


def test_linux_enumeration_falls_back_to_indices(monkeypatch):
    """A Pi with no by-id directory must still enumerate something."""
    monkeypatch.setattr(sources.sys, "platform", "linux")
    monkeypatch.setattr(sources.os.path, "isdir", lambda p: False)
    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda *a: FakeCap())

    found = sources.list_devices(max_index=2)
    assert [d["index"] for d in found] == [0, 1]


def test_windows_enumeration_still_returns_indices(monkeypatch):
    monkeypatch.setattr(sources.sys, "platform", "win32")
    monkeypatch.setattr(sources.cv2, "VideoCapture", lambda *a: FakeCap())
    found = sources.list_devices(max_index=2)
    assert [d["index"] for d in found] == [0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sources_v4l2.py -v`
Expected: `test_v4l2_capture_requests_mjpg_and_resolution` FAILS with `KeyError` on `CAP_PROP_FOURCC`.

- [ ] **Step 3: Configure the V4L2 capture**

In `setup/app/sources.py`, add near the top after the existing imports:

```python
from config import settings as _settings
```

Verify the module does not already import settings first: `grep -n "settings" app/sources.py`.

Then add above `open_capture()`:

```python
def _configure_v4l2(cap: "cv2.VideoCapture") -> "cv2.VideoCapture":
    """Ask a V4L2 camera for MJPG at the configured resolution.

    Four uncompressed 1280x800 streams exceed the Pi 5's shared USB3
    bandwidth and the later cameras simply fail to open. MJPG moves the
    decode cost onto the CPU, which is the cheaper of the two problems.

    Best-effort: a camera that refuses a mode keeps its default, and the
    frame size the pipeline sees comes from the frame itself, never from
    these values.
    """
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, _settings.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _settings.FRAME_HEIGHT)
    return cap
```

Then in `open_capture()`'s device branch, wrap both non-Windows returns:

```python
        if "path" in src:
            return _configure_v4l2(cv2.VideoCapture(src["path"]))
        if sys.platform == "win32":
            return cv2.VideoCapture(src["index"], cv2.CAP_DSHOW)
        return _configure_v4l2(cv2.VideoCapture(src["index"]))
```

Leave the file branch untouched.

- [ ] **Step 4: Rewrite `list_devices()`**

Replace `list_devices()` entirely:

```python
BY_ID_DIR = "/dev/v4l/by-id"


def _probe(arg) -> tuple[int, int] | None:
    """Open a source just long enough to learn its frame size."""
    cap = cv2.VideoCapture(arg, cv2.CAP_DSHOW) if sys.platform == "win32" \
        else cv2.VideoCapture(arg)
    try:
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame.shape[1], frame.shape[0]
    finally:
        cap.release()


def list_devices(max_index: int = 4) -> list[dict]:
    """Probe capture sources. Slow-ish, so the UI should call it on demand.

    On Linux, prefer the /dev/v4l/by-id symlinks: udev reorders the integer
    indices across reboots, so an index stored in Mongo can point at a
    different physical camera after a power cycle.
    """
    found = []
    if sys.platform != "win32" and os.path.isdir(BY_ID_DIR):
        for name in sorted(os.listdir(BY_ID_DIR)):
            path = f"{BY_ID_DIR}/{name}"
            size = _probe(path)
            if size:
                found.append({"path": path, "name": name,
                              "width": size[0], "height": size[1]})
        return found
    for i in range(max_index):
        size = _probe(i)
        if size:
            found.append({"index": i, "width": size[0], "height": size[1]})
    return found
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_sources_v4l2.py -v`
Expected: all 6 PASS.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest -q`
Expected: no new failures.

- [ ] **Step 7: Verify on Windows with real hardware**

Run `python run_server.py`, open the Cameras page, click the device-probe button. Expected: your USB HBVCAM and laptop camera still enumerate as `device 0` / `device 1` with correct resolutions, and selecting one still yields a frame. The Windows path must be behaviorally identical to before this task.

- [ ] **Step 8: Commit**

```bash
cd /c/thesis
git add setup/app/sources.py setup/tests/test_sources_v4l2.py
git commit -m "feat(sources): MJPG V4L2 capture and stable by-id enumeration"
```

---

### Task 6: UI — same-origin API and path-addressed devices

**Files:**
- Create: `C:\thesis_ui\badminton\.env.production`
- Modify: `C:\thesis_ui\badminton\src\lib\api.ts:1-2` (API base), `:72` (`CaptureDevice`)
- Modify: `C:\thesis_ui\badminton\src\lib\socket.ts:8` (websocket URL)
- Modify: `C:\thesis_ui\badminton\src\pages\Cameras.tsx:471`, `:501-506` (device picker)

**Interfaces:**
- Consumes: Task 5's two `list_devices()` shapes.
- Produces: nothing later tasks read from code. Task 7's `build_ui.ps1` consumes `.env.production` implicitly by running `vite build`.

**Note on testing:** this repo has **no test runner** — `package.json` has no vitest or jest, and adding one is out of scope. The gate for this task is `npm run build` (which runs `tsc -b`, so type errors fail the build) plus the explicit manual checks below. Do not claim this task verified without running them.

- [ ] **Step 1: Add the production env file**

Create `C:\thesis_ui\badminton\.env.production`:

```
# Production = the FastAPI server serves this bundle itself, so the API is
# same-origin and the base must be empty. api.ts uses ?? (nullish
# coalescing), so an empty string is kept rather than replaced by the
# localhost:8000 dev default.
VITE_API_URL=
```

Vite loads `.env.production` automatically for `vite build`. `npm run dev` uses `.env`, which keeps `VITE_API_URL=http://localhost:8000`.

- [ ] **Step 2: Derive the websocket URL from the page origin**

In `src/lib/socket.ts`, replace line 8:

```ts
  const url = API_URL.replace(/^http/, "ws") + "/ws";
```

with:

```ts
  // Same-origin production build: API_URL is "", so derive the origin from
  // the page rather than relying on relative-WebSocket-URL resolution.
  const url = API_URL
    ? API_URL.replace(/^http/, "ws") + "/ws"
    : `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
```

- [ ] **Step 3: Widen the `CaptureDevice` type**

In `src/lib/api.ts`, replace line 72:

```ts
export interface CaptureDevice { index: number; width: number; height: number }
```

with:

```ts
// Windows enumerates by capture index; Linux by stable /dev/v4l/by-id path
// (udev reorders indices across reboots). Exactly one of index/path is set.
export interface CaptureDevice {
  index?: number;
  path?: string;
  name?: string;
  width: number;
  height: number;
}
```

Also widen `setSource`'s body type on line 156 — it already accepts `path?: string`, so confirm with `grep -n "setSource" src/lib/api.ts` and change nothing if `path` is present.

- [ ] **Step 4: Send whichever identity the device carries**

In `src/pages/Cameras.tsx`, replace the device `<option>` block (lines ~503-506):

```tsx
                      {devices.map((d) => (
                        <option key={`d${d.index}`} value={JSON.stringify({ kind: "device", index: d.index })}>
                          device {d.index} ({d.width}×{d.height})
```

with:

```tsx
                      {devices.map((d) => (
                        <option
                          key={d.path ?? `d${d.index}`}
                          value={JSON.stringify(
                            d.path
                              ? { kind: "device", path: d.path }
                              : { kind: "device", index: d.index },
                          )}
                        >
                          {d.name ?? `device ${d.index}`} ({d.width}×{d.height})
```

Then at line ~471, the current-source comparison builds `JSON.stringify({ kind: "device", index: cur.index })`. Replace with:

```tsx
            ? JSON.stringify(cur.path && cur.kind === "device"
                ? { kind: "device", path: cur.path }
                : { kind: "device", index: cur.index })
```

Read the surrounding ternary before editing — it also handles the `file` kind, which must keep working unchanged.

- [ ] **Step 5: Build to verify types**

Run: `cd C:\thesis_ui\badminton && npm run build`
Expected: `tsc -b` passes and Vite writes `dist/`. Any `CaptureDevice.index` is `possibly undefined` error means a call site still assumes an index — fix it rather than casting.

- [ ] **Step 6: Manually verify the dev flow (index path)**

Run `npm run dev` plus `python run_server.py` in `C:\thesis\setup`. In the browser:
- The dashboard loads and populates (proves `.env` still supplies the dev base).
- The Cameras page device probe lists `device 0` / `device 1` and selecting one shows a frame.
- Browser devtools Network tab shows the websocket connected to `ws://localhost:8000/ws`.

- [ ] **Step 7: Manually verify the production bundle (same-origin path)**

```bash
cd /c/thesis_ui/badminton && npm run build
mkdir -p /c/thesis/setup/app/static/ui
cp -r dist/* /c/thesis/setup/app/static/ui/
cd /c/thesis/setup && python run_server.py
```

Open `http://localhost:8000` (not 5173). Expected: the UI loads, the dashboard populates, devtools shows API calls going to `http://localhost:8000/api/...` with **no** CORS preflights, and the websocket connected to `ws://localhost:8000/ws`. This is the first end-to-end proof of the whole design.

Then clean up: `rm -rf /c/thesis/setup/app/static/ui`.

- [ ] **Step 8: Commit (UI repo)**

```bash
cd /c/thesis_ui/badminton
git add .env.production src/lib/api.ts src/lib/socket.ts src/pages/Cameras.tsx
git commit -m "feat: same-origin API base and path-addressed capture devices"
```

---

### Task 7: The UI build step

**Files:**
- Create: `setup/scripts/build_ui.ps1`
- Modify: `setup/.gitignore` — an explicit un-ignore comment for the bundle
- Modify: `setup/README.md` — the update cycle

**Interfaces:**
- Consumes: Task 3's `ui_static.UI_DIR` (`app/static/ui`), Task 6's `.env.production`.
- Produces: the committed `setup/app/static/ui/` bundle that Task 8's `install.sh` verifies.

- [ ] **Step 1: Write the build script**

Create `setup/scripts/build_ui.ps1`:

```powershell
# Build the React UI and stage it where FastAPI serves it from.
#
# The bundle is committed so the Raspberry Pi needs no Node toolchain: the Pi's
# whole update cycle is `git pull` + `systemctl restart aerosense`. That means a
# UI change is not shipped until this script has run AND the result is
# committed.
#
# Usage:  pwsh scripts/build_ui.ps1

$ErrorActionPreference = "Stop"

$UiRepo = "C:\thesis_ui\badminton"
$Target = Join-Path $PSScriptRoot "..\app\static\ui"

if (-not (Test-Path $UiRepo)) { throw "UI repo not found at $UiRepo" }

Write-Host "==> building $UiRepo"
Push-Location $UiRepo
try {
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "npm run build failed ($LASTEXITCODE)" }
}
finally { Pop-Location }

$Dist = Join-Path $UiRepo "dist"
if (-not (Test-Path (Join-Path $Dist "index.html"))) {
    throw "no index.html in $Dist - the build produced nothing"
}

Write-Host "==> staging into $Target"
if (Test-Path $Target) { Remove-Item -Recurse -Force $Target }
New-Item -ItemType Directory -Force -Path $Target | Out-Null
Copy-Item -Recurse -Force (Join-Path $Dist "*") $Target

$n = (Get-ChildItem -Recurse -File $Target | Measure-Object).Count
Write-Host "==> staged $n files"
Write-Host ""
Write-Host "Now commit the bundle, then on the Pi:"
Write-Host "  git pull && sudo systemctl restart aerosense"
```

- [ ] **Step 2: Run it**

Run: `cd C:\thesis\setup && pwsh scripts/build_ui.ps1`
Expected: the build runs, staging reports a file count of at least 3 (`index.html`, a JS chunk, a CSS chunk), and `app/static/ui/index.html` exists.

- [ ] **Step 3: Confirm git will actually track the bundle**

Run: `cd /c/thesis && git check-ignore -v setup/app/static/ui/index.html; git status --short setup/app/static/ui | head`
Expected: `check-ignore` prints nothing and exits non-zero (not ignored), and `git status` lists the bundle files as untracked. If anything is ignored, stop — the whole delivery mechanism depends on this.

- [ ] **Step 4: Document the bundle in .gitignore**

Add to `setup/.gitignore`:

```
# The built UI bundle IS committed on purpose: it is how the Raspberry Pi gets
# the frontend without a Node toolchain. Regenerate it only with
# scripts/build_ui.ps1 - never hand-edit app/static/ui.
!app/static/ui/
```

- [ ] **Step 5: Verify the server serves the real bundle**

Run: `cd C:\thesis\setup && python run_server.py`, then open `http://localhost:8000`.
Expected: the log line `[UI] bundle mounted at /`, the real AeroSense UI loads, and the dashboard populates from the API.

- [ ] **Step 6: Document the update cycle**

Add to `setup/README.md` under a new `## Updating the Pi` heading:

```markdown
The Pi runs the committed UI bundle, not a dev server. To ship a UI change:

    pwsh scripts/build_ui.ps1      # rebuild + stage app/static/ui
    git add setup/app/static/ui && git commit
    # then on the Pi:
    git pull && git lfs pull && sudo systemctl restart aerosense

A UI change that skips `build_ui.ps1` is not shipped, however green the tests are.
```

- [ ] **Step 7: Commit**

```bash
cd /c/thesis
git add setup/scripts/build_ui.ps1 setup/.gitignore setup/README.md setup/app/static/ui
git commit -m "feat(deploy): build_ui.ps1 and commit the UI bundle for the Pi"
```

---

### Task 8: Provisioning — systemd units and install.sh

**Files:**
- Create: `setup/deploy/aerosense.service`
- Create: `setup/deploy/kiosk.service`
- Create: `setup/deploy/install.sh`
- Create: `setup/deploy/PI-SETUP.md`
- Modify: `setup/requirements.txt` — add `httpx`

**Interfaces:**
- Consumes: Task 1's LFS weights, Task 2's `pipeline._load` / `DEFAULT_BACKEND` (for export warm-up), Task 3's `app/static/ui` mount, Task 7's committed bundle.
- Produces: a provisioned Pi for Task 9.

- [ ] **Step 1: Close the httpx dependency gap**

`app/esp32_camera_client.py:16` imports `httpx`, which is absent from
`requirements.txt` and currently resolves only transitively (via starlette's
`TestClient` extra). A clean Pi venv would fail at ESP32 face enrollment. Add
to `setup/requirements.txt`, under the web-API section:

```
httpx        # app/esp32_camera_client.py — snapshot pull from the ESP32-CAM
```

- [ ] **Step 2: Verify the gap was real**

Run: `python -c "import httpx, importlib.metadata as m; print(m.version('httpx'))"`
Expected: a version prints (it is installed here). The point of Step 1 is the *clean* venv in Step 6, not this machine — but confirm `grep -n httpx requirements.txt` now finds it.

- [ ] **Step 3: Write the API unit**

Create `setup/deploy/aerosense.service`:

```ini
[Unit]
Description=AeroSense backend (FastAPI + drill engine)
# Mongo must be accepting connections before app.db builds its client.
After=network-online.target mongod.service
Wants=network-online.target mongod.service

[Service]
Type=simple
User=aerosense
# Relative weight paths in config/settings.py only resolve from setup/.
WorkingDirectory=/opt/aerosense/setup
Environment=PYTHONUNBUFFERED=1
# Bound to loopback: the ESP32-CAM integration is outbound-only
# (app/esp32_camera_client.py GETs http://ESP32_CAM_IP/capture), so nothing
# needs to reach this port from the LAN.
ExecStart=/opt/aerosense/setup/.venv/bin/uvicorn app.server:app --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 4: Write the kiosk unit**

Create `setup/deploy/kiosk.service`:

```ini
[Unit]
Description=AeroSense kiosk display
After=aerosense.service graphical-session.target
Wants=aerosense.service
PartOf=graphical-session.target

[Service]
Type=simple
User=aerosense
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/aerosense/.Xauthority
# --app drops all browser chrome; the flags after it suppress the restore
# bubble and the "unsupported flag" infobar that would otherwise sit on top
# of the UI after an unclean shutdown.
ExecStart=/usr/bin/chromium-browser \
  --app=http://localhost:8000 \
  --start-fullscreen \
  --kiosk \
  --noerrdialogs \
  --disable-infobars \
  --disable-session-crashed-bubble \
  --check-for-update-interval=31536000 \
  --autoplay-policy=no-user-gesture-required
Restart=always
RestartSec=3

[Install]
WantedBy=graphical-session.target
```

- [ ] **Step 5: Write install.sh**

Create `setup/deploy/install.sh`:

```bash
#!/usr/bin/env bash
# Provision a Raspberry Pi 5 to run AeroSense as a kiosk appliance.
#
#   sudo ./deploy/install.sh
#
# Idempotent: safe to re-run after a git pull.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
RUN_USER="${RUN_USER:-aerosense}"

log() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run with sudo"
[ "$(uname -m)" = "aarch64" ] || die "expected aarch64, got $(uname -m) - use Pi OS 64-bit"

# ── the committed UI bundle ─────────────────────────────────
log "checking the UI bundle"
[ -f "$ROOT/app/static/ui/index.html" ] || die \
  "app/static/ui/index.html is missing. The bundle is built on the Windows dev
   machine with scripts/build_ui.ps1 and committed. Run it, commit, and git pull."

# ── LFS-tracked model weights ───────────────────────────────
log "pulling LFS objects (model weights, face ONNX)"
command -v git-lfs >/dev/null || apt-get install -y git-lfs
sudo -u "$RUN_USER" git -C "$ROOT/.." lfs pull
for w in models/shuttle_clear_badminton_p2.pt models/shuttle_lines_stock_n.pt \
         models/yolov8n-pose.pt models/face_detection_yunet_2023mar.onnx \
         models/face_recognition_sface_2021dec.onnx; do
  [ -s "$ROOT/$w" ] || die "$w missing or empty after git lfs pull"
  # An unfetched LFS pointer is a ~130-byte text file, not a model.
  [ "$(stat -c%s "$ROOT/$w")" -gt 100000 ] || die "$w is an unfetched LFS pointer"
done

# ── system packages ─────────────────────────────────────────
log "installing system packages"
apt-get update
# python3-opencv from apt, not pip: a pip opencv-python build on the Pi is slow
# and frequently fails. The venv is created --system-site-packages so it is
# visible.
apt-get install -y python3-venv python3-pip python3-opencv chromium-browser \
                   v4l-utils curl gnupg

# ── MongoDB 7.0 (arm64) ─────────────────────────────────────
if ! command -v mongod >/dev/null; then
  log "installing MongoDB 7.0"
  # The Pi 5's Cortex-A76 is ARMv8.2-A, which Mongo 6+ requires. A Pi 4's
  # Cortex-A72 (ARMv8.0) cannot run this and would need Mongo 4.4.
  curl -fsSL https://pgp.mongodb.com/server-7.0.asc \
    | gpg --dearmor -o /usr/share/keyrings/mongodb-server-7.0.gpg
  echo "deb [arch=arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg] \
https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" \
    > /etc/apt/sources.list.d/mongodb-org-7.0.list
  apt-get update
  apt-get install -y mongodb-org
fi
systemctl enable --now mongod
mongosh --quiet --eval 'db.runCommand({ping:1}).ok' | grep -q 1 \
  || die "mongod is not answering"

# ── Python environment ──────────────────────────────────────
log "creating the venv"
[ -d "$VENV" ] || sudo -u "$RUN_USER" python3 -m venv --system-site-packages "$VENV"
sudo -u "$RUN_USER" "$VENV/bin/pip" install --upgrade pip
# opencv-python comes from apt (above); dropping it here avoids a source build.
sudo -u "$RUN_USER" grep -v '^opencv-python' "$ROOT/requirements.txt" \
  > /tmp/req-pi.txt
sudo -u "$RUN_USER" "$VENV/bin/pip" install -r /tmp/req-pi.txt
sudo -u "$RUN_USER" "$VENV/bin/pip" install ncnn
sudo -u "$RUN_USER" "$VENV/bin/python" -c "import cv2, httpx, ultralytics; print('deps ok', cv2.__version__)"

# ── .env ────────────────────────────────────────────────────
if [ ! -f "$ROOT/.env" ]; then
  log "writing .env"
  cat > "$ROOT/.env" <<'ENV'
ROBOFLOW_API_KEY=
MONGO_URL=mongodb://localhost:27017/
MONGO_DB=aerosense
ENV
  chown "$RUN_USER" "$ROOT/.env"
  echo "    NOTE: .env is gitignored. Add your ROBOFLOW_API_KEY by hand."
fi

# ── warm the NCNN exports ───────────────────────────────────
# First use triggers an export that takes minutes. Doing it here means the
# first drill does not stall.
log "warming NCNN exports (several minutes)"
cd "$ROOT"
sudo -u "$RUN_USER" "$VENV/bin/python" - <<'PY'
from app import pipeline
from config import settings
print("backend:", pipeline.DEFAULT_BACKEND)
assert pipeline.DEFAULT_BACKEND == "ncnn", pipeline.DEFAULT_BACKEND
for key in ("shuttle", "landed", "pose"):
    cfg = pipeline.MODELS[key]
    imgsz = settings.PI_IMGSZ[key]
    print(f"  {key} @ {imgsz} ...", flush=True)
    pipeline._load(cfg["weights"], imgsz, cfg["task"], "ncnn")
print("exports ready")
PY

# ── systemd units ───────────────────────────────────────────
log "installing systemd units"
install -m644 "$ROOT/deploy/aerosense.service" /etc/systemd/system/
install -m644 "$ROOT/deploy/kiosk.service"     /etc/systemd/system/
systemctl daemon-reload
systemctl enable aerosense.service kiosk.service
systemctl restart aerosense.service

log "waiting for the API"
for _ in $(seq 30); do
  if curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -fsS http://127.0.0.1:8000/api/health || die "the API never came up: journalctl -u aerosense -n 50"
echo

log "done. Reboot to bring up the kiosk: sudo reboot"
echo "    Cameras still need assigning - see deploy/PI-SETUP.md."
```

Make it executable: `chmod +x setup/deploy/install.sh`.

- [ ] **Step 6: Shell-check the script**

Run: `bash -n setup/deploy/install.sh && shellcheck setup/deploy/install.sh || true`
Expected: `bash -n` reports no syntax errors. Fix any shellcheck error-level findings; warnings are discretionary.

- [ ] **Step 7: Write PI-SETUP.md**

Create `setup/deploy/PI-SETUP.md`:

```markdown
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
set the hostname, enable SSH, and create the user **`aerosense`** — the systemd
units run as that user and reference `/home/aerosense/.Xauthority`. If you use
a different name, set `RUN_USER` when running `install.sh` and edit both unit
files.

Then, on the Pi:

    sudo raspi-config
      System Options  -> Boot / Auto Login -> Desktop Autologin
      Advanced        -> Wayland           -> X11

X11 rather than Wayland: `kiosk.service` sets `DISPLAY=:0` and `XAUTHORITY`,
which is the X11 contract.

## 2. Clone and install

    sudo apt-get install -y git git-lfs
    sudo mkdir -p /opt/aerosense && sudo chown aerosense /opt/aerosense
    git clone <your remote> /opt/aerosense
    cd /opt/aerosense/setup
    sudo ./deploy/install.sh

`/opt/aerosense` is baked into `aerosense.service`'s `WorkingDirectory`. To put
it elsewhere, edit both unit files.

Add your Roboflow key to `/opt/aerosense/setup/.env` afterwards — that file is
gitignored, so `install.sh` writes it with an empty key.

## 3. Assign the four cameras

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

## 4. ESP32-CAM

`config/settings.py` pins `ESP32_CAM_IP = "10.200.33.50"`. The Pi must be on
that same subnet. Check with:

    curl -o /dev/null -w '%{http_code}\n' http://10.200.33.50/capture

## 5. Reboot

    sudo reboot

The Pi should come up into the fullscreen UI with no interaction.

## Operating it

    systemctl status aerosense kiosk mongod
    journalctl -u aerosense -f          # backend logs
    sudo systemctl restart kiosk        # just reload the display

    # ship an update (UI bundle is built on the Windows box)
    cd /opt/aerosense && git pull && git lfs pull
    sudo systemctl restart aerosense

## Troubleshooting

| Symptom | Check |
|---|---|
| Black screen, no UI | `systemctl status kiosk`; is autologin to desktop on? Is it X11, not Wayland? |
| UI loads, no data | `curl localhost:8000/api/health`; `journalctl -u aerosense -n 50` |
| `"mongo": false` in health | `systemctl status mongod`; on a Pi 4 Mongo 7 cannot run at all |
| Only 1-2 cameras give frames | USB bandwidth. Confirm MJPG: `v4l2-ctl -d <dev> --list-formats`. Spread cameras across both USB3 and USB2 ports |
| Cameras swapped after reboot | A slot is stored as an `index`, not a by-id `path`. Reassign it in the UI |
| Everything is slow, then slower | Thermal throttling: `vcgencmd measure_temp`, `vcgencmd get_throttled` (nonzero = throttled) |
| First drill stalls for minutes | An NCNN export was not warmed. Re-run `install.sh` |
```

- [ ] **Step 8: Commit**

```bash
cd /c/thesis
git add setup/deploy setup/requirements.txt
git commit -m "feat(deploy): systemd units, install.sh and Pi setup guide"
```

---

### Task 9: Pi bring-up and throughput measurement

This is the task that runs on real hardware. Its deliverable is a provisioned Pi **and a measurement document** — the numbers are the input to the AI-HAT decision the spec deliberately deferred.

**Files:**
- Create: `setup/docs/pi-throughput.md` (measurements)
- Modify: `setup/deploy/PI-SETUP.md` — fill in the camera mapping table

**Interfaces:**
- Consumes: everything above.
- Produces: measured ms/frame per model, the input to the throughput decision.

- [ ] **Step 1: Provision the Pi**

Follow `deploy/PI-SETUP.md` steps 1-2 on real hardware.
Expected: `install.sh` completes and prints the health JSON.

- [ ] **Step 2: Verify health**

Run on the Pi: `curl -s localhost:8000/api/health`
Expected: `"mongo": true`, a `"db": "aerosense"`, an engine state, and `"face": true`. A `false` face value means the ONNX models did not arrive — recheck `git lfs pull`.

- [ ] **Step 3: Verify the backend picked NCNN**

Run: `/opt/aerosense/setup/.venv/bin/python -c "from app import pipeline; print(pipeline.DEFAULT_BACKEND)"` from `/opt/aerosense/setup`.
Expected: `ncnn`.

- [ ] **Step 4: Assign and smoke-test all four cameras**

Assign each slot in the UI per PI-SETUP.md step 3, then:

```bash
for c in front left right back; do
  echo -n "$c: "
  curl -s -o /dev/null -w '%{http_code}\n' "http://localhost:8000/api/cameras/$c/frame"
done
```

Expected: `200` four times. A `503` means that slot's source is unset or the device is not opening — check `ls -l /dev/v4l/by-id/`. Fill in the PI-SETUP.md mapping table with what you assigned.

- [ ] **Step 5: Confirm MJPG is actually in use**

Run: `v4l2-ctl -d /dev/video0 --get-fmt-video`
Expected: `Pixel Format: 'MJPG'`. If it reports `YUYV`, the camera refused MJPG and will eat USB bandwidth — note it in the measurement doc, because it explains later camera failures.

- [ ] **Step 6: Measure, one camera at a time**

Start one camera on each model in turn from the Cameras page and read the per-worker stats:

```bash
curl -s localhost:8000/api/pipeline | python3 -m json.tool
```

The fields to record are `infer_ms` (inference cost per frame), `infer_fps` (achieved inference rate), and `fps` (capture rate). Do this for `pose`, then `shuttle`, then `landed`, one camera at a time.

- [ ] **Step 7: Measure all four concurrently**

Start all four cameras on `pose`, let it settle 60 seconds, then:

```bash
curl -s localhost:8000/api/pipeline | python3 -m json.tool
vcgencmd measure_temp
vcgencmd get_throttled
free -m
```

Expected: `get_throttled` reads `0x0`. Anything else means the numbers are throttled and not representative — improve cooling and re-measure.

- [ ] **Step 8: Write up the measurements**

Create `setup/docs/pi-throughput.md` with: the Pi model and cooling, the date, a table of model / imgsz / ms-per-frame / achieved-FPS for both the single-camera and four-camera cases, the x86 OpenVINO figures from START.md alongside for comparison, peak temperature and throttle status, and a one-paragraph verdict on whether the drill loop is usable at `FPS_TARGET = 10`.

- [ ] **Step 9: Cold-boot test**

Run: `sudo reboot`
Expected: the Pi comes up into the fullscreen UI with no keyboard interaction, no browser chrome, and no terminal visible. The dashboard populates. This is the acceptance test for the whole plan.

- [ ] **Step 10: Commit**

```bash
cd /opt/aerosense
git add setup/docs/pi-throughput.md setup/deploy/PI-SETUP.md
git commit -m "docs: measured Pi 5 inference throughput and camera mapping"
```

---

## Decision point after Task 9

The spec deferred the throughput question deliberately. With Task 9's numbers in hand, choose:

1. **Usable as measured** — done.
2. **Close but slow** — pull the two cheap levers before spending money: `pipeline.py`'s existing court crop (START.md measures pose at 30 ms cropped versus roughly twice that on full frames), and staggering the four cameras rather than running them concurrently.
3. **Not close** — the Raspberry Pi AI HAT+ (Hailo-8). This is a separate plan: recompiling every weight to `.hef` through Hailo's x86 Docker toolchain, plus a third `pipeline.py` backend alongside `openvino` and `ncnn`. Task 2's `resolve_backend()` is the seam it plugs into.
