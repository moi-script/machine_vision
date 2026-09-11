# PyInstaller spec for the packaged Windows build.
#
#   pyinstaller scripts/aerosense.spec --noconfirm      (run from setup/)
#
# onedir, deliberately: a onefile build unpacks its entire ~1.5 GB payload to
# a temp directory on EVERY launch, which takes minutes and hammers the disk.
# onedir starts immediately; scripts/build_windows_exe.ps1 zips it afterwards
# for distribution.
#
# WHAT NEEDS COLLECTING BY HAND
# ultralytics and torch both load things PyInstaller's static analysis cannot
# see: ultralytics reads .yaml model/tracker configs from its package data at
# runtime, and torch ships DLLs plus C++ extensions that are dlopen'd rather
# than imported. collect_all() pulls the whole package - crude, but the
# alternative is discovering each missing file one crash at a time.

from PyInstaller.utils.hooks import collect_all, collect_submodules

import os

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

datas = [
    # Relative paths in config/settings.py resolve against the working
    # directory, which aerosense_app.py sets to the bundle root.
    (os.path.join(ROOT, "models"), "models"),
    (os.path.join(ROOT, "app", "static"), "app/static"),
]
binaries = []
hiddenimports = [
    # Routers are pulled in inside function bodies in app/server.py, so the
    # static analyser does not always see them.
    *collect_submodules("app"),
    *collect_submodules("config"),
    *collect_submodules("utils"),
    "uvicorn.logging",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "pymongo",
    "httpx",
]

# collect_all() only for packages PyInstaller has no hook for. ultralytics
# reads .yaml model and tracker configs from its package data at runtime,
# which static analysis cannot see, so it genuinely needs the sweep.
#
# torch, torchvision, cv2 and openvino deliberately do NOT get collect_all:
# PyInstaller ships hooks for them, and sweeping a 454 MB torch tree forces
# analysis of thousands of modules. A first attempt at this build was killed
# by the OS partway through torch's analysis on a machine with ~0.8 GB free.
# The bundled hooks pull the same DLLs and extensions for a fraction of the
# peak memory, and produce a smaller bundle.
for pkg in ("ultralytics",):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # a missing optional package is not fatal
        print(f"[spec] skipping {pkg}: {exc}")

# Named so the standard hooks are applied; not swept.
hiddenimports += ["torch", "torchvision", "cv2", "openvino"]


a = Analysis(
    [os.path.join(ROOT, "scripts", "aerosense_app.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Trimming what ultralytics only needs for training/plotting. The app
    # only ever runs inference.
    excludes=["tkinter", "PyQt5", "PySide2", "notebook", "IPython", "pytest"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AeroSense",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX on torch's DLLs is slow and breaks some of them
    console=True,       # the console carries the MongoDB / port diagnostics
    icon=os.path.join(ROOT, "app", "static", "ui", "icon-512.png"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AeroSense",
)
