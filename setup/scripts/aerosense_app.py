"""Frozen entry point for the packaged Windows build.

WHY THIS EXISTS RATHER THAN run_server.py
run_server.py starts uvicorn with the import STRING "app.server:app". Uvicorn
resolves that by importing the module at runtime, which works from a source
checkout but not inside a PyInstaller bundle, where the module graph is frozen
and there is no importable package directory to walk. Here the app object is
imported directly and handed to uvicorn as an object.

It also does the two things a double-clicked app has to do for itself:
check that MongoDB is actually reachable (the app cannot function without it,
and the bundle does not contain it), and open a browser window once the API
answers.
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"


def _bundle_root() -> str:
    """Directory holding the bundled data files.

    PyInstaller unpacks data into sys._MEIPASS; from source this is setup/.
    Relative paths in config/settings.py (models/, app/static/ui) resolve
    against the working directory, so it has to become this.
    """
    if getattr(sys, "frozen", False):
        return sys._MEIPASS  # type: ignore[attr-defined]
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) != 0


def _check_mongo() -> tuple[bool, str]:
    """MongoDB is a prerequisite, not part of this bundle.

    Bundling mongod would add ~50 MB and raise licensing questions, so the
    packaged app expects a local MongoDB. Saying so plainly here is much
    kinder than letting every page fail with an empty list.
    """
    try:
        from pymongo import MongoClient
        from pymongo.errors import PyMongoError
    except Exception as exc:  # pragma: no cover - import guard
        return False, f"pymongo missing from the bundle: {exc}"
    try:
        MongoClient(
            os.environ.get("MONGO_URL", "mongodb://localhost:27017/"),
            serverSelectionTimeoutMS=3000,
        ).admin.command("ping")
        return True, ""
    except PyMongoError as exc:
        return False, str(exc)


def _open_browser_when_ready() -> None:
    """Open the UI once the API answers, so the first paint is not an error."""
    import urllib.error
    import urllib.request

    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{URL}/api/health", timeout=2):
                break
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    webbrowser.open(URL)


def main() -> int:
    root = _bundle_root()
    os.chdir(root)

    print("AeroSense")
    print("=" * 60)

    ok, why = _check_mongo()
    if not ok:
        print()
        print("  MongoDB is not reachable, and this app cannot run without it.")
        print("  It stores every player, session, calibration and camera setup.")
        print()
        print(f"  Reason: {why}")
        print()
        print("  Install MongoDB Community Edition, make sure the service is")
        print("  running, then start AeroSense again:")
        print("      https://www.mongodb.com/try/download/community")
        print()
        input("  Press Enter to close...")
        return 1

    if not _port_is_free(HOST, PORT):
        print()
        print(f"  Port {PORT} is already in use, so AeroSense cannot start.")
        print("  Something else is serving on it - possibly another copy of")
        print("  this app. Close it, or find the culprit with:")
        print(f"      netstat -ano | findstr :{PORT}")
        print()
        input("  Press Enter to close...")
        return 1

    print(f"  Starting on {URL} - close this window to stop.")
    print("=" * 60)

    threading.Thread(target=_open_browser_when_ready, daemon=True).start()

    import uvicorn
    from app.server import app  # imported as an object, not an import string

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
