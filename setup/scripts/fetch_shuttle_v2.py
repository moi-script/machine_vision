# ============================================================
# fetch_shuttle_v2.py — Download the v2 shuttlecock dataset from Roboflow.
#
# Why a new dataset: the v1 dataset (datasets/shuttlecock-1) was 422 frames
# from ONE clip, and its val split sat 0-1 frames from a train frame, so
# mAP50=0.898 measured memorisation, not generalisation. Against a real webcam
# it hit 4/175 frames at conf>=0.05. See docs/shuttle-v2-dataset.md.
#
# Default source is diwenne/smashspeed: ~13.4k images across many different
# gyms, floor colours, lighting rigs and camera angles — amateur hall footage,
# which is much closer to a feeder-mounted camera than broadcast TV coverage.
# It also ships background frames with no shuttle, which suppress false
# positives.
#
# Writes to datasets/shuttle-v2/. NOTHING under datasets/shuttlecock-1/,
# runs/shuttle/ or models/shuttlecock.pt is touched.
#
# Usage:  python scripts/fetch_shuttle_v2.py
#         python scripts/fetch_shuttle_v2.py --project mathieu-cartron/shuttlecock-cqzy3 --version 1
# ============================================================

import argparse
import os
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(REPO, "datasets", "shuttle-v2")

DEFAULT_PROJECT = "diwenne/smashspeed"
DEFAULT_VERSION = 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=DEFAULT_PROJECT,
                    help="workspace/project slug from the Universe URL")
    ap.add_argument("--version", type=int, default=DEFAULT_VERSION)
    ap.add_argument("--format", default="yolov8")
    ap.add_argument("--dest", default=DEST,
                    help="extract here instead of datasets/shuttle-v2. Use this "
                         "to fetch into a fresh directory rather than unzipping "
                         "on top of an existing export and mixing two datasets.")
    args = ap.parse_args()
    dest = args.dest

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(REPO, ".env"))
    except ImportError:
        pass
    key = os.getenv("ROBOFLOW_API_KEY")
    if not key:
        print("[ERROR] ROBOFLOW_API_KEY not set (put it in .env)", file=sys.stderr)
        return 1

    import urllib.request

    url = (f"https://api.roboflow.com/{args.project}/{args.version}/"
           f"{args.format}?api_key={key}")
    print(f"[FETCH] {args.project} v{args.version} ({args.format})")

    try:
        import json
        meta = json.load(urllib.request.urlopen(url, timeout=120))
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] could not reach api.roboflow.com: {exc}", file=sys.stderr)
        print("        If this is an SSL/connection error, download the ZIP "
              "manually from the Universe page and unzip it into:", file=sys.stderr)
        print(f"        {dest}", file=sys.stderr)
        return 1

    link = meta.get("export", {}).get("link")
    if not link:
        print(f"[ERROR] no export link in response: {meta}", file=sys.stderr)
        return 1

    os.makedirs(dest, exist_ok=True)

    # Stream to disk rather than .read()-ing the whole export into RAM. A 13k
    # image export is over a gigabyte: buffering it gives no progress for many
    # minutes and loses everything if the process is interrupted.
    zip_path = dest.rstrip("/\\") + ".zip"
    resume_from = os.path.getsize(zip_path) if os.path.isfile(zip_path) else 0

    req = urllib.request.Request(link)
    if resume_from:
        req.add_header("Range", f"bytes={resume_from}-")
        print(f"[FETCH] resuming from {resume_from / 1e6:.1f} MB")

    resp = urllib.request.urlopen(req, timeout=1800)
    partial = resp.status == 206
    total = int(resp.headers.get("Content-Length") or 0)
    if partial:
        total += resume_from
    elif resume_from:
        print("[FETCH] server ignored resume, restarting download")
        resume_from = 0

    mode = "ab" if (partial and resume_from) else "wb"
    done = resume_from if mode == "ab" else 0
    last_pct = -5
    with open(zip_path, mode) as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                p = int(100 * done / total)
                if p >= last_pct + 5:
                    last_pct = p
                    print(f"[FETCH] {p:3d}%  {done / 1e6:7.1f} / {total / 1e6:.1f} MB",
                          flush=True)
            elif done // (50 << 20) != (done - len(chunk)) // (50 << 20):
                print(f"[FETCH] {done / 1e6:7.1f} MB", flush=True)

    print(f"[FETCH] downloaded {done / 1e6:.1f} MB -> {zip_path}")
    print("[FETCH] extracting…", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    os.remove(zip_path)

    n = sum(len(fs) for _, _, fs in os.walk(dest) if fs)
    print(f"[DONE] extracted {n} files -> {dest}")
    print("[NEXT] python scripts/resplit_by_clip.py   # fixes the leaky split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
