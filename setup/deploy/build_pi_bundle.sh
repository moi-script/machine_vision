#!/usr/bin/env bash
# Build a distributable AeroSense bundle ON a Raspberry Pi 5.
#
#   ./deploy/build_pi_bundle.sh
#
# WHY THIS CANNOT BE RUN ON THE WINDOWS DEV MACHINE
# PyInstaller does not cross-compile. It bundles the interpreter and the
# native libraries of the machine it runs on, so an ARM64 Linux artifact can
# only be produced on ARM64 Linux. There is no flag for this; the Windows
# build (scripts/build_windows_exe.ps1) and this script are separate builds of
# the same source, each run on its own target.
#
# WHAT THIS PRODUCES, AND WHEN YOU WANT IT
# A tarball of the source, the LFS model weights and the committed UI bundle -
# everything install.sh needs - so a Pi with no network access to GitHub can
# still be provisioned from a USB stick. For a Pi that CAN reach GitHub, skip
# this: `git clone` plus `sudo ./deploy/install.sh` is the supported path and
# stays up to date.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/dist}"
STAMP="$(date +%Y%m%d)"
NAME="aerosense-rasp_v-${STAMP}-arm64"

log() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[ "$(uname -m)" = "aarch64" ] || die "expected aarch64, got $(uname -m).
  PyInstaller cannot cross-compile - run this on the Pi itself."

# The bundle is only useful if the pieces that arrive via Git LFS are real
# files rather than pointers, and if the UI has been staged from the dev box.
log "checking the payload"
[ -f "$ROOT/app/static/ui/index.html" ] || die \
  "app/static/ui is missing - it is built on the Windows machine with
   scripts/build_ui.ps1 and committed. Run git pull."
for w in models/shuttle_clear_badminton_p2.pt models/shuttle_lines_stock_n.pt \
         models/yolov8n-pose.pt models/face_detection_yunet_2023mar.onnx \
         models/face_recognition_sface_2021dec.onnx; do
  [ -s "$ROOT/$w" ] || die "$w is missing - run: git lfs pull"
  # An unfetched LFS pointer is a ~130-byte text file, not a model.
  [ "$(stat -c%s "$ROOT/$w")" -gt 100000 ] || die "$w is an unfetched LFS pointer - run: git lfs pull"
done

mkdir -p "$OUT"
log "packing $NAME.tar.gz"
tar -czf "$OUT/$NAME.tar.gz" \
    -C "$ROOT/.." \
    --exclude='setup/.venv' \
    --exclude='setup/dist' \
    --exclude='setup/build' \
    --exclude='setup/runs' \
    --exclude='setup/datasets' \
    --exclude='setup/__pycache__' \
    --exclude='**/__pycache__' \
    --exclude='setup/.pytest_cache' \
    --exclude='**/*_ncnn_model' \
    --exclude='**/*_openvino_model' \
    setup

size="$(du -h "$OUT/$NAME.tar.gz" | cut -f1)"
log "done: $OUT/$NAME.tar.gz ($size)"
cat <<EOF

To provision an offline Pi from this tarball:

    tar xzf $NAME.tar.gz -C /opt/aerosense --strip-components=0
    cd /opt/aerosense/setup
    sudo ./deploy/install.sh

install.sh still needs a network for apt and the MongoDB repository. Only the
application, its weights and the UI come from the tarball.
EOF
