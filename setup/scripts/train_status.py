# ============================================================
# train_status.py — Show progress of the local shuttlecock training run.
#
# The raw ultralytics log is full of carriage-return progress bars and is
# painful to read. Ultralytics also writes one clean row per epoch to
# results.csv, which is what this reads.
#
# Usage:  python scripts/train_status.py
# ============================================================

import csv
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(REPO, "runs", "shuttle", "yolov8n-640")
CSV = os.path.join(RUN, "results.csv")
TOTAL_EPOCHS = 120


def main() -> int:
    if not os.path.isfile(CSV):
        print("No results.csv yet — training is still warming up.")
        return 0

    with open(CSV) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("results.csv is empty — first epoch not finished yet.")
        return 0

    done = len(rows)
    # Per-epoch cost must come from the DELTA between consecutive rows, not
    # cumulative_time/epochs: `time` resets to 0 whenever a run is resumed, so
    # the cumulative form reports nonsense after a --resume.
    deltas = []
    for a, b in zip(rows, rows[1:]):
        d = float(b["time"]) - float(a["time"])
        if d > 0:  # skip the negative jump at a resume boundary
            deltas.append(d)
    per = sorted(deltas)[len(deltas) // 2] if deltas else 0.0  # median
    left = (TOTAL_EPOCHS - done) * per / 3600

    print(f"epoch {done}/{TOTAL_EPOCHS}   {per:.0f}s/epoch (median)")
    print(f"remaining ~{left:.1f} h if it runs the full 120 "
          f"(patience=30 may stop it sooner)")
    print()
    print(f"{'epoch':<7}{'mAP50':<10}{'mAP50-95':<11}{'precision':<11}{'recall':<8}")
    for r in rows[-10:]:
        print(f"{r['epoch'].strip():<7}"
              f"{float(r['metrics/mAP50(B)']):<10.4f}"
              f"{float(r['metrics/mAP50-95(B)']):<11.4f}"
              f"{float(r['metrics/precision(B)']):<11.3f}"
              f"{float(r['metrics/recall(B)']):<8.3f}")

    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    print()
    print(f"best epoch so far: {best['epoch'].strip()}  "
          f"mAP50={float(best['metrics/mAP50(B)']):.4f}  "
          f"mAP50-95={float(best['metrics/mAP50-95(B)']):.4f}")
    print(f"weights: {os.path.join(RUN, 'weights', 'best.pt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
