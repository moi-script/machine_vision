"""Build two splits of the same auto-labelled clip, to expose frame leakage.

- "random"   : frames shuffled, then split 80/20. This is the WRONG way, and is
               included precisely so its inflated score can be compared.
- "temporal" : the last N% of the clip's timeline is held out as validation, so
               no validation frame has a near-duplicate neighbour in training.

Both draw from the identical pool of images, so any difference in reported
metrics is attributable to the split alone.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def write_split(root: Path, name: str, train: list[Path], valid: list[Path]) -> None:
    out = root / name
    if out.exists():
        shutil.rmtree(out)
    for part, items in (("train", train), ("valid", valid)):
        (out / part / "images").mkdir(parents=True, exist_ok=True)
        (out / part / "labels").mkdir(parents=True, exist_ok=True)
        for image in items:
            label = image.parent.parent / "labels" / f"{image.stem}.txt"
            shutil.copy2(image, out / part / "images" / image.name)
            shutil.copy2(label, out / part / "labels" / label.name)

    # Absolute paths: Ultralytics resolves relative entries against the yaml's
    # own directory, not the working directory.
    (out / "data.yaml").write_text(
        f"train: {(out / 'train' / 'images').resolve().as_posix()}\n"
        f"val: {(out / 'valid' / 'images').resolve().as_posix()}\n"
        "\nnc: 1\nnames: ['shuttlecock']\n"
    )
    print(f"{name:9s} train={len(train):4d} valid={len(valid):4d} -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="datasets/scene-dummy/raw")
    parser.add_argument("--out", default="datasets/scene-dummy")
    parser.add_argument("--holdout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw = Path(args.raw)
    images = sorted((raw / "images").glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images under {raw / 'images'}")

    # Frame index is encoded in the filename, so sorted order is time order.
    cut = int(len(images) * (1 - args.holdout))
    write_split(Path(args.out), "temporal", images[:cut], images[cut:])

    shuffled = list(images)
    random.Random(args.seed).shuffle(shuffled)
    split = int(len(shuffled) * 0.8)
    write_split(Path(args.out), "random", shuffled[:split], shuffled[split:])


if __name__ == "__main__":
    main()
