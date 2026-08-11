"""Combine the two recording sessions into one training set.

Session 1 (raw)  : close range, grey wall, olive shirt. Labels verified.
Session 2 (v2)   : varied distance, white wall, black shirt. Mostly pre-labels.

Two deliberate choices:

* Session 2's EMPTY label files are dropped rather than used as negatives. They
  were produced by a model that cannot see distant shuttles, so a large share of
  them contain a shuttle that simply was not detected. Feeding those in as
  background would teach the exact opposite of what this session is for.
  Session 1 supplies verified negatives instead.

* Validation is the LAST portion of session 2 by time. With only two sessions
  there is no fully held-out session available, so this number is still
  optimistic -- same room, same lighting. Treat cross-domain evals as the honest
  generalisation measure.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def collect(root: Path, keep_empty: bool) -> list[tuple[Path, Path]]:
    pairs = []
    for image in sorted((root / "images").glob("*.jpg")):
        label = root / "labels" / f"{image.stem}.txt"
        if not label.exists():
            continue
        has_box = any(line.strip() for line in label.read_text().splitlines())
        if has_box or keep_empty:
            pairs.append((image, label))
    return pairs


def write(out: Path, part: str, pairs: list[tuple[Path, Path]]) -> None:
    (out / part / "images").mkdir(parents=True, exist_ok=True)
    (out / part / "labels").mkdir(parents=True, exist_ok=True)
    for image, label in pairs:
        shutil.copy2(image, out / part / "images" / image.name)
        shutil.copy2(label, out / part / "labels" / label.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session1", default="datasets/scene-dummy/raw")
    parser.add_argument("--session2", default="datasets/scene-dummy/roboflow_upload")
    parser.add_argument("--out", default="datasets/scene-dummy/combined")
    parser.add_argument("--holdout", type=float, default=0.3)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    s1 = collect(Path(args.session1), keep_empty=True)
    s2 = collect(Path(args.session2), keep_empty=False)

    cut = int(len(s2) * (1 - args.holdout))
    train = s1 + s2[:cut]
    valid = s2[cut:]

    write(out, "train", train)
    write(out, "valid", valid)

    (out / "data.yaml").write_text(
        f"train: {(out / 'train' / 'images').resolve().as_posix()}\n"
        f"val: {(out / 'valid' / 'images').resolve().as_posix()}\n"
        "\nnc: 1\nnames: ['shuttlecock']\n"
    )

    s1_neg = sum(1 for _, l in s1 if not l.read_text().strip())
    print(f"session 1 : {len(s1)} frames ({s1_neg} negatives)")
    print(f"session 2 : {len(s2)} frames with boxes (empties dropped)")
    print(f"train     : {len(train)}")
    print(f"valid     : {len(valid)}  (last {args.holdout:.0%} of session 2)")
    print(f"out       : {out}")


if __name__ == "__main__":
    main()
