# ============================================================
# sketch_frame.py — Pull one frame out of a video and draw on it, so the marks
# come back as coordinates rather than as a picture.
#
# WHY NOT JUST MARK IT IN PAINT
# A painted image has to be colour-matched afterwards to recover where the marks
# are, which is lossy and guesses at anti-aliased edges. Here every stroke is
# recorded as it is drawn, so the JSON holds the exact pixel path. The rendered
# PNG is written too, but only so a human can see what was marked.
#
# COORDINATES ARE ALWAYS IN ORIGINAL FRAME SPACE
# --scale only magnifies the window. A point clicked at display (900, 400) with
# --scale 1.5 is stored as (600, 267). Getting this wrong would silently offset
# every calibration point, so the conversion happens once, at input.
#
# WHY THE ENHANCE TOGGLE EXISTS
# This footage is dim and the court lines are barely above the floor in
# brightness. Pressing `e` swaps in a CLAHE-boosted background so the lines can
# actually be seen while marking. It changes the picture, never the coordinates.
#
# Usage:
#   python scripts/feeder_court/sketch_frame.py
#   python scripts/feeder_court/sketch_frame.py --frame 300 --scale 1.5
#
# Mouse:  drag        freehand stroke        (line mode)
#         click       drop a numbered point  (point mode)
#
# Keys:   r g b y  colour: red / green / blue / yellow
#         p        toggle point <-> line mode
#         e        toggle enhanced background
#         [ ]      brush thinner / thicker
#         u        undo last stroke
#         c        clear everything
#         s        SAVE  (png + json)
#         q / ESC  quit
# ============================================================

import argparse
import json
import os
import time

import cv2
import numpy as np

DEFAULT_VIDEO = "datasets/vid_source/clear_badminton_dataset_for_collab/lines.mp4"
DEFAULT_OUT = "datasets/clear_badminton/calibration"
WINDOW = "sketch — r/g/b/y colour, p point-mode, e enhance, u undo, s save, q quit"

# BGR. Red and green first: those are the two asked for, the rest are spare.
COLORS = {
    "r": ("red", (0, 0, 255)),
    "g": ("green", (0, 255, 0)),
    "b": ("blue", (255, 128, 0)),
    "y": ("yellow", (0, 255, 255)),
}


class Sketch:
    """Strokes in ORIGINAL image coordinates, newest last."""

    def __init__(self):
        self.strokes: list[dict] = []
        self.active: dict | None = None

    def begin(self, pt, color_key, mode, width):
        self.active = {"color": color_key, "mode": mode, "width": width, "points": [pt]}

    def extend(self, pt):
        if self.active is not None:
            # Skip sub-pixel jitter so the JSON stays readable.
            last = self.active["points"][-1]
            if abs(pt[0] - last[0]) + abs(pt[1] - last[1]) >= 2:
                self.active["points"].append(pt)

    def end(self):
        if self.active is not None:
            self.strokes.append(self.active)
            self.active = None

    def undo(self):
        if self.strokes:
            self.strokes.pop()

    def clear(self):
        self.strokes.clear()
        self.active = None


def enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def render(base, sketch: Sketch, scale: float, status: str):
    canvas = base.copy()
    todo = sketch.strokes + ([sketch.active] if sketch.active else [])
    point_no = 0
    for st in todo:
        _, bgr = COLORS[st["color"]]
        pts = st["points"]
        if st["mode"] == "point":
            point_no += 1
            x, y = pts[0]
            cv2.drawMarker(canvas, (x, y), bgr, cv2.MARKER_CROSS, 18, st["width"] + 1)
            cv2.circle(canvas, (x, y), 9, bgr, 1, cv2.LINE_AA)
            cv2.putText(canvas, str(point_no), (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2, cv2.LINE_AA)
        else:
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i - 1], pts[i], bgr, st["width"], cv2.LINE_AA)

    view = canvas if scale == 1.0 else cv2.resize(
        canvas, (int(canvas.shape[1] * scale), int(canvas.shape[0] * scale)),
        interpolation=cv2.INTER_CUBIC)

    pad = 8
    cv2.rectangle(view, (0, 0), (view.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(view, status, (pad, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return canvas, view


def main() -> None:
    ap = argparse.ArgumentParser(description="Sketch on one video frame and save the coordinates.")
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--frame", type=int, default=300,
                    help="which frame to pull; 300 has the emptiest floor in lines.mp4")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--name", default=None, help="basename for the saved files")
    ap.add_argument("--scale", type=float, default=1.4,
                    help="window magnification; coordinates stay in original space")
    ap.add_argument("--width", type=int, default=2, help="starting brush width")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(root)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(args.frame, total - 1)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {args.frame} of {total}")

    plain = frame.copy()
    boosted = enhance(frame)
    h, w = frame.shape[:2]

    name = args.name or f"{os.path.splitext(os.path.basename(args.video))[0]}_f{args.frame}_sketch"
    os.makedirs(args.out, exist_ok=True)

    state = {"color": "r", "mode": "line", "width": args.width,
             "enhanced": True, "drawing": False}
    sketch = Sketch()

    def to_original(x, y):
        return (int(round(x / args.scale)), int(round(y / args.scale)))

    def on_mouse(event, x, y, flags, _param):
        pt = to_original(x, y)
        pt = (max(0, min(pt[0], w - 1)), max(0, min(pt[1], h - 1)))
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            sketch.begin(pt, state["color"], state["mode"], state["width"])
            if state["mode"] == "point":
                sketch.end()
                state["drawing"] = False
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            sketch.extend(pt)
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            sketch.extend(pt)
            sketch.end()
            state["drawing"] = False

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(f"frame {args.frame}/{total}  {w}x{h}  window scale {args.scale}")
    print("draw, then press s to save. q quits.")

    saved = 0
    last_save = 0.0
    while True:
        base = boosted if state["enhanced"] else plain
        pts_count = sum(1 for s in sketch.strokes if s["mode"] == "point")
        status = (f"[{COLORS[state['color']][0]}] mode={state['mode']} w={state['width']} "
                  f"{'enhanced' if state['enhanced'] else 'plain'} | "
                  f"strokes={len(sketch.strokes)} points={pts_count} | "
                  f"r g b y  p  e  [ ]  u  c  s  q")
        canvas, view = render(base, sketch, args.scale, status)
        cv2.imshow(WINDOW, view)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("r"), ord("g"), ord("b"), ord("y")):
            state["color"] = chr(key)
        elif key == ord("p"):
            state["mode"] = "point" if state["mode"] == "line" else "line"
        elif key == ord("e"):
            state["enhanced"] = not state["enhanced"]
        elif key == ord("["):
            state["width"] = max(1, state["width"] - 1)
        elif key == ord("]"):
            state["width"] = min(12, state["width"] + 1)
        elif key == ord("u"):
            sketch.undo()
        elif key == ord("c"):
            sketch.clear()
        elif key == ord("s"):
            # waitKey polls every 20 ms, so a normal keypress is delivered many
            # times over. Without this the last session wrote 14 identical
            # copies of one sketch.
            if time.perf_counter() - last_save < 0.6:
                continue
            last_save = time.perf_counter()
            saved += 1
            stem = os.path.join(args.out, name if saved == 1 else f"{name}_{saved}")
            cv2.imwrite(f"{stem}.png", canvas)
            # Also write the clean frame so the marks can be re-checked later.
            cv2.imwrite(f"{stem}_source.jpg", plain)
            payload = {
                "video": args.video,
                "frame": args.frame,
                "image_size": [w, h],
                "strokes": [
                    {"color": COLORS[s["color"]][0], "mode": s["mode"],
                     "width": s["width"],
                     "points": [[int(p[0]), int(p[1])] for p in s["points"]]}
                    for s in sketch.strokes
                ],
            }
            with open(f"{stem}.json", "w") as fh:
                json.dump(payload, fh, indent=2)
            by_color: dict[str, int] = {}
            for s in payload["strokes"]:
                by_color[s["color"]] = by_color.get(s["color"], 0) + 1
            print(f"saved {stem}.png / .json  ({len(payload['strokes'])} strokes {by_color})",
                  flush=True)

    cv2.destroyAllWindows()
    if saved == 0:
        print("nothing saved (press s before quitting)")


if __name__ == "__main__":
    main()
