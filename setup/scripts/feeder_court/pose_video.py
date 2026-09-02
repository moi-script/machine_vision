# ============================================================
# pose_video.py — Run COCO-pretrained YOLO pose over a court video, live, at
# 30 fps, showing only the players on our court.
#
# WHY THERE IS NO TRAINING STEP HERE
# COCO contains `person` with all 17 keypoints across ~150k annotated people,
# so yolov8n-pose already knows this task. Measured on 80 frames of this venue
# with no fine-tuning: 358 people, 4.5/frame, never zero, mean keypoint
# confidence 0.744, court players 0.78-0.88. Shuttlecock needed training
# because COCO has no shuttlecock; player does not, because COCO has person.
#
# HOW 280 ms/frame BECAME 30 ms
# Measured on this i3-1215U, full 1280x720 frame:
#     torch  1280   172 ms   3.5 people/frame
#     torch   960   109 ms   3.8
#     torch   640    83 ms   1.6   <- downscaling loses over half the people
#     torch   480    59 ms   0.1   <- and then nearly all of them
# Shrinking the whole frame destroys detection because a 128 px player becomes
# 64 px and then 48 px. Cropping instead keeps the players at native scale and
# throws away only sky, ceiling and spectators:
#     crop + openvino 640   29.7 ms   1.9 people/frame   median height 117 px
# Same player pixels, a third of the input area, and OpenVINO on top. 33.7 fps.
#
# WHY THE CROP IS A STAND-IN, NOT THE ANSWER
# CROP is hand-measured for dataset_court1's camera and is wrong for any other
# camera position. The real filter is geometric: project each person's ankle
# midpoint through the court homography and keep whoever is standing inside the
# court polygon. That works for all four cameras and needs no per-clip tuning.
# This crop buys a working demo until the homography exists.
#
# Usage:
#   python scripts/feeder_court/pose_video.py --show --fps 30
#   python scripts/feeder_court/pose_video.py --show --top-n 0   # everyone in the crop
#   python scripts/feeder_court/pose_video.py --show --crop 0,0,1280,720
#
# Keys:  q / ESC  quit          SPACE  pause
# ============================================================

import argparse
import os
import time

import cv2

DEFAULT_VIDEO = "datasets/vid_source/clear_badminton_dataset_for_collab/dataset_court1.mp4"
DEFAULT_WEIGHTS = "yolov8n-pose.pt"
# Hand-measured off a gridded frame of dataset_court1: the near half-court plus
# a margin, stopping at the net line. Everything above y=330 is the far court,
# the spectator tables and the ceiling.
DEFAULT_CROP = (200, 330, 1120, 720)
WINDOW = "feeder_court pose"
HUD_COLOR = (255, 255, 255)
CROP_COLOR = (90, 90, 90)


def load_model(YOLO, weights: str, backend: str, imgsz: int):
    """Load pose weights, exporting to OpenVINO on first use for that imgsz.

    OpenVINO bakes the input size in, so each imgsz needs its own export
    directory — reusing a 640 export at 1280 silently runs at 640.
    """
    if backend == "torch":
        return YOLO(weights)
    stem = os.path.splitext(weights)[0]
    ov_dir = f"{stem}_{imgsz}_openvino_model"
    if not os.path.isdir(ov_dir):
        print(f"[EXPORT] building OpenVINO pose model for imgsz={imgsz}...")
        produced = YOLO(weights).export(format="openvino", imgsz=imgsz, half=False)
        os.rename(str(produced), ov_dir)
        print(f"[EXPORT] {ov_dir}")
    return YOLO(ov_dir, task="pose")


def parse_crop(text: str) -> tuple[int, int, int, int]:
    parts = [int(v) for v in text.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise SystemExit("--crop wants x1,y1,x2,y2")
    return tuple(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Live pose detection over a court video.")
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="applied to the CROP, not the frame; 640 keeps players near native scale")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--crop", default=",".join(str(v) for v in DEFAULT_CROP),
                    help="x1,y1,x2,y2 region to search; the rest of the frame is ignored")
    ap.add_argument("--top-n", type=int, default=1, dest="top_n",
                    help="keep only the N largest people (1 = the near player); 0 keeps all")
    ap.add_argument("--backend", choices=["torch", "openvino"], default="openvino")
    ap.add_argument("--show", action="store_true", help="live window")
    ap.add_argument("--out", default=None, help="write the annotated video here")
    ap.add_argument("--fps", type=float, default=None,
                    help="pace the window to this rate; 30 is now reachable")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(root)
    if not os.path.exists(args.video):
        raise SystemExit(f"missing video: {args.video}")

    cx1, cy1, cx2, cy2 = parse_crop(args.crop)

    from ultralytics import YOLO  # imported late so --help stays instant
    model = load_model(YOLO, args.weights, args.backend, args.imgsz)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))

    print(f"{args.video}  {total} frames  {w}x{h}  crop={cx1},{cy1},{cx2},{cy2}  "
          f"imgsz={args.imgsz} [{args.backend}]  top_n={args.top_n}")
    frames = people = 0
    t0 = time.perf_counter()

    while True:
        started = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break

        crop = frame[cy1:cy2, cx1:cx2]
        result = model.predict(crop, imgsz=args.imgsz, conf=args.conf,
                               classes=[0], verbose=False)[0]

        # Select by index so boxes and their keypoints stay in step — rebuilding
        # either one on its own desynchronises the skeletons from the people.
        order = sorted(
            range(len(result.boxes)),
            key=lambda i: -( (lambda b: (b[2] - b[0]) * (b[3] - b[1]))(result.boxes.xyxy[i].tolist()) ),
        )
        keep = order[:args.top_n] if args.top_n else order

        annotated = (result[keep] if len(keep) != len(result.boxes) else result).plot()
        view = frame.copy()
        view[cy1:cy2, cx1:cx2] = annotated
        cv2.rectangle(view, (cx1, cy1), (cx2, cy2), CROP_COLOR, 1)

        people += len(keep)
        frames += 1
        cv2.putText(view, f"f{frames}/{total}  people={len(keep)}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, HUD_COLOR, 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(view)

        if args.show:
            cv2.imshow(WINDOW, view)
            wait_ms = 1
            if args.fps:
                spent = (time.perf_counter() - started) * 1000.0
                wait_ms = max(int(1000.0 / args.fps - spent), 1)
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                while True:
                    k2 = cv2.waitKey(50) & 0xFF
                    if k2 in (ord(" "), ord("q"), 27):
                        break
                if k2 in (ord("q"), 27):
                    break

        if frames % 200 == 0:
            print(f"  {frames}/{total} frames, {people} people", flush=True)

    elapsed = time.perf_counter() - t0
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    ms = 1000 * elapsed / frames if frames else 0
    print(f"\nframes {frames}  people {people}  mean {people / frames if frames else 0:.1f}/frame  "
          f"{ms:.1f} ms/frame ({1000 / ms if ms else 0:.1f} fps)")
    if args.out:
        print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
