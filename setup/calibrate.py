# ============================================================
# calibrate.py — Run this FIRST before main.py
#
# This tool helps you set up your camera by clicking the 4 corners
# of the trainee's half-court:
#   1. NET_LEFT      (net meets the left sideline)
#   2. NET_RIGHT     (net meets the right sideline)
#   3. BASELINE_RIGHT (far baseline meets the right sideline)
#   4. BASELINE_LEFT (far baseline meets the left sideline)
#
# How to use:
#   1. python calibrate.py
#   2. Click the 4 corners in the order listed above
#   3. Copy the printed COURT_CORNERS = [...] value
#   4. Paste it into config/settings.py
#
# All court geometry (net line, sides, 6 zones) is derived from these
# 4 corners via homography, so it works for any camera angle.
#
# Controls:
#   Left click  = print (x, y) coordinates
#   S           = save current frame as calibration.jpg
#   Q           = quit
# ============================================================

import argparse
import cv2
from config.settings import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, GRAYSCALE, VIDEO_SOURCE,
)

click_points = []


def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))
        print(f"  Clicked: ({x}, {y})  — total points: {len(click_points)}")

        # Give hints based on how many points clicked
        hints = [
            "→ 1: NET_LEFT  (net meets the LEFT sideline)",
            "→ 2: NET_RIGHT (net meets the RIGHT sideline)",
            "→ 3: BASELINE_RIGHT (far baseline meets RIGHT sideline)",
            "→ 4: BASELINE_LEFT  (far baseline meets LEFT sideline)",
        ]
        if len(click_points) <= len(hints):
            print(f"  Hint: {hints[len(click_points)-1]}")


def run_calibration(source):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    is_file = not isinstance(source, int)

    # For a video file, grab ONE frame and freeze it so corners are clicked on a
    # static image. For a live webcam, keep streaming.
    frozen = None
    if is_file:
        ret, frozen = cap.read()
        if not ret:
            print(f"[ERROR] Could not read a frame from {source!r}")
            cap.release()
            return
        if GRAYSCALE and len(frozen.shape) == 2:
            frozen = cv2.cvtColor(frozen, cv2.COLOR_GRAY2BGR)

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_mouse_click)

    src_kind = "webcam index" if isinstance(source, int) else "video file"
    print(f"\n[SOURCE] Using {src_kind}: {source}")
    print("[CALIBRATION] Click the 4 corners of the TRAINEE's half-court:")
    print("  1. NET_LEFT      (net meets the left sideline)")
    print("  2. NET_RIGHT     (net meets the right sideline)")
    print("  3. BASELINE_RIGHT(far baseline meets the right sideline)")
    print("  4. BASELINE_LEFT (far baseline meets the left sideline)")
    print("\n  Press S to save frame | Q to quit\n")

    while True:
        if is_file:
            frame = frozen.copy()   # same static frame every loop
        else:
            ret, frame = cap.read()
            if not ret:
                break
            if GRAYSCALE and len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Draw all clicked points
        for i, (px, py) in enumerate(click_points):
            cv2.circle(frame, (px, py), 6, (0, 255, 255), -1)
            cv2.putText(frame, str(i + 1), (px + 8, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw crosshair hint
        cv2.putText(frame,
                    "Left click to mark points | S = save | Q = quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("calibration.jpg", frame)
            print("[SAVED] calibration.jpg saved")

    cap.release()
    cv2.destroyAllWindows()

    print("\n[SUMMARY] Your clicked points:")
    for i, (px, py) in enumerate(click_points):
        print(f"  Point {i+1}: ({px}, {py})")

    if len(click_points) >= 4:
        c = click_points[:4]
        print("\n[COURT_CORNERS suggestion] paste into config/settings.py:")
        print(f"COURT_CORNERS = [{c[0]}, {c[1]}, {c[2]}, {c[3]}]")
    else:
        print("\n[WARN] Need 4 corner clicks to produce COURT_CORNERS.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Court corner calibration")
    parser.add_argument("--source", default=None,
                        help="Video file path to calibrate on instead of the "
                             "webcam (overrides VIDEO_SOURCE / CAMERA_INDEX)")
    args = parser.parse_args()

    if args.source is not None:
        source = args.source
    elif VIDEO_SOURCE is not None:
        source = VIDEO_SOURCE
    else:
        source = CAMERA_INDEX

    run_calibration(source)
