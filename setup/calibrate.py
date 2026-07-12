# ============================================================
# calibrate.py — Run this FIRST before main.py
#
# This tool helps you find the correct pixel values for:
#   - COURT_ZONE (x1, y1, x2, y2)
#   - NET_X (vertical line position)
#   - PLAYER_ZONES (6 zone boundaries)
#
# How to use:
#   1. python calibrate.py
#   2. Click on corners of your court in the window
#   3. Note the (x, y) values printed in terminal
#   4. Copy those values into config/settings.py
#
# Controls:
#   Left click  = print (x, y) coordinates
#   S           = save current frame as calibration.jpg
#   Q           = quit
# ============================================================

import cv2
from config.settings import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, GRAYSCALE

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


def run_calibration():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_mouse_click)

    print("\n[CALIBRATION] Click the 4 corners of the TRAINEE's half-court:")
    print("  1. NET_LEFT      (net meets the left sideline)")
    print("  2. NET_RIGHT     (net meets the right sideline)")
    print("  3. BASELINE_RIGHT(far baseline meets the right sideline)")
    print("  4. BASELINE_LEFT (far baseline meets the left sideline)")
    print("\n  Press S to save frame | Q to quit\n")

    while True:
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
    run_calibration()
