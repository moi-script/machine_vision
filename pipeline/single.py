import cv2
import time
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────
MODEL_PATH  = "best.pt"
CAMERA_ID   = 0
INPUT_SIZE  = 640        # try 320 if CPU is slow
CONF_THRESH = 0.4
FRAME_SKIP  = 2          # infer every N frames
TARGET_FPS  = 30
# ────────────────────────────────────────────────────────

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count  = 0
annotated    = None
t_prev       = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % FRAME_SKIP == 0:
        small     = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        results   = model(small, conf=CONF_THRESH, verbose=False)
        annotated = results[0].plot()
        annotated = cv2.resize(annotated, (frame.shape[1], frame.shape[0]))
    elif annotated is None:
        annotated = frame.copy()   # nothing inferred yet

    # FPS overlay
    t_now        = time.time()
    fps          = 1.0 / (t_now - t_prev + 1e-9)
    t_prev       = t_now
    cv2.putText(annotated, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Badminton Vision", annotated)

    elapsed  = time.time() - t_now
    wait_ms  = max(1, int((1 / TARGET_FPS - elapsed) * 1000))
    if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


# Now this can detect real time object that was provided
# it does not track yet, position, history, speed computation
# it was needed from some algorithm like 

# SORT
# DeepSORT
# ByteTrack (best modern choice)

