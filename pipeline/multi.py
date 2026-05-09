import cv2
import time
import threading
import queue
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────
MODEL_PATH  = "best.pt"
CAMERA_ID   = 0
INPUT_SIZE  = 640
CONF_THRESH = 0.4
TARGET_FPS  = 30
# ────────────────────────────────────────────────────────

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_queue  = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
stop_event   = threading.Event()

def capture_loop():
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break
        # drop old frame, put new one (always freshest frame)
        if frame_queue.full():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        frame_queue.put(frame)

def infer_loop():
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue
        small   = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        results = model(small, conf=CONF_THRESH, verbose=False)
        out     = results[0].plot()
        out     = cv2.resize(out, (frame.shape[1], frame.shape[0]))
        if result_queue.full():
            try: result_queue.get_nowait()
            except queue.Empty: pass
        result_queue.put(out)

threading.Thread(target=capture_loop, daemon=True).start()
threading.Thread(target=infer_loop,   daemon=True).start()

annotated = None
t_prev    = time.time()

while not stop_event.is_set():
    try:
        annotated = result_queue.get(timeout=0.05)
    except queue.Empty:
        if annotated is None:
            continue   # nothing to show yet

    t_now = time.time()
    fps   = 1.0 / (t_now - t_prev + 1e-9)
    t_prev = t_now
    cv2.putText(annotated, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Badminton Vision", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        stop_event.set()
        break

cap.release()
cv2.destroyAllWindows()


# Now this can detect real time object that was provided
# it does not track yet, position, history, speed computation
# it was needed from some algorithm like 

# SORT
# DeepSORT
# ByteTrack (best modern choice)