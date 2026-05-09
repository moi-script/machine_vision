

# pip install opencv-python ultralytics



import cv2
from ultralytics import YOLO

model = YOLO("best.pt")  # your exported model

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated = results[0].plot()

    cv2.imshow("Live Detection", annotated)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()