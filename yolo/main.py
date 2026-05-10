from ultralytics import YOLO
import cv2

model = YOLO("best.pt")  # your trained model
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OPTIONAL: resize for speed
    frame_resized = cv2.resize(frame, (640, 480))

    results = model(frame_resized)

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])

            if conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                cv2.circle(frame_resized, (cx, cy), 5, (0,255,0), -1)
                cv2.rectangle(frame_resized, (x1,y1), (x2,y2), (0,255,0), 2)

                print(f"x={cx}, y={cy}")

    cv2.imshow("Detection", frame_resized)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()