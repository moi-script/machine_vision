
# OpenCV

import cv2

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('ov9281_output.avi', fourcc, 60.0, (1280, 720))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)
    cv2.imshow("Recording", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()