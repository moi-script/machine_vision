# open CV


import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    cv2.imwrite("snapshot.png", frame)

cap.release()