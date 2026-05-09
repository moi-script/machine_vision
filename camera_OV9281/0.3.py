import cv2


# // 640×480 @ higher stability
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)


# 0.5MP mode (640×360 widescreen)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
# cap.set(cv2.CAP_PROP_FPS, 60)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("0.3MP Mode (640x480)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# 640×480
# MJPEG mode
# 60 FPS (or highest stable)


# cap = cv2.VideoCapture(0)

# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FPS, 60)