#PW2 Camera mounting

import cv2
from picamera2 import Picamera2
import numpy as np

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"size": (320, 240)})
)
picam2.start()

LEFT_LIMIT = 0.45
RIGHT_LIMIT = 0.55

while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    h, w = thresh.shape
    roi = thresh[int(h*0.6):h, :]

    contours, _ = cv2.findContours(
        roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])

            # Draw center line
            cv2.line(frame, (cx, 0), (cx, h), (0, 255, 0), 2)

            # Draw limits
            cv2.line(frame, (int(w*LEFT_LIMIT), 0), (int(w*LEFT_LIMIT), h), (255, 0, 0), 2)
            cv2.line(frame, (int(w*RIGHT_LIMIT), 0), (int(w*RIGHT_LIMIT), h), (0, 0, 255), 2)

    cv2.imshow("Limit Tuning", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()

