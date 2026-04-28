#PW2 Line Detection with PID


import cv2
import numpy as np
from picamera2 import Picamera2
import pigpio
import time

# ================= GPIO =================
IN1 = 17
IN2 = 18
IN3 = 22
IN4 = 23
ENA = 12
ENB = 13

# ================= PID PARAMETERS =================
Kp = 3.0
Ki = 0.002
Kd = 0.8

BASE_SPEED = 150
MAX_SPEED = 250
MAX_CONTROL = 150
ROI_HEIGHT = 150

# ================= INIT =================
pi = pigpio.pi()
if not pi.connected:
    print("pigpio not connected")
    exit()

for pin in [IN1, IN2, IN3, IN4, ENA, ENB]:
    pi.set_mode(pin, pigpio.OUTPUT)

pi.set_PWM_frequency(ENA, 1000)
pi.set_PWM_frequency(ENB, 1000)

def set_motor(left_speed, right_speed):

    left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
    right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))

    if left_speed >= 0:
        pi.write(IN1, 0)
        pi.write(IN2, 1)
        pi.set_PWM_dutycycle(ENA, int(left_speed))
    else:
        pi.write(IN1, 1)
        pi.write(IN2, 0)
        pi.set_PWM_dutycycle(ENA, int(abs(left_speed)))

    if right_speed >= 0:
        pi.write(IN3, 0)
        pi.write(IN4, 1)
        pi.set_PWM_dutycycle(ENB, int(right_speed))
    else:
        pi.write(IN3, 1)
        pi.write(IN4, 0)
        pi.set_PWM_dutycycle(ENB, int(abs(right_speed)))
# ================= CAMERA =================
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"size": (320, 240)})
)
picam2.start()

lower_black = np.array([0, 0, 0])
upper_black = np.array([179, 255, 160])

# ================= PID STATE =================
previous_error = 0
integral = 0
last_error = 0

print("Two-point PID tracking started. Press q to quit.")

while True:

    frame = picam2.capture_array()
    debug_frame = frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_black, upper_black)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    height, width = mask.shape
    roi_mask = mask[height-ROI_HEIGHT:height, :]

    frame_center = width // 2
    cv2.line(debug_frame, (frame_center, 0), (frame_center, height), (0,255,0), 2)


    cv2.imshow("ROI", roi_mask)
    cv2.imshow("DEBUG", debug_frame)

    if cv2.waitKey(1) == ord('q'):
        break
# ===== Two-point detection =====
    bottom_strip = roi_mask[-140:-115, :]
    middle_strip = roi_mask[-80:-55, :]

    bottom_indices = np.where(bottom_strip == 255)
    middle_indices = np.where(middle_strip == 255)

    # Draw strip regions on debug frame
    cv2.rectangle(debug_frame,
                  (0, height-140),
                  (width, height-115),
                  (255, 0, 255), 2)

    cv2.rectangle(debug_frame,
                  (0, height-80),
                  (width, height-55),
                  (0, 255, 255), 2)

    if len(bottom_indices[1]) > 0:

        bottom_cx = int(np.mean(bottom_indices[1]))
        bottom_error = bottom_cx - frame_center

        if len(middle_indices[1]) > 0:
            middle_cx = int(np.mean(middle_indices[1]))
            middle_error = middle_cx - frame_center
        else:
            middle_cx = bottom_cx
            middle_error = bottom_error

        error = 0.3 * bottom_error + 0.7 * middle_error
        last_error = error


        # ===== PID =====
        integral += error
        derivative = error - previous_error
        previous_error = error

        integral = max(-1000, min(1000, integral))

        turn = Kp * error + Ki * integral + Kd * derivative
        turn = max(-MAX_CONTROL, min(MAX_CONTROL, turn))

        left_speed = BASE_SPEED + turn
        right_speed = BASE_SPEED - turn

        set_motor(left_speed, right_speed)

        # ===== Draw bottom point =====
        bottom_y = height - 140 + 12   # middle of strip
        cv2.circle(debug_frame, (bottom_cx, bottom_y), 6, (0,0,255), -1)
        cv2.putText(debug_frame, "Bottom",
                    (bottom_cx + 5, bottom_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,0,255), 1)

        # ===== Draw middle point =====
        middle_y = height - 80 + 12
        cv2.circle(debug_frame, (middle_cx, middle_y), 6, (255,0,0), -1)
        cv2.putText(debug_frame, "Middle",
                    (middle_cx + 5, middle_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,0,0), 1)

        # Draw weighted error info
        cv2.putText(debug_frame,
                    f"Error: {int(error)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

    else:
        if last_error > 0:
            set_motor(175, -175)
        else:
            set_motor(-175, 175)
            
    cv2.imshow("ROI", roi_mask)
    cv2.imshow("DEBUG", debug_frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
set_motor(0,0)
pi.stop()
cv2.destroyAllWindows()
picam2.stop()
