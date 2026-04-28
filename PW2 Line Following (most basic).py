#PW2 Line following (most basic)


import cv2
from picamera2 import Picamera2
import numpy as np
import RPi.GPIO as GPIO
import time

# ======================
# GPIO SETUP (EDIT PINS IF NEEDED)
# ======================

IN1 = 17
IN2 = 27
IN3 = 22
IN4 = 23
ENA = 12   # PWM Left
ENB = 13   # PWM Right

GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

pwm_left = GPIO.PWM(ENA, 1000)
pwm_right = GPIO.PWM(ENB, 1000)

pwm_left.start(0)
pwm_right.start(0)

# ======================
# CAMERA SETUP
# ======================

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"size": (320, 240)})
)
picam2.start()

# ======================
# CONTROL PARAMETERS (TUNE THESE)
# ======================

BASE_SPEED = 40     # Try 30–60
KP = 0.4            # Try 0.2 – 1.0

# ======================
# MOTOR FUNCTIONS
# ======================

def forward(left_speed, right_speed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

    pwm_left.ChangeDutyCycle(left_speed)
    pwm_right.ChangeDutyCycle(right_speed)

def stop():
    pwm_left.ChangeDutyCycle(0)
    pwm_right.ChangeDutyCycle(0)

# ======================
# MAIN LOOP
# ======================

try:
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

                # Convert ROI x to full-frame reference
                error = cx - (w / 2)

                # Proportional control
                correction = KP * error

                left_speed = BASE_SPEED - correction
                right_speed = BASE_SPEED + correction

                # Clamp speeds (0–100)
                left_speed = max(0, min(100, left_speed))
                right_speed = max(0, min(100, right_speed))

                forward(left_speed, right_speed)

        else:
            # If no line detected
            stop()

        cv2.imshow("Line Following", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
    picam2.stop()
