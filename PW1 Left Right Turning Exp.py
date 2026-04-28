# -*- coding: utf-8 -*-

import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# ================= GPIO PINS =================
IN1 = 17
IN2 = 18
IN3 = 22
IN4 = 23
ENA = 12
ENB = 13

GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

# ================= CONSTANTS =================
PWM_FREQ = 1000
TURN_SPEED = 60

LEFT_TURN_TIME = {
    10: 0.18,
    20: 0.29,
    30: 0.36,
    40: 0.50,
    45: 0.53,
    50: 0.56,
    60: 0.69,
    70: 0.72,
    80: 0.91,
    90: 1.1,
    180: 2.25,
    360: 4.6
}

RIGHT_TURN_TIME = {
    10: 0.18,
    20: 0.27,
    30: 0.36,
    40: 0.45,
    45: 0.50,
    50: 0.53,
    60: 0.63,
    70: 0.72,
    80: 0.81,
    90: 1.10,
    180: 2.35,
    360: 4.50
}

# ================= PWM SETUP =================
pwmA = GPIO.PWM(ENA, PWM_FREQ)
pwmB = GPIO.PWM(ENB, PWM_FREQ)

pwmA.start(0)
pwmB.start(0)

# ================= FUNCTIONS =================
def stop():
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)

def left_by_angle(angle):
    if angle not in LEFT_TURN_TIME:
        print("Angle not found in LEFT data")
        return

    duration = LEFT_TURN_TIME[angle]

    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

    pwmA.ChangeDutyCycle(TURN_SPEED)
    pwmB.ChangeDutyCycle(TURN_SPEED)

    time.sleep(duration)
    stop()

    print("Left turn", angle, "deg completed")

def right_by_angle(angle):
    if angle not in RIGHT_TURN_TIME:
        print("Angle not found in RIGHT data")
        return

    duration = RIGHT_TURN_TIME[angle]

    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

    pwmA.ChangeDutyCycle(TURN_SPEED)
    pwmB.ChangeDutyCycle(TURN_SPEED)

    time.sleep(duration)
    stop()

    print("Right turn", angle, "deg completed")

# ================= MAIN LOOP =================
try:
    while True:
        cmd = input("l / r / q : ").lower()

        if cmd == "l":
            angle = int(input("Angle: "))
            left_by_angle(angle)

        elif cmd == "r":
            angle = int(input("Angle: "))
            right_by_angle(angle)

        elif cmd == "q":
            break

        else:
            print("Invalid command")

except KeyboardInterrupt:
    print("Stopped")

finally:
    stop()
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()
    print("GPIO cleaned")
