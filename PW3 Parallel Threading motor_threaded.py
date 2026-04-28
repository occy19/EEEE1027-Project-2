"""
motor_ctrl.py — L298N motor driver using pigpio.
Split from original line detection code.
"""

import pigpio

# ================= GPIO PINS =================
IN1 = 17
IN2 = 18
IN3 = 22
IN4 = 23
ENA = 12
ENB = 13

MAX_SPEED = 180

pi = pigpio.pi()
if not pi.connected:
    print("pigpio not connected")
    exit()

for pin in [IN1, IN2, IN3, IN4, ENA, ENB]:
    pi.set_mode(pin, pigpio.OUTPUT)

pi.set_PWM_frequency(ENA, 500)
pi.set_PWM_frequency(ENB, 500)


def set_motor(left_speed, right_speed):
    left_speed  = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
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


def stop_motors():
    set_motor(0, 0)


def cleanup():
    stop_motors()
    pi.stop()
