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

PWM_FREQ = 1000  # Hz

# ================= AVERAGE SPEED DATA (cm/s) =================
AVG_SPEED = {
    50: 37.1,
    60: 45.9,
    70: 52.0,
    80: 57.44,
    90: 61.12,
    100: 67.93
}

# ================= PWM SETUP =================
pwmA = GPIO.PWM(ENA, PWM_FREQ)
pwmB = GPIO.PWM(ENB, PWM_FREQ)
pwmA.start(0)
pwmB.start(0)

# ================= MOTOR FUNCTIONS =================
def stop():
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)

def forward(duty_cycle, duration):
    # Motor forward
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

    pwmA.ChangeDutyCycle(duty_cycle)
    pwmB.ChangeDutyCycle(duty_cycle)

    time.sleep(duration)
    stop()

# ================= MAIN LOOP =================
try:
    while True:
        # ===== Prompt for duty cycle =====
        dc_input = input("Enter duty cycle (50/60/70/80/90/100), press Enter to skip: ")
        if dc_input == "":
            break

        try:
            dc = int(dc_input)
        except ValueError:
            print("Invalid input. Enter a number like 50, 60, ...")
            continue

        if dc not in AVG_SPEED:
            print("Invalid duty cycle. Choose from 50, 60, 70, 80, 90, 100.")
            continue

        # ===== Calculate minimum distance for 1s motion =====
        speed = AVG_SPEED[dc]
        min_distance = speed * 1.0  # cm

        # ===== Prompt for distance =====
        dist_input = input(
            f"Enter distance in cm (must be at least {round(min_distance, 2)} cm): "
        )
        if dist_input == "":
            continue

        try:
            distance = float(dist_input)
        except ValueError:
            print("Invalid input. Enter a numeric value for distance.")
            continue

        if distance < min_distance:
            print(f"Distance too small. For {dc}% DC, minimum distance is {round(min_distance, 2)} cm")
            continue

        # ===== Calculate time needed =====
        time_needed = distance / speed
        print(f"Moving at {dc}% DC for {round(time_needed, 2)} seconds")

        # ===== Move motor =====
        forward(dc, time_needed)

except KeyboardInterrupt:
    print("Program stopped by user")

finally:
    stop()
    GPIO.cleanup()
    print("GPIO cleaned up")
