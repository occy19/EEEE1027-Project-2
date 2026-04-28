import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# ================= PIN DEFINITIONS =================
IN1 = 17
IN2 = 18
IN3 = 22
IN4 = 23
ENA = 12
ENB = 13

GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

# ================= EXPERIMENT CONSTANTS =================
freq = 1000
MOVE_SPEED = 100       # duty cycle for forward/back (%)
TURN_SPEED = 60       # duty cycle for turning (%)
MOVE_TIME = 1   # seconds

# ================= PWM SETUP (DUMMY INIT) =================
pwmA = GPIO.PWM(ENA, 100)   # temp frequency
pwmB = GPIO.PWM(ENB, 100)
pwmA.start(0)
pwmB.start(0)

# ================= MOTOR FUNCTIONS =================
def stop():
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)

def forward(speed=MOVE_SPEED, duration=MOVE_TIME):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwmA.ChangeDutyCycle(speed)
    pwmB.ChangeDutyCycle(speed)
    time.sleep(duration)
    stop()

def backward(speed=MOVE_SPEED, duration=MOVE_TIME):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(speed)
    pwmB.ChangeDutyCycle(speed)
    time.sleep(duration)
    stop()

def left(speed=TURN_SPEED, duration=MOVE_TIME):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(speed)
    pwmB.ChangeDutyCycle(speed)
    time.sleep(duration)
    stop()

def right(speed=TURN_SPEED, duration=MOVE_TIME):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwmA.ChangeDutyCycle(speed)
    pwmB.ChangeDutyCycle(speed)
    time.sleep(duration)
    stop()

# ================= MANUAL EXPERIMENT CONTROL =================
try:
    print("\n=== Time-based Motor Experiment ===")

    while True:
        
        pwmA.ChangeFrequency(freq)
        pwmB.ChangeFrequency(freq)


        print("Choose mode:")
        print("F = Forward | B = Backward | L = Left | R = Right | Q = Quit")
        cmd = input("Mode: ").lower()

        if cmd == "f":
            forward()
            print("Forward done")
        elif cmd == "b":
            backward()
            print("Backward done")
        elif cmd == "l":
            left()
            print("Left turn done")
        elif cmd == "r":
            right()
            print("Right turn done")
        else:
            print("Exiting experiment")
            break
        

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    stop()
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()
    print("GPIO cleaned up")
