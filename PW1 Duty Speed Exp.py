import numpy as np

# ================= DATA =================
duty_cycles = np.array([50, 60, 70, 80, 90, 100])
speeds = np.array([37.1, 45.9, 52.0, 57.44, 61.12, 67.93])

# ================= HELPER FUNCTIONS =================
def speed_to_duty(speed_value):
    """Given speed, return interpolated duty cycle"""
    if speed_value < speeds[0] or speed_value > speeds[-1]:
        print("Speed out of range ({0} - {1} cm/s)".format(speeds[0], speeds[-1]))
        return None
    duty = np.interp(speed_value, speeds, duty_cycles)
    return duty

def duty_to_speed(duty_value):
    """Given duty cycle, return interpolated speed"""
    if duty_value < duty_cycles[0] or duty_value > duty_cycles[-1]:
        print("Duty cycle out of range ({0} - {1} %)".format(duty_cycles[0], duty_cycles[-1]))
        return None
    speed = np.interp(duty_value, duty_cycles, speeds)
    return speed

# ================= MAIN LOOP =================
try:
    while True:
        mode = input("Select mode: 1 = speed to duty, 2 = duty to speed, Enter to exit: ")
        if mode == "":
            break

        if mode == "1":
            speed_input = input("Enter speed (cm/s): ")
            try:
                speed_val = float(speed_input)
            except ValueError:
                print("Invalid input. Enter a number.")
                continue
            duty_val = speed_to_duty(speed_val)
            if duty_val is not None:
                print("Required duty cycle is approximately {0:.2f} %".format(duty_val))

        elif mode == "2":
            duty_input = input("Enter duty cycle (%): ")
            try:
                duty_val = float(duty_input)
            except ValueError:
                print("Invalid input. Enter a number.")
                continue
            speed_val = duty_to_speed(duty_val)
            if speed_val is not None:
                print("Expected speed is approximately {0:.2f} cm/s".format(speed_val))

        else:
            print("Invalid mode. Choose 1 or 2.")

except KeyboardInterrupt:
    print("\nProgram stopped by user")
