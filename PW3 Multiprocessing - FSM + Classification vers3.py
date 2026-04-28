#!/usr/bin/env python3
"""
robot_v4.py  –  Multiprocessing line-following + symbol-detection robot.

Colour-line FSM states:
  S_IDLE           Normal black-line PID + Black Shortcut Exit Detection.
  S_COLOR_APPROACH Red only: steer toward red centroid until area cap.
  S_COLOR_CLASSIFY Red only: shape-check → arrow (hand off) or line (→ FOLLOW).
  S_COLOR_FOLLOW   Red/Yellow: PID on colour contour, black ignored.
  S_COLOR_EXIT     Spin in entry direction until black found.
  S_YELLOW_APPROACH Yellow only: PID toward yellow centroid until area cap.
  S_YELLOW_CLASSIFY Yellow only: "Slow roll" while ORB checks for Caution sign.
"""

from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import time
import pigpio
from collections import deque
import multiprocessing as mp
from multiprocessing import shared_memory
import os

# ══════════════════════════════════════════════════════════════════════
# GPIO PINS  (BCM, pigpio)
# ══════════════════════════════════════════════════════════════════════
IN1, IN2 = 17, 18
IN3, IN4 = 22, 23
ENA      = 12
ENB      = 13

MAX_SPEED = 180

# ══════════════════════════════════════════════════════════════════════
# FRAME CONFIG
# ══════════════════════════════════════════════════════════════════════
FRAME_W      = 480
FRAME_H      = 360
FRAME_SHAPE  = (FRAME_H, FRAME_W, 3)
FRAME_NBYTES = FRAME_H * FRAME_W * 3

LINE_DISP_SHAPE  = (FRAME_H, FRAME_W, 3)
LINE_DISP_NBYTES = FRAME_H * FRAME_W * 3

IMG_DISP_SHAPE  = (FRAME_H, FRAME_W, 3)
IMG_DISP_NBYTES = FRAME_H * FRAME_W * 3

TARGET_FPS        = 30
TARGET_FRAME_TIME = 1.0 / TARGET_FPS

# ══════════════════════════════════════════════════════════════════════
# PID / SPEED CONSTANTS
# ══════════════════════════════════════════════════════════════════════
Kp          = 1.2
Ki          = 0.001
Kd          = 1.0
BASE_SPEED  = 100
MAX_CONTROL = 90

ACTION_DURATION = {
    'Hand':    2.0,
    'Caution': 2.0,
    'Recycle': 3.0,
}

SPIN_SPEED         = 160
RED_SPIN_SPEED     = 150
COOLDOWN_AFTER_ACT = 1.5

# ── Colour-line tuning ────────────────────────────────────────────────
MIN_COLOR_LINE_AREA      = 6000
APPROACH_SPEED           = 150
MIN_YELLOW_AREA          = 500
MIN_YELLOW_CLASSIFY_AREA = 6000

ENTRY_CONFIRM_FRAMES = 3
LOST_TOLERANCE       = 4
EXIT_SPIN_TIMEOUT    = 3.0

# ══════════════════════════════════════════════════════════════════════
# SYMBOL & COLOR CONFIG
# ══════════════════════════════════════════════════════════════════════
SAMPLE_DICT = {
    "Caution":     (["templates/Caution.jpg"],     30),
    "Fingerprint": (["templates/Fingerprint.jpg"], 40),
    "Hand":        (["templates/Hand.jpg"],        35),
    "QR":          (["templates/QR.jpg", "templates/QR1.jpg",
                     "templates/QR2.jpg", "templates/QR3.jpg"], 20),
    "Recycle":     (["templates/Recycle.jpg", "templates/Recycle1.jpg",
                     "templates/Recycle2.jpg", "templates/Recycle3.jpg",
                     "templates/Recycle4.jpg"], 25),
}

# Standalone sign colours used by image_worker (Arrows)
COLOR_RANGES = {
    "Green":  {"space": "HSV", "lower": np.array([34, 140,  80]), "upper": np.array([74, 200, 180])},
    "Red":    {"space": "HSV", "lower": np.array([ 0, 200, 110]), "upper": np.array([179, 255, 230])},
    "Blue":   {"space": "LAB", "lower": np.array([104, 215, 115]), "upper": np.array([124, 255, 205])},
    "Orange": {"space": "HSV", "lower": np.array([ 2, 200, 180]), "upper": np.array([ 22, 255, 255])},
}

# Floor/track colours used by line_worker (Black, Red shortcut, Yellow shortcut)
LINE_COLOUR_RANGES = {
    "Red":    {"lower_1": np.array([0, 100, 100]),  "lower_2": np.array([160, 100, 100]),
               "upper_1": np.array([10, 255, 255]), "upper_2": np.array([180, 255, 255])},
    "Yellow": {"lower": np.array([23, 80, 80]),   "upper": np.array([40, 255, 255])},
    "Black":  {"lower": np.array([0, 0, 0]),      "upper": np.array([180, 255, 70])},
}

MIN_SCENE_KPS = 20

S_IDLE            = 0
S_COLOR_APPROACH  = 1   
S_COLOR_CLASSIFY  = 2   
S_COLOR_FOLLOW    = 3
S_COLOR_EXIT      = 4
S_YELLOW_CLASSIFY = 5   
S_YELLOW_APPROACH = 6   

CMD_NONE       = 0
CMD_EXIT_LEFT  = 1
CMD_EXIT_RIGHT = 2
CMD_ARROW_DONE = 3

# ══════════════════════════════════════════════════════════════════════
# MOTORS
# ══════════════════════════════════════════════════════════════════════
def setup_pigpio():
    pi = pigpio.pi()
    if not pi.connected: raise RuntimeError("Cannot connect to pigpiod!")
    for pin in (IN1, IN2, IN3, IN4, ENA, ENB):
        pi.set_mode(pin, pigpio.OUTPUT)
    pi.set_PWM_frequency(ENA, 1000)
    pi.set_PWM_frequency(ENB, 1000)
    return pi

def _set_motor_pi(pi, left_speed, right_speed):
    left_speed  = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
    right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))
    if left_speed >= 0:
        pi.write(IN1, 0); pi.write(IN2, 1); pi.set_PWM_dutycycle(ENA, int(left_speed))
    else:
        pi.write(IN1, 1); pi.write(IN2, 0); pi.set_PWM_dutycycle(ENA, int(abs(left_speed)))
    if right_speed >= 0:
        pi.write(IN3, 0); pi.write(IN4, 1); pi.set_PWM_dutycycle(ENB, int(right_speed))
    else:
        pi.write(IN3, 1); pi.write(IN4, 0); pi.set_PWM_dutycycle(ENB, int(abs(right_speed)))

def _stop_motors_pi(pi):
    _set_motor_pi(pi, 0, 0)

def best_contour(mask):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours: return None, 0
    c = max(contours, key=cv.contourArea)
    return c, cv.contourArea(c)

def _detect_shape(contour):
    """
    Returns ("Arrow", "Left"|"Right"|"Up") or ("Unknown", None).
    """
    peri = cv.arcLength(contour, True)
    if peri == 0: return "Unknown", None

    approx    = cv.approxPolyDP(contour, 0.02 * peri, True)
    v         = len(approx)
    area      = cv.contourArea(contour)
    hull_area = cv.contourArea(cv.convexHull(contour))
    if hull_area == 0: return "Unknown", None

    solidity  = area / hull_area
    is_convex = cv.isContourConvex(approx)
    circ      = 4 * np.pi * area / (peri * peri)

    if not (4 <= v <= 15 and not is_convex and 0.30 <= solidity <= 0.85 and circ >= 0.05):
        return "Unknown", None

    M = cv.moments(contour)
    if M["m00"] == 0: return "Arrow", "Unknown"
    cx_ = int(M["m10"] / M["m00"])
    cy_ = int(M["m01"] / M["m00"])

    hull_idx = cv.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3: return "Arrow", "Unknown"
    try:
        defects = cv.convexityDefects(contour, hull_idx)
    except cv.error:
        return "Arrow", "Unknown"
    if defects is None or len(defects) == 0: return "Arrow", "Unknown"

    max_depth, notch_x, notch_y = -1, cx_, cy_
    for defect in defects:
        s, e, f, d = defect[0]
        depth = d / 256.0
        if depth > max_depth:
            max_depth = depth
            notch_x   = contour[f][0][0]
            notch_y   = contour[f][0][1]

    # Compare X-notch distance vs Y-notch distance to see if arrow points horizontal or vertical
    dx = notch_x - cx_
    dy = notch_y - cy_

    if abs(dx) > abs(dy):
        return "Arrow", ("Left" if dx > 0 else "Right")
    else:
        # An arrow pointing Up has its tail pointing Down (notch Y is below centroid Y)
        # Note: image y-axis goes down, so bottom of image is higher Y value.
        return "Arrow", ("Up" if dy > 0 else "Down")


# ══════════════════════════════════════════════════════════════════════
# LINE WORKER
# ══════════════════════════════════════════════════════════════════════
def line_worker(
    shm_name, frame_lock, shared_fid, my_fid,
    out_left, out_right, out_has_line,
    out_turn_cmd, out_colour_label, is_on_color_line,
    out_label, active_shortcut_dir, disp_shm_name, disp_lock,
):
    shm      = shared_memory.SharedMemory(name=shm_name)
    fbuf     = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)
    disp_shm = shared_memory.SharedMemory(name=disp_shm_name)
    disp_buf = np.ndarray(LINE_DISP_SHAPE, dtype=np.uint8, buffer=disp_shm.buf)

    prev_error      = 0.0
    integral        = 0.0
    last_error      = 0.0

    fsm_state       = S_IDLE
    active_colour   = None    
    entry_direction = None    
    entry_confirm_n = 0
    lost_frames     = 0
    yellow_wait_frames = 0

    local_shortcut_dir = 0
    shortcut_frames    = 0

    def _write_colour_label(s: str):
        enc = s.encode()[:63]
        out_colour_label.raw = enc + b'\x00' * (64 - len(enc))

    _write_colour_label("")
    print("[LineWorker] Started.")

    while True:
        fid = shared_fid.value
        if fid == my_fid.value:
            time.sleep(0.002)
            continue
        my_fid.value = fid

        with frame_lock:
            frame_rgb = fbuf.copy()

        frame_bgr    = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        height, width = frame_bgr.shape[:2]
        frame_center  = width // 2

        hsv = cv.cvtColor(cv.GaussianBlur(frame_bgr, (3, 3), 0), cv.COLOR_BGR2HSV)

        # ── Black, Red, Yellow Masks ──────────────────────────────────
        mask_black  = cv.inRange(hsv, LINE_COLOUR_RANGES["Black"]["lower"],
                                      LINE_COLOUR_RANGES["Black"]["upper"])
        mask_yellow = cv.inRange(hsv, LINE_COLOUR_RANGES["Yellow"]["lower"],
                                      LINE_COLOUR_RANGES["Yellow"]["upper"])
        mask_red    = cv.bitwise_or(
            cv.inRange(hsv, LINE_COLOUR_RANGES["Red"]["lower_1"],
                            LINE_COLOUR_RANGES["Red"]["upper_1"]),
            cv.inRange(hsv, LINE_COLOUR_RANGES["Red"]["lower_2"],
                            LINE_COLOUR_RANGES["Red"]["upper_2"]),
        )

        mask_black[0:150, :] = 0
        mask_yellow[0:150, :] = 0
        mask_red[0:150, :] = 0

        cnt_black,  area_black  = best_contour(mask_black)
        cnt_yellow, area_yellow = best_contour(mask_yellow)
        cnt_red,    area_red    = best_contour(mask_red)

        has_red    = area_red > 0
        has_yellow = area_yellow >= MIN_YELLOW_AREA

        if fsm_state in (S_COLOR_APPROACH, S_COLOR_CLASSIFY, S_COLOR_FOLLOW, S_COLOR_EXIT, S_YELLOW_APPROACH, S_YELLOW_CLASSIFY):
            if active_colour == "Red":
                cnt_colour, area_colour, seen_colour = cnt_red, area_red, "Red" if has_red else None
            else:
                cnt_colour, area_colour, seen_colour = cnt_yellow, area_yellow, "Yellow" if has_yellow else None
        else:
            if has_red:
                seen_colour, cnt_colour, area_colour = "Red", cnt_red, area_red
            elif has_yellow:
                seen_colour, cnt_colour, area_colour = "Yellow", cnt_yellow, area_yellow
            else:
                seen_colour, cnt_colour, area_colour = None, None, 0

        left_out, right_out = 0.0, 0.0
        has_line = False

        # ── FSM ───────────────────────────────────────────────────────
        if fsm_state == S_IDLE:
            is_on_color_line.value = False

            # Manage Black Shortcut state
            if active_shortcut_dir.value != local_shortcut_dir:
                local_shortcut_dir = active_shortcut_dir.value
                shortcut_frames = 0
            
            if local_shortcut_dir != 0:
                shortcut_frames += 1
                # Wait ~1.5 seconds into the shortcut before scanning for T-Junction
                if shortcut_frames > 45 and cnt_black is not None:
                    x, y, w, h = cv.boundingRect(cnt_black)
                    # If line spreads horizontally across the screen -> Junction Exit
                    if w > 250 and (w / max(1, h)) > 1.5 and area_black > 4000:
                        print(f"[LineWorker] T-Junction Exit detected! Spitting out.")
                        out_turn_cmd.value = CMD_EXIT_LEFT if local_shortcut_dir == 1 else CMD_EXIT_RIGHT
                        active_shortcut_dir.value = 0
                        local_shortcut_dir = 0

            if seen_colour == "Red":
                active_colour = "Red"
                fsm_state     = S_COLOR_APPROACH
                print(f"[LineWorker] Red spotted — APPROACHING")

            elif seen_colour == "Yellow":
                active_colour = "Yellow"
                fsm_state     = S_YELLOW_APPROACH
                print(f"[LineWorker] Yellow spotted — APPROACHING")

            else:
                # Normal black PID
                if cnt_black is not None:
                    has_line = True
                    M  = cv.moments(cnt_black)
                    cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else frame_center
                    cv.drawContours(frame_bgr, [cnt_black], -1, (0, 255, 0), 3)
                    error      = frame_center - cx
                    last_error = error
                    integral   = max(-500, min(500, integral + error))
                    derivative = error - prev_error
                    prev_error = error
                    turn       = max(-MAX_CONTROL, min(MAX_CONTROL,
                                     Kp * error + Ki * integral + Kd * derivative))
                    dyn_base   = max(30, BASE_SPEED - abs(turn) * 0.3)
                    left_out   = dyn_base + turn
                    right_out  = dyn_base - turn
                else:
                    left_out, right_out = (150, -150) if last_error > 0 else (-150, 150)

        elif fsm_state == S_COLOR_APPROACH:
            is_on_color_line.value = False
            if cnt_colour is None or area_colour == 0:
                fsm_state, active_colour = S_IDLE, None
            elif area_colour >= MIN_COLOR_LINE_AREA:
                fsm_state = S_COLOR_CLASSIFY
            else:
                M = cv.moments(cnt_colour)
                cx_c = int(M['m10'] / M['m00']) if M['m00'] != 0 else frame_center
                has_line = True
                cv.drawContours(frame_bgr, [cnt_colour], -1, (0, 165, 255), 2)
                error      = frame_center - cx_c
                last_error = error
                integral   = max(-500, min(500, integral + error))
                derivative = error - prev_error
                prev_error = error
                turn       = max(-MAX_CONTROL, min(MAX_CONTROL, Kp * error + Ki * integral + Kd * derivative))
                dyn_base   = max(30, APPROACH_SPEED - abs(turn) * 0.3)
                left_out, right_out = dyn_base + turn, dyn_base - turn

        elif fsm_state == S_COLOR_CLASSIFY:
            is_on_color_line.value = False
            if cnt_colour is None or area_colour == 0:
                fsm_state = S_COLOR_APPROACH
            else:
                shape, direction = _detect_shape(cnt_colour)
                if shape == "Arrow" and direction not in (None, "Unknown"):
                    print(f"[LineWorker] Red ARROW → {direction}. Handing off.")
                    _write_colour_label(direction)
                    out_turn_cmd.value = CMD_ARROW_DONE
                    fsm_state          = S_IDLE
                    active_colour      = None
                else:
                    print(f"[LineWorker] Red LINE confirmed. FOLLOWING.")
                    fsm_state, entry_direction, lost_frames = S_COLOR_FOLLOW, None, 0

        elif fsm_state == S_YELLOW_APPROACH:
            is_on_color_line.value = False
            if cnt_colour is None or area_colour == 0:
                fsm_state, active_colour = S_IDLE, None
            elif area_colour >= MIN_YELLOW_CLASSIFY_AREA:
                fsm_state, yellow_wait_frames = S_YELLOW_CLASSIFY, 0
            else:
                M = cv.moments(cnt_colour)
                cx_c = int(M['m10'] / M['m00']) if M['m00'] != 0 else frame_center
                has_line = True
                cv.drawContours(frame_bgr, [cnt_colour], -1, (0, 255, 255), 2)
                error      = frame_center - cx_c
                last_error = error
                integral   = max(-500, min(500, integral + error))
                derivative = error - prev_error
                prev_error = error
                turn       = max(-MAX_CONTROL, min(MAX_CONTROL, Kp * error + Ki * integral + Kd * derivative))
                dyn_base   = max(30, APPROACH_SPEED - abs(turn) * 0.3)
                left_out, right_out = dyn_base + turn, dyn_base - turn

        elif fsm_state == S_YELLOW_CLASSIFY:
            is_on_color_line.value = False
            if cnt_colour is None or area_colour == 0:
                fsm_state, active_colour = S_IDLE, None
            else:
                yellow_wait_frames += 1
                M = cv.moments(cnt_colour)
                cx_c = int(M['m10'] / M['m00']) if M['m00'] != 0 else frame_center
                has_line = True
                cv.drawContours(frame_bgr, [cnt_colour], -1, (0, 255, 255), 2)
                error      = frame_center - cx_c
                last_error = error
                integral   = max(-500, min(500, integral + error))
                derivative = error - prev_error
                prev_error = error
                turn       = max(-MAX_CONTROL, min(MAX_CONTROL, Kp * error + Ki * integral + Kd * derivative))
                dyn_base   = max(25, BASE_SPEED * 0.6 - abs(turn) * 0.3)
                left_out, right_out = dyn_base + turn, dyn_base - turn

                current_label = out_label.raw.rstrip(b'\x00').decode(errors='ignore')
                if current_label == "Caution":
                    fsm_state, active_colour = S_IDLE, None
                elif yellow_wait_frames > 10:
                    fsm_state, entry_direction, lost_frames = S_COLOR_FOLLOW, None, 0

        elif fsm_state == S_COLOR_FOLLOW:
            is_on_color_line.value = True
            if cnt_colour is not None and area_colour > 0:
                lost_frames = 0
                M = cv.moments(cnt_colour)
                cx_c = int(M['m10'] / M['m00']) if M['m00'] != 0 else frame_center
                draw_col = (0, 0, 255) if active_colour == "Red" else (0, 255, 255)
                cv.drawContours(frame_bgr, [cnt_colour], -1, draw_col, 3)
                has_line = True
                error      = frame_center - cx_c
                last_error = error
                integral   = max(-500, min(500, integral + error))
                derivative = error - prev_error
                prev_error = error
                turn       = max(-MAX_CONTROL, min(MAX_CONTROL, Kp * error + Ki * integral + Kd * derivative))
                dyn_base   = max(30, BASE_SPEED - abs(turn) * 0.3)
                left_out, right_out = dyn_base + turn, dyn_base - turn

                if entry_direction is None:
                    candidate = "Left" if cx_c < frame_center else "Right"
                    entry_confirm_n += 1
                    if entry_confirm_n >= ENTRY_CONFIRM_FRAMES:
                        entry_direction = candidate
            else:
                lost_frames += 1
                left_out, right_out, has_line = out_left.value, out_right.value, True
                if lost_frames > LOST_TOLERANCE:
                    fsm_state = S_COLOR_EXIT
                    out_turn_cmd.value = CMD_EXIT_LEFT if entry_direction == "Left" else CMD_EXIT_RIGHT

        elif fsm_state == S_COLOR_EXIT:
            is_on_color_line.value = True
            left_out, right_out, has_line = 0.0, 0.0, False
            if out_turn_cmd.value == CMD_NONE:
                fsm_state, active_colour, entry_direction, lost_frames, integral, prev_error, last_error = S_IDLE, None, None, 0, 0.0, 0.0, 0.0

        out_left.value     = float(left_out)
        out_right.value    = float(right_out)
        out_has_line.value = has_line

        with disp_lock:
            state_names = {0: "IDLE", 1: "APPROACH", 2: "CLASSIFY", 3: "FOLLOW", 4: "EXIT", 5: "Y_SLOW", 6: "Y_APPR"}
            cv.putText(frame_bgr, f"FSM:{state_names.get(fsm_state,'?')} {active_colour or ''}", (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            if entry_direction: cv.putText(frame_bgr, f"ENTRY:{entry_direction}", (5, 40), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
            np.copyto(disp_buf, frame_bgr)


# ══════════════════════════════════════════════════════════════════════
# IMAGE-RECOGNITION WORKER
# ══════════════════════════════════════════════════════════════════════
def image_worker(
    shm_name, frame_lock, shared_fid, my_fid,
    out_found, out_label,
    is_on_color_line, colour_label_arr,
    disp_shm_name, disp_lock,
):
    shm      = shared_memory.SharedMemory(name=shm_name)
    fbuf     = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)
    disp_shm = shared_memory.SharedMemory(name=disp_shm_name)
    disp_buf = np.ndarray(IMG_DISP_SHAPE, dtype=np.uint8, buffer=disp_shm.buf)

    orb = cv.ORB_create(nfeatures=500, scaleFactor=1.2)
    bf  = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    reference_data = {}
    for sym_name, (paths, thresh) in SAMPLE_DICT.items():
        des_list = []
        for path in paths:
            if os.path.exists(path):
                img = cv.imread(path, cv.IMREAD_GRAYSCALE)
                if img is not None:
                    kp, des = orb.detectAndCompute(img, None)
                    if des is not None: des_list.append(des)
        reference_data[sym_name] = {"des_list": des_list, "thresh": thresh}

    label_history    = deque(maxlen=3)
    cooldown_counter = 0

    def _write(found: bool, label: str):
        out_found.value = found
        enc = label.encode()[:63]
        out_label.raw = enc + b'\x00' * (64 - len(enc))

    while True:
        fid = shared_fid.value
        if fid == my_fid.value:
            time.sleep(0.005)
            continue
        my_fid.value = fid

        colour_arrow = colour_label_arr.raw.rstrip(b'\x00').decode(errors='replace')
        if colour_arrow in ("Left", "Right", "Up"):
            colour_label_arr.raw = b'\x00' * 64
            if cooldown_counter == 0:
                _write(True, colour_arrow)
                cooldown_counter = 15
            continue

        with frame_lock:
            frame_rgb = fbuf.copy()

        frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        h, w      = frame_bgr.shape[:2]
        blurred   = cv.GaussianBlur(frame_bgr, (5, 5), 0)
        hsv       = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
        lab       = cv.cvtColor(blurred, cv.COLOR_BGR2LAB)

        found = False
        label = ""

        # ORB Check
        gray_scene   = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        kp_s, des_s  = orb.detectAndCompute(gray_scene, None)
        matched_name = None

        if reference_data and des_s is not None and len(kp_s) >= MIN_SCENE_KPS:
            for sym_name, ref in reference_data.items():
                for ref_des in ref["des_list"]:
                    matches = bf.knnMatch(ref_des, des_s, k=2)
                    good = sum(1 for m in matches if len(m) == 2 and m[0].distance < 0.80 * m[1].distance)
                    if good >= ref["thresh"]:
                        matched_name = sym_name
                        break
                if matched_name: break

        if matched_name:
            label = matched_name
            found = True
        elif not is_on_color_line.value:
            # Color blob arrow check
            all_candidates = []
            for colour_name, params in COLOR_RANGES.items():
                if params.get("space") == "LAB": mask = cv.inRange(lab, params["lower"], params["upper"])
                else:                            mask = cv.inRange(hsv, params["lower"], params["upper"])
                mask = cv.erode(mask,  None, iterations=1)
                mask = cv.dilate(mask, None, iterations=2)
                for cnt in cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]:
                    area = cv.contourArea(cnt)
                    if 1500 <= area <= 25000:
                        all_candidates.append((area, cnt, colour_name))

            all_candidates.sort(key=lambda x: x[0], reverse=True)
            for area, cnt, colour_name in all_candidates[:3]:
                x_b, y_b, w_b, h_b = cv.boundingRect(cnt)
                if (x_b < 15 or y_b < 15 or (x_b + w_b) > (w - 15) or (y_b + h_b) > (h - 15)): continue
                aspect_ratio = float(w_b) / float(h_b) if h_b > 0 else 0
                if aspect_ratio < 0.4 or aspect_ratio > 2.5: continue
                shape, direction = _detect_shape(cnt)
                if shape == "Arrow":
                    label = direction
                    found = True
                    break

        safe_label = ""
        if found and cooldown_counter == 0:
            label_history.append(label)
            if (len(label_history) == label_history.maxlen and len(set(label_history)) == 1):
                safe_label = label_history[0]
                label_history.clear()
                cooldown_counter = 10
        else:
            if cooldown_counter > 0: cooldown_counter -= 1
            label_history.clear()

        display = frame_bgr.copy()
        if safe_label: cv.putText(display, safe_label, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 2)
        elif label:    cv.putText(display, f"? {label}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2)

        with disp_lock: np.copyto(disp_buf, display)
        _write(bool(safe_label), safe_label)


# ══════════════════════════════════════════════════════════════════════
# MAIN EXECUTIONS
# ══════════════════════════════════════════════════════════════════════
def _execute_arrow_turn(pi, direction_str, picam2):
    print(f"\n[main] ARROW TURN {direction_str} ...\n")
    
    if direction_str == "Up":
        # Simply drive straight forward to cross the junction.
        # Adjust time.sleep(0.7) if it crosses too far or too short.
        _set_motor_pi(pi, BASE_SPEED, BASE_SPEED)
        time.sleep(0.7)
        _stop_motors_pi(pi)
        return

    # For Left/Right turns:
    if direction_str == "Left":
        l_spd, r_spd = -SPIN_SPEED, SPIN_SPEED
    else:
        l_spd, r_spd =  SPIN_SPEED, -SPIN_SPEED

    _set_motor_pi(pi, l_spd, r_spd)
    time.sleep(0.7)  # Blind turn fix to face away from the arrow

    start = time.time()
    found_black = False
    while time.time() - start < EXIT_SPIN_TIMEOUT:
        rgb = picam2.capture_array()
        if rgb.ndim == 3 and rgb.shape[2] == 4: rgb = rgb[:, :, :3]
        frame_bgr  = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        hsv        = cv.cvtColor(cv.GaussianBlur(frame_bgr, (3, 3), 0), cv.COLOR_BGR2HSV)
        mask_black = cv.inRange(hsv, LINE_COLOUR_RANGES["Black"]["lower"], LINE_COLOUR_RANGES["Black"]["upper"])
        mask_black[0:150, :] = 0
        _, area_black = best_contour(mask_black)
        if area_black > 3000:
            found_black = True
            break
        time.sleep(0.01)

    _stop_motors_pi(pi)
    time.sleep(0.15)


def _execute_exit_spin(pi, direction_str, picam2):
    print(f"\n[main] EXIT SPIN {direction_str} ...\n")
    l_spd = -RED_SPIN_SPEED if direction_str == "Left" else RED_SPIN_SPEED
    r_spd =  RED_SPIN_SPEED if direction_str == "Left" else -RED_SPIN_SPEED
    _set_motor_pi(pi, l_spd, r_spd)

    start = time.time()
    found_black = False
    while time.time() - start < EXIT_SPIN_TIMEOUT:
        rgb = picam2.capture_array()
        if rgb.ndim == 3 and rgb.shape[2] == 4: rgb = rgb[:, :, :3]
        frame_bgr  = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        hsv        = cv.cvtColor(cv.GaussianBlur(frame_bgr, (3, 3), 0), cv.COLOR_BGR2HSV)
        mask_black = cv.inRange(hsv, LINE_COLOUR_RANGES["Black"]["lower"], LINE_COLOUR_RANGES["Black"]["upper"])
        mask_black[0:150, :] = 0
        _, area_black = best_contour(mask_black)
        if area_black > 3000:
            found_black = True
            break
        time.sleep(0.01)

    _stop_motors_pi(pi)
    time.sleep(0.15)


ENABLE_GUI = True

def main():
    shm  = shared_memory.SharedMemory(create=True, size=FRAME_NBYTES)
    fbuf = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)

    line_disp_shm  = shared_memory.SharedMemory(create=True, size=LINE_DISP_NBYTES)
    img_disp_shm   = shared_memory.SharedMemory(create=True, size=IMG_DISP_NBYTES)
    line_disp_buf  = np.ndarray(LINE_DISP_SHAPE, dtype=np.uint8, buffer=line_disp_shm.buf)
    img_disp_buf   = np.ndarray(IMG_DISP_SHAPE,  dtype=np.uint8, buffer=img_disp_shm.buf)
    line_disp_lock = mp.Lock()
    img_disp_lock  = mp.Lock()

    frame_lock = mp.Lock()
    shared_fid = mp.Value('i',  0)
    line_fid   = mp.Value('i', -1)
    img_fid    = mp.Value('i', -1)

    out_left            = mp.Value('d', 0.0)
    out_right           = mp.Value('d', 0.0)
    out_has_line        = mp.Value('b', False)
    out_found           = mp.Value('b', False)
    out_label           = mp.Array('c', 64)
    out_turn_cmd        = mp.Value('i', CMD_NONE)
    out_colour_label    = mp.Array('c', 64)
    is_on_color_line    = mp.Value('b', False)
    
    # 0 = None, 1 = Left, 2 = Right
    active_shortcut_dir = mp.Value('i', 0)

    p_line = mp.Process(
        target=line_worker,
        args=(shm.name, frame_lock, shared_fid, line_fid,
              out_left, out_right, out_has_line,
              out_turn_cmd, out_colour_label, is_on_color_line,
              out_label, active_shortcut_dir, line_disp_shm.name, line_disp_lock),
        daemon=True, name="LineWorker",
    )
    p_img = mp.Process(
        target=image_worker,
        args=(shm.name, frame_lock, shared_fid, img_fid,
              out_found, out_label,
              is_on_color_line, out_colour_label,
              img_disp_shm.name, img_disp_lock),
        daemon=True, name="ImgWorker",
    )
    p_line.start()
    p_img.start()

    pi = setup_pigpio()
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (FRAME_W, FRAME_H)}))
    picam2.start()
    time.sleep(2.0)

    try:
        meta = picam2.capture_metadata()
        picam2.set_controls({
            "AeEnable":     False,
            "ExposureTime": meta.get("ExposureTime", 20000),
            "AnalogueGain": meta.get("AnalogueGain", 1.0),
            "AwbEnable":    False,
            "ColourGains":  meta.get("ColourGains", (1.5, 1.5)),
        })
    except Exception as e:
        print(f"[main] Warning – could not lock exposure/AWB: {e}")

    robot_state    = 'LINE_FOLLOW'
    action_start   = 0.0
    cooldown_until = 0.0

    print("[main] Running. Press 'q' to quit.")

    try:
        while True:
            loop_start = time.time()

            rgb = picam2.capture_array()
            if rgb.ndim == 3 and rgb.shape[2] == 4: rgb = rgb[:, :, :3]
            with frame_lock: np.copyto(fbuf, rgb)
            shared_fid.value += 1

            left     = out_left.value
            right    = out_right.value
            has_line = bool(out_has_line.value)
            found    = bool(out_found.value)
            label    = out_label.raw.rstrip(b'\x00').decode(errors='replace')
            now      = time.time()

            turn_cmd = out_turn_cmd.value
            if turn_cmd in (CMD_EXIT_LEFT, CMD_EXIT_RIGHT):
                out_turn_cmd.value = CMD_NONE
                direction_str = "Left" if turn_cmd == CMD_EXIT_LEFT else "Right"
                _stop_motors_pi(pi)
                _execute_exit_spin(pi, direction_str, picam2)
                cooldown_until = time.time() + COOLDOWN_AFTER_ACT
                elapsed = time.time() - loop_start
                if (TARGET_FRAME_TIME - elapsed) > 0: time.sleep(TARGET_FRAME_TIME - elapsed)
                continue
            elif turn_cmd == CMD_ARROW_DONE:
                out_turn_cmd.value = CMD_NONE

            in_cooldown = now < cooldown_until

            if found and label and robot_state == 'LINE_FOLLOW' and not in_cooldown:
                # Terminal-only prints, no pausing line tracking
                if label in ('QR', 'Fingerprint'):
                    print(f"[main] >>> Action: {label} detected. Logging and continuing.")
                    cooldown_until = now + COOLDOWN_AFTER_ACT
                else:
                    robot_state  = label
                    action_start = now
                    print(f"[main] >>> Action: {label}")

            if robot_state != 'LINE_FOLLOW':
                duration = ACTION_DURATION.get(robot_state, 0.0)
                if now - action_start >= duration:
                    robot_state    = 'LINE_FOLLOW'
                    cooldown_until = now + COOLDOWN_AFTER_ACT
                    print("[main] Back to line tracking.")

            if robot_state == 'LINE_FOLLOW':
                _set_motor_pi(pi, left, right)
            elif robot_state in ('Hand', 'Caution'):
                _stop_motors_pi(pi)
            elif robot_state == 'Recycle':
                _set_motor_pi(pi, SPIN_SPEED, -SPIN_SPEED)
            elif robot_state in ('Left', 'Right', 'Up'):
                _stop_motors_pi(pi)
                _execute_arrow_turn(pi, robot_state, picam2)
                
                # Remember standard turns for Black Shortcuts
                if robot_state == 'Left':  active_shortcut_dir.value = 1
                if robot_state == 'Right': active_shortcut_dir.value = 2
                
                robot_state    = 'LINE_FOLLOW'
                cooldown_until = time.time() + COOLDOWN_AFTER_ACT

            if ENABLE_GUI:
                with line_disp_lock: cv.imshow("Line View", line_disp_buf.copy())
                with img_disp_lock:  cv.imshow("Symbol View", img_disp_buf.copy())
                if cv.waitKey(1) & 0xFF == ord('q'): break

            elapsed = time.time() - loop_start
            if (TARGET_FRAME_TIME - elapsed) > 0: time.sleep(TARGET_FRAME_TIME - elapsed)

    except KeyboardInterrupt: print("\n[main] Ctrl+C – shutting down.")
    except Exception as e:
        import traceback; traceback.print_exc()
    finally:
        print("[main] Cleaning up…")
        try: _stop_motors_pi(pi); pi.stop()
        except: pass
        p_line.terminate(); p_line.join()
        p_img.terminate();  p_img.join()
        try: shm.unlink(); line_disp_shm.unlink(); img_disp_shm.unlink()
        except: pass
        try: picam2.stop()
        except: pass
        if ENABLE_GUI: cv.destroyAllWindows()
        print("[main] Done.")

if __name__ == "__main__":
    mp.set_start_method("forkserver")
    main()
