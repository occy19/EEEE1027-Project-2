f#!/usr/bin/env python3
"""
robot_v4.py  –  Multiprocessing line-following + symbol-detection robot.

Key changes vs ccy2.py:
  • Yellow line  — NO approach/classify phase. Any yellow area detected
                   goes directly to S_COLOR_FOLLOW. Yellow caution sign
                   is caught by ORB template matching in image_worker and
                   is not affected by the line FSM.
  • Red line     — Approach + classify unchanged from ccy2.py (arrow-vs-line
                   determination, min-cap area gate, etc.).
  • FPS control  — Set TARGET_FPS at the top. Main loop throttles itself
                   with a sleep to hit that rate. Workers are frame-ID
                   driven so they follow automatically.

Colour-line FSM states:
  S_IDLE           Normal black-line PID.
  S_COLOR_APPROACH Red only: steer toward red centroid until area ≥ MIN_COLOR_LINE_AREA.
  S_COLOR_CLASSIFY Red only: shape-check → arrow (hand off) or line (→ FOLLOW).
  S_COLOR_FOLLOW   PID on colour contour, black ignored, entry direction locked.
  S_COLOR_EXIT     Spin in entry direction until black found, then back to IDLE.
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

# ══════════════════════════════════════════════════════════════════════
# FPS CONTROL
# Change TARGET_FPS to whatever rate you want (e.g. 20, 25, 30).
# The main loop will sleep to hit this target each iteration.
# ══════════════════════════════════════════════════════════════════════
TARGET_FPS        = 30
TARGET_FRAME_TIME = 1.0 / TARGET_FPS   # seconds per frame

# ══════════════════════════════════════════════════════════════════════
# PID / SPEED CONSTANTS
# ══════════════════════════════════════════════════════════════════════
Kp          = 1.2
Ki          = 0.001
Kd          = 1.0
BASE_SPEED  = 100
MAX_CONTROL = 90

# ── Action durations (seconds) ────────────────────────────────────────
ACTION_DURATION = {
    'Hand':        2.0,
    'Caution':     2.0,
    'QR':          1.0,
    'Fingerprint': 1.0,
    'Recycle':     3.0,
    'Left':        0.9,
    'Right':       0.9,
}

SPIN_SPEED         = 160
RED_SPIN_SPEED     = 150
COOLDOWN_AFTER_ACT = 1.5

# ── Colour-line tuning ────────────────────────────────────────────────
# RED: area gate before classify begins (approach phase).
MIN_COLOR_LINE_AREA = 6000
# RED: area that conclusively means "line" when shape is ambiguous.
LINE_AREA_THRESHOLD = 8000
# RED: slow creep speed during approach.
APPROACH_SPEED      = 150

# YELLOW: minimum area to trigger direct line-follow (skip approach/classify).
# Set low so even a partially visible yellow line starts following immediately.
MIN_YELLOW_AREA     = 500

# Entry direction: frames of stable centroid before locking.
ENTRY_CONFIRM_FRAMES = 3
# Consecutive frames of colour loss tolerated before declaring line end.
LOST_TOLERANCE       = 4
# Maximum seconds the exit spin will run searching for black.
EXIT_SPIN_TIMEOUT    = 3.0

# ══════════════════════════════════════════════════════════════════════
# SYMBOL & COLOR CONFIG
# ══════════════════════════════════════════════════════════════════════
SAMPLE_DICT = {
    "Caution":     (["templates/Caution.jpg"],     30),
    "Fingerprint": (["templates/Fingerprint.jpg"], 40),
    "Hand":        (["templates/Hand.jpg"],        50),
    "QR":          (["templates/QR.jpg", "templates/QR1.jpg",
                     "templates/QR2.jpg", "templates/QR3.jpg"], 30),
    "Recycle":     (["templates/Recycle.jpg", "templates/Recycle1.jpg",
                     "templates/Recycle2.jpg", "templates/Recycle3.jpg",
                     "templates/Recycle4.jpg"], 35),
}

# Arrow-sign colours used by image_worker
COLOR_RANGES = {
    "Green":  {"space": "HSV", "lower": np.array([34, 140,  80]), "upper": np.array([74, 200, 180])},
    "Red":    {"space": "HSV", "lower": np.array([ 0, 200, 110]), "upper": np.array([179, 255, 230])},
    "Blue":   {"space": "LAB", "lower": np.array([104, 215, 115]), "upper": np.array([124, 255, 205])},
    "Orange": {"space": "HSV", "lower": np.array([ 2, 200, 180]), "upper": np.array([ 22, 255, 255])},
}

# Track-line colours used by line_worker
LINE_COLOUR_RANGES = {
    "Red":    {"lower_1": np.array([0, 100, 100]),  "lower_2": np.array([160, 100, 100]),
               "upper_1": np.array([10, 255, 255]), "upper_2": np.array([180, 255, 255])},
    "Yellow": {"lower": np.array([20, 80, 80]),  "upper": np.array([40, 255, 255])},
    "Black":  {"lower": np.array([0, 0, 0]),     "upper": np.array([180, 255, 70])},
}

MIN_SCENE_KPS = 20

# ── FSM state IDs ─────────────────────────────────────────────────────
S_IDLE           = 0
S_COLOR_APPROACH = 1   # Red only
S_COLOR_CLASSIFY = 2   # Red only
S_COLOR_FOLLOW   = 3
S_COLOR_EXIT     = 4

# ── Commands from line_worker → main() ────────────────────────────────
CMD_NONE       = 0
CMD_EXIT_LEFT  = 1
CMD_EXIT_RIGHT = 2
CMD_ARROW_DONE = 3   # Red classified as arrow; image_worker picks up direction

# ══════════════════════════════════════════════════════════════════════
# PIGPIO MOTOR HELPERS
# ══════════════════════════════════════════════════════════════════════
def setup_pigpio():
    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError("Cannot connect to pigpiod – is it running?")
    for pin in (IN1, IN2, IN3, IN4, ENA, ENB):
        pi.set_mode(pin, pigpio.OUTPUT)
    pi.set_PWM_frequency(ENA, 1000)
    pi.set_PWM_frequency(ENB, 1000)
    return pi

def _set_motor_pi(pi, left_speed, right_speed):
    left_speed  = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
    right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))
    if left_speed >= 0:
        pi.write(IN1, 0); pi.write(IN2, 1)
        pi.set_PWM_dutycycle(ENA, int(left_speed))
    else:
        pi.write(IN1, 1); pi.write(IN2, 0)
        pi.set_PWM_dutycycle(ENA, int(abs(left_speed)))
    if right_speed >= 0:
        pi.write(IN3, 0); pi.write(IN4, 1)
        pi.set_PWM_dutycycle(ENB, int(right_speed))
    else:
        pi.write(IN3, 1); pi.write(IN4, 0)
        pi.set_PWM_dutycycle(ENB, int(abs(right_speed)))

def _stop_motors_pi(pi):
    _set_motor_pi(pi, 0, 0)

# ══════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════
def best_contour(mask):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0
    c = max(contours, key=cv.contourArea)
    return c, cv.contourArea(c)

# ══════════════════════════════════════════════════════════════════════
# ARROW SHAPE + DIRECTION DETECTOR  (Red only)
# ══════════════════════════════════════════════════════════════════════
def _detect_shape(contour):
    """
    Returns ("Arrow", "Left"|"Right") or ("Unknown", None).
    Uses deepest convex-hull defect (tail notch) for direction:
        notch RIGHT of centroid → tip points LEFT  → "Left"
        notch LEFT  of centroid → tip points RIGHT → "Right"
    """
    peri = cv.arcLength(contour, True)
    if peri == 0:
        return "Unknown", None

    approx    = cv.approxPolyDP(contour, 0.02 * peri, True)
    v         = len(approx)
    area      = cv.contourArea(contour)
    hull_area = cv.contourArea(cv.convexHull(contour))
    if hull_area == 0:
        return "Unknown", None

    solidity  = area / hull_area
    is_convex = cv.isContourConvex(approx)
    circ      = 4 * np.pi * area / (peri * peri)

    if not (4 <= v <= 15 and not is_convex and 0.30 <= solidity <= 0.85 and circ >= 0.05):
        return "Unknown", None

    M = cv.moments(contour)
    if M["m00"] == 0:
        return "Arrow", "Unknown"
    cx_ = int(M["m10"] / M["m00"])

    hull_idx = cv.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return "Arrow", "Unknown"
    try:
        defects = cv.convexityDefects(contour, hull_idx)
    except cv.error:
        return "Arrow", "Unknown"
    if defects is None or len(defects) == 0:
        return "Arrow", "Unknown"

    max_depth, notch_x = -1, cx_
    for defect in defects:
        s, e, f, d = defect[0]
        depth = d / 256.0
        if depth > max_depth:
            max_depth = depth
            notch_x   = contour[f][0][0]

    return "Arrow", ("Left" if notch_x > cx_ else "Right")

# ══════════════════════════════════════════════════════════════════════
# LINE-FOLLOWING WORKER PROCESS
# ══════════════════════════════════════════════════════════════════════
def line_worker(
    shm_name, frame_lock, shared_fid, my_fid,
    out_left, out_right, out_has_line,
    out_turn_cmd,      # CMD_* → consumed by main()
    out_colour_label,  # written when red classified as arrow
    is_on_color_line,  # True during COLOR_FOLLOW / COLOR_EXIT
    disp_shm_name, disp_lock,
):
    shm      = shared_memory.SharedMemory(name=shm_name)
    fbuf     = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)
    disp_shm = shared_memory.SharedMemory(name=disp_shm_name)
    disp_buf = np.ndarray(LINE_DISP_SHAPE, dtype=np.uint8, buffer=disp_shm.buf)

    prev_error      = 0.0
    integral        = 0.0
    last_error      = 0.0

    fsm_state       = S_IDLE
    active_colour   = None    # "Red" | "Yellow"
    entry_direction = None    # "Left" | "Right"
    entry_confirm_n = 0
    lost_frames     = 0

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

        # ── Masks ─────────────────────────────────────────────────────
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

        # Horse blinders
        mask_black [0:150, :] = 0
        mask_yellow[0:150, :] = 0
        mask_red   [0:150, :] = 0

        cnt_black,  area_black  = best_contour(mask_black)
        cnt_yellow, area_yellow = best_contour(mask_yellow)
        cnt_red,    area_red    = best_contour(mask_red)

        # ── Colour selection for FSM ───────────────────────────────────
        # Red and yellow handled with different thresholds.
        # Red needs area > 0 to start approach.
        # Yellow needs area > MIN_YELLOW_AREA to start follow directly.
        has_red    = area_red    > 0
        has_yellow = area_yellow >= MIN_YELLOW_AREA

        # Determine which colour contour is "active" for current FSM state
        if fsm_state in (S_COLOR_APPROACH, S_COLOR_CLASSIFY, S_COLOR_FOLLOW, S_COLOR_EXIT):
            # Already tracking — use whichever we locked onto
            if active_colour == "Red":
                cnt_colour  = cnt_red
                area_colour = area_red
                seen_colour = "Red" if has_red else None
            else:  # Yellow
                cnt_colour  = cnt_yellow
                area_colour = area_yellow
                seen_colour = "Yellow" if has_yellow else None
        else:
            # IDLE — pick whichever appears first (red takes priority if both)
            if has_red:
                seen_colour = "Red"
                cnt_colour  = cnt_red
                area_colour = area_red
            elif has_yellow:
                seen_colour = "Yellow"
                cnt_colour  = cnt_yellow
                area_colour = area_yellow
            else:
                seen_colour = None
                cnt_colour  = None
                area_colour = 0

        # ── Default frame outputs ──────────────────────────────────────
        left_out, right_out = 0.0, 0.0
        has_line = False

        # ══════════════════════════════════════════════════════════════
        # FSM
        # ══════════════════════════════════════════════════════════════

        # ── S_IDLE ────────────────────────────────────────────────────
        if fsm_state == S_IDLE:
            is_on_color_line.value = False

            if seen_colour == "Red":
                # Red: start approach phase
                active_colour = "Red"
                fsm_state     = S_COLOR_APPROACH
                print(f"[LineWorker] Red spotted — APPROACHING")

            elif seen_colour == "Yellow":
                # Yellow: skip approach/classify, go straight to follow
                active_colour   = "Yellow"
                entry_direction = None
                entry_confirm_n = 0
                lost_frames     = 0
                fsm_state       = S_COLOR_FOLLOW
                print(f"[LineWorker] Yellow spotted — FOLLOW directly")

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

        # ── S_COLOR_APPROACH  (Red only) ──────────────────────────────
        elif fsm_state == S_COLOR_APPROACH:
            is_on_color_line.value = False

            if not has_red or cnt_colour is None:
                print("[LineWorker] Red lost during approach — IDLE")
                fsm_state     = S_IDLE
                active_colour = None
            elif area_colour >= MIN_COLOR_LINE_AREA:
                fsm_state = S_COLOR_CLASSIFY
                print(f"[LineWorker] Min cap reached ({area_colour}px) — CLASSIFYING")
            else:
                M = cv.moments(cnt_colour)
                cx_c = int(M['m10'] / M['m00']) if M['m00'] != 0 else frame_center
                if cx_c < frame_center:
                    left_out, right_out = APPROACH_SPEED , -APPROACH_SPEED
                else:
                    left_out, right_out = -APPROACH_SPEED, APPROACH_SPEED 
                has_line = True
                cv.drawContours(frame_bgr, [cnt_colour], -1, (0, 165, 255), 2)

        # ── S_COLOR_CLASSIFY  (Red only) ──────────────────────────────
        elif fsm_state == S_COLOR_CLASSIFY:
            is_on_color_line.value = False

            if not has_red or cnt_colour is None:
                fsm_state = S_COLOR_APPROACH
            else:
                shape, direction = _detect_shape(cnt_colour)

                if shape == "Arrow" and direction not in (None, "Unknown"):
                    print(f"[LineWorker] Red ARROW → {direction}. Back to IDLE.")
                    _write_colour_label(direction)
                    out_turn_cmd.value = CMD_ARROW_DONE
                    fsm_state          = S_IDLE
                    active_colour      = None
                    entry_direction    = None
                    entry_confirm_n    = 0

                else:
                    # Not an arrow (shape Unknown or above area threshold) → LINE
                    print(f"[LineWorker] Red LINE confirmed. FOLLOWING.")
                    fsm_state       = S_COLOR_FOLLOW
                    entry_direction = None
                    entry_confirm_n = 0
                    lost_frames     = 0

        # ── S_COLOR_FOLLOW  (Red and Yellow) ──────────────────────────
        elif fsm_state == S_COLOR_FOLLOW:
            is_on_color_line.value = True

            colour_present = (
                (active_colour == "Red"    and has_red    and cnt_colour is not None) or
                (active_colour == "Yellow" and has_yellow and cnt_colour is not None)
            )

            if colour_present:
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
                turn       = max(-MAX_CONTROL, min(MAX_CONTROL,
                                 Kp * error + Ki * integral + Kd * derivative))
                dyn_base   = max(30, BASE_SPEED - abs(turn) * 0.3)
                left_out   = dyn_base + turn
                right_out  = dyn_base - turn

                # Lock entry direction after ENTRY_CONFIRM_FRAMES
                if entry_direction is None:
                    candidate = "Left" if cx_c < frame_center else "Right"
                    entry_confirm_n += 1
                    if entry_confirm_n >= ENTRY_CONFIRM_FRAMES:
                        entry_direction = candidate
                        print(f"[LineWorker] Entry direction LOCKED: {entry_direction}")
            else:
                lost_frames += 1
                left_out  = out_left.value
                right_out = out_right.value
                has_line  = True

                if lost_frames > LOST_TOLERANCE:
                    print(f"[LineWorker] {active_colour} LOST — EXIT {entry_direction}")
                    fsm_state = S_COLOR_EXIT
                    if entry_direction == "Left":
                        out_turn_cmd.value = CMD_EXIT_LEFT
                    elif entry_direction == "Right":
                        out_turn_cmd.value = CMD_EXIT_RIGHT
                    else:
                        out_turn_cmd.value = CMD_EXIT_RIGHT  # fallback

        # ── S_COLOR_EXIT ──────────────────────────────────────────────
        elif fsm_state == S_COLOR_EXIT:
            is_on_color_line.value = True
            left_out  = 0.0
            right_out = 0.0
            has_line  = False

            # Wait for main() to clear the command after spin completes
            if out_turn_cmd.value == CMD_NONE:
                print(f"[LineWorker] Exit done — IDLE. Forget {active_colour}.")
                fsm_state       = S_IDLE
                active_colour   = None
                entry_direction = None
                entry_confirm_n = 0
                lost_frames     = 0
                integral        = 0.0
                prev_error      = 0.0
                last_error      = 0.0

        # ── Write outputs ──────────────────────────────────────────────
        out_left.value     = float(left_out)
        out_right.value    = float(right_out)
        out_has_line.value = has_line

        with disp_lock:
            state_names = {
                S_IDLE: "IDLE", S_COLOR_APPROACH: "APPROACH",
                S_COLOR_CLASSIFY: "CLASSIFY", S_COLOR_FOLLOW: "FOLLOW",
                S_COLOR_EXIT: "EXIT",
            }
            cv.putText(frame_bgr, f"FSM:{state_names.get(fsm_state,'?')} {active_colour or ''}",
                       (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            if entry_direction:
                cv.putText(frame_bgr, f"ENTRY:{entry_direction}",
                           (5, 40), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
            np.copyto(disp_buf, frame_bgr)


# ══════════════════════════════════════════════════════════════════════
# IMAGE-RECOGNITION WORKER PROCESS
# ══════════════════════════════════════════════════════════════════════
def image_worker(
    shm_name, frame_lock, shared_fid, my_fid,
    out_found, out_label,
    is_on_color_line,
    colour_label_arr,   # written by line_worker when red arrow classified
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
                    if des is not None:
                        des_list.append(des)
            else:
                print(f"[ImgWorker] WARNING: template not found – '{path}'.")
        reference_data[sym_name] = {"des_list": des_list, "thresh": thresh}

    label_history    = deque(maxlen=3)
    cooldown_counter = 0

    def _write(found: bool, label: str):
        out_found.value = found
        enc = label.encode()[:63]
        out_label.raw = enc + b'\x00' * (64 - len(enc))

    print("[ImgWorker] Started.")

    while True:
        fid = shared_fid.value
        if fid == my_fid.value:
            time.sleep(0.005)
            continue
        my_fid.value = fid

        # ── Red arrow handed off from line_worker ─────────────────────
        colour_arrow = colour_label_arr.raw.rstrip(b'\x00').decode(errors='replace')
        if colour_arrow in ("Left", "Right"):
            colour_label_arr.raw = b'\x00' * 64   # consume
            if cooldown_counter == 0:
                print(f"[ImgWorker] Red arrow from line_worker: {colour_arrow}")
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

        # ── ORB symbol detection ──────────────────────────────────────
        gray_scene   = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        kp_s, des_s  = orb.detectAndCompute(gray_scene, None)
        matched_name = None

        if reference_data and des_s is not None and len(kp_s) >= MIN_SCENE_KPS:
            for sym_name, ref in reference_data.items():
                for ref_des in ref["des_list"]:
                    matches = bf.knnMatch(ref_des, des_s, k=2)
                    good = sum(
                        1 for m in matches
                        if len(m) == 2 and m[0].distance < 0.80 * m[1].distance
                    )
                    if good >= ref["thresh"]:
                        matched_name = sym_name
                        break
                if matched_name:
                    break

        if matched_name:
            label = matched_name
            found = True
        elif not is_on_color_line.value:
            # Arrow detection only when NOT following a colour line
            all_candidates = []
            for colour_name, params in COLOR_RANGES.items():
                if params.get("space") == "LAB":
                    mask = cv.inRange(lab, params["lower"], params["upper"])
                else:
                    mask = cv.inRange(hsv, params["lower"], params["upper"])
                mask = cv.erode(mask,  None, iterations=1)
                mask = cv.dilate(mask, None, iterations=2)
                for cnt in cv.findContours(
                        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]:
                    area = cv.contourArea(cnt)
                    if 1500 <= area <= 25000:
                        all_candidates.append((area, cnt, colour_name))

            all_candidates.sort(key=lambda x: x[0], reverse=True)
            for area, cnt, colour_name in all_candidates[:3]:
                x_b, y_b, w_b, h_b = cv.boundingRect(cnt)
                if (x_b < 15 or y_b < 15 or
                        (x_b + w_b) > (w - 15) or (y_b + h_b) > (h - 15)):
                    continue
                aspect_ratio = float(w_b) / float(h_b) if h_b > 0 else 0
                if aspect_ratio < 0.4 or aspect_ratio > 2.5:
                    continue
                shape, direction = _detect_shape(cnt)
                if shape == "Arrow":
                    label = direction
                    found = True
                    break

        # ── Confirmation buffer ───────────────────────────────────────
        safe_label = ""
        if found and cooldown_counter == 0:
            label_history.append(label)
            if (len(label_history) == label_history.maxlen and
                    len(set(label_history)) == 1):
                safe_label = label_history[0]
                print(f"[ImgWorker] Detected: {safe_label}")
                label_history.clear()
                cooldown_counter = 10
        else:
            if cooldown_counter > 0:
                cooldown_counter -= 1
            label_history.clear()

        display = frame_bgr.copy()
        if safe_label:
            cv.putText(display, safe_label, (10, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 2)
        elif label:
            cv.putText(display, f"? {label}", (10, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2)

        with disp_lock:
            np.copyto(disp_buf, display)
        _write(bool(safe_label), safe_label)


# ══════════════════════════════════════════════════════════════════════
# EXIT SPIN  — spin until black found or timeout
# ══════════════════════════════════════════════════════════════════════
def _execute_exit_spin(pi, direction_str, picam2):
    print(f"\n[main] EXIT SPIN {direction_str} — searching for black line…\n")
    l_spd = -RED_SPIN_SPEED if direction_str == "Left" else RED_SPIN_SPEED
    r_spd =  RED_SPIN_SPEED if direction_str == "Left" else -RED_SPIN_SPEED
    _set_motor_pi(pi, l_spd, r_spd)

    start       = time.time()
    found_black = False
    while time.time() - start < EXIT_SPIN_TIMEOUT:
        rgb = picam2.capture_array()
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        frame_bgr  = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        hsv        = cv.cvtColor(cv.GaussianBlur(frame_bgr, (3, 3), 0), cv.COLOR_BGR2HSV)
        mask_black = cv.inRange(hsv, LINE_COLOUR_RANGES["Black"]["lower"],
                                      LINE_COLOUR_RANGES["Black"]["upper"])
        mask_black[0:150, :] = 0
        _, area_black = best_contour(mask_black)
        if area_black > 3000:
            found_black = True
            break
        time.sleep(0.01)

    _stop_motors_pi(pi)
    print("[main] Black line found.\n" if found_black else "[main] Timeout — resuming.\n")
    time.sleep(0.15)


# ══════════════════════════════════════════════════════════════════════
# MAIN PROCESS
# ══════════════════════════════════════════════════════════════════════
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

    out_left         = mp.Value('d', 0.0)
    out_right        = mp.Value('d', 0.0)
    out_has_line     = mp.Value('b', False)
    out_found        = mp.Value('b', False)
    out_label        = mp.Array('c', 64)
    out_turn_cmd     = mp.Value('i', CMD_NONE)
    out_colour_label = mp.Array('c', 64)
    is_on_color_line = mp.Value('b', False)

    p_line = mp.Process(
        target=line_worker,
        args=(shm.name, frame_lock, shared_fid, line_fid,
              out_left, out_right, out_has_line,
              out_turn_cmd, out_colour_label, is_on_color_line,
              line_disp_shm.name, line_disp_lock),
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
    print(f"[main] Workers started. Target FPS: {TARGET_FPS}")

    pi = setup_pigpio()
    print("[main] pigpio connected.")

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"size": (FRAME_W, FRAME_H)}))
    picam2.start()
    time.sleep(2.0)

    try:
        meta            = picam2.capture_metadata()
        locked_exposure = meta.get("ExposureTime", 20000)
        locked_gain     = meta.get("AnalogueGain",  1.0)
        picam2.set_controls({
            "AeEnable":     False,
            "ExposureTime": locked_exposure,
            "AnalogueGain": locked_gain,
        })
    except Exception as e:
        print(f"[main] Warning – could not lock exposure: {e}")

    robot_state    = 'LINE_FOLLOW'
    action_start   = 0.0
    cooldown_until = 0.0

    print("[main] Running. Press 'q' to quit.")

    try:
        while True:
            loop_start = time.time()   # ← FPS throttle reference

            rgb = picam2.capture_array()
            if rgb.ndim == 3 and rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]

            with frame_lock:
                np.copyto(fbuf, rgb)
            shared_fid.value += 1

            left     = out_left.value
            right    = out_right.value
            has_line = bool(out_has_line.value)
            found    = bool(out_found.value)
            label    = out_label.raw.rstrip(b'\x00').decode(errors='replace')
            now      = time.time()

            # ── Colour-line commands ───────────────────────────────────
            turn_cmd = out_turn_cmd.value

            if turn_cmd in (CMD_EXIT_LEFT, CMD_EXIT_RIGHT):
                out_turn_cmd.value = CMD_NONE   # clear before spin so FSM resets
                direction_str = "Left" if turn_cmd == CMD_EXIT_LEFT else "Right"
                _stop_motors_pi(pi)
                _execute_exit_spin(pi, direction_str, picam2)
                cooldown_until = time.time() + COOLDOWN_AFTER_ACT
                # Recalculate elapsed so FPS sleep accounts for spin time
                elapsed = time.time() - loop_start
                sleep_t = TARGET_FRAME_TIME - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
                continue

            elif turn_cmd == CMD_ARROW_DONE:
                out_turn_cmd.value = CMD_NONE   # image_worker picks up colour_label

            # ── Symbol / arrow actions ─────────────────────────────────
            in_cooldown = now < cooldown_until

            if found and label and robot_state == 'LINE_FOLLOW' and not in_cooldown:
                if label == 'Up':
                    cooldown_until = now + COOLDOWN_AFTER_ACT
                else:
                    robot_state  = label
                    action_start = now
                    print(f"[main] >>> Action: {label}")

            if robot_state != 'LINE_FOLLOW':
                duration = ACTION_DURATION.get(robot_state, 2.0)
                if now - action_start >= duration:
                    robot_state    = 'LINE_FOLLOW'
                    cooldown_until = now + COOLDOWN_AFTER_ACT
                    print("[main] Back to line tracking.")

            # ── Motor output ───────────────────────────────────────────
            if robot_state == 'LINE_FOLLOW':
                _set_motor_pi(pi, left, right)
            elif robot_state in ('Hand', 'Caution'):
                _stop_motors_pi(pi)
            elif robot_state == 'Recycle':
                _set_motor_pi(pi, SPIN_SPEED, -SPIN_SPEED)
            elif robot_state == 'Left':
                _set_motor_pi(pi, -SPIN_SPEED, SPIN_SPEED)
            elif robot_state == 'Right':
                _set_motor_pi(pi, SPIN_SPEED, -SPIN_SPEED)

            # ── GUI ────────────────────────────────────────────────────
            if ENABLE_GUI:
                with line_disp_lock: line_frame = line_disp_buf.copy()
                with img_disp_lock:  img_frame  = img_disp_buf.copy()
                cv.imshow("Line View",   line_frame)
                cv.imshow("Symbol View", img_frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

            # ── FPS throttle ───────────────────────────────────────────
            elapsed = time.time() - loop_start
            sleep_t = TARGET_FRAME_TIME - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[main] Ctrl+C – shutting down.")
    except Exception as e:
        import traceback
        print(f"[main] FATAL: {e}")
        traceback.print_exc()
    finally:
        print("[main] Cleaning up…")
        try:
            _stop_motors_pi(pi)
            pi.stop()
        except Exception:
            pass
        p_line.terminate(); p_line.join()
        p_img.terminate();  p_img.join()
        try:
            shm.close();           shm.unlink()
            line_disp_shm.close(); line_disp_shm.unlink()
            img_disp_shm.close();  img_disp_shm.unlink()
        except Exception:
            pass
        try:
            picam2.stop()
        except Exception:
            pass
        if ENABLE_GUI:
            cv.destroyAllWindows()
        print("[main] Done.")


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    main()
