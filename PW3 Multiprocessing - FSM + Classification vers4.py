#!/usr/bin/env python3
"""
robot_v5.py  –  Multiprocessing line-following + symbol-detection robot.

Changes vs ccy3__1_.py  (problems 1,2,3,4,6 only — prob 5 deferred):
  1. Arrow turn now arcs forward (one motor full, other reduced) instead of
     spinning in place.  Constants: ARROW_ARC_OUTER / ARROW_ARC_INNER.
  2. After red arrow classified+done, a per-frame cooldown in line_worker
     (RED_ARROW_COOLDOWN_FRAMES) blocks re-entering S_COLOR_APPROACH so the
     car can't immediately re-detect the same red region as a line.
  3. S_COLOR_APPROACH uses pure-ratio steering (no PID / no MAX_CONTROL cap)
     so it always steers hard enough to close in on the red line regardless
     of approach angle.
  4. After Caution or red-arrow action completes, a shared flag
     colour_approach_blocked is set True for COLOUR_BLOCK_AFTER_ACTION
     seconds.  line_worker S_IDLE skips colour approach while blocked,
     so a nearby yellow line doesn't get swallowed by the post-action
     cooldown window.  The flag auto-clears in main() after the timer.
  6. MIN_SCENE_KPS lowered 20→10, Lowe's ratio 0.80→0.75,
     label_history maxlen 3→2 for faster symbol detection at speed.
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

# ── FIX 1: Arrow arc-turn speeds ─────────────────────────────────────
# Outer wheel runs at ARROW_ARC_OUTER, inner wheel at ARROW_ARC_INNER.
# Car moves forward while turning — increase ARROW_ARC_OUTER or decrease
# ARROW_ARC_INNER to tighten the arc radius.
ARROW_ARC_OUTER  = SPIN_SPEED          # 160  — the faster side
ARROW_ARC_INNER  = int(SPIN_SPEED * 0.25)  # 40 — the slower forward side
EXIT_SPIN_TIMEOUT = 3.0

# ── FIX 2: Red-arrow re-detect cooldown (frames) ─────────────────────
# After a red arrow is classified, block re-entering S_COLOR_APPROACH
# for this many frames so the car cannot immediately re-read the same
# red region as a line.
RED_ARROW_COOLDOWN_FRAMES = 20

# ── FIX 4: Colour-approach block after action (seconds) ──────────────
# After Caution or red-arrow action, colour approach is blocked for this
# duration.  Keep short so a nearby yellow line is not skipped entirely.
COLOUR_BLOCK_AFTER_ACTION = 2.0

# ── Colour-line tuning ────────────────────────────────────────────────
MIN_COLOR_LINE_AREA      = 6000
APPROACH_SPEED           = 150
MIN_YELLOW_AREA          = 500
MIN_YELLOW_CLASSIFY_AREA = 6000

ENTRY_CONFIRM_FRAMES = 3
LOST_TOLERANCE       = 4

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

COLOR_RANGES = {
    "Green":  {"space": "HSV", "lower": np.array([34, 140,  80]), "upper": np.array([74, 200, 180])},
    "Red":    {"space": "HSV", "lower": np.array([ 0, 200, 110]), "upper": np.array([179, 255, 230])},
    "Blue":   {"space": "LAB", "lower": np.array([104, 215, 115]), "upper": np.array([124, 255, 205])},
    "Orange": {"space": "HSV", "lower": np.array([ 2, 200, 180]), "upper": np.array([ 22, 255, 255])},
}

LINE_COLOUR_RANGES = {
    "Red":    {"lower_1": np.array([0, 100, 100]),  "lower_2": np.array([160, 100, 100]),
               "upper_1": np.array([10, 255, 255]), "upper_2": np.array([180, 255, 255])},
    "Yellow": {"lower": np.array([23, 80, 80]),  "upper": np.array([40, 255, 255])},
    "Black":  {"lower": np.array([0, 0, 0]),     "upper": np.array([180, 255, 70])},
}

# FIX 6: lower scene KP threshold and Lowe ratio for faster detection at speed
MIN_SCENE_KPS  = 10      # was 20
LOWE_RATIO     = 0.75    # was 0.80
CONFIRM_FRAMES = 2       # label_history maxlen; was 3

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
    """Returns ("Arrow", "Left"|"Right"|"Up") or ("Unknown", None)."""
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
    dx = notch_x - cx_
    dy = notch_y - cy_
    if abs(dx) > abs(dy):
        return "Arrow", ("Left" if dx > 0 else "Right")
    else:
        return "Arrow", ("Up" if dy > 0 else "Down")


# ══════════════════════════════════════════════════════════════════════
# LINE WORKER
# ══════════════════════════════════════════════════════════════════════
def line_worker(
    shm_name, frame_lock, shared_fid, my_fid,
    out_left, out_right, out_has_line,
    out_turn_cmd, out_colour_label, is_on_color_line,
    out_label, active_shortcut_dir,
    colour_approach_blocked,          # FIX 4: shared flag from main()
    disp_shm_name, disp_lock,
):
    shm      = shared_memory.SharedMemory(name=shm_name)
    fbuf     = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)
    disp_shm = shared_memory.SharedMemory(name=disp_shm_name)
    disp_buf = np.ndarray(LINE_DISP_SHAPE, dtype=np.uint8, buffer=disp_shm.buf)

    prev_error      = 0.0
    integral        = 0.0
    last_error      = 0.0

    fsm_state          = S_IDLE
    active_colour      = None
    entry_direction    = None
    entry_confirm_n    = 0
    lost_frames        = 0
    yellow_wait_frames = 0

    local_shortcut_dir = 0
    shortcut_frames    = 0

    # FIX 2: frames remaining before red approach is allowed again
    red_arrow_cooldown = 0

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

        frame_bgr     = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        height, width = frame_bgr.shape[:2]
        frame_center  = width // 2

        hsv = cv.cvtColor(cv.GaussianBlur(frame_bgr, (3, 3), 0), cv.COLOR_BGR2HSV)

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

        mask_black [0:150, :] = 0
        mask_yellow[0:150, :] = 0
        mask_red   [0:150, :] = 0

        cnt_black,  area_black  = best_contour(mask_black)
        cnt_yellow, area_yellow = best_contour(mask_yellow)
        cnt_red,    area_red    = best_contour(mask_red)

        has_red    = area_red > 0
        has_yellow = area_yellow >= MIN_YELLOW_AREA

        if fsm_state in (S_COLOR_APPROACH, S_COLOR_CLASSIFY, S_COLOR_FOLLOW,
                         S_COLOR_EXIT, S_YELLOW_APPROACH, S_YELLOW_CLASSIFY):
            if active_colour == "Red":
                cnt_colour, area_colour = cnt_red,    area_red
                seen_colour = "Red"    if has_red    else None
            else:
                cnt_colour, area_colour = cnt_yellow, area_yellow
                seen_colour = "Yellow" if has_yellow else None
        else:
            if has_red:
                seen_colour, cnt_colour, area_colour = "Red",    cnt_red,    area_red
            elif has_yellow:
                seen_colour, cnt_colour, area_colour = "Yellow", cnt_yellow, area_yellow
            else:
                seen_colour, cnt_colour, area_colour = None, None, 0

        left_out, right_out = 0.0, 0.0
        has_line = False

        # ── Decrement red-arrow cooldown every frame ──────────────────
        if red_arrow_cooldown > 0:
            red_arrow_cooldown -= 1

        # ══════════════════════════════════════════════════════════════
        # FSM
        # ══════════════════════════════════════════════════════════════

        if fsm_state == S_IDLE:
            is_on_color_line.value = False

            # Black-shortcut junction exit detection (unchanged)
            if active_shortcut_dir.value != local_shortcut_dir:
                local_shortcut_dir = active_shortcut_dir.value
                shortcut_frames    = 0
            if local_shortcut_dir != 0:
                shortcut_frames += 1
                if shortcut_frames > 45 and cnt_black is not None:
                    x, y, w, h = cv.boundingRect(cnt_black)
                    if w > 250 and (w / max(1, h)) > 1.5 and area_black > 4000:
                        print("[LineWorker] T-Junction Exit detected!")
                        out_turn_cmd.value = CMD_EXIT_LEFT if local_shortcut_dir == 1 else CMD_EXIT_RIGHT
                        active_shortcut_dir.value = 0
                        local_shortcut_dir = 0

            # FIX 4: only attempt colour approach when not blocked by main()
            colour_blocked = bool(colour_approach_blocked.value)

            if seen_colour == "Red" and not colour_blocked and red_arrow_cooldown == 0:
                # FIX 2: red_arrow_cooldown must be 0 before re-entering approach
                active_colour = "Red"
                fsm_state     = S_COLOR_APPROACH
                print("[LineWorker] Red spotted — APPROACHING")

            elif seen_colour == "Yellow" and not colour_blocked:
                active_colour = "Yellow"
                fsm_state     = S_YELLOW_APPROACH
                print("[LineWorker] Yellow spotted — APPROACHING")

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

        # ── S_COLOR_APPROACH (Red) ────────────────────────────────────
        elif fsm_state == S_COLOR_APPROACH:
            is_on_color_line.value = False
            if cnt_colour is None or area_colour == 0:
                fsm_state, active_colour = S_IDLE, None
            elif area_colour >= MIN_COLOR_LINE_AREA:
                fsm_state = S_COLOR_CLASSIFY
                print(f"[LineWorker] Red min-cap reached ({area_colour}px) — CLASSIFYING")
            else:
                # FIX 3: pure-ratio proportional steer — no PID, no MAX_CONTROL cap.
                # Guarantees aggressive enough steering at any approach angle.
                M    = cv.moments(cnt_colour)
                cx_c = int(M['m10'] / M['m00']) if M['m00'] != 0 else frame_center
                cv.drawContours(frame_bgr, [cnt_colour], -1, (0, 165, 255), 2)
                has_line = True
                if cx_c < frame_center:
                    # Red is LEFT of center → steer left
                    left_out  = APPROACH_SPEED * 0.4
                    right_out = APPROACH_SPEED
                else:
                    # Red is RIGHT of center → steer right
                    left_out  = APPROACH_SPEED
                    right_out = APPROACH_SPEED * 0.4

        # ── S_COLOR_CLASSIFY (Red) ────────────────────────────────────
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
                    # FIX 2: arm the cooldown so this red region can't re-trigger
                    red_arrow_cooldown = RED_ARROW_COOLDOWN_FRAMES
                    fsm_state          = S_IDLE
                    active_colour      = None
                else:
                    print("[LineWorker] Red LINE confirmed. FOLLOWING.")
                    fsm_state, entry_direction, entry_confirm_n, lost_frames = \
                        S_COLOR_FOLLOW, None, 0, 0

        # ── S_YELLOW_APPROACH ─────────────────────────────────────────
        elif fsm_state == S_YELLOW_APPROACH:
            is_on_color_line.value = False
            if cnt_colour is None or area_colour == 0:
                fsm_state, active_colour = S_IDLE, None
            elif area_colour >= MIN_YELLOW_CLASSIFY_AREA:
                fsm_state, yellow_wait_frames = S_YELLOW_CLASSIFY, 0
            else:
                M    = cv.moments(cnt_colour)
                cx_c = int(M['m10'] / M['m00']) if M['m00'] != 0 else frame_center
                has_line = True
                cv.drawContours(frame_bgr, [cnt_colour], -1, (0, 255, 255), 2)
                error      = frame_center - cx_c
                last_error = error
                integral   = max(-500, min(500, integral + error))
                derivative = error - prev_error
                prev_error = error
                turn       = max(-MAX_CONTROL, min(MAX_CONTROL,
                                 Kp * error + Ki * integral + Kd * derivative))
                dyn_base   = max(30, APPROACH_SPEED - abs(turn) * 0.3)
                left_out, right_out = dyn_base + turn, dyn_base - turn

        # ── S_YELLOW_CLASSIFY ─────────────────────────────────────────
        elif fsm_state == S_YELLOW_CLASSIFY:
            is_on_color_line.value = False
            if cnt_colour is None or area_colour == 0:
                fsm_state, active_colour = S_IDLE, None
            else:
                yellow_wait_frames += 1
                M    = cv.moments(cnt_colour)
                cx_c = int(M['m10'] / M['m00']) if M['m00'] != 0 else frame_center
                has_line = True
                cv.drawContours(frame_bgr, [cnt_colour], -1, (0, 255, 255), 2)
                error      = frame_center - cx_c
                last_error = error
                integral   = max(-500, min(500, integral + error))
                derivative = error - prev_error
                prev_error = error
                turn       = max(-MAX_CONTROL, min(MAX_CONTROL,
                                 Kp * error + Ki * integral + Kd * derivative))
                dyn_base   = max(25, BASE_SPEED * 0.6 - abs(turn) * 0.3)
                left_out, right_out = dyn_base + turn, dyn_base - turn

                current_label = out_label.raw.rstrip(b'\x00').decode(errors='ignore')
                if current_label == "Caution":
                    # FIX 4: Caution confirmed — stop motors, reset to IDLE.
                    # main() will pick up the label and execute the stop action.
                    # colour_approach_blocked is set by main() after action ends.
                    print("[LineWorker] Caution detected in classify — stopping, back to IDLE.")
                    left_out, right_out, has_line = 0.0, 0.0, False
                    fsm_state, active_colour = S_IDLE, None
                elif yellow_wait_frames > 25:   # ~0.83 s at 30 fps (was 10)
                    print("[LineWorker] No Caution found — treating as Yellow LINE.")
                    fsm_state, entry_direction, entry_confirm_n, lost_frames = \
                        S_COLOR_FOLLOW, None, 0, 0

        # ── S_COLOR_FOLLOW (Red / Yellow) ─────────────────────────────
        elif fsm_state == S_COLOR_FOLLOW:
            is_on_color_line.value = True
            if cnt_colour is not None and area_colour > 0:
                lost_frames = 0
                M    = cv.moments(cnt_colour)
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
                left_out, right_out = dyn_base + turn, dyn_base - turn
                if entry_direction is None:
                    candidate = "Left" if cx_c < frame_center else "Right"
                    entry_confirm_n += 1
                    if entry_confirm_n >= ENTRY_CONFIRM_FRAMES:
                        entry_direction = candidate
                        print(f"[LineWorker] Entry direction LOCKED: {entry_direction}")
            else:
                lost_frames += 1
                left_out, right_out, has_line = out_left.value, out_right.value, True
                if lost_frames > LOST_TOLERANCE:
                    print(f"[LineWorker] {active_colour} LOST — EXIT {entry_direction}")
                    fsm_state = S_COLOR_EXIT
                    out_turn_cmd.value = CMD_EXIT_LEFT if entry_direction == "Left" \
                                         else CMD_EXIT_RIGHT

        # ── S_COLOR_EXIT ──────────────────────────────────────────────
        elif fsm_state == S_COLOR_EXIT:
            is_on_color_line.value = True
            left_out, right_out, has_line = 0.0, 0.0, False
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

        out_left.value     = float(left_out)
        out_right.value    = float(right_out)
        out_has_line.value = has_line

        with disp_lock:
            state_names = {0: "IDLE", 1: "R_APPR", 2: "R_CLASS",
                           3: "FOLLOW", 4: "EXIT", 5: "Y_CLASS", 6: "Y_APPR"}
            cv.putText(frame_bgr,
                       f"FSM:{state_names.get(fsm_state,'?')} {active_colour or ''}  "
                       f"RCD:{red_arrow_cooldown}",
                       (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            if entry_direction:
                cv.putText(frame_bgr, f"ENTRY:{entry_direction}",
                           (5, 40), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
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

    # FIX 6: maxlen 2 — only 2 consecutive matching frames needed
    label_history    = deque(maxlen=CONFIRM_FRAMES)
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

        # ── ORB symbol detection ──────────────────────────────────────
        gray_scene   = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        kp_s, des_s  = orb.detectAndCompute(gray_scene, None)
        matched_name = None

        # FIX 6: MIN_SCENE_KPS=10, LOWE_RATIO=0.75
        if reference_data and des_s is not None and len(kp_s) >= MIN_SCENE_KPS:
            for sym_name, ref in reference_data.items():
                for ref_des in ref["des_list"]:
                    matches = bf.knnMatch(ref_des, des_s, k=2)
                    good = sum(1 for m in matches
                               if len(m) == 2 and m[0].distance < LOWE_RATIO * m[1].distance)
                    if good >= ref["thresh"]:
                        matched_name = sym_name
                        break
                if matched_name: break

        if matched_name:
            label = matched_name
            found = True
        elif not is_on_color_line.value:
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
            if len(label_history) == label_history.maxlen and len(set(label_history)) == 1:
                safe_label = label_history[0]
                print(f"[ImgWorker] Confirmed: {safe_label}")
                label_history.clear()
                cooldown_counter = 10
        else:
            if cooldown_counter > 0: cooldown_counter -= 1
            label_history.clear()

        display = frame_bgr.copy()
        if safe_label: cv.putText(display, safe_label,    (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 2)
        elif label:    cv.putText(display, f"? {label}",  (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2)

        with disp_lock: np.copyto(disp_buf, display)
        _write(bool(safe_label), safe_label)


# ══════════════════════════════════════════════════════════════════════
# ARROW ARC TURN  (FIX 1)
# ══════════════════════════════════════════════════════════════════════
def _execute_arrow_turn(pi, direction_str, picam2):
    """
    Arc turn: outer wheel = ARROW_ARC_OUTER, inner wheel = ARROW_ARC_INNER
    (both positive so car moves forward while turning).
    After arc, keep spinning until black line found or timeout.
    """
    print(f"\n[main] ARROW ARC TURN {direction_str}\n")

    if direction_str == "Up":
        _set_motor_pi(pi, BASE_SPEED, BASE_SPEED)
        time.sleep(0.7)
        _stop_motors_pi(pi)
        return

    # Arc: outer wheel drives at full arc speed, inner at reduced speed
    # Left turn  → right wheel is outer (faster), left wheel is inner (slower)
    # Right turn → left wheel is outer (faster), right wheel is inner (slower)
    if direction_str == "Left":
        l_spd, r_spd = ARROW_ARC_INNER, ARROW_ARC_OUTER
    else:
        l_spd, r_spd = ARROW_ARC_OUTER, ARROW_ARC_INNER

    _set_motor_pi(pi, l_spd, r_spd)

    # Seek black line while arcing — stop as soon as it's found
    start       = time.time()
    found_black = False
    while time.time() - start < EXIT_SPIN_TIMEOUT:
        rgb = picam2.capture_array()
        if rgb.ndim == 3 and rgb.shape[2] == 4: rgb = rgb[:, :, :3]
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
    print("[main] Black found.\n" if found_black else "[main] Timeout.\n")
    time.sleep(0.15)


# ══════════════════════════════════════════════════════════════════════
# COLOUR-LINE EXIT SPIN  (unchanged)
# ══════════════════════════════════════════════════════════════════════
def _execute_exit_spin(pi, direction_str, picam2):
    print(f"\n[main] EXIT SPIN {direction_str}\n")
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
        mask_black = cv.inRange(hsv, LINE_COLOUR_RANGES["Black"]["lower"],
                                      LINE_COLOUR_RANGES["Black"]["upper"])
        mask_black[0:150, :] = 0
        _, area_black = best_contour(mask_black)
        if area_black > 3000:
            found_black = True
            break
        time.sleep(0.01)

    _stop_motors_pi(pi)
    print("[main] Black found.\n" if found_black else "[main] Timeout.\n")
    time.sleep(0.15)


# ══════════════════════════════════════════════════════════════════════
# MAIN
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

    out_left             = mp.Value('d', 0.0)
    out_right            = mp.Value('d', 0.0)
    out_has_line         = mp.Value('b', False)
    out_found            = mp.Value('b', False)
    out_label            = mp.Array('c', 64)
    out_turn_cmd         = mp.Value('i', CMD_NONE)
    out_colour_label     = mp.Array('c', 64)
    is_on_color_line     = mp.Value('b', False)
    active_shortcut_dir  = mp.Value('i', 0)
    # FIX 4: shared flag — main() sets True after Caution/arrow action,
    # clears after COLOUR_BLOCK_AFTER_ACTION seconds
    colour_approach_blocked = mp.Value('b', False)

    p_line = mp.Process(
        target=line_worker,
        args=(shm.name, frame_lock, shared_fid, line_fid,
              out_left, out_right, out_has_line,
              out_turn_cmd, out_colour_label, is_on_color_line,
              out_label, active_shortcut_dir,
              colour_approach_blocked,
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
    # FIX 4: timestamp when colour approach block should lift
    colour_block_until = 0.0

    print(f"[main] Running. TARGET_FPS={TARGET_FPS}. Press 'q' to quit.")

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

            # FIX 4: manage colour_approach_blocked flag
            if colour_approach_blocked.value and now >= colour_block_until:
                colour_approach_blocked.value = False

            # ── colour-line commands from line_worker ──────────────────
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
                if label in ('QR', 'Fingerprint'):
                    print(f"[main] >>> {label} detected — logging, continuing.")
                    cooldown_until = now + COOLDOWN_AFTER_ACT
                else:
                    robot_state  = label
                    action_start = now
                    print(f"[main] >>> Action: {label}")

            if robot_state != 'LINE_FOLLOW':
                duration = ACTION_DURATION.get(robot_state, 0.0)
                if now - action_start >= duration:
                    # FIX 4: after Caution action ends, block colour approach
                    if robot_state == 'Caution':
                        colour_approach_blocked.value = True
                        colour_block_until = time.time() + COLOUR_BLOCK_AFTER_ACTION
                        print(f"[main] Caution done — colour approach blocked for "
                              f"{COLOUR_BLOCK_AFTER_ACTION}s")
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
                # FIX 4: after arrow action, block colour approach
                colour_approach_blocked.value = True
                colour_block_until = time.time() + COLOUR_BLOCK_AFTER_ACTION
                print(f"[main] Arrow done — colour approach blocked for "
                      f"{COLOUR_BLOCK_AFTER_ACTION}s")
                if robot_state == 'Left':  active_shortcut_dir.value = 1
                if robot_state == 'Right': active_shortcut_dir.value = 2
                robot_state    = 'LINE_FOLLOW'
                cooldown_until = time.time() + COOLDOWN_AFTER_ACT

            if ENABLE_GUI:
                with line_disp_lock: cv.imshow("Line View",   line_disp_buf.copy())
                with img_disp_lock:  cv.imshow("Symbol View", img_disp_buf.copy())
                if cv.waitKey(1) & 0xFF == ord('q'): break

            elapsed = time.time() - loop_start
            if (TARGET_FRAME_TIME - elapsed) > 0:
                time.sleep(TARGET_FRAME_TIME - elapsed)

    except KeyboardInterrupt:
        print("\n[main] Ctrl+C – shutting down.")
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
