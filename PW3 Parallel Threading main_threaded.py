"""
main_threaded.py
 
Thread layout
─────────────
  T0  Camera capture   — picamera2, writes latest frame to SharedState
  T1  Line follower    — reads frame when new, runs PID, writes steering
  T2  Symbol detector  — reads frame when new, runs ORB+colour, writes symbol
  T3  Motor controller — reads steering + symbol, drives L298N at 50Hz
  Main                 — GUI only (optional)
 
Key design decisions
─────────────────────
- Version-gated frame reads: each consumer passes the version it last saw to
  get_frame_if_new(). If nothing changed it returns immediately without
  acquiring the main lock.
 
- T1 does NOT sleep between iterations; it blocks on frame freshness instead.
  As soon as T0 deposits a new frame, T1 wakes on the next tight loop tick.
 
- T2 polls at SYMBOL_POLL_SLEEP (0.05s) — only actually runs ORB+colour when
  the frame is genuinely new.
 
- draw_orb is passed True only every ORB_DRAW_EVERY calls (~5Hz at ~12fps).
 
- CONFIRM_COUNT = 2. Two consecutive same-label frames required to confirm.
 
Changes from previous version
──────────────────────────────
- FPS print removed from T0 (no longer clutters terminal).
- Fingerprint and QR no longer stop the car — they print to terminal only.
- ACTION_DURATION entries for Fingerprint/QR kept for reference but the motor
  logic in T3 now treats them as pass-through (car keeps following the line).
"""
 
import cv2 as cv
import threading
import time
 
from picamera2     import Picamera2
from shared_state  import SharedState
from line_detector import process_frame
from symbol_detect import classify_symbol, ROI_X1, ROI_X2, ROI_Y1, ROI_Y2
from motor_ctrl    import set_motor, stop_motors, cleanup
 
# ── Camera ────────────────────────────────────────────────────────────────────
FRAME_W = 640
FRAME_H = 360
 
# ── Symbol detection ──────────────────────────────────────────────────────────
CONFIRM_COUNT     = 1       # two consecutive same-label frames to confirm
SYMBOL_POLL_SLEEP = 0.05    # T2 wakes every 50ms, skips if frame not new
#ORB_DRAW_EVERY    = 6       # draw ORB debug frame every N T2 iterations (~5Hz)
 
# ── Motor / timing ────────────────────────────────────────────────────────────
MOTOR_SLEEP           = 0.02    # 50Hz motor loop
COOLDOWN_AFTER_ACTION = 2.0
 
ACTION_DURATION = {
    'Hand':        2.0,
    'Caution':     2.0,
    'QR':          0.0,   # no stop — terminal print only
    'Fingerprint': 0.0,   # no stop — terminal print only
    'Recycle':     2.5,
    'Left':        0.9,
    'Right':       0.9,
}
 
SPIN_SPEED = 160
 
# ── GUI ───────────────────────────────────────────────────────────────────────
# Set False when running headless (SSH). Removes all imshow/waitKey overhead.
ENABLE_GUI = True
 
 
# ════════════════════════════════════════════════════════════════════════════════
# T0 — Camera capture
# ════════════════════════════════════════════════════════════════════════════════
 
def thread_camera(state: SharedState, picam2):
    print("[T0] Camera thread started.")
 
    while state.is_running():
        frame_rgb = picam2.capture_array()
        bgr       = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        state.put_frame(bgr)
 
    print("[T0] Camera thread stopped.")
 
 
# ════════════════════════════════════════════════════════════════════════════════
# T1 — Line follower
# ════════════════════════════════════════════════════════════════════════════════
 
def thread_line_follow(state: SharedState):
    prev_error = 0
    integral   = 0
    last_error = 0
    last_ver   = 0
 
    print("[T1] Line follower started.")
 
    while state.is_running():
        frame, new_ver = state.get_frame_if_new(last_ver)
 
        if frame is None:
            time.sleep(0.002)
            continue
 
        last_ver = new_ver
        left, right, prev_error, integral, last_error, \
            debug_frame, _mask, _found = process_frame(
                frame, prev_error, integral, last_error
            )
 
        state.set_steering(left, right)
        if ENABLE_GUI:
            state.set_debug_frame(debug_frame)
 
    print("[T1] Line follower stopped.")
 
 
# ════════════════════════════════════════════════════════════════════════════════
# T2 — Symbol detector
# ════════════════════════════════════════════════════════════════════════════════

def thread_symbol_detect(state: SharedState):
    candidate  = 'NONE'
    streak     = 0
    last_ver   = 0

    print("[T2] Symbol detector started.")

    while state.is_running():
        if state.in_cooldown():
            candidate = 'NONE'
            streak    = 0
            time.sleep(SYMBOL_POLL_SLEEP)
            continue

        frame, new_ver = state.get_frame_if_new(last_ver)

        if frame is None:
            time.sleep(SYMBOL_POLL_SLEEP)
            continue

        last_ver   = new_ver
        frame_copy = frame.copy()

        # ── 1. Show the Clean ROI Screen (No Dots) ──
        if ENABLE_GUI:
            # Crop the frame using the coordinates imported from symbol_detect.py
            roi_clean = frame_copy[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
            state.set_orb_frame(roi_clean)

        # ── 2. Run Classification (draw_orb=False stops the dots) ──
        label = classify_symbol(frame_copy, state, draw_orb=False)

        # ── 3. Confirm and Print Detection ──
        if label != 'NONE':
            if label == candidate:
                streak += 1
            else:
                candidate = label
                streak    = 1

            if streak >= CONFIRM_COUNT:
                print(f"[T2] 🎯 Detected & Confirmed: {candidate}") # PRINTS TO TERMINAL
                state.set_symbol(candidate)
                candidate = 'NONE'
                streak    = 0
        else:
            streak = max(0, streak - 1)

        time.sleep(SYMBOL_POLL_SLEEP)

    print("[T2] Symbol detector stopped.")
 
# ════════════════════════════════════════════════════════════════════════════════
# T3 — Motor controller
# ════════════════════════════════════════════════════════════════════════════════
 
def thread_motor_ctrl(state: SharedState):
    print("[T3] Motor controller started.")
 
    while state.is_running():
        now                       = time.time()
        left, right               = state.get_steering()
        symbol                    = state.get_symbol()
        robot_state, action_start = state.get_robot_state()
 
        # ── Symbol → state transition ─────────────────────────────────────────
        if symbol != 'NONE' and robot_state == 'LINE_FOLLOW':
            if symbol == 'Up':
                # 'Up' is a no-op signal — just clear and cool down
                state.clear_symbol()
                state.start_cooldown(COOLDOWN_AFTER_ACTION)
 
            elif symbol in ('QR', 'Fingerprint'):
                # These are informational only — car keeps following the line.
                print(f"[T3] Detected (no stop): {symbol}")
                state.clear_symbol()
                # Short cooldown so we don't spam the terminal.
                state.start_cooldown(1.0)
 
            else:
                state.set_robot_state(symbol)
                state.clear_symbol()
                robot_state  = symbol
                action_start = now
 
        # ── Action timeout → return to LINE_FOLLOW ────────────────────────────
        if robot_state != 'LINE_FOLLOW':
            duration = ACTION_DURATION.get(robot_state, 2.0)
            if now - action_start >= duration:
                state.set_robot_state('LINE_FOLLOW')
                state.start_cooldown(COOLDOWN_AFTER_ACTION)
                robot_state = 'LINE_FOLLOW'
 
        # ── Drive motors ──────────────────────────────────────────────────────
        if robot_state == 'LINE_FOLLOW':
            set_motor(left, right)
        elif robot_state in ('Hand', 'Caution'):
            stop_motors()
        elif robot_state == 'Recycle':
            set_motor(SPIN_SPEED, -SPIN_SPEED)
        elif robot_state == 'Left':
            set_motor(-SPIN_SPEED, SPIN_SPEED)
        elif robot_state == 'Right':
            set_motor(SPIN_SPEED, -SPIN_SPEED)
 
        time.sleep(MOTOR_SLEEP)
 
    stop_motors()
    print("[T3] Motor controller stopped.")
 
 
# ════════════════════════════════════════════════════════════════════════════════
# Main — camera init, thread launch, GUI
# ════════════════════════════════════════════════════════════════════════════════
 
def main():
    state = SharedState()
 
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_W, FRAME_H)})
    config["main"]["fps"] = 30
    picam2.configure(config)
    picam2.start()
 
    time.sleep(1.0)   # wait for sensor to stabilise
 
    threads = [
        threading.Thread(target=thread_camera,        args=(state, picam2), daemon=True, name="T0-cam"),
        threading.Thread(target=thread_line_follow,   args=(state,),        daemon=True, name="T1-pid"),
        threading.Thread(target=thread_symbol_detect, args=(state,),        daemon=True, name="T2-sym"),
        threading.Thread(target=thread_motor_ctrl,    args=(state,),        daemon=True, name="T3-mot"),
    ]
 
    for t in threads:
        t.start()
 
    if ENABLE_GUI:
        print("[Main] Running. Press 'q' to quit.")
    else:
        print("[Main] Running headless. Ctrl+C to quit.")
 
    try:
        while state.is_running():
            if ENABLE_GUI:
                dbg = state.get_debug_frame()
                if dbg is not None:
                    cv.imshow("Line view", dbg)
                orb = state.get_orb_frame()
                if orb is not None:
                    cv.imshow("ORB ROI", orb)
                if cv.waitKey(10) == ord('q'):
                    break
            else:
                time.sleep(0.1)
 
    except KeyboardInterrupt:
        pass
 
    finally:
        print("[Main] Shutting down...")
        state.stop_all()
        time.sleep(0.2)
        picam2.stop()
        cleanup()
        if ENABLE_GUI:
            cv.destroyAllWindows()
        print("[Main] Done.")
 
 
if __name__ == '__main__':
    main()
 
