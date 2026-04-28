"""
line_detector.py  —  3-strip lookahead PID for 480×360
 
Root cause of shortcutting on curves
─────────────────────────────────────
The old code used two strips both near the bottom of frame, weighted 70% near   
/ 30% mid. On a curve the near strip centroid is already deep into the turn,
so the robot steers toward *where the line is now* rather than *where it's
going*. The car cuts the inside of every curve.
 
Fix: three strips — near, mid, far — with dynamic weighting.
- Straight (|error| < 30): weight far heavily so the car sees the curve coming
  and begins steering early.
- Turning (|error| >= 30): shift weight toward near so the car doesn't
  over-steer when it's already mid-turn.
 
This gives genuine predictive steering on curves without needing a more
complex path model.
 
PID notes
─────────
- Kp lowered slightly: the far strip already moves the setpoint earlier, so
  you don't need as aggressive a proportional response.
- Kd raised: damping matters more now that we're steering earlier.
- BASE_SPEED is kept low so the car has time to react; raise it once the
  curve tracking is confirmed good.
- MAX_CONTROL set to 80 (was 65) to allow sharper turns when needed.
"""
 
import cv2
import numpy as np
 
# ── PID gains ────────────────────────────────────────────────────────────────
Kp = 0.55
Ki = 0.00005
Kd = 1.5
 
BASE_SPEED   = 90     # conservative — increase once curve tracking is solid
MAX_CONTROL  = 80
 
# ── Black line detection ──────────────────────────────────────────────────────
# Value < 90 catches a slightly wider range of blacks without grabbing shadows.
# If floor shadows re-appear, lower upper_black[2] back to 80.
lower_black = np.array([0,   0,   0])
upper_black = np.array([179, 255, 90])
 
# ── Strip positions (y measured from bottom of 360px frame) ──────────────────
#   near: immediate steering feedback
#   mid:  medium lookahead
#   far:  curve preview — the most important addition
NEAR_Y1, NEAR_Y2 = 310, 340   # rows 310–340  (near bottom)
MID_Y1,  MID_Y2  = 260, 290   # rows 260–290
FAR_Y1,  FAR_Y2  = 200, 230   # rows 200–230  (further ahead)
 
# ── Dynamic weight thresholds ─────────────────────────────────────────────────
# When |error| is below this, the car is roughly straight → lean on far strip
# When |error| is above this, the car is in a turn    → lean on near strip
STRAIGHT_THRESHOLD = 30
 
# Weights [near, mid, far] for straight and turning modes
W_STRAIGHT = (0.2, 0.3, 0.5)
W_TURNING  = (0.6, 0.3, 0.1)
 
 
def process_frame(frame, previous_error, integral, last_error):
    debug_frame = frame.copy()
    height, width = frame.shape[:2]
    frame_center  = width // 2   # 240
 
    # ── 1. Black mask ─────────────────────────────────────────────────────────
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
 
    # ── 2. Extract strip centroids ────────────────────────────────────────────
    def strip_cx(y1, y2):
        """Return mean x of white pixels in mask[y1:y2], or None."""
        idx = np.where(mask[y1:y2, :] == 255)
        if len(idx[1]) == 0:
            return None
        return float(np.mean(idx[1]))
 
    near_cx = strip_cx(NEAR_Y1, NEAR_Y2)
    mid_cx  = strip_cx(MID_Y1,  MID_Y2)
    far_cx  = strip_cx(FAR_Y1,  FAR_Y2)
 
    # ── 3. Detection logic ────────────────────────────────────────────────────
    if near_cx is None:
        # Line lost — spin toward the direction it was last seen
        spin = 150 if last_error > 0 else -150
        return spin, -spin, previous_error, integral, last_error, debug_frame, mask, False
 
    # Fill missing strips with the nearest known centroid
    if mid_cx is None:
        mid_cx = near_cx
    if far_cx is None:
        far_cx = mid_cx   # not near_cx — keeps some lookahead character
 
    # Dynamic weighting based on current turn magnitude
    if abs(previous_error) < STRAIGHT_THRESHOLD:
        wn, wm, wf = W_STRAIGHT
    else:
        wn, wm, wf = W_TURNING
 
    current_cx = wn * near_cx + wm * mid_cx + wf * far_cx
 
    # ── 4. PID ───────────────────────────────────────────────────────────────
    error      = frame_center - current_cx
    last_error = error
 
    integral  += error
    integral   = max(-500, min(500, integral))   # anti-windup
 
    derivative     = error - previous_error
    previous_error = error
 
    turn = Kp * error + Ki * integral + Kd * derivative
    turn = max(-MAX_CONTROL, min(MAX_CONTROL, turn))
 
    left_speed  = BASE_SPEED + turn
    right_speed = BASE_SPEED - turn
 
    # ── 5. Debug overlay ──────────────────────────────────────────────────────
#    cv2.line(debug_frame, (frame_center, 0), (frame_center, height), (0, 255, 0), 1)
 
    # Strip boxes: near=magenta, mid=cyan, far=yellow
#    cv2.rectangle(debug_frame, (0, NEAR_Y1), (width, NEAR_Y2), (255, 0, 255), 2)
#    cv2.rectangle(debug_frame, (0, MID_Y1),  (width, MID_Y2),  (0, 255, 255), 2)
#    cv2.rectangle(debug_frame, (0, FAR_Y1),  (width, FAR_Y2),  (0, 255, 0),   2)
 
    # Centroids
#    cv2.circle(debug_frame, (int(near_cx), (NEAR_Y1+NEAR_Y2)//2), 5, (255, 0, 255), -1)
#    cv2.circle(debug_frame, (int(mid_cx),  (MID_Y1+MID_Y2)//2),   5, (0, 255, 255), -1)
#    cv2.circle(debug_frame, (int(far_cx),  (FAR_Y1+FAR_Y2)//2),   5, (0, 255, 0),   -1)
#    cv2.circle(debug_frame, (int(current_cx), height-15),          7, (255, 255, 255), -1)
 
    mode = "STR" if abs(previous_error) < STRAIGHT_THRESHOLD else "TRN"
    cv2.putText(debug_frame, f"Err:{int(error)} {mode}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
 
    return left_speed, right_speed, error, integral, last_error, debug_frame, mask, True
 
