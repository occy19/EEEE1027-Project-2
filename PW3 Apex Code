from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import time
import pigpio
from collections import deque
import multiprocessing as mp
from multiprocessing import shared_memory

# ══════════════════════════════════════════════════════════════
# PIN DECLARATIONS  (BCM)
# ══════════════════════════════════════════════════════════════
IN1, IN2 = 17, 18   # motor_a (Left) direction
IN3, IN4 = 22, 23   # motor_b (Right) direction
ENA, ENB = 12, 13   # PWM enable

# ══════════════════════════════════════════════════════════════
# FRAME CONFIG
# ══════════════════════════════════════════════════════════════
FRAME_W      = 480
FRAME_H      = 360
FRAME_SHAPE  = (FRAME_H, FRAME_W, 3)          # RGB uint8
FRAME_NBYTES = FRAME_H * FRAME_W * 3

# Display output frames (BGR, written by workers, read by main)
LINE_DISP_SHAPE  = (180, FRAME_W, 3)          # cropped region [120:360]
LINE_DISP_NBYTES = 180 * FRAME_W * 3
IMG_DISP_SHAPE   = (FRAME_H, FRAME_W, 3)      # full frame
IMG_DISP_NBYTES  = FRAME_H * FRAME_W * 3

KP, KI, KD         = 1.3, 0.001, 0.8
X_CENTRE_REF        = 240
Y_CENTRE_REF        = 180

# ══════════════════════════════════════════════════════════════
# IMAGE-RECOGNITION CONFIG
# ══════════════════════════════════════════════════════════════
TEMPLATES = {
    0:        (["templates/Hand.jpg"],        30),
    1: (["templates/Fingerprint.jpg"], 30),
    2:          (["templates/QR.jpg", "templates/QR1.jpg", "templates/QR2.jpg", "templates/QR3.jpg"], 15),
    3:     (["templates/Recycle.jpg", "templates/Recycle1.jpg", "templates/Recycle2.jpg", "templates/Recycle3.jpg", "templates/Recycle4.jpg"], 15),
    4:     (["templates/Caution.jpg","templates/Caution1.jpg", "templates/Caution2.jpg", "templates/Caution3.jpg"], 20),
}
SYMBOL = {
    0: "Hand",
    1: "Fingerprint",
    2: "QR",
    3: "Recycle",
    4: "Caution",
}

IMAGE_COLOUR_RANGES = {
    "Green":    {"space": "HSV", "lower": np.array([35,  30,  30]), "upper": np.array([90,  255, 255])}, 
    "Yellow":   {"space": "HSV", "lower": np.array([15, 30,  35]), "upper": np.array([55,  255, 255])}, 
    "Purple":   {"space": "LAB", "lower": np.array([0, 145,  60 ]), "upper": np.array([255, 195, 135])}, 
    "Blue":{"space": "LAB", "lower": np.array([20 , 90 , 50]), "upper": np.array([200,150,110])},
    "Red":      {"space": "LAB", "lower": np.array([0 , 160, 130]), "upper": np.array([255, 255, 180])}, 
    "Orange":   {"space": "LAB", "lower": np.array([15, 140, 145 ]), "upper": np.array([255, 190, 210])},
}

LINE_COLOUR_RANGES = {
    "Red":      {"lower_1": np.array([0, 100, 100]), "lower_2": np.array([160, 100, 100]), "upper_1": np.array([10, 255, 255]), "upper_2": np.array([180, 255, 255])},
    "Yellow":   {"lower": np.array([20, 80, 80]), "upper": np.array([40, 255, 255])},
    "Black":   {"lower": np.array([0, 0, 0]), "upper": np.array([180, 255, 70])}
}

LABEL_TO_INSTRUCTION = {
    "Arrow (Left)":    "TURN_LEFT",
    "Arrow (Right)":   "TURN_RIGHT",
    "Arrow (Up)":      "MOVE_FORWARD",
    "Hand":    "STOP",
    "Caution":  "STOP",
    "Recycle": "360-TURN",
}

# ══════════════════════════════════════════════════════════════
# MOTOR   (main process only) - PIGPIO VERSION
# ══════════════════════════════════════════════════════════════
def setup_gpio():
    pi = pigpio.pi()
    if not pi.connected:
        print("ERROR: pigpiod not running! Run 'sudo pigpiod' first.")
        return None
        
    for pin in (IN1, IN2, IN3, IN4, ENA, ENB):
        pi.set_mode(pin, pigpio.OUTPUT)
        
    pi.set_PWM_frequency(ENA, 1000)
    pi.set_PWM_frequency(ENB, 1000)
    return pi

def _drive_side(pi, in_a, in_b, en_pin, speed):
    # Convert 0-100 speed to 0-255 duty cycle for PIGPIO
    duty = int(max(0, min(100, abs(speed))) * 2.55)
    
    if speed < 0:
        pi.write(in_a, 1)
        pi.write(in_b, 0)
    else:
        pi.write(in_a, 0)
        pi.write(in_b, 1)
        
    pi.set_PWM_dutycycle(en_pin, duty)

def move_forward(pi, speed_motor_a, speed_motor_b):
    _drive_side(pi, IN1, IN2, ENA, speed_motor_a)
    _drive_side(pi, IN3, IN4, ENB, speed_motor_b)

def stop_motors(pi):
    for pin in (IN1, IN2, IN3, IN4):
        pi.write(pin, 0)
    pi.set_PWM_dutycycle(ENA, 0)
    pi.set_PWM_dutycycle(ENB, 0)

# ══════════════════════════════════════════════════════════════
# SHARED HELPERS 
# ══════════════════════════════════════════════════════════════
def best_contour(mask):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0
    c = max(contours, key=cv.contourArea)
    return c, cv.contourArea(c)

def _write_str(arr: mp.Array, text: str, max_bytes: int):
    enc = text.encode()[:max_bytes - 1]
    arr.raw = enc + b'\x00' * (max_bytes - len(enc))

def _read_str(arr: mp.Array) -> str:
    return arr.raw.rstrip(b'\x00').decode(errors='replace')

# ══════════════════════════════════════════════════════════════
# LINE-FOLLOWING WORKER PROCESS
# ══════════════════════════════════════════════════════════════
def line_worker(
    shm_name,           
    frame_lock,         
    shared_fid,         
    my_fid,             
    out_pid,            
    out_cx,             
    out_cy,             
    out_has_line,       
    out_lineArea,
    out_is_priority,
    out_turn_cmd,
    disp_shm_name,      
    disp_lock,          
):
    shm  = shared_memory.SharedMemory(name=shm_name)
    fbuf = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)

    disp_shm  = shared_memory.SharedMemory(name=disp_shm_name)
    disp_buf  = np.ndarray(LINE_DISP_SHAPE, dtype=np.uint8, buffer=disp_shm.buf)

    pid_state = {'last_error': 0.0, 'integral': 0.0, 'last_time': time.monotonic()}
    pid_out = 0.0
    cx, cy = X_CENTRE_REF, 150          
    
    prev_fps_time = time.monotonic()
    smoothed_fps = 0.0

    lane_memory = None
    red_left_votes = 0
    red_right_votes = 0
    was_on_red = False
    was_on_yellow = False

    while True:
        fid = shared_fid.value
        if fid == my_fid.value:
            time.sleep(0.002)
            continue
        my_fid.value = fid

        now = time.monotonic()
        dt_fps = now - prev_fps_time
        prev_fps_time = now
        if dt_fps > 0:
            current_fps = 1.0 / dt_fps
            smoothed_fps = (0.9 * smoothed_fps) + (0.1 * current_fps)

        with frame_lock:
            frame = fbuf.copy()

        crop_rgb = frame[180:360, :]
        crop_bgr = cv.cvtColor(crop_rgb, cv.COLOR_RGB2BGR)

        blur = cv.GaussianBlur(crop_rgb, (3, 3), 0)
        hsv  = cv.cvtColor(blur, cv.COLOR_RGB2HSV)
        
        mask_black = cv.inRange(hsv, LINE_COLOUR_RANGES["Black"]["lower"], LINE_COLOUR_RANGES["Black"]["upper"])
        mask_yellow = cv.inRange(hsv, LINE_COLOUR_RANGES["Yellow"]["lower"], LINE_COLOUR_RANGES["Yellow"]["upper"])
        mask_red = cv.bitwise_or(cv.inRange(hsv, LINE_COLOUR_RANGES["Red"]["lower_1"], LINE_COLOUR_RANGES["Red"]["upper_1"]), 
                                 cv.inRange(hsv, LINE_COLOUR_RANGES["Red"]["lower_2"], LINE_COLOUR_RANGES["Red"]["upper_2"]))
 
        cnt_black,  area_black  = best_contour(mask_black)
        cnt_yellow, area_yellow = best_contour(mask_yellow)
        cnt_red,    area_red    = best_contour(mask_red)
 
        # Bounding box sizing for priority line confirmation
        def get_bbox(cnt): return cv.boundingRect(cnt) if cnt is not None else (0, 0, 0, 0)
        xr, yr, wr, hr = get_bbox(cnt_red)
        xy, yy, wy, hy = get_bbox(cnt_yellow)
        xb, yb, wb, hb = get_bbox(cnt_black)
        
        ay_ratio = float(wy) / hy if hy > 0 else 0
        if area_red > 4500 and hr > 40 and wr > 40:
            cnts = [cnt_red] if cnt_red is not None else []
            follow_colour = "Red"
            draw_colour   = (0, 0, 255)   
        elif area_yellow > 4500 and hy > 40 and wy>40 :
            print(f"{hy}")
            cnts = [cnt_yellow] if cnt_yellow is not None else []
            follow_colour = "Yellow"
            draw_colour   = (0, 255, 255) 
        
                
        elif area_black > 4500:
            cnts = [cnt_black] if cnt_black is not None else []
            follow_colour = "Black"
            draw_colour   = (0, 255, 0)     
        else:
            cnts = []
            follow_colour = "None"
            draw_colour   = (0, 255, 0)
            
        if follow_colour != "Yellow":
            was_on_yellow = False
            
        has_line = False
        current_area = 0.0
        
        if cnts:
            has_line = True
            
            largest_contour = max(cnts, key=cv.contourArea)
            area = cv.contourArea(largest_contour)
            current_area = area

            x, y, w, h = cv.boundingRect(largest_contour)
            cv.rectangle(crop_bgr, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv.putText(crop_bgr, f"w:{w} h:{h}", (x, max(10, y - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            M = cv.moments(largest_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + (w // 2), y + (h // 2)

            cv.drawContours(crop_bgr, [largest_contour], -1, draw_colour, 2)

            error = cx- X_CENTRE_REF 
            now   = time.monotonic()
            dt    = now - pid_state['last_time']
            pid_state['last_time'] = now
                
            P = KP * error
            pid_state['integral'] += error * dt
            I = KI * pid_state['integral']
            D = (KD * (error - pid_state['last_error']) / dt) if dt > 0.01 else 0.0
            pid_state['last_error'] = error
            pid_out = P + I + D
        else:
            pid_state['last_time'] = time.monotonic()

        # Lane Memory Logic
        if follow_colour == "Red":
            if lane_memory is None:
                cx_blk = X_CENTRE_REF  
                if cnt_black is not None and area_black > 1000:
                    M_blk = cv.moments(cnt_black)
                    if M_blk['m00'] != 0:
                        cx_blk = int(M_blk['m10'] / M_blk['m00'])

                if cx < cx_blk: red_left_votes += 1
                else: red_right_votes += 1
                    
                if (red_left_votes - red_right_votes) >= 5:
                    lane_memory = "Left"
                    print(f"\n[line_worker] VOTING COMPLETE: Locked LEFT (+{red_left_votes - red_right_votes} lead)\n")
                elif (red_right_votes - red_left_votes) >= 5:
                    lane_memory = "Right"
                    print(f"\n[line_worker] VOTING COMPLETE: Locked RIGHT (+{red_right_votes - red_left_votes} lead)\n")
            was_on_red = True
            
        elif follow_colour == "Black":
            if was_on_red:
                if lane_memory == "Left": out_turn_cmd.value = 1  
                elif lane_memory == "Right": out_turn_cmd.value = 2  
                was_on_red = False; lane_memory = None; red_left_votes = 0; red_right_votes = 0
        else:
            
            was_on_red = False
                
        cv.circle(crop_bgr, (X_CENTRE_REF, Y_CENTRE_REF), 5, (0, 255, 255), -1)
        cv.circle(crop_bgr, (cx, cy), 5, (0, 0, 255), -1)
        cv.putText(crop_bgr, f"FPS: {int(smoothed_fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        with disp_lock:
            np.copyto(disp_buf, crop_bgr)

        out_pid.value = pid_out
        out_cx.value = cx
        out_cy.value = cy
        out_is_priority.value = (follow_colour in ["Red", "Yellow"])
        out_lineArea.value = current_area 
        out_has_line.value = has_line


# ══════════════════════════════════════════════════════════════
# IMAGE-RECOGNITION   
# ══════════════════════════════════════════════════════════════
def orb_match_symbol(bf, ref_entries, des_scene, threshold):
    for ref in ref_entries:
        if ref["des"] is None: continue
        matches = bf.knnMatch(ref["des"], des_scene, k=2)
        good = sum(1 for m in matches if len(m) == 2 and m[0].distance < 0.72 * m[1].distance)
        if good >= threshold: return True, good
    return False, 0

def _detect_shape(contour):
    area = cv.contourArea(contour)
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * peri, True)
    v = len(approx)
    hull_area = cv.contourArea(cv.convexHull(contour))
    if hull_area == 0: return "Unknown", None
    hull = cv.convexHull(contour)
    hull_peri = cv.arcLength(hull, True)
    convexity = hull_peri / peri if peri > 0 else 0.0
    solidity = area / hull_area
    is_convex = cv.isContourConvex(approx)
    circ = 4 * np.pi * area / (peri * peri)
    x, y, w, h = cv.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    if 7 <= v <= 11 and 0.4 <= convexity <= 0.80 and 0.2 <= solidity <= 0.5 and circ >= 0.15 and 0.6 <= aspect_ratio <= 1.6 :
        M = cv.moments(contour)
        if M["m00"] == 0: return "Arrow", "Unknown"
        cx_ = int(M["m10"] / M["m00"])
        cy_ = int(M["m01"] / M["m00"])
        far = max(contour, key=lambda p: (p[0][0]-cx_)**2 + (p[0][1]-cy_)**2)
        dx, dy = far[0][0] - cx_, far[0][1] - cy_
        
        if abs(dx) > abs(dy): 
            direction = "Left" if dx > 0 else "Right"
            return "Arrow", direction
    
    return "Unknown", None


# ══════════════════════════════════════════════════════════════
# IMAGE-RECOGNITION WORKER PROCESS
# ══════════════════════════════════════════════════════════════
def image_worker(
    shm_name, frame_lock, shared_fid, my_fid,
    out_found, out_label, out_instruction, out_instruction_ready,
    disp_shm_name, disp_lock, out_is_priority,    
):
    shm  = shared_memory.SharedMemory(name=shm_name)
    fbuf = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)

    disp_shm = shared_memory.SharedMemory(name=disp_shm_name)
    disp_buf = np.ndarray(IMG_DISP_SHAPE, dtype=np.uint8, buffer=disp_shm.buf)

    orb = cv.ORB_create(nfeatures=500, nlevels=8, fastThreshold=17)
    bf  = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    reference_data = []
    for symbol_id, (img_files, threshold) in TEMPLATES.items():
        refs = []
        for img_file in img_files:
            img = cv.imread(img_file, cv.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[img_worker] WARNING: reference image '{img_file}' not found – skipping.")
                refs.append({"filename": img_file, "kp": None, "des": None})
                continue
            kp, des = orb.detectAndCompute(img, None)
            refs.append({"filename": img_file, "kp": kp, "des": des})
            print(f"[img_worker] Loaded '{img_file}' — {len(kp)} keypoints.")
 
        reference_data.append({"id": symbol_id, "name": SYMBOL[symbol_id], "threshold": threshold, "refs": refs})
 
    ref_by_id = {entry["id"]: entry for entry in reference_data}
    label_history = deque(maxlen = 3)

    cooldown_counter = 0
    missed_frames = 0

    while True:
        fid = shared_fid.value
        if fid == my_fid.value:
            time.sleep(0.002)
            continue
        my_fid.value = fid

        with frame_lock: frame = fbuf.copy()
        display_bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        found = False
        label = ""
        instruction = ""
        best_contour_for_display = None
        
        current_is_priority = bool(out_is_priority.value)
        
        blurred = cv.GaussianBlur(frame, (3, 3), 0)
        HSV = cv.cvtColor(blurred, cv.COLOR_RGB2HSV)
        LAB = cv.cvtColor(blurred, cv.COLOR_RGB2LAB)
        
        all_candidates = []
        for colour_name, params in IMAGE_COLOUR_RANGES.items():
            src  = HSV if params["space"] == "HSV" else LAB
            mask = cv.inRange(src, params["lower"], params["upper"])
            for cnt in cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]:
                a = cv.contourArea(cnt)
                if a >= 400: all_candidates.append((a, cnt, colour_name))
        
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = all_candidates[:3]

        colour_to_ids = {"Yellow": [4], "Blue": [1, 2], "Purple": [1, 2], "Green": [0, 3]}
        orb_eligible = any(colour in colour_to_ids for _, _, colour in top_candidates)
        
        des_s = None
        if orb_eligible:
            gray_scene = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            _, des_s   = orb.detectAndCompute(gray_scene, None)

        for area, cnt, detected_colour in top_candidates: 
            x, y, w, h = cv.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            if detected_colour in colour_to_ids and des_s is not None:
                if detected_colour in colour_to_ids and des_s is not None:
                    for sym_id in colour_to_ids[detected_colour]:
                        entry = ref_by_id[sym_id]
                        matched, good_count = orb_match_symbol(bf, entry["refs"], des_s, entry["threshold"])
                        if matched:
                            label = entry["name"]; found = True; best_contour_for_display = cnt
                            break 
            if not found and not current_is_priority:
                shape, direction = _detect_shape(cnt)
                if shape != "Unknown":
                    label = shape + (f" ({direction})" if direction else "")
                    found = True; best_contour_for_display = cnt
            if found: break 

        if found and cooldown_counter == 0:
            label_history.append(label)
            missed_frames = 0
            if len(label_history) == label_history.maxlen and len(set(label_history)) == 1:
                confirmed_label = label_history[0]
                instruction     = LABEL_TO_INSTRUCTION.get(confirmed_label, "")

                print(f"----------\n[img_worker] Detected : {confirmed_label}")
                if instruction: print(f"[img_worker] Instruction: {instruction}")
                print(f"----------")

                label_history.clear()
                cooldown_counter = 8
        else:
            if cooldown_counter > 0: cooldown_counter -= 1
            missed_frames += 1
            if missed_frames > 4:
                label_history.clear()
                instruction = ""   

        if current_is_priority:
            cv.putText(display_bgr, "Line Following Only", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        if best_contour_for_display is not None:
            x, y, w, h = cv.boundingRect(best_contour_for_display)
            cv.rectangle(display_bgr, (x, y), (x + w, y + h), (255, 0, 0), 3)
            if label:
                cv.putText(display_bgr, label, (x, max(10, y - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        elif label:
            cv.putText(display_bgr, label, (10, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        with disp_lock: np.copyto(disp_buf, display_bgr)

        out_found.value = found
        _write_str(out_label, label, 64)
        if instruction:
            _write_str(out_instruction, instruction, 32)
            out_instruction_ready.value = True


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    shm  = shared_memory.SharedMemory(create=True, size=FRAME_NBYTES)
    fbuf = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)

    line_disp_shm = shared_memory.SharedMemory(create=True, size=LINE_DISP_NBYTES)
    img_disp_shm  = shared_memory.SharedMemory(create=True, size=IMG_DISP_NBYTES)
    line_disp_buf = np.ndarray(LINE_DISP_SHAPE, dtype=np.uint8, buffer=line_disp_shm.buf)
    img_disp_buf  = np.ndarray(IMG_DISP_SHAPE,  dtype=np.uint8, buffer=img_disp_shm.buf)
    line_disp_lock, img_disp_lock = mp.Lock(), mp.Lock()

    frame_lock = mp.Lock()
    shared_fid = mp.Value('i',  0)   
    line_fid   = mp.Value('i', -1)   
    img_fid    = mp.Value('i', -1)   

    out_pid = mp.Value('d', 0.0); out_cx = mp.Value('i', X_CENTRE_REF); out_cy = mp.Value('i', Y_CENTRE_REF)
    out_lineArea = mp.Value('d', 0.0); out_has_line = mp.Value('b', False)
    
    out_found = mp.Value('b', False); out_label = mp.Array('c', 64)
    out_instruction = mp.Array('c', 32); out_instruction_ready = mp.Value('b', False) 
    out_is_priority = mp.Value('b', False); out_turn_cmd = mp.Value('i', 0) 

    p_line = mp.Process(
        target=line_worker,
        args=(shm.name, frame_lock, shared_fid, line_fid, out_pid, out_cx, out_cy, out_has_line, out_lineArea, out_is_priority, out_turn_cmd, line_disp_shm.name, line_disp_lock),
        daemon=True, name="LineWorker"
    )
    p_img = mp.Process(
        target=image_worker,
        args=(shm.name, frame_lock, shared_fid, img_fid, out_found, out_label, out_instruction, out_instruction_ready, img_disp_shm.name, img_disp_lock, out_is_priority),
        daemon=True, name="ImgWorker"
    )
    p_line.start(); p_img.start()
    print("[main] Workers started.")

    # Set up motors safely
    pi = setup_gpio()
    if pi is None:
        print("[main] Exiting safely. Fix the motor daemon and try again.")
        return # This instantly stops the rest of the code from running

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (FRAME_W, FRAME_H)}))
    picam2.start()
    time.sleep(2.0)                       
    print("[main] Camera ready. Press ESC to quit.")

    line_loss_counter = 0
    active_instruction = ""
    
    try:
        while True:
            RGB = picam2.capture_array()
            if RGB.ndim == 3 and RGB.shape[2] == 4: RGB = RGB[:, :, :3]

            with frame_lock: np.copyto(fbuf, RGB)
            shared_fid.value += 1          

            pid, cx, cy = out_pid.value, out_cx.value, out_cy.value
            has_line, found = bool(out_has_line.value), bool(out_found.value)
            label = _read_str(out_label)
            current_is_priority = bool(out_is_priority.value)
            
            new_instr= ""
            if out_instruction_ready.value:
                new_instr = _read_str(out_instruction)
                out_instruction_ready.value = False   

            turn_cmd = out_turn_cmd.value
            if turn_cmd != 0:
                if turn_cmd == 1:
                    print("[main] Exiting priority line — turning LEFT")
                    move_forward(pi, 160, -150); time.sleep(0.7)
                elif turn_cmd == 2:
                    print("[main] Exiting priority line — turning RIGHT")
                    move_forward(pi, -150, 160); time.sleep(0.7)
                out_turn_cmd.value = 0

            if new_instr:
                active_instruction = new_instr
                print(f"[main] Instruction '{active_instruction}' confirmed.")
                _write_str(out_instruction, "", 32)   

            if active_instruction:
                if active_instruction == "TURN_LEFT":
                    move_forward(pi, 150, -150); time.sleep(0.5)
                    stop_motors(pi); active_instruction = ""
                elif active_instruction == "TURN_RIGHT":

                    move_forward(pi, -150, 150); time.sleep(0.5)
                    stop_motors(pi); active_instruction = ""
                elif active_instruction == "MOVE_FORWARD":
                    move_forward(pi,120, 120); time.sleep(1.0)
                    active_instruction = ""
                elif active_instruction == "STOP":
                    stop_motors(pi); time.sleep(2)
                    move_forward(pi, 150, 150); time.sleep(0.2)
                    active_instruction = ""
                elif active_instruction == "360-TURN":
                    move_forward(pi, -170, 170); time.sleep(2.0)
                    stop_motors(pi); active_instruction = ""
            else:
                if current_is_priority:
                    L_base, R_base = 120, 120  # Slower scan speed for ORB to catch up
                else:
                    L_base = 130
                    R_base = 130

                if has_line:
                    move_forward(pi, L_base - pid, R_base + pid)
                    line_loss_counter = 0
                else:
                    if line_loss_counter <= 8:
                        move_forward(pi, 110, 110)
                        line_loss_counter += 1
                    else:
                        if pid > 0: move_forward(pi, -145, 155)
                        else: move_forward(pi,  155, -145)
                    
            with line_disp_lock: line_frame = line_disp_buf.copy()
            with img_disp_lock: img_frame = img_disp_buf.copy()

            if active_instruction:
                cv.putText(line_frame, f"CMD: {active_instruction}", (10, 220), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv.putText(img_frame, f"CMD: {active_instruction}", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv.imshow("Line Following", line_frame)
            # cv.imshow("Image Detection", img_frame)

            if cv.waitKey(1) & 0xFF == 27: break

    except KeyboardInterrupt: print("\n[main] Ctrl+C received — stopping.")
    except Exception as e:
        import traceback; print(f"\n[main] ERROR: {e}"); traceback.print_exc()
    finally:
        print("[main] Shutting down…")
        # 1. Stop the physical wheels from spinning
        try: stop_motors(pi)
        except: pass

        # 2. Close the PIGPIO daemon connection properly 
        try: pi.stop()
        except: pass

        p_line.terminate(); p_line.join()
        p_img.terminate(); p_img.join()

        try: shm.close(); shm.unlink(); line_disp_shm.close(); line_disp_shm.unlink(); img_disp_shm.close(); img_disp_shm.unlink()
        except: pass

        try: picam2.stop()
        except: pass

        cv.destroyAllWindows()
        print("[main] Done.")

if __name__ == "__main__":
    # Change from "forkserver" to "spawn"
    mp.set_start_method("spawn", force=True) 
    main()
