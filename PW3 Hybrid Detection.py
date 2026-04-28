#PW3 No line detection - Hybrid Detection Method

import cv2 as cv
import numpy as np
from picamera2 import Picamera2

# 1. SETUP: Symbols and exact filenames
SAMPLE_DICT = {
    "Caution":     ("templates/Caution.jpg", 40),
    "Fingerprint": ("templates/Fingerprint.jpg", 45),
    "Hand":        ("templates/Hand.jpg", 30),
    "QR":          ("templates/QR.jpg", 50),
    "Recycle":     ("templates/Recycle.jpg", 35)
}

# 2. COLOR RANGES: All converted to HSV for consistency
COLOR_RANGES = {
    "Green":  {"lower": np.array([34, 80, 63]),   "upper": np.array([74, 255, 255])},
    "Red":    {"lower": np.array([160, 100, 50]), "upper": np.array([179, 255, 255])},
    "Blue":   {"lower": np.array([100, 150, 50]), "upper": np.array([140, 255, 255])},
    "Orange": {"lower": np.array([5, 150, 150]),  "upper": np.array([15, 255, 255])}
}

def get_arrow_direction(contour, cx, cy):
    """Calculates direction using the farthest point from the center."""
    farthest = max(contour, key=lambda p: (p[0][0]-cx)**2 + (p[0][1]-cy)**2)
    dx, dy = farthest[0][0] - cx, farthest[0][1] - cy
    
    if abs(dx) > abs(dy):
        # FIXED: Swapped to fix "terbalik" Left/Right
        return "Left" if dx > 0 else "Right"
    else:
        # FIXED: Swapped to fix "terbalik" Up/Down
        return "Up" if dy > 0 else "Down"

def main():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()

    orb = cv.ORB_create(nfeatures=1000, scaleFactor=1.2) 
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    
    reference_data = []
    for name, (img_file, threshold) in SAMPLE_DICT.items():
        img = cv.imread(img_file, cv.IMREAD_GRAYSCALE)
        if img is not None:
            kp, des = orb.detectAndCompute(img, None)
            reference_data.append({"name": name, "des": des, "thresh": threshold})

    print("--- MAZE SYSTEM START ---")

    try:
        while True:
            frame = picam2.capture_array()
            # CRITICAL: Convert to BGR so it matches your Calibration HSV values
            bgr_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            display_frame = bgr_frame.copy()
            gray = cv.cvtColor(bgr_frame, cv.COLOR_BGR2GRAY)
            
            # --- STEP 1: ORB SYMBOL MATCHING ---
            kp_scene, des_scene = orb.detectAndCompute(gray, None)
            symbol_found = False

            if des_scene is not None:
                for ref in reference_data:
                    matches = bf.knnMatch(ref["des"], des_scene, k=2)
                    # Using 0.85 ratio for better matching at 174cm distance
                    good = sum(1 for m in matches if len(m) == 2 and m[0].distance < 0.85 * m[1].distance)
                    
                    if good > ref["thresh"]:
                        cv.putText(display_frame, f"SYMBOL: {ref['name'].upper()}", (20, 50), 
                                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                        symbol_found = True
                        break 

            # --- STEP 2: MULTI-COLOR SHAPE DETECTION ---
            if not symbol_found:
                # Pre-blur the frame once
                blurred = cv.GaussianBlur(bgr_frame, (5, 5), 0)
                hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
                
                # Check each color in your dictionary
                for color_name, r in COLOR_RANGES.items():
                    mask = cv.inRange(hsv, r["lower"], r["upper"])
                    
                    # Clean up small noise/dots
                    mask = cv.erode(mask, None, iterations=1)
                    mask = cv.dilate(mask, None, iterations=1)
                    
                    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_cnt = max(contours, key=cv.contourArea)
                        # Filter out tiny specks, focus on the arrow
                        if cv.contourArea(largest_cnt) > 1200:
                            peri = cv.arcLength(largest_cnt, True)
                            approx = cv.approxPolyDP(largest_cnt, 0.02 * peri, True)
                            
                            # Arrow vertex check (7-12 covers most angles)
                            if 7 <= len(approx) <= 12:
                                M = cv.moments(largest_cnt)
                                if M["m00"] != 0:
                                    cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                                    direct = get_arrow_direction(largest_cnt, cx, cy)
                                    
                                    # Highlight the arrow and the direction
                                    cv.drawContours(display_frame, [largest_cnt], -1, (0, 255, 0), 2)
                                    cv.putText(display_frame, f"{color_name} ARROW: {direct}", (20, 100), 
                                                cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                    break # Stop looking for other colors once one is found

            cv.imshow("Maze Detection System", display_frame)
            if cv.waitKey(1) & 0xFF == 27: break
    finally:
        picam2.stop()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
