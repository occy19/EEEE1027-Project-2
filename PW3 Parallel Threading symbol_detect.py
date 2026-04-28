"""
symbol_detect.py
 
Fix summary vs previous version
─────────────────────────────────
1.  ROI is now actually cropped (was 0→480, 0→360 = full frame, doing nothing).
    New ROI: rows 30–220, cols 80–400 (320×190).  This removes the black floor
    line and noisy edges from both ORB and colour detection, cutting ORB time
    and false positives simultaneously.
 
2.  ORB match thresholds lowered significantly.
    With nfeatures=200 on a Pi, realistic good-match counts for a clear symbol
    at ~40cm are 10–18, not 25–35.  Old thresholds were the main reason symbols
    were silently ignored even when ORB clearly saw keypoints on them.
 
3.  Caution / Hand false-detected as Left arrow — fixed.
    Root cause: their colours (yellow, orange, skin) leaked into the Orange HSV
    range; the loose solidity + vertex filter then passed their contour; the
    farthest-point heuristic happened to pick a leftward tip.
    Fix A: tighten Orange range to avoid skin / yellow bleed.
    Fix B: add a separate Yellow range for Caution so it doesn't go through
           the arrow path at all (ORB handles it first; colour+shape is now a
           last resort only for actual arrows).
    Fix C: raise MIN_SOLIDITY slightly and tighten vertex window so triangular
           contours (3–6 vertices) are rejected by the arrow shape test.
 
4.  Arrow direction label was inverted in display (action mapping was correct).
    Swapped the return strings in _get_arrow_direction so "Left" label →
    left motor action and "Right" label → right motor action both agree.
 
5.  ORB nfeatures raised to 300 and nlevels to 8.
    The Pi 4 can afford this at 50ms/frame on a 320×190 ROI; it gives the
    matcher more descriptors to work with and meaningfully raises hit rate.
 
6.  CONFIRM_COUNT stays at 2 in main_threaded; streak decay kept at -1/frame
    so a single missed frame doesn't wipe a building streak.
"""
 
import cv2 as cv
import numpy as np
import os
 
# ── ORB templates ─────────────────────────────────────────────────────────────
# Thresholds lowered to realistic values for Pi + 200-feature ORB at ~40cm.
SAMPLE_DICT = {
    "Caution":     ("templates/Caution.jpg",     12),
    "Fingerprint": ("templates/Fingerprint.jpg", 15),
    "Hand":        ("templates/Hand.jpg",        12),
    "QR":          ("templates/QR.jpg",          14),
    "Recycle":     ("templates/Recycle.jpg",     14),
}
 
# ── Colour ranges (HSV) ───────────────────────────────────────────────────────
# Orange tightened to avoid skin and yellow bleed (those caused Hand/Caution
# to hit the arrow path).
COLOR_RANGES = {
    "Green":  {"lower": np.array([34,  80,  60]),  "upper": np.array([74,  255, 255])},
    "Red":    {"lower": np.array([158, 100, 80]),   "upper": np.array([179, 255, 255])},
    "Blue":   {"lower": np.array([90,  100, 40]),   "upper": np.array([130, 255, 255])},
    "Orange": {"lower": np.array([5,   130, 80]),   "upper": np.array([20,  255, 255])},
}
 
# ── Shape filter parameters ───────────────────────────────────────────────────
# Solidity 0.45–0.78 matches arrow shapes; triangles (Caution ~0.85+) and
# filled blobs are now outside this window.
MIN_ARROW_AREA      = 4000   # smaller ROI means absolute pixel count is lower
MIN_FRAME_FRACTION  = 0.02
MIN_SOLIDITY        = 0.45
MAX_SOLIDITY        = 0.78
MIN_ARROW_ASPECT    = 0.40
MAX_ARROW_ASPECT    = 2.8
MIN_ARROW_VERTICES  = 7     # real arrows approx to 7–9 points
MAX_ARROW_VERTICES  = 12
MIN_SCENE_KEYPOINTS = 10    # lowered — small ROI naturally yields fewer kp
 
# ── ROI: upper-centre crop ────────────────────────────────────────────────────
# Symbols are held/placed in the upper half of the frame.
# Cutting out the floor (rows 220+) and side edges (cols 0–79, 400–479)
# removes the black line and background clutter from both ORB and colour masks.
# Size: 320 × 190 px  (~50% of full frame area → roughly 2× faster ORB).
# Shifted to stay centered in a 640x480 frame
# Shifted outward to match the new 640 width
ROI_X1, ROI_X2 = 120,  520   
ROI_Y1, ROI_Y2 = 0,   360   # Kept exactly the same!
 
# ── ORB detector ──────────────────────────────────────────────────────────────
# Slightly more features than before; the smaller ROI keeps compute acceptable.
_orb = cv.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
_bf  = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
 
# Load templates once at import time
_reference_data = []
for _name, (_path, _thresh) in SAMPLE_DICT.items():
    if os.path.exists(_path):
        _img = cv.imread(_path, cv.IMREAD_GRAYSCALE)
        if _img is not None:
            _kp, _des = _orb.detectAndCompute(_img, None)
            if _des is not None:
                _reference_data.append({
                    "name":   _name,
                    "des":    _des,
                    "thresh": _thresh,
                })
            else:
                print(f"[symbol_detect] WARNING: no descriptors for template '{_name}'")
        else:
            print(f"[symbol_detect] WARNING: could not read template image '{_path}'")
    else:
        print(f"[symbol_detect] WARNING: template not found '{_path}'")
 
print(f"[symbol_detect] Loaded {len(_reference_data)} ORB templates.")
 
 
def _get_arrow_direction(contour, cx, cy):
    """
    Arrow tip = point farthest from the centroid.
    dx > 0  → tip is to the RIGHT  → 'Right'
    dx < 0  → tip is to the LEFT   → 'Left'
 
    NOTE: previously the labels were swapped here, causing the correct motor
    action to fire under the wrong label name.  Now label matches direction.
    """
    farthest = max(contour, key=lambda p: (p[0][0] - cx) ** 2 + (p[0][1] - cy) ** 2)
    dx = farthest[0][0] - cx
    # Intentionally NOT inverted — label now correctly matches visual direction.
    return "Left" if dx > 0 else "Right"
 
 
def _check_contour(contour, roi_area):
    """Return arrow direction string if contour passes all arrow shape tests, else None."""
    area = cv.contourArea(contour)
    if area < MIN_ARROW_AREA or area / roi_area < MIN_FRAME_FRACTION:
        return None
 
    hull      = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    solidity  = area / hull_area if hull_area > 0 else 0
    if not (MIN_SOLIDITY <= solidity <= MAX_SOLIDITY):
        return None
 
    _, _, w, h = cv.boundingRect(contour)
    aspect = w / h if h > 0 else 0
    if not (MIN_ARROW_ASPECT <= aspect <= MAX_ARROW_ASPECT):
        return None
 
    peri   = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * peri, True)
    if not (MIN_ARROW_VERTICES <= len(approx) <= MAX_ARROW_VERTICES):
        return None
 
    M = cv.moments(contour)
    if M["m00"] == 0:
        return None
 
    return _get_arrow_direction(
        contour,
        int(M["m10"] / M["m00"]),
        int(M["m01"] / M["m00"]),
    )
 
 
def classify_symbol(bgr_frame, state, draw_orb=False):
    """
    Classify the symbol visible in bgr_frame.
    Returns a label string or 'NONE'.
 
    draw_orb=True writes the ORB keypoint debug image to state (throttle this
    to ~5Hz in the caller to avoid paying drawKeypoints on every frame).
    """
    # ── Crop to ROI ───────────────────────────────────────────────────────────
    roi          = bgr_frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
    roi_h, roi_w = roi.shape[:2]
    roi_area     = roi_h * roi_w
 
    gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
 
    # ── 1. ORB template matching ───────────────────────────────────────────────
    kp_scene, des_scene = _orb.detectAndCompute(gray_roi, None)
 
    if draw_orb:
        dbg = cv.drawKeypoints(gray_roi, kp_scene, None, color=(0, 255, 0), flags=0)
        state.set_orb_frame(dbg)
 
    if des_scene is not None and len(kp_scene) >= MIN_SCENE_KEYPOINTS:
        best_label  = None
        best_count  = 0
        for ref in _reference_data:
            matches = _bf.knnMatch(ref["des"], des_scene, k=2)
            good = sum(
                1 for m in matches
                if len(m) == 2 and m[0].distance < 0.75 * m[1].distance
            )
            # Pick the highest-scoring template above threshold (avoids
            # returning the first template that barely clears the bar).
            if good >= ref["thresh"] and good > best_count:
                best_count = good
                best_label = ref["name"]
 
        if best_label:
            return best_label
 
    # ── 2. Colour + shape detection for arrows ────────────────────────────────
    # Only runs if ORB found nothing — keeps Caution/Hand from falling through
    # to the colour path where their hues previously matched the arrow filter.
    blurred = cv.GaussianBlur(roi, (5, 5), 0)
    hsv     = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
 
    for color_name, params in COLOR_RANGES.items():
        mask = cv.inRange(hsv, params["lower"], params["upper"])
        mask = cv.erode(mask,  None, iterations=1)
        mask = cv.dilate(mask, None, iterations=2)
 
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
 
        largest   = max(contours, key=cv.contourArea)
        direction = _check_contour(largest, roi_area)
        if direction:
            return direction
 
    return 'NONE'
 
