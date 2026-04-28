import cv2 as cv
import numpy as np
import time
from picamera2 import Picamera2

# ══════════════════════════════════════════════════════════════
# PARAMETERS FROM LH.py
# ══════════════════════════════════════════════════════════════
IMAGE_COLOUR_RANGES = {
    "Green":    {"space": "HSV", "lower": np.array([35,  40,  40]), "upper": np.array([90,  255, 255])}, 
    "Yellow":   {"space": "HSV", "lower": np.array([25, 40,  40]), "upper": np.array([40,  255, 255])}, 
    "Purple":   {"space": "LAB", "lower": np.array([0, 145,  60 ]), "upper": np.array([255, 195, 135])}, 
    "Blue/Teal":{"space": "LAB", "lower": np.array([20 , 90 , 50]), "upper": np.array([200,150,110])},
    "Red":      {"space": "LAB", "lower": np.array([0 , 160, 130]), "upper": np.array([255, 255, 180])}, 
    "Orange":   {"space": "LAB", "lower": np.array([0, 125, 145 ]), "upper": np.array([255, 190, 210])},
}

def get_diagnostics(cnt):
    area = cv.contourArea(cnt)
    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
    v = len(approx)
    
    # Numeric Convexity
    hull = cv.convexHull(cnt)
    hull_peri = cv.arcLength(hull, True)
    convexity = hull_peri / peri if peri > 0 else 0.0
    
    # Bounding Box Metrics (W3 Logic)
    x, y, w, h = cv.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0
    solidity = area / (w * h) if (w * h) > 0 else 0
    circ = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
    
    return {
        "A": int(area), "V": v, "H": h, "W": w,
        "Conv": round(convexity, 2), "Sol": round(solidity, 2),
        "AR": round(aspect_ratio, 2), "Circ": round(circ, 2)
    }

def main():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (480, 360)}))
    picam2.start()
    
    print("--- DIAGNOSTIC MODE ---")
    print("Point camera at symbol/line.")
    print("Press 'p' to PRINT stats for the largest object.")
    print("Press 'ESC' to quit.")

    try:
        while True:
            frame = picam2.capture_array()
            if frame.shape[2] == 4: frame = frame[:, :, :3]
            
            display_bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
            lab = cv.cvtColor(frame, cv.COLOR_RGB2LAB)
            
            all_objects = []
            for name, p in IMAGE_COLOUR_RANGES.items():
                src = hsv if p["space"] == "HSV" else lab
                mask = cv.inRange(src, p["lower"], p["upper"])
                contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    if cv.contourArea(cnt) > 500:
                        all_objects.append((cnt, name))
            
            all_objects.sort(key=lambda x: cv.contourArea(x[0]), reverse=True)
            
            # Show live stats for top candidate
            if all_objects:
                cnt, color_name = all_objects[0]
                d = get_diagnostics(cnt)
                x, y, w, h = cv.boundingRect(cnt)
                
                # Visual Feedback
                cv.drawContours(display_bgr, [cnt], -1, (0, 255, 0), 2)
                cv.rectangle(display_bgr, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                info_str = f"{color_name} | A:{d['A']} W:{d['W']} H:{d['H']} V:{d['V']} Conv:{d['Conv']}"
                cv.putText(display_bgr, info_str, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                w3_str = f"AR:{d['AR']} Sol:{d['Sol']} Circ:{d['Circ']}"
                cv.putText(display_bgr, w3_str, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv.imshow("LH Diagnostic Tool", display_bgr)
            key = cv.waitKey(1)
            
            if key == ord('p') and all_objects:
                cnt, name = all_objects[0]
                d = get_diagnostics(cnt)
                print(f"\n--- OBJECT DATA [{name}] ---")
                print(f"Area: {d['A']} | Vertices: {d['V']} || Width: {d['W']}| Height: {d['H']}")
                print(f"Convexity: {d['Conv']} | Solidity: {d['Sol']}")
                print(f"Aspect Ratio: {d['AR']} | Circularity: {d['Circ']}")
                
            elif key == 27: # ESC
                break
                
    finally:
        picam2.stop()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
