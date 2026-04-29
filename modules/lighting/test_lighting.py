"""
Standalone test runner for LightingDetector.
Opens the webcam and overlays all perceptual metrics on the live feed.
Press Q to quit.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from modules.lighting.lighting_detector import LightingDetector

detector = LightingDetector()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: could not open camera.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

font  = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (80, 200, 80)
RED   = (60, 60, 220)
AMBER = (0, 180, 220)
BLUE  = (220, 120, 50)
WHITE = (240, 240, 240)
DARK  = (30, 30, 30)

print("Lighting detector running — press Q to quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = detector.analyze(frame)
    ex     = result.extra or {}

    # Draw face bounding box
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = detector._get_face_region(gray)
    if face is not None:
        fx, fy, fw, fh = face
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), GREEN, 2)
        cv2.putText(frame, "face region", (fx, fy - 8), font, 0.42, GREEN, 1)

    # Status bar
    cv2.rectangle(frame, (0, 0), (640, 28), DARK, -1)
    issue_type   = ex.get("issue_type", "none")
    status_color = GREEN if result.is_ok else RED
    status_text  = "OK" if result.is_ok else f"WARN: {issue_type.upper()}"
    cv2.putText(frame, status_text, (8, 19), font, 0.58, status_color, 2)

    region_tag = ex.get("region_used", "?")
    face_tag   = f"{'face' if ex.get('face_detected') else 'fallback'} ({region_tag})"
    cv2.putText(frame, f"region: {face_tag}", (230, 19), font, 0.45, WHITE, 1)

    # Metrics panel
    cv2.rectangle(frame, (0, 385), (640, 480), DARK, -1)

    # Calibration status / color temperature indicator
    if ex.get("issue_type") == "calibrating":
        prog = ex.get("calib_progress", "")
        cv2.putText(frame, f"calibrating... {prog}", (8, 404), font, 0.46, AMBER, 1)
    else:
        ct       = ex.get("color_temp", "?")
        b_val    = ex.get("ambient_b", 0)
        b_delta  = ex.get("b_delta", 0)
        baseline = ex.get("baseline_b", 0)
        ct_color = BLUE if ct == "cool" else (AMBER if ct == "warm" else WHITE)
        cv2.putText(frame,
            f"color temp: {ct}  b*={b_val:.1f}  delta={b_delta:+.1f}  baseline={baseline:.1f}",
            (8, 404), font, 0.44, ct_color, 1)

    # L* brightness metrics
    lines = [
        (f"subject L*: {ex.get('subject_L', 0):.1f}   "
         f"ambient L*: {ex.get('ambient_L', 0):.1f}   "
         f"contrast L*: {ex.get('contrast_L', 0):.1f}"),
        (f"dark bg%: {ex.get('dark_fraction', 0)*100:.0f}   "
         f"bright bg%: {ex.get('bright_fraction', 0)*100:.0f}   "
         f"conf: {result.confidence:.0%}"),
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (8, 423 + i * 20), font, 0.44, WHITE, 1)

    # Warning message
    if not result.is_ok:
        msg = result.warning_message[:82] + ("..." if len(result.warning_message) > 82 else "")
        cv2.putText(frame, msg, (8, 370), font, 0.42, RED, 1)

    cv2.imshow("Lighting Detector Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
