"""
Standalone test runner for LightingDetector.
Opens the webcam and overlays detection results on the live feed.
Press Q to quit.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from modules.lighting.lighting_detector import LightingDetector

detector = LightingDetector()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: could not open camera.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

font      = cv2.FONT_HERSHEY_SIMPLEX
GREEN     = (80, 200, 80)
RED       = (60, 60, 220)
YELLOW    = (0, 200, 220)
WHITE     = (240, 240, 240)
DARK      = (30, 30, 30)

print("Lighting detector running — press Q to quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = detector.analyze(frame)
    ex     = result.extra or {}

    # Draw face bounding box if detector found one
    import numpy as np
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face  = detector._get_face_region(gray)
    if face is not None:
        fx, fy, fw, fh = face
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), GREEN, 2)
        cv2.putText(frame, "face region", (fx, fy - 8), font, 0.45, GREEN, 1)

    # Status bar background
    cv2.rectangle(frame, (0, 0), (640, 30), DARK, -1)

    status_color = GREEN if result.is_ok else RED
    status_text  = "OK" if result.is_ok else f"WARN  {ex.get('issue_type', '').upper()}"
    cv2.putText(frame, status_text, (8, 20), font, 0.6, status_color, 2)

    region_label = ex.get("region_used", "?")
    face_tag     = "face" if ex.get("face_detected") else f"fallback ({region_label})"
    cv2.putText(frame, f"region: {face_tag}", (200, 20), font, 0.5, WHITE, 1)

    # Metrics panel bottom
    cv2.rectangle(frame, (0, 390), (640, 480), DARK, -1)

    lines = [
        f"mean: {ex.get('mean_brightness', 0):.0f}   "
        f"ambient dark%: {ex.get('ambient_dark_ratio', 0)*100:.0f}   "
        f"ambient bright%: {ex.get('ambient_bright_ratio', 0)*100:.0f}",

        f"subject: {ex.get('subject_brightness', 0):.0f}   "
        f"background: {ex.get('periphery_brightness', 0):.0f}   "
        f"contrast: {ex.get('contrast_difference', 0):.0f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (8, 410 + i * 22), font, 0.45, WHITE, 1)

    # Warning message (wraps onto two lines if long)
    if not result.is_ok:
        msg   = result.warning_message
        short = msg[:80] + ("..." if len(msg) > 80 else "")
        cv2.putText(frame, short, (8, 370), font, 0.45, RED, 1)

    cv2.imshow("Lighting Detector Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
