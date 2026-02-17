import numpy as np
import cv2
from core.base_detector import BaseDetector, DetectionResult
from config import LIGHTING_DARK_THRESHOLD, LIGHTING_CONTRAST_THRESHOLD

class LightingDetector(BaseDetector):

    def analyze(self, frame: np.ndarray) -> DetectionResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        is_ok = mean_brightness >= LIGHTING_DARK_THRESHOLD

        return DetectionResult(
            module_name     = "lighting",
            is_ok           = is_ok,
            warning_message = "Room is too dark — consider turning on a light.",
            confidence      = min(1.0, (LIGHTING_DARK_THRESHOLD - mean_brightness)
                                  / LIGHTING_DARK_THRESHOLD) if not is_ok else 1.0,
            extra           = {"mean_brightness": mean_brightness}
        )