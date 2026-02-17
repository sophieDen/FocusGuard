import numpy as np
from core.base_detector import BaseDetector, DetectionResult

class GazeDetector(BaseDetector):

    def analyze(self, frame: np.ndarray) -> DetectionResult:
        # implements gaze + drowsiness logic here
        # Use config.py for GAZE_* thresholds
        
        return DetectionResult(
            module_name     = "gaze",
            is_ok           = True,          # placeholder CHANGEEEE
            warning_message = "",
            confidence      = 1.0,
        )