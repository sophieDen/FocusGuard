from core.monitor import FocusGuardMonitor
from modules.lighting.lighting_detector import LightingDetector
from modules.gaze.gaze_detector import GazeDetector
from modules.posture.posture_detector import PostureDetector

if __name__ == "__main__":
    detectors = [
        LightingDetector(),
        GazeDetector(),
        PostureDetector(),
    ]
    
    monitor = FocusGuardMonitor(detectors)
    monitor.run()