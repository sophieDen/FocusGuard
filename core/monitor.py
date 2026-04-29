import cv2
import time
import importlib
from alert import AlertManager
from .base_detector import BaseDetector
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS

class FocusGuardMonitor:
    """
    Loads all detector modules, takes webcam frames,
    runs every module on each frame, passes results
    to AlertManager.
    """

    def __init__(self, detectors: list[BaseDetector]):
        self.detectors = detectors
        self.alert_manager = AlertManager()
        self._frame_interval = 1.0 / TARGET_FPS

    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        print("press Q to quit\n")
        while True:
            start = time.time()
            # Read frame from cam
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            # Run every module on this frame
            for detector in self.detectors:
                try:
                    result = detector.analyze(frame)
                    self.alert_manager.process(result)
                except Exception as e:
                    print(f"error {detector} failed: {e}")

            # show raw feed while developing
            cv2.imshow("FocusGuard", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Pace to target FPS
            elapsed = time.time() - start
            sleep_for = self._frame_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

        cap.release()
        cv2.destroyAllWindows()