import cv2
import time
import math as m
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from core.base_detector import BaseDetector, DetectionResult


def findDistance(x1, y1, x2, y2):
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def findAngle(x1, y1, x2, y2):
    theta  = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree


class PostureDetector(BaseDetector):

    # seconds of bad posture before warning fires
    BAD_POSTURE_SECONDS  = 3
    PHONE_SECONDS        = 5
    NO_CHAIR_SECONDS     = 5

    # posture thresholds
    Z_DELTA_THRESHOLD    = 0.02
    NECK_MAX             = 35
    NECK_MIN             = 15
    SHOULDER_DIFF_MAX    = 20
    OFFSET_MARGIN        = 0.15

    def __init__(self):
        # ── Pose landmarker (IMAGE mode — one frame at a time) ──────────────
        pose_base = python.BaseOptions(model_asset_path="models/pose_landmarker_full.task")
        pose_opts = vision.PoseLandmarkerOptions(
            base_options=pose_base,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_opts)

        # ── Object detector (IMAGE mode) ────────────────────────────────────
        obj_base = python.BaseOptions(model_asset_path="models/efficientdet_lite2.tflite")
        obj_opts = vision.ObjectDetectorOptions(
            base_options=obj_base,
            running_mode=vision.RunningMode.IMAGE,
            max_results=5,
            score_threshold=0.5,
            category_allowlist=["cell phone", "chair"],
        )
        self.obj_detector = vision.ObjectDetector.create_from_options(obj_opts)

        # ── State ────────────────────────────────────────────────────────────
        self.bad_frames               = 0
        self.good_frames              = 0
        self.fps_estimate             = 15          # updated each frame
        self._frame_count             = 0
        self._start_time              = time.time()

        self.phone_first_detected_time  = None
        self.chair_missing_start_time   = None
        self._last_warning              = ""        # track last warning to re-trigger on change

        # ── Calibration & Smoothing State ────────────────────────────────────
        self.is_calibrated          = False
        self.base_depth             = 0.0
        self.base_neck              = 0.0
        self.base_shoulder_lvl      = 0.0
        self.base_offset            = 0.0
        self.depth_history          = []
        self._calibrate_next_frame  = False  # set via request_calibration()

    def request_calibration(self):
        """Call this (e.g. on a keypress) to capture the baseline on the next frame."""
        self._calibrate_next_frame = True

    # =========================================================================
    #   analyze() — called per-frame by the main pipeline
    # =========================================================================

    def analyze(self, frame: np.ndarray) -> DetectionResult:
        self._frame_count += 1

        # rolling FPS estimate
        elapsed = time.time() - self._start_time
        if elapsed > 0:
            self.fps_estimate = self._frame_count / elapsed

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # ── Object detection ─────────────────────────────────────────────────
        obj_result = self.obj_detector.detect(mp_image)
        obj_warning = self._check_objects(obj_result)
        if obj_warning:
            return DetectionResult(
                module_name="posture", is_ok=False,
                warning_message=obj_warning, confidence=0.9
            )

        # ── Pose detection ───────────────────────────────────────────────────
        pose_result = self.pose_detector.detect(mp_image)
        if not pose_result.pose_landmarks:
            return DetectionResult(
                module_name="posture", is_ok=True,
                warning_message="", confidence=1.0
            )

        return self._check_posture(pose_result, frame)

    # =========================================================================
    #   Object detection helper
    # =========================================================================

    def _check_objects(self, obj_result) -> str:
        """Returns a warning string if objects warrant one, else empty string."""
        now                        = time.time()
        phone_detected_this_frame  = False
        chair_detected_this_frame  = False

        if obj_result.detections:
            for det in obj_result.detections:
                name = det.categories[0].category_name
                if name == "cell phone":
                    phone_detected_this_frame = True
                elif name == "chair":
                    chair_detected_this_frame = True

        # phone
        if phone_detected_this_frame:
            if self.phone_first_detected_time is None:
                self.phone_first_detected_time = now
            elif now - self.phone_first_detected_time >= self.PHONE_SECONDS:
                return "Put away the phone!"
        else:
            self.phone_first_detected_time = None

        # chair
        if not chair_detected_this_frame:
            if self.chair_missing_start_time is None:
                self.chair_missing_start_time = now
            elif now - self.chair_missing_start_time >= self.NO_CHAIR_SECONDS:
                return "Go find a nice chair!"
        else:
            self.chair_missing_start_time = None

        return ""

    # =========================================================================
    #   Posture check helper
    # =========================================================================

    def _check_posture(self, pose_result, frame) -> DetectionResult:
        lm     = pose_result.pose_landmarks[0]
        h, w   = frame.shape[:2]
        fps    = max(self.fps_estimate, 1)

        try:
            l_shldr_x = int(lm[11].x * w);  l_shldr_y = int(lm[11].y * h);  l_shldr_z = lm[11].z
            r_shldr_x = int(lm[12].x * w);  r_shldr_y = int(lm[12].y * h);  r_shldr_z = lm[12].z
            l_ear_x   = int(lm[7].x  * w);  l_ear_y   = int(lm[7].y  * h)
            nose_z    = lm[0].z

            m_shldr_z           = (l_shldr_z + r_shldr_z) / 2

            # --- Moving Average Filter for Z-axis noise ---
            raw_depth_diff = nose_z - m_shldr_z
            self.depth_history.append(raw_depth_diff)
            if len(self.depth_history) > 10:
                self.depth_history.pop(0)

            depth_diff = sum(self.depth_history) / len(self.depth_history)
            # ----------------------------------------------

            shoulder_lvl_diff   = abs(l_shldr_y - r_shldr_y)
            offset              = findDistance(lm[11].x, lm[11].y, lm[12].x, lm[12].y)
            neck_inclination    = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)

        except Exception as e:
            print(f"[posture] landmark error: {e}")
            return DetectionResult(module_name="posture", is_ok=True, warning_message="", confidence=1.0)

        # ── Calibration Trigger ──────────────────────────────────────────────
        if self._calibrate_next_frame:
            self.base_depth            = depth_diff
            self.base_neck             = neck_inclination
            self.base_shoulder_lvl     = shoulder_lvl_diff
            self.base_offset           = offset
            self.is_calibrated         = True
            self._calibrate_next_frame = False

        if not self.is_calibrated:
            return DetectionResult(
                module_name="posture", is_ok=False,
                warning_message="SIT STRAIGHT & PRESS 'C' TO CALIBRATE", confidence=1.0
            )

        # ── Classify posture ─────────────────────────────────────────────────
        z_delta = self.base_depth - depth_diff
        issue = ""

        if offset > (self.base_offset + self.OFFSET_MARGIN):
            issue = "Too close, please move back."
        elif offset < (self.base_offset - self.OFFSET_MARGIN):
            issue = "Too far, please move closer."
        elif z_delta > self.Z_DELTA_THRESHOLD:
            issue = "Don't lean forward (Turtle neck)!"
        elif abs(shoulder_lvl_diff - self.base_shoulder_lvl) > self.SHOULDER_DIFF_MAX:
            issue = "Your shoulders are uneven."
        elif neck_inclination > self.NECK_MAX:
            issue = "Straighten your neck."
        elif neck_inclination < self.NECK_MIN:
            issue = "Straighten your neck."

        if issue:
            self.good_frames  = 0
            self.bad_frames  += 1
        else:
            self.good_frames += 1
            self.bad_frames   = 0
            self._last_warning = ""   # reset so next bad state re-triggers

        bad_time = self.bad_frames / fps

        if bad_time >= self.BAD_POSTURE_SECONDS:
            # only surface if warning is new or has changed
            if issue != self._last_warning:
                self._last_warning = issue
                return DetectionResult(
                    module_name="posture", is_ok=False,
                    warning_message=issue, confidence=0.9
                )

        return DetectionResult(module_name="posture", is_ok=True, warning_message="", confidence=1.0)
