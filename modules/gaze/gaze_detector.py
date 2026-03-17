import numpy as np
from core.base_detector import BaseDetector, DetectionResult
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
from collections import deque
from alert import AlertManager
from config import GAZE_DROWSY_EAR_THRESHOLD, GAZE_DROWSY_SECONDS, GAZE_DISTRACTION_SECONDS

# IN SECONDS — lower these for faster testing
STARING_DURATION  = 7
SLEEPING_DURATION = 3
D_GAZE_DURATION   = 5

EAR_THRESHOLD     = 0.20   # raised slightly from 0.15 — more reliable closed-eye detection
ASSUMED_FPS       = 25     # fallback until real fps is measured


class GazeDetector(BaseDetector):
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path="models/face_landmarker.task")
        options      = vision.FaceLandmarkerOptions(base_options=base_options)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        self.lm_le_range      = []
        self.lm_re_range      = []
        self.gaze_down_frames = 0
        self.staring_seconds  = 0
        self.num_frames_count = 0
        self.start_time       = time.time()
        self.alert            = AlertManager()

        # Rolling FPS: timestamps of last N frames
        self._frame_times = deque(maxlen=60)

    # =========================================================================
    #   FPS — rolling window instead of session average
    # =========================================================================

    def get_fps(self):
        """Estimate FPS from the last 60 frame timestamps."""
        now = time.time()
        self._frame_times.append(now)
        if len(self._frame_times) < 2:
            return ASSUMED_FPS
        elapsed = self._frame_times[-1] - self._frame_times[0]
        if elapsed <= 0:
            return ASSUMED_FPS
        return (len(self._frame_times) - 1) / elapsed

    # kept for run() compatibility
    def seconds_to_frames(self, num_frames_count, start_time, user_second_amount=10):
        current_time = time.time()
        t_passed     = current_time - start_time
        fps          = num_frames_count / (t_passed + 1e-8)
        return fps * user_second_amount

    # =========================================================================
    #   Landmark helpers
    # =========================================================================

    def landmark_vals(self, frame):
        if frame is None:
            return None
        mp_frame  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        return self.face_landmarker.detect(mp_frame)

    def full_landmark_mapping(self, lm_result, frame):
        full_landmarks = lm_result.face_landmarks[0]
        h, w, _ = frame.shape
        for lm in full_landmarks:
            pass  # cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 1, (0,255,0), -1)

    def eyes_landmark_mapping(self, lm_result, frame):
        lm_le = [463,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382,362]
        lm_re = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]
        full_landmarks = lm_result.face_landmarks[0]
        h, w, _ = frame.shape
        for idx in lm_le + lm_re:
            lm = full_landmarks[idx]
            pass  # cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 1, (0,0,255), -1)

    # =========================================================================
    #   Eye Aspect Ratio (EAR)
    # =========================================================================

    def eye_range_vals(self, lm_result, frame):
        fl = lm_result.face_landmarks[0]

        def ear(p1i, p2i, p3i, p4i, p5i, p6i):
            p1 = np.array([fl[p1i].x, fl[p1i].y])
            p2 = np.array([fl[p2i].x, fl[p2i].y])
            p3 = np.array([fl[p3i].x, fl[p3i].y])
            p4 = np.array([fl[p4i].x, fl[p4i].y])
            p5 = np.array([fl[p5i].x, fl[p5i].y])
            p6 = np.array([fl[p6i].x, fl[p6i].y])
            return (np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)) / (2 * np.linalg.norm(p1-p4))

        le_EAR = ear(362, 385, 387, 263, 373, 380)
        re_EAR = ear(33,  160, 158, 133, 153, 144)
        return le_EAR, re_EAR

    # =========================================================================
    #   Drowsiness detection
    # =========================================================================

    def eye_sleeping_threshold(self, lm_le_range, lm_re_range, fps,
                                threshold=EAR_THRESHOLD, s_duration=SLEEPING_DURATION):
        n_frames = fps * s_duration
        avg_le   = np.mean(lm_le_range[-int(n_frames):]) if len(lm_le_range) >= int(n_frames) else None
        avg_re   = np.mean(lm_re_range[-int(n_frames):]) if len(lm_re_range) >= int(n_frames) else None

        if avg_le is None or avg_re is None:
            return False  # not enough data yet

        result = avg_le <= threshold or avg_re <= threshold
        print(f"[sleep] n_frames={n_frames:.0f} buf={len(lm_re_range)} "
              f"avg_le={avg_le:.3f} avg_re={avg_re:.3f} thresh={threshold} → {result}")
        return result

    # =========================================================================
    #   Blink / staring detection
    # =========================================================================

    def eye_blink_threshold(self, lm_le_range, lm_re_range, fps,
                             threshold_frames=3, n_threshold=EAR_THRESHOLD, b_duration=1):
        n_frames     = fps * b_duration
        le           = np.array(lm_le_range).flatten()
        re           = np.array(lm_re_range).flatten()
        le_threshold = np.sum(le[-threshold_frames:] <= n_threshold)
        re_threshold = np.sum(re[-threshold_frames:] <= n_threshold)

        if le_threshold >= threshold_frames and re_threshold >= threshold_frames and len(re) > n_frames - 10:
            return True
        return False

    def eye_staring_tracker(self, blinked, fps, staring_duration=STARING_DURATION):
        n_frames = fps * staring_duration
        if blinked:
            self.staring_seconds = 0
            return False
        self.staring_seconds += 1
        return self.staring_seconds > n_frames

    # =========================================================================
    #   Gaze direction & duration
    # =========================================================================

    def gaze_direct_detect(self, lm_result, frame, c_threshold=0.5):
        h, w, _ = frame.shape
        fl       = lm_result.face_landmarks[0]

        def gaze_vec(inner_i, outer_i, iris_i):
            inner  = np.array([fl[inner_i].x * w, fl[inner_i].y * h])
            outer  = np.array([fl[outer_i].x * w, fl[outer_i].y * h])
            iris   = np.array([fl[iris_i].x  * w, fl[iris_i].y  * h])
            centre = (inner + outer) / 2
            vec    = iris - centre
            norm   = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec

        le_gaze   = gaze_vec(133, 33,  468)
        re_gaze   = gaze_vec(362, 263, 473)
        combined  = le_gaze + re_gaze
        norm      = np.linalg.norm(combined)
        full_gaze = combined / norm if norm > 0 else combined

        up    = full_gaze[1] < -c_threshold
        down  = full_gaze[1] >  c_threshold
        right = full_gaze[0] < -c_threshold
        left  = full_gaze[0] >  c_threshold

        ud_abs = np.abs(full_gaze[1])
        lr_abs = np.abs(full_gaze[0])

        if not any([up, down, left, right]):  return 1, full_gaze  # centre
        elif up   and ud_abs > lr_abs:        return 2, full_gaze  # up
        elif down and ud_abs > lr_abs:        return 3, full_gaze  # down
        elif left:                            return 4, full_gaze  # left IRL
        elif right:                           return 5, full_gaze  # right IRL
        else:                                 return 1, full_gaze  # diagonal → centre

    def gaze_duration_detect(self, lm_result, frame, fps,
                              gaze_threshold=D_GAZE_DURATION, c_threshold=0.5):
        direction, full_gaze = self.gaze_direct_detect(lm_result, frame, c_threshold=c_threshold)
        threshold_n_frames   = fps * gaze_threshold

        # up (2) excluded — iris naturally sits above eye centre when looking at screen
        looking_away = direction in (3, 4, 5)
        print(f"[gaze_dir] direction={direction} gaze={np.round(full_gaze,3)} looking_away={looking_away}")

        if looking_away:
            self.gaze_down_frames += 1
            if self.gaze_down_frames >= threshold_n_frames:
                return True
        else:
            self.gaze_down_frames = 0
        return False

    # =========================================================================
    #   Rolling temporal memory
    # =========================================================================

    def rolling_temporal_memory(self, lm_le_range, lm_re_range, n_frames=50):
        max_buf = int(n_frames) + 10  # keep a little extra headroom
        if len(lm_le_range) > max_buf:
            del lm_le_range[:len(lm_le_range) - max_buf]
        if len(lm_re_range) > max_buf:
            del lm_re_range[:len(lm_re_range) - max_buf]

    # =========================================================================
    #   analyze() — called per-frame by the main pipeline
    # =========================================================================

    def analyze(self, frame: np.ndarray) -> DetectionResult:
        self.num_frames_count += 1
        fps       = self.get_fps()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm_result = self.landmark_vals(frame_rgb)

        if not lm_result or len(lm_result.face_landmarks) == 0:
            print("[gaze] No face detected")
            return DetectionResult(module_name="gaze", is_ok=True, warning_message="", confidence=1.0)

        le_ear, re_ear = self.eye_range_vals(lm_result, frame_rgb)
        self.lm_le_range.append(le_ear)
        self.lm_re_range.append(re_ear)
        self.rolling_temporal_memory(
            self.lm_le_range, self.lm_re_range,
            n_frames=fps * max(SLEEPING_DURATION, STARING_DURATION, D_GAZE_DURATION)
        )

        print(f"[gaze] fps={fps:.1f} buf={len(self.lm_re_range)} "
              f"le={le_ear:.3f} re={re_ear:.3f}")

        if self.eye_sleeping_threshold(self.lm_le_range, self.lm_re_range, fps):
            return DetectionResult(module_name="gaze", is_ok=False,
                                   warning_message="You look tired!", confidence=0.9)

        blinked = self.eye_blink_threshold(self.lm_le_range, self.lm_re_range,
                                            fps, threshold_frames=1)
        staring = self.eye_staring_tracker(blinked, fps)
        print(f"[gaze] blinked={blinked} staring={staring} staring_frames={self.staring_seconds}")
        if staring:
            return DetectionResult(module_name="gaze", is_ok=False,
                                   warning_message="Remember to blink!", confidence=0.9)

        gaze_away = self.gaze_duration_detect(lm_result, frame_rgb, fps)
        print(f"[gaze] gaze_away={gaze_away} gaze_down_frames={self.gaze_down_frames}")
        if gaze_away:
            return DetectionResult(module_name="gaze", is_ok=False,
                                   warning_message="You've been looking away from the screen too long!", confidence=0.9)

        return DetectionResult(module_name="gaze", is_ok=True, warning_message="", confidence=1.0)

    # =========================================================================
    #   run() — standalone webcam loop for testing
    # =========================================================================

    def run(self):
        feed = cv2.VideoCapture(0)
        while True:
            self.num_frames_count += 1
            good, frame = feed.read()
            if not good:
                break

            fps       = self.get_fps()
            lm_result = self.landmark_vals(frame)

            if lm_result and len(lm_result.face_landmarks) >= 1:
                le_ear, re_ear = self.eye_range_vals(lm_result, frame)
                self.lm_le_range.append(le_ear)
                self.lm_re_range.append(re_ear)
                self.rolling_temporal_memory(
                    self.lm_le_range, self.lm_re_range,
                    n_frames=fps * max(SLEEPING_DURATION, STARING_DURATION, D_GAZE_DURATION)
                )

                if self.eye_sleeping_threshold(self.lm_le_range, self.lm_re_range, fps):
                    self.alert.process(DetectionResult(module_name="gaze", is_ok=False,
                                                       warning_message="You look drowsy", confidence=0.9))

                blinked = self.eye_blink_threshold(self.lm_le_range, self.lm_re_range, fps, threshold_frames=1)
                if self.eye_staring_tracker(blinked, fps):
                    self.alert.process(DetectionResult(module_name="gaze", is_ok=False,
                                                       warning_message="Remember to blink!", confidence=0.9))

                if self.gaze_duration_detect(lm_result, frame, fps):
                    self.alert.process(DetectionResult(module_name="gaze", is_ok=False,
                                                       warning_message="You've been looking away from the screen too long.",
                                                       confidence=0.9))

            cv2.imshow("Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        feed.release()


if __name__ == '__main__':
    GazeDetector().run()