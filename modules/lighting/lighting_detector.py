"""
Analyzes ambient lighting conditions using histogram analysis.
Detects three problematic scenarios:
1. Overall too dark (eye strain, poor visibility)
2. Overall too bright (glare, discomfort)
3. High contrast (bright screen in dark room - worst for eyes)

Contrast detection uses a face-detected bounding box as the subject region
when a face is visible, falling back to a fixed center circle otherwise.
This makes the high-contrast check semantically accurate: it measures the
brightness difference the user's eyes are actually experiencing.
"""

import numpy as np
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_detector import BaseDetector, DetectionResult
import config


class LightingDetector(BaseDetector):
    """
    Histogram-based lighting analysis for workspace environments.

    The detector analyzes the distribution of pixel intensities to determine
    if lighting conditions are suitable for productive work.

    For contrast detection, a Haar cascade face detector is used to locate
    the user's face precisely. The face region is compared against the
    surrounding background to measure contrast. When no face is detected,
    a fixed central region is used as fallback.
    """

    def __init__(self):
        self.frame_count = 0
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self._face_cascade = cv2.CascadeClassifier(cascade_path)

    # =========================================================================
    #   Face detection helper
    # =========================================================================

    def _get_face_region(self, gray: np.ndarray):
        """
        Returns (x, y, w, h) of the largest detected face, or None.
        Uses OpenCV Haar cascade — no extra model file required.
        """
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    # =========================================================================
    #   Subject / background brightness split
    # =========================================================================

    def _analyze_regions(self, gray: np.ndarray):
        """
        Splits the frame into a subject region (face or center fallback) and
        background, then returns brightness and pixel-ratio metrics computed
        on the background only.

        Returns:
            subject_brightness   — mean brightness of face / center region
            bg_brightness        — mean brightness of everything else
            ambient_dark_ratio   — fraction of background pixels that are dark (0-85)
            ambient_bright_ratio — fraction of background pixels that are bright (170-255)
            face_detected        — bool
            region_label         — "face" or "center"

        Computing pixel ratios on background-only pixels is important: a
        screen-lit face in a dark room would otherwise push bright_pixels_ratio
        above the threshold and incorrectly trigger "too bright" instead of
        "high contrast" or "too dark".
        """
        face = self._get_face_region(gray)

        if face is not None:
            fx, fy, fw, fh = face
            face_pixels = gray[fy:fy + fh, fx:fx + fw]

            bg_mask = np.ones_like(gray, dtype=bool)
            bg_mask[fy:fy + fh, fx:fx + fw] = False
            bg_pixels = gray[bg_mask]
        else:
            # Fallback: fixed center circle
            h, w = gray.shape
            center_size = int(min(h, w) * config.LIGHTING_CENTER_RATIO)
            cy, cx = h // 2, w // 2
            half = center_size // 2

            face_pixels = gray[cy - half: cy + half, cx - half: cx + half]

            bg_mask = np.ones_like(gray, dtype=bool)
            bg_mask[cy - half: cy + half, cx - half: cx + half] = False
            bg_pixels = gray[bg_mask]

        # Ambient histogram from background pixels only
        bg_hist = np.bincount(bg_pixels.flatten(), minlength=256).astype(float)
        bg_hist /= bg_hist.sum()
        ambient_dark_ratio   = float(np.sum(bg_hist[:config.LIGHTING_DARK_PIXEL_THRESHOLD]))
        ambient_bright_ratio = float(np.sum(bg_hist[config.LIGHTING_BRIGHT_PIXEL_THRESHOLD:]))

        return (
            float(np.mean(face_pixels)),
            float(np.mean(bg_pixels)),
            ambient_dark_ratio,
            ambient_bright_ratio,
            face is not None,
            "face" if face is not None else "center",
        )

    # =========================================================================
    #   Main analysis
    # =========================================================================

    def analyze(self, frame: np.ndarray) -> DetectionResult:
        """
        Analyze lighting conditions from a webcam frame.

        Args:
            frame: BGR image from webcam

        Returns:
            DetectionResult indicating lighting quality
        """
        self.frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
        histogram = histogram.flatten() / histogram.sum()

        mean_brightness   = float(np.mean(gray))
        median_brightness = float(np.median(gray))
        std_brightness    = float(np.std(gray))

        (subject_brightness, bg_brightness,
         ambient_dark_ratio, ambient_bright_ratio,
         face_detected, region_label) = self._analyze_regions(gray)

        contrast_difference = subject_brightness - bg_brightness

        base_extra = {
            "face_detected":        face_detected,
            "region_used":          region_label,
            "subject_brightness":   subject_brightness,
            "periphery_brightness": bg_brightness,
            "contrast_difference":  contrast_difference,
            "mean_brightness":      mean_brightness,
            "median_brightness":    median_brightness,
            "std_brightness":       std_brightness,
            "ambient_dark_ratio":   ambient_dark_ratio,
            "ambient_bright_ratio": ambient_bright_ratio,
            "histogram":            histogram.tolist(),
        }

        # 1. High contrast (most harmful) — bright face/screen in dark room
        if (contrast_difference > config.LIGHTING_CONTRAST_THRESHOLD and
                bg_brightness < config.LIGHTING_DARK_THRESHOLD):

            if face_detected:
                msg = (
                    f"Your face is brightly lit ({subject_brightness:.0f}) "
                    f"but the room is dark ({bg_brightness:.0f}). "
                    "This causes severe eye strain. Turn on ambient lighting!"
                )
            else:
                msg = (
                    f"High contrast detected: screen area is bright ({subject_brightness:.0f}) "
                    f"but the room is dark ({bg_brightness:.0f}). "
                    "This causes severe eye strain. Turn on ambient lighting!"
                )

            return DetectionResult(
                module_name="lighting",
                is_ok=False,
                warning_message=msg,
                confidence=self._calculate_confidence(
                    contrast_difference,
                    config.LIGHTING_CONTRAST_THRESHOLD,
                    upper_bound=150
                ),
                extra={"issue_type": "high_contrast", **base_extra}
            )

        # 2. Overall too dark — uses background-only metrics so a screen-lit
        #    face doesn't mask a dark ambient environment
        if (ambient_dark_ratio > config.LIGHTING_LOW_INTENSITY_RATIO or
                bg_brightness < config.LIGHTING_DARK_THRESHOLD):
            return DetectionResult(
                module_name="lighting",
                is_ok=False,
                warning_message=(
                    f"Room is too dark (ambient brightness: {bg_brightness:.0f}/255, "
                    f"{ambient_dark_ratio * 100:.0f}% dark pixels). "
                    "Increase ambient lighting to reduce eye strain."
                ),
                confidence=max(
                    ambient_dark_ratio / config.LIGHTING_LOW_INTENSITY_RATIO,
                    (config.LIGHTING_DARK_THRESHOLD - bg_brightness) / config.LIGHTING_DARK_THRESHOLD
                ),
                extra={"issue_type": "too_dark", **base_extra}
            )

        # 3. Overall too bright — also background-only so face doesn't skew it
        if (ambient_bright_ratio > config.LIGHTING_HIGH_INTENSITY_RATIO or
                bg_brightness > config.LIGHTING_BRIGHT_THRESHOLD):
            return DetectionResult(
                module_name="lighting",
                is_ok=False,
                warning_message=(
                    f"Room is too bright (ambient brightness: {bg_brightness:.0f}/255, "
                    f"{ambient_bright_ratio * 100:.0f}% bright pixels). "
                    "Reduce glare or close blinds to prevent eye discomfort."
                ),
                confidence=max(
                    ambient_bright_ratio / config.LIGHTING_HIGH_INTENSITY_RATIO,
                    (bg_brightness - config.LIGHTING_BRIGHT_THRESHOLD) / (255 - config.LIGHTING_BRIGHT_THRESHOLD)
                ),
                extra={"issue_type": "too_bright", **base_extra}
            )

        # All good
        return DetectionResult(
            module_name="lighting",
            is_ok=True,
            warning_message="",
            confidence=1.0,
            extra={"issue_type": "none", **base_extra}
        )

    # =========================================================================
    #   Confidence scoring
    # =========================================================================

    def _calculate_confidence(self, value: float, threshold: float,
                               upper_bound: float = None) -> float:
        """
        Normalize how far a value exceeds a threshold onto a 0.0–1.0 scale.
        """
        if upper_bound is None:
            upper_bound = threshold * 2
        normalized = (value - threshold) / (upper_bound - threshold)
        return min(1.0, max(0.0, normalized))
