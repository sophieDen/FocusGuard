"""
Analyzes ambient lighting conditions 
"""

import numpy as np
import cv2
import sys
import os
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_detector import BaseDetector, DetectionResult
import config


class LightingDetector(BaseDetector):

    HISTORY_LEN = 10  # n frames to smooth before deciding
    L_DARK_THRESHOLD = 25 # if L* below - too dark
    L_BRIGHT_THRESHOLD = 78 # above - too bright
    L_CONTRAST_THRESHOLD = 32
    L_DARK_PIXEL_FRACTION = 0.65  # fraction of bg pixels with L* < L_DARK_THRESHOLD
    L_BRIGHT_PIXEL_FRACTION = 0.45  # fraction of bg pixels with L* > L_BRIGHT_THRESHOLD
    CALIBRATION_FRAMES = 30
    B_COOL_DELTA = 6 # b* drop from baseline that triggers cool-light alert
    B_LABEL_DELTA = 4 # b* delta for the warm/cool display label

    def __init__(self):
        self.frame_count = 0
        self._history = deque(maxlen=self.HISTORY_LEN)
        self._calib_buf = []
        self._baseline_b = None
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self._face_cascade = cv2.CascadeClassifier(cascade_path)

    #   Face detection
    def _get_face_region(self, gray: np.ndarray):
        """Returns (x,y,w,h) of the face"""
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    #   Perceptual region analysis
    def _analyze_regions(self, frame: np.ndarray, gray: np.ndarray):
        """
        Converts the frame to CIELAB and extracts perceptual metrics for the
        face or geometric centre region and the bg separately.

        Returns:
            subject_L mean L* of subject region (0–100)
            ambient_L mean L* of bg (0–100)
            ambient_b mean b* of bg (−128 to +127)
            dark_fraction fraction of bg pixels with L* < L_DARK_THRESHOLD
            bright_fraction fraction of bg pixels with L* > L_BRIGHT_THRESHOLD
            face_detected bool
            region_label "face" or "center"
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        L_ch = lab[:, :, 0] # luminance
        b_ch = lab[:, :, 2] # yellow-blue axis

        face = self._get_face_region(gray)

        if face is not None:
            fx, fy, fw, fh = face
            subject_mask = np.zeros(gray.shape, dtype=bool)
            subject_mask[fy:fy + fh, fx:fx + fw] = True
            face_detected = True
            region_label = "face"
        else:
            h, w = gray.shape
            sz = int(min(h, w) * config.LIGHTING_CENTER_RATIO)
            cy, cx = h // 2, w // 2
            half = sz // 2
            subject_mask = np.zeros(gray.shape, dtype=bool)
            subject_mask[cy - half:cy + half, cx - half:cx + half] = True
            face_detected = False
            region_label = "center"

        bg_mask = ~subject_mask

        # Convert to standard perceptual ranges
        subject_L = float(np.mean(L_ch[subject_mask])) / 2.55
        ambient_L = float(np.mean(L_ch[bg_mask])) / 2.55
        ambient_b = float(np.mean(b_ch[bg_mask])) - 128.0

        # Pixel fraction metrics on background only
        bg_L = L_ch[bg_mask]
        dark_l_raw = self.L_DARK_THRESHOLD * 2.55
        bright_l_raw = self.L_BRIGHT_THRESHOLD * 2.55
        dark_fraction = float(np.sum(bg_L < dark_l_raw) / len(bg_L))
        bright_fraction = float(np.sum(bg_L > bright_l_raw) / len(bg_L))

        return subject_L, ambient_L, ambient_b, dark_fraction, bright_fraction, face_detected, region_label

    def analyze(self, frame: np.ndarray) -> DetectionResult:
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        (subject_L, ambient_L, ambient_b, dark_fraction, bright_fraction, face_detected, region_label) = self._analyze_regions(frame, gray)

        contrast_L = subject_L - ambient_L

        self._history.append({
            "subject_L": subject_L,
            "ambient_L": ambient_L,
            "ambient_b": ambient_b,
            "contrast_L": contrast_L,
            "dark_fraction": dark_fraction,
            "bright_fraction": bright_fraction,
        })

        # Calibration
        # Collecting raw b* values for the first CALIBRATION_FRAMES frames to establish baseline
        if self._baseline_b is None:
            self._calib_buf.append(ambient_b)
            if len(self._calib_buf) >= self.CALIBRATION_FRAMES:
                self._baseline_b = float(np.mean(self._calib_buf))
            return DetectionResult(
                module_name="lighting", is_ok=True,
                warning_message="",  confidence=1.0,
                extra={
                    "issue_type": "calibrating",
                    "calib_progress": f"{len(self._calib_buf)}/{self.CALIBRATION_FRAMES}",
                }
            )

        # Waiting for the rolling window to fill then decide
        if len(self._history) < 3:
            return DetectionResult(
                module_name="lighting", is_ok=True,
                warning_message="", confidence=1.0
            )

        # Smoothed metrics
        s_subject_L = float(np.mean([h["subject_L"] for h in self._history]))
        s_ambient_L = float(np.mean([h["ambient_L"] for h in self._history]))
        s_ambient_b = float(np.mean([h["ambient_b"] for h in self._history]))
        s_contrast_L = float(np.mean([h["contrast_L"] for h in self._history]))
        s_dark_frac = float(np.mean([h["dark_fraction"] for h in self._history]))
        s_bright_frac = float(np.mean([h["bright_fraction"] for h in self._history]))

        # Colour temperature relative to calibrated baseline
        b_delta = s_ambient_b - self._baseline_b
        color_temp = ("cool"    if b_delta < -self.B_LABEL_DELTA
                      else "warm" if b_delta >  self.B_LABEL_DELTA
                      else "neutral")

        base_extra = {
            "face_detected": face_detected,
            "region_used": region_label,
            "subject_L": round(s_subject_L,  1),
            "ambient_L": round(s_ambient_L,  1),
            "ambient_b": round(s_ambient_b,  1),
            "b_delta": round(b_delta,       1),
            "baseline_b": round(self._baseline_b, 1),
            "contrast_L": round(s_contrast_L, 1),
            "dark_fraction": round(s_dark_frac,  3),
            "bright_fraction": round(s_bright_frac, 3),
            "color_temp": color_temp,
        }

        # High contrast 
        if s_contrast_L > self.L_CONTRAST_THRESHOLD and s_ambient_L < self.L_DARK_THRESHOLD:
            subject_label = "face" if face_detected else "screen area"
            msg = (
                f"Your {subject_label} is brightly lit, but the room is dark"
                "This causes severe eye strain. Turn on ambient lighting!"
            )
            return DetectionResult(
                module_name="lighting", is_ok=False,
                warning_message=msg,
                confidence=self._confidence(s_contrast_L, self.L_CONTRAST_THRESHOLD, 70),
                extra={"issue_type": "high_contrast", **base_extra}
            )

        # Too dark
        if s_dark_frac > self.L_DARK_PIXEL_FRACTION or s_ambient_L < self.L_DARK_THRESHOLD:
            msg = (
                f"Room is too dark"
                f"{s_dark_frac * 100:.0f}% of background underlit). "
                "Increase ambient lighting to reduce eye strain."
            )
            return DetectionResult(
                module_name="lighting", is_ok=False,
                warning_message=msg,
                confidence=max(
                    s_dark_frac / self.L_DARK_PIXEL_FRACTION,
                    (self.L_DARK_THRESHOLD - s_ambient_L) / self.L_DARK_THRESHOLD
                ),
                extra={"issue_type": "too_dark", **base_extra}
            )

        # Too bright
        if s_bright_frac > self.L_BRIGHT_PIXEL_FRACTION or s_ambient_L > self.L_BRIGHT_THRESHOLD:
            msg = (
                f"Room is too bright"
                f"{s_bright_frac * 100:.0f}% of background overlit). "
                "Reduce glare or close blinds."
            )
            return DetectionResult(
                module_name="lighting", is_ok=False,
                warning_message=msg,
                confidence=max(
                    s_bright_frac / self.L_BRIGHT_PIXEL_FRACTION,
                    (s_ambient_L - self.L_BRIGHT_THRESHOLD) / (100 - self.L_BRIGHT_THRESHOLD)
                ),
                extra={"issue_type": "too_bright", **base_extra}
            )

        # Cool/blue-shifted light
        # triggered when b* drops B_COOL_DELTA units below the calibrated baseline
        # compensates camera's auto-white-balance offset
        if b_delta < -self.B_COOL_DELTA:
            msg = (
                f"Cool blue-shifted lighting detected "
                "Blue-enriched light suppresses melatonin and can cause eye strain "
                "Consider switching to a warmer light."
            )
            return DetectionResult(
                module_name="lighting", is_ok=False,
                warning_message=msg,
                confidence=self._confidence(abs(b_delta), self.B_COOL_DELTA, 20),
                extra={"issue_type": "cool_light", **base_extra}
            )

        return DetectionResult(
            module_name="lighting", is_ok=True,
            warning_message="", confidence=1.0,
            extra={"issue_type": "none", **base_extra}
        )

    #   Confidence calc
    def _confidence(self, value: float, threshold: float, upper_bound: float) -> float:
        """Normalises how far a value exceeds threshold onto 0.0–1.0."""
        return min(1.0, max(0.0, (value - threshold) / (upper_bound - threshold)))
