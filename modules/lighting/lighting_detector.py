"""
Analyzes ambient lighting conditions using histogram analysis.
Detects three problematic scenarios:
1. Overall too dark (eye strain, poor visibility)
2. Overall too bright (glare, discomfort)
3. High contrast (bright screen in dark room - worst for eyes)
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
    """
    
    def __init__(self):
        """Initialize the lighting detector."""
        self.frame_count = 0
        
    def analyze(self, frame: np.ndarray) -> DetectionResult:
        """
        Analyze lighting conditions from a webcam frame.
        
        Args:
            frame: BGR image from webcam
            
        Returns:
            DetectionResult indicating lighting quality
        """
        self.frame_count += 1
        
        # Convert to grayscale for luminance analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram (256 bins for 0-255 intensity range)
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
        histogram = histogram.flatten() / histogram.sum()  # Normalize to probabilities
        
        # Overall brightness metrics
        mean_brightness = float(np.mean(gray))
        median_brightness = float(np.median(gray))
        std_brightness = float(np.std(gray))
        
        # Histogram-based analysis
        dark_pixels_ratio = np.sum(histogram[:config.LIGHTING_DARK_PIXEL_THRESHOLD])
        bright_pixels_ratio = np.sum(histogram[config.LIGHTING_BRIGHT_PIXEL_THRESHOLD:])
        
        # Contrast analysis: compare center (face/screen) vs periphery (ambient)
        h, w = gray.shape
        center_size = int(min(h, w) * config.LIGHTING_CENTER_RATIO)
        center_y, center_x = h // 2, w // 2
        
        center_region = gray[
            center_y - center_size//2 : center_y + center_size//2,
            center_x - center_size//2 : center_x + center_size//2
        ]
        
        # Create periphery mask (everything except center)
        periphery_mask = np.ones_like(gray, dtype=bool)
        periphery_mask[
            center_y - center_size//2 : center_y + center_size//2,
            center_x - center_size//2 : center_x + center_size//2
        ] = False
        periphery = gray[periphery_mask]
        
        center_brightness = float(np.mean(center_region))
        periphery_brightness = float(np.mean(periphery))
        contrast_difference = center_brightness - periphery_brightness
        
        # Decision logic - priority order matters
        # 1. High contrast (most harmful) - bright screen in dark room
        if (contrast_difference > config.LIGHTING_CONTRAST_THRESHOLD and 
            periphery_brightness < config.LIGHTING_DARK_THRESHOLD):
            return DetectionResult(
                module_name="lighting",
                is_ok=False,
                warning_message=(
                    f"High contrast detected: Your screen is bright ({center_brightness:.0f}) "
                    f"but the room is dark ({periphery_brightness:.0f}). "
                    "This causes severe eye strain. Turn on ambient lighting!"
                ),
                confidence=self._calculate_confidence(
                    contrast_difference, 
                    config.LIGHTING_CONTRAST_THRESHOLD, 
                    upper_bound=150
                ),
                extra={
                    "issue_type": "high_contrast",
                    "mean_brightness": mean_brightness,
                    "median_brightness": median_brightness,
                    "std_brightness": std_brightness,
                    "center_brightness": center_brightness,
                    "periphery_brightness": periphery_brightness,
                    "contrast_difference": contrast_difference,
                    "dark_pixels_ratio": dark_pixels_ratio,
                    "bright_pixels_ratio": bright_pixels_ratio,
                    "histogram": histogram.tolist()
                }
            )
        
        # 2. Overall too dark
        if (dark_pixels_ratio > config.LIGHTING_LOW_INTENSITY_RATIO or 
            mean_brightness < config.LIGHTING_DARK_THRESHOLD):
            return DetectionResult(
                module_name="lighting",
                is_ok=False,
                warning_message=(
                    f"Room is too dark (brightness: {mean_brightness:.0f}/255, "
                    f"{dark_pixels_ratio*100:.0f}% dark pixels). "
                    "Increase ambient lighting to reduce eye strain."
                ),
                confidence=max(
                    dark_pixels_ratio / config.LIGHTING_LOW_INTENSITY_RATIO,
                    (config.LIGHTING_DARK_THRESHOLD - mean_brightness) / config.LIGHTING_DARK_THRESHOLD
                ),
                extra={
                    "issue_type": "too_dark",
                    "mean_brightness": mean_brightness,
                    "median_brightness": median_brightness,
                    "std_brightness": std_brightness,
                    "dark_pixels_ratio": dark_pixels_ratio,
                    "bright_pixels_ratio": bright_pixels_ratio,
                    "histogram": histogram.tolist()
                }
            )
        
        # 3. Overall too bright
        if (bright_pixels_ratio > config.LIGHTING_HIGH_INTENSITY_RATIO or 
            mean_brightness > config.LIGHTING_BRIGHT_THRESHOLD):
            return DetectionResult(
                module_name="lighting",
                is_ok=False,
                warning_message=(
                    f"Room is too bright (brightness: {mean_brightness:.0f}/255, "
                    f"{bright_pixels_ratio*100:.0f}% bright pixels). "
                    "Reduce glare or close blinds to prevent eye discomfort."
                ),
                confidence=max(
                    bright_pixels_ratio / config.LIGHTING_HIGH_INTENSITY_RATIO,
                    (mean_brightness - config.LIGHTING_BRIGHT_THRESHOLD) / (255 - config.LIGHTING_BRIGHT_THRESHOLD)
                ),
                extra={
                    "issue_type": "too_bright",
                    "mean_brightness": mean_brightness,
                    "median_brightness": median_brightness,
                    "std_brightness": std_brightness,
                    "dark_pixels_ratio": dark_pixels_ratio,
                    "bright_pixels_ratio": bright_pixels_ratio,
                    "histogram": histogram.tolist()
                }
            )
        
        # All good!
        return DetectionResult(
            module_name="lighting",
            is_ok=True,
            warning_message="",
            confidence=1.0,
            extra={
                "issue_type": "none",
                "mean_brightness": mean_brightness,
                "median_brightness": median_brightness,
                "std_brightness": std_brightness,
                "center_brightness": center_brightness,
                "periphery_brightness": periphery_brightness,
                "contrast_difference": contrast_difference,
                "dark_pixels_ratio": dark_pixels_ratio,
                "bright_pixels_ratio": bright_pixels_ratio,
                "histogram": histogram.tolist()
            }
        )
    
    def _calculate_confidence(self, value: float, threshold: float, 
                             upper_bound: float = None) -> float:
        """
        Calculate confidence score (0.0 to 1.0) based on how far a value
        exceeds a threshold.
        
        Args:
            value: The measured value
            threshold: The threshold that was exceeded
            upper_bound: Optional upper limit for normalization
            
        Returns:
            Confidence between 0.0 and 1.0
        """
        if upper_bound is None:
            upper_bound = threshold * 2
            
        # How far past threshold (0 = at threshold, 1 = at upper_bound)
        normalized = (value - threshold) / (upper_bound - threshold)
        return min(1.0, max(0.0, normalized))