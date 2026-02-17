from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class DetectionResult:
    """
    The standard result struct every module must return.
    Nobody changes this without telling the whole team.
    """
    module_name: str          # lighting, gaze or posture
    is_ok: bool               # True = no problem, False = warning needed
    warning_message: str      # Human-readable, shown in UI/terminal
    confidence: float         # 0.0 to 1.0 — how sure the module is
    extra: Optional[dict] = None  # Module-specific extras (optional)

class BaseDetector(ABC):
    """
    Every module MUST inherit this and implement analyze().
    That's the only rule.
    """
    
    @abstractmethod
    def analyze(self, frame: np.ndarray) -> DetectionResult:
        """
        Takes a webcam frame, returns a DetectionResult.
        This is the only method the monitor will ever call.
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"