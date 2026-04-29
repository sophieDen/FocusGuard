from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class DetectionResult:
    """
    The standard result struct every module must return.
    """
    module_name: str
    is_ok: bool #True means no problem, False - warning needed
    warning_message: str
    confidence: float # how sure the module is 0.0 to 1.0
    extra: Optional[dict] = None  # optional

class BaseDetector(ABC):
    """
    Every module inherits this and implement analyze().
    """
    
    @abstractmethod
    def analyze(self, frame: np.ndarray) -> DetectionResult:
        """
        Takes a webcam frame, returns a DetectionResult.
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"