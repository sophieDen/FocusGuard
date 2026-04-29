import time
from core.base_detector import DetectionResult

class AlertManager:
    """
    Receives results from all modules and decides when to
    send a warning. Prevents alert spam.
    """

    def __init__(self, cooldown_seconds: int = 20):
        self.cooldown = cooldown_seconds
        self._last_alerted: dict[str, float] = {}

    def process(self, result: DetectionResult) -> bool:
        """
        Returns True if a warning should be sent, False if still in cooldown or no issue.
        """
        if result.is_ok:
            return False

        now = time.time()
        last = self._last_alerted.get(result.module_name, 0)

        if now - last >= self.cooldown:
            self._last_alerted[result.module_name] = now
            self._display(result)
            return True

        return False
    
    def _display(self, result: DetectionResult):
        tag = result.module_name.upper()
        print(f"{tag} WARNING {result.warning_message} "
              f"(confidence: {result.confidence:.0%})")