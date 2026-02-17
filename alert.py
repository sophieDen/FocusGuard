import time
from .base_detector import DetectionResult

class AlertManager:
    """
    Receives results from all modules and decides when to
    actually surface a warning. Prevents alert spam.
    """

    def __init__(self, cooldown_seconds: int = 20):
        self.cooldown = cooldown_seconds
        self._last_alerted: dict[str, float] = {}

    def process(self, result: DetectionResult) -> bool:
        """
        Returns True if a warning was surfaced (i.e. the user
        should be notified), False if still in cooldown or no issue.
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