import threading
import time

try:
    import msvcrt
except ImportError:
    msvcrt = None

from game_automation.core.agent import BaseAgent
from game_automation.core.events import PerceptionState, StateBus


class AtlasRacerAgent(BaseAgent):
    """Swipe horizontally on the device using the left and right arrow keys."""

    def __init__(self, phone, state_bus: StateBus, duration_ms: int = 300):
        super().__init__(phone, state_bus, tick_hz=1)
        if duration_ms <= 0:
            raise ValueError("Swipe duration must be a positive number of milliseconds.")
        self.duration_ms = duration_ms
        self._thread: threading.Thread | None = None

    def start(self):
        if msvcrt is None:
            raise RuntimeError("AtlasRacerAgent requires a Windows console for arrow-key input.")
        if self.phone.resolution is None:
            raise RuntimeError("Device resolution is unavailable; connect the device before starting the agent.")
        if self._thread and self._thread.is_alive():
            return
        self.running = True
        self._thread = threading.Thread(target=self._input_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=0.2)

    def tick(self, state: PerceptionState):
        return

    def _input_loop(self):
        width, height = self.phone.resolution
        left_x = int(width * 0.2)
        right_x = int(width * 0.8)
        center_y = height // 2
        print("AtlasRacerAgent ready. Use the left/right arrow keys to swipe (Ctrl+C to exit).")

        while self.running:
            if not msvcrt.kbhit():
                time.sleep(0.01)
                continue

            key = msvcrt.getwch()
            if key in ("\x00", "\xe0"):
                key = msvcrt.getwch()
                if key == "K":
                    self.phone.swipe((right_x, center_y), (left_x, center_y), self.duration_ms)
                elif key == "M":
                    self.phone.swipe((left_x, center_y), (right_x, center_y), self.duration_ms)
