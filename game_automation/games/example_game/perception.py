import numpy as np
import time

from game_automation.core.perception import BasePerception
from game_automation.core.events import PerceptionState

class ExampleGamePerception(BasePerception):
    def process(self, frame: np.ndarray) -> PerceptionState:
        h, w, _ = frame.shape
        # Dummy bbox in center
        cx, cy = w // 2, h // 2
        bbox = (cx - 50, cy - 50, cx + 50, cy + 50)

        state = PerceptionState(
            timestamp=time.time(),
            data={
                "info": "example_game",
                "bbox": bbox,
            },
        )
        self.bus.publish(state)
        return state
        