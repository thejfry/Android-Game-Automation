from game_automation.core.agent import BaseAgent
from game_automation.core.events import PerceptionState

class ExampleGameAgent(BaseAgent):
    def tick(self, state: PerceptionState):
        bbox = state.data.get("bbox")
        if bbox:
            # Example: tap center of bbox every tick
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            self.phone.tap(cx, cy)
            