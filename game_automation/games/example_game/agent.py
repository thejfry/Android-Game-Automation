from game_automation.core.agent import BaseAgent
from game_automation.core.device import MyPhone
from game_automation.core.events import PerceptionState, StateBus
from game_automation.core.tap_executor import TapExecutor

class ExampleGameAgent(BaseAgent):
    def __init__(self, phone: MyPhone, state_bus: StateBus, tick_hz: int = 10):
        super().__init__(phone, state_bus, tick_hz=tick_hz)
        self.tap_executor = TapExecutor(phone)

    def tick(self, state: PerceptionState):
        bbox = state.data.get("bbox")
        if bbox:
            x1, y1, x2, y2 = bbox
            self.tap_executor.tap_device_point(
                ((x1 + x2) // 2, (y1 + y2) // 2),
                ((x2 - x1) * 0.2, (y2 - y1) * 0.2),
                description="example-game target",
            )
            