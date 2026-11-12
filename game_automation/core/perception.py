from abc import ABC, abstractmethod
import time
import numpy as np

from .events import PerceptionState, StateBus

class BasePerception(ABC):
    def __init__(self, bus: StateBus):
        self.bus = bus

    @abstractmethod
    def process(self, frame: np.ndarray) -> PerceptionState:
        ...

class NoopPerception(BasePerception):
    def process(self, frame: np.ndarray) -> PerceptionState:
        state = PerceptionState(
            timestamp=time.time(),
            data={"info": "noop", "frame_shape": frame.shape},
        )
        self.bus.publish(state)
        return state
        