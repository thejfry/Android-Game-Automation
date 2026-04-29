from abc import ABC, abstractmethod
import time
import numpy as np

from .device import MyPhone
from .events import StateBus, PerceptionState

class BaseAgent(ABC):
    def __init__(self, phone: MyPhone, state_bus: StateBus, tick_hz: int = 10):
        self.phone = phone
        self.bus = state_bus
        self.period = 1.0 / tick_hz
        self.running = False

    def start(self):
        self.running = True
        self._loop()

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            t0 = time.time()
            state = self.bus.latest()
            if state is not None:
                self.tick(state)
            dt = time.time() - t0
            if dt < self.period:
                time.sleep(self.period - dt)

    @abstractmethod
    def tick(self, state: PerceptionState):
        ...
