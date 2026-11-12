from dataclasses import dataclass, field
from typing import Any, Dict
import threading

@dataclass
class PerceptionState:
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)

class StateBus:
    """Thread-safe latest-state store."""
    def __init__(self):
        self._lock = threading.Lock()
        self._state: PerceptionState | None = None

    def publish(self, state: PerceptionState):
        with self._lock:
            self._state = state

    def latest(self) -> PerceptionState | None:
        with self._lock:
            return self._state
            