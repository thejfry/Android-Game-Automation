import time
import threading
from typing import Optional

import cv2
import numpy as np

from .events import StateBus, PerceptionState
from .frame_stream import FrameStream

class HUD:
    def __init__(self, frame_stream: FrameStream, state_bus: StateBus, target_hz: int = 20):
        self.stream = frame_stream
        self.bus = state_bus
        self.period = 1.0 / target_hz
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            t0 = time.time()
            frame = self.stream.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            state: Optional[PerceptionState] = self.bus.latest()
            self._draw(frame, state)

            cv2.imshow("HUD", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stop()

            dt = time.time() - t0
            if dt < self.period:
                time.sleep(self.period - dt)

        cv2.destroyAllWindows()

    def _draw(self, frame: np.ndarray, state: Optional[PerceptionState]):
        h, w, _ = frame.shape
        cv2.putText(frame, "HUD ONLINE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if state is not None:
            cv2.putText(frame, f"t={state.timestamp:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            info = state.data.get("info", "")
            cv2.putText(frame, f"info: {info}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            bbox = state.data.get("bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                