import logging
import threading
from queue import Empty, Queue
from typing import Optional, Protocol

import adbutils
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ScreenshotDevice(Protocol):
    def shell(
        self,
        command: str,
        *,
        encoding: None,
        timeout: float,
    ) -> bytes: ...


class FrameStream:
    """Continuously capture BGR screenshots from an Android device over ADB."""

    def __init__(
        self,
        max_queue: int = 1,
        max_size: int = 720,
        device: ScreenshotDevice | None = None,
    ):
        if max_size < 1:
            raise ValueError("Maximum frame size must be positive.")
        self.queue: Queue[np.ndarray] = Queue(maxsize=max_queue)
        self.max_size = max_size
        self.device = device
        self.running = False
        self._stop_event = threading.Event()
        self._reader_thread: threading.Thread | None = None

    def start(self):
        """Start the background ADB screenshot reader."""
        if self.running:
            return
        if self.device is None:
            self.device = adbutils.adb.device()
        self._stop_event.clear()
        self.running = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        logger.info("Started Android screen capture over ADB.")

    def stop(self):
        self.running = False
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)
            self._reader_thread = None

    def _reader_loop(self):
        first_frame = True
        while self.running:
            try:
                image_data = self.device.shell(
                    "screencap -p",
                    encoding=None,
                    timeout=5,
                )
                frame = cv2.imdecode(
                    np.frombuffer(image_data, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if frame is None:
                    raise RuntimeError("ADB returned an invalid PNG screenshot.")
                height, width = frame.shape[:2]
                largest_dimension = max(width, height)
                if largest_dimension > self.max_size:
                    scale = self.max_size / largest_dimension
                    frame = cv2.resize(
                        frame,
                        (round(width * scale), round(height * scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                if first_frame:
                    logger.info(
                        "Android screen capture is active (%dx%d).",
                        frame.shape[1],
                        frame.shape[0],
                    )
                    first_frame = False

                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except Empty:
                        pass
                self.queue.put(frame)
            except Exception:
                if not self.running:
                    break
                logger.exception("Failed to capture the Android screen; retrying shortly.")
                self._stop_event.wait(1)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Non-blocking fetch of the most recent screenshot."""
        try:
            return self.queue.get_nowait()
        except Empty:
            return None
