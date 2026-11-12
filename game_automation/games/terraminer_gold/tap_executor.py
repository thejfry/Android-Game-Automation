"""Coordinate mapping and bounded, jittered Android taps."""

import logging
import random

from game_automation.core.device import MyPhone

logger = logging.getLogger(__name__)


class TapExecutor:
    def __init__(
        self,
        phone: MyPhone,
        tap_log_level: int = logging.INFO,
    ):
        self.phone = phone
        self.tap_log_level = tap_log_level

    def tap_point(
        self,
        point: tuple[float, float],
        frame_size: tuple[int, int],
        jitter_extent: tuple[float, float] = (0.0, 0.0),
        description: str = "screen point",
    ) -> tuple[int, int]:
        frame_width, frame_height = self._validate_dimensions(frame_size)
        device_width, device_height = self._device_resolution()
        jitter_x, jitter_y = jitter_extent
        if jitter_x < 0 or jitter_y < 0:
            raise ValueError("Tap jitter extents must not be negative.")
        frame_x = point[0] + random.uniform(-jitter_x, jitter_x)
        frame_y = point[1] + random.uniform(-jitter_y, jitter_y)
        return self._tap_frame_coordinate(
            frame_x,
            frame_y,
            frame_width,
            frame_height,
            device_width,
            device_height,
            description,
        )

    def tap_bounds(
        self,
        bounds: tuple[int, int, int, int],
        frame_size: tuple[int, int],
        description: str,
    ) -> tuple[int, int]:
        frame_width, frame_height = self._validate_dimensions(frame_size)
        device_width, device_height = self._device_resolution()
        x, y, width, height = bounds
        if width <= 0 or height <= 0:
            raise ValueError("Tap bounds must have positive width and height.")
        if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
            raise ValueError("Tap bounds must lie within the frame.")
        frame_x = random.randint(x, x + width - 1)
        frame_y = random.randint(y, y + height - 1)
        return self._tap_frame_coordinate(
            frame_x,
            frame_y,
            frame_width,
            frame_height,
            device_width,
            device_height,
            description,
        )

    @staticmethod
    def _validate_dimensions(frame_size: tuple[int, int]) -> tuple[int, int]:
        frame_width, frame_height = frame_size
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame dimensions must be positive.")
        return frame_width, frame_height

    def _device_resolution(self) -> tuple[int, int]:
        if self.phone.resolution is None:
            raise RuntimeError("Device resolution is unavailable.")
        device_width, device_height = self.phone.resolution
        if device_width <= 0 or device_height <= 0:
            raise RuntimeError("Device resolution must be positive.")
        return device_width, device_height

    def _tap_frame_coordinate(
        self,
        frame_x: float,
        frame_y: float,
        frame_width: int,
        frame_height: int,
        device_width: int,
        device_height: int,
        description: str,
    ) -> tuple[int, int]:
        tap_x = min(device_width - 1, max(0, round(frame_x * device_width / frame_width)))
        tap_y = min(device_height - 1, max(0, round(frame_y * device_height / frame_height)))
        logger.log(
            self.tap_log_level,
            "Tapping %s at device position (%d, %d).",
            description,
            tap_x,
            tap_y,
        )
        self.phone.tap(tap_x, tap_y)
        return tap_x, tap_y
