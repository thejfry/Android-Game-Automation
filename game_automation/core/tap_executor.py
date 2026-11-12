"""Shared coordinate mapping and configurable, bounded tap randomization."""

import logging
import random
import time

from .device import MyPhone

logger = logging.getLogger(__name__)


class TapExecutor:
    """Send taps with separately configurable position and timing randomness."""

    def __init__(
        self,
        phone: MyPhone,
        tap_log_level: int = logging.INFO,
        timing_jitter_fraction: float = 0.2,
    ):
        if timing_jitter_fraction < 0:
            raise ValueError("Timing jitter fraction must not be negative.")
        self.phone = phone
        self.tap_log_level = tap_log_level
        self.timing_jitter_fraction = timing_jitter_fraction

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

    def tap_device_point(
        self,
        point: tuple[int, int],
        jitter_extent: tuple[float, float] = (0.0, 0.0),
        description: str = "device point",
    ) -> tuple[int, int]:
        """Tap a device-coordinate point with bounded spatial jitter."""
        device_width, device_height = self._device_resolution()
        jitter_x, jitter_y = jitter_extent
        if jitter_x < 0 or jitter_y < 0:
            raise ValueError("Tap jitter extents must not be negative.")
        tap_x = round(point[0] + random.uniform(-jitter_x, jitter_x))
        tap_y = round(point[1] + random.uniform(-jitter_y, jitter_y))
        tap_x = min(device_width - 1, max(0, tap_x))
        tap_y = min(device_height - 1, max(0, tap_y))
        logger.log(
            self.tap_log_level,
            "Tapping %s at device position (%d, %d).",
            description,
            tap_x,
            tap_y,
        )
        self.phone.tap(tap_x, tap_y)
        return tap_x, tap_y

    def wait_between_taps(
        self,
        interval_seconds: float,
        timing_jitter_fraction: float | None = None,
    ) -> float:
        """Sleep for an interval with optional independent timing jitter."""
        duration = self.randomized_interval(interval_seconds, timing_jitter_fraction)
        time.sleep(duration)
        return duration

    def randomized_interval(
        self,
        interval_seconds: float,
        timing_jitter_fraction: float | None = None,
    ) -> float:
        """Return a randomized interval without sleeping."""
        if interval_seconds < 0:
            raise ValueError("Tap interval must not be negative.")
        jitter_fraction = (
            self.timing_jitter_fraction
            if timing_jitter_fraction is None
            else timing_jitter_fraction
        )
        if jitter_fraction < 0:
            raise ValueError("Timing jitter fraction must not be negative.")
        jitter = interval_seconds * jitter_fraction
        duration = random.uniform(
            max(0.0, interval_seconds - jitter),
            interval_seconds + jitter,
        )
        return duration

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
