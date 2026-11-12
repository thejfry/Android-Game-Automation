"""Visual detection of Terraminer Rock completion and level-up controls."""

from dataclasses import dataclass
from enum import Enum
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RewardMatchKind(Enum):
    PLAY_AGAIN = "play_again"
    LEVEL_UP_PANEL = "level_up_panel"


@dataclass(frozen=True)
class RewardMatch:
    kind: RewardMatchKind
    button_bounds: tuple[int, int, int, int] | None = None


class RewardScreenDetector:
    """Find the standard replay button or the separate level-up prompt."""

    @staticmethod
    def detect(frame: np.ndarray) -> RewardMatch | None:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Reward-screen detection requires a BGR image frame.")

        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(
            hsv,
            np.array((38, 150, 120), dtype=np.uint8),
            np.array((90, 255, 255), dtype=np.uint8),
        )
        green_components = _components(green_mask)
        for x, y, box_width, box_height, area in green_components:
            if (
                area >= height * width * 0.008
                and box_width >= width * 0.35
                and box_height >= height * 0.055
                and height * 0.42 <= y + box_height / 2 <= height * 0.8
            ):
                orange_mask = cv2.inRange(
                    hsv,
                    np.array((5, 135, 130), dtype=np.uint8),
                    np.array((22, 255, 255), dtype=np.uint8),
                )
                for orange_x, orange_y, orange_width, orange_height, orange_area in _components(
                    orange_mask
                ):
                    center_x = orange_x + orange_width / 2
                    center_y = orange_y + orange_height / 2
                    if (
                        x <= center_x <= x + box_width
                        and y <= center_y <= y + box_height
                        and orange_area >= height * width * 0.001
                        and orange_width >= box_width * 0.4
                        and orange_height >= box_height * 0.2
                        and orange_height <= box_height * 0.85
                    ):
                        logger.debug(
                            "Detected Rock level-up panel and nested Play Level button."
                        )
                        return RewardMatch(RewardMatchKind.LEVEL_UP_PANEL)

        yellow_mask = cv2.inRange(
            hsv,
            np.array((18, 150, 150), dtype=np.uint8),
            np.array((40, 255, 255), dtype=np.uint8),
        )
        for x, y, box_width, box_height, area in _components(yellow_mask):
            center_y = y + box_height / 2
            aspect_ratio = box_width / max(box_height, 1)
            if (
                area >= height * width * 0.001
                and box_width >= width * 0.2
                and box_height >= height * 0.035
                and 1.7 <= aspect_ratio <= 5.5
                and height * 0.48 <= center_y <= height * 0.78
            ):
                bounds = (x, y, box_width, box_height)
                logger.debug(
                    "Detected Rock Play Again button bounding box "
                    "(x=%d, y=%d, width=%d, height=%d).",
                    *bounds,
                )
                return RewardMatch(RewardMatchKind.PLAY_AGAIN, bounds)
        return None


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    return [
        tuple(int(value) for value in stats[label])
        for label in range(1, count)
    ]
