"""Visual detection of Terraminer Gold reward-screen controls."""

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
    """Recognize reward actions without treating ordinary board regions as UI."""

    @staticmethod
    def detect(frame: np.ndarray) -> RewardMatch | None:
        if frame is None or frame.ndim != 3:
            raise ValueError("Reward-screen detection requires a BGR image frame.")

        height, width = frame.shape[:2]
        lower_start = round(height * 0.5)
        lower_screen = frame[lower_start:, :]
        if lower_screen.size == 0:
            return None
        hsv = cv2.cvtColor(lower_screen, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(
            hsv,
            np.array((14, 100, 150), dtype=np.uint8),
            np.array((40, 255, 255), dtype=np.uint8),
        )
        button_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))
        yellow_mask = cv2.morphologyEx(
            yellow_mask,
            cv2.MORPH_CLOSE,
            button_kernel,
        )
        count, _, stats, centroids = cv2.connectedComponentsWithStats(
            yellow_mask,
            connectivity=8,
        )
        minimum_area = height * width * 0.0015
        for label in range(1, count):
            x, y, box_width, box_height, area = stats[label]
            center_y = lower_start + centroids[label][1]
            aspect_ratio = box_width / max(box_height, 1)
            bottom_button = (
                area >= minimum_area
                and box_width >= width * 0.18
                and box_height >= height * 0.035
                and 1.8 <= aspect_ratio <= 8.0
                and center_y >= height * 0.78
                and y + lower_start + box_height >= height * 0.82
            )
            if bottom_button:
                bounds = (
                    int(x),
                    int(y + lower_start),
                    int(box_width),
                    int(box_height),
                )
                logger.debug(
                    "Reward screen matched Play Again button bounding box "
                    "(x=%d, y=%d, width=%d, height=%d).",
                    *bounds,
                )
                return RewardMatch(RewardMatchKind.PLAY_AGAIN, bounds)

        panel_mask = cv2.inRange(
            hsv,
            np.array((20, 140, 190), dtype=np.uint8),
            np.array((40, 255, 255), dtype=np.uint8),
        )
        panel_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))
        panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_CLOSE, panel_kernel)
        panel_count, _, panel_stats, panel_centroids = cv2.connectedComponentsWithStats(
            panel_mask,
            connectivity=8,
        )
        orange_mask = cv2.inRange(
            hsv,
            np.array((5, 140, 180), dtype=np.uint8),
            np.array((19, 255, 255), dtype=np.uint8),
        )
        orange_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, orange_kernel)
        orange_count, _, orange_stats, orange_centroids = cv2.connectedComponentsWithStats(
            orange_mask,
            connectivity=8,
        )
        for panel_label in range(1, panel_count):
            panel_x, panel_y, panel_width, panel_height, panel_area = panel_stats[panel_label]
            panel_center_x = panel_centroids[panel_label][0]
            panel_center_y = lower_start + panel_centroids[panel_label][1]
            panel_ratio = panel_width / max(panel_height, 1)
            if not (
                panel_area >= height * width * 0.004
                and panel_width >= width * 0.4
                and panel_height >= height * 0.05
                and 2.0 <= panel_ratio <= 9.0
                and width * 0.2 <= panel_center_x <= width * 0.8
                and height * 0.56 <= panel_center_y <= height * 0.76
                and panel_y + lower_start + panel_height >= height * 0.62
            ):
                continue

            for orange_label in range(1, orange_count):
                orange_x, orange_y, orange_width, orange_height, orange_area = orange_stats[orange_label]
                orange_center_x, orange_center_y_local = orange_centroids[orange_label]
                orange_center_y = lower_start + orange_center_y_local
                contained = (
                    panel_x <= orange_center_x <= panel_x + panel_width
                    and panel_y + lower_start <= orange_center_y
                    <= panel_y + lower_start + panel_height
                )
                button_sized = (
                    orange_area >= height * width * 0.001
                    and orange_width >= panel_width * 0.55
                    and orange_height >= panel_height * 0.2
                    and orange_height <= panel_height * 0.8
                )
                if contained and button_sized:
                    logger.debug(
                        "Reward screen matched Level Up panel bounding box "
                        "(x=%d, y=%d, width=%d, height=%d) with Watch Ad button "
                        "bounding box (x=%d, y=%d, width=%d, height=%d).",
                        panel_x,
                        panel_y + lower_start,
                        panel_width,
                        panel_height,
                        orange_x,
                        orange_y + lower_start,
                        orange_width,
                        orange_height,
                    )
                    return RewardMatch(RewardMatchKind.LEVEL_UP_PANEL)
        return None
