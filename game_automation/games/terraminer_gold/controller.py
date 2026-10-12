"""Lifecycle controller for solving a Terraminer Gold level."""

from enum import Enum
import logging
import time

import cv2
import numpy as np

from game_automation.core.events import PerceptionState

from .agent import TerraminerGoldAgent
from .perception import BoardNotDetectedError, TerraminerGoldPerception

logger = logging.getLogger(__name__)

SETTLING_SECONDS = 2.0
REWARD_CONFIRMATION_FRAMES = 2


class ControllerState(Enum):
    WAITING_FOR_BOARD = "WaitingForBoard"
    PLANNING = "Planning"
    EXECUTING = "Executing"
    SETTLING = "Settling"
    VERIFYING = "Verifying"
    WAITING_FOR_COMPLETION = "WaitingForCompletion"
    REWARD_SCREEN = "RewardScreen"
    FAILED = "Failed"
    STOPPED = "Stopped"


class TerraminerGoldController:
    """Coordinates board solving with the game's animation and reward flow."""

    def __init__(
        self,
        agent: TerraminerGoldAgent,
        perception: TerraminerGoldPerception,
        settling_seconds: float = SETTLING_SECONDS,
    ):
        if settling_seconds < 0:
            raise ValueError("Settling duration must not be negative.")
        self.agent = agent
        self.perception = perception
        self.settling_seconds = settling_seconds
        self.state = ControllerState.WAITING_FOR_BOARD
        self._settling_started_at: float | None = None
        self._last_detection_error: str | None = None
        self._reward_screen_frames = 0
        logger.info("Controller state: %s.", self.state.value)

    @property
    def finished(self) -> bool:
        return self.state in (
            ControllerState.REWARD_SCREEN,
            ControllerState.FAILED,
            ControllerState.STOPPED,
        )

    def process_frame(self, frame: np.ndarray) -> None:
        if self.finished:
            return

        if self._is_reward_screen(frame):
            self._reward_screen_frames += 1
        else:
            self._reward_screen_frames = 0
        if self._reward_screen_frames >= REWARD_CONFIRMATION_FRAMES:
            self._transition(ControllerState.REWARD_SCREEN)
            logger.info("Reward screen confirmed; stopping without sending further taps.")
            return
        if self._reward_screen_frames:
            return

        if self.state is ControllerState.WAITING_FOR_BOARD:
            self._process_playable_board(frame)
        elif self.state is ControllerState.SETTLING:
            self._advance_settling()
        elif self.state is ControllerState.VERIFYING:
            self._verify_board(frame)
        elif self.state is ControllerState.WAITING_FOR_COMPLETION:
            return

    def _process_playable_board(self, frame: np.ndarray) -> None:
        try:
            state = self.perception.process(frame)
        except BoardNotDetectedError as error:
            message = str(error)
            if message != self._last_detection_error:
                logger.info("Waiting for board detection: %s", message)
                self._last_detection_error = message
            return

        self._last_detection_error = None
        self._transition(ControllerState.PLANNING)
        try:
            self.agent.tick(
                state,
                before_execution=lambda: self._transition(ControllerState.EXECUTING),
            )
        except RuntimeError:
            self._transition(ControllerState.FAILED)
            raise
        if self.agent.dry_run:
            self._transition(ControllerState.STOPPED)
            return
        self._settling_started_at = time.monotonic()
        self._transition(ControllerState.SETTLING)

    def _advance_settling(self) -> None:
        if self._settling_started_at is None:
            raise RuntimeError("Settling state has no start time.")
        if time.monotonic() - self._settling_started_at >= self.settling_seconds:
            self._transition(ControllerState.VERIFYING)

    def _verify_board(self, frame: np.ndarray) -> None:
        try:
            state: PerceptionState = self.perception.process(frame)
        except BoardNotDetectedError as error:
            logger.info(
                "Board is not currently detectable during verification (%s); "
                "waiting for the game's completion screen.",
                error,
            )
            self._transition(ControllerState.WAITING_FOR_COMPLETION)
            return

        solution = self.agent._last_solution
        if solution is None:
            raise RuntimeError("Cannot verify completion because no route was executed.")
        if self.agent._is_path_connected(state.data["board"], solution):
            logger.info(
                "Commanded route geometry is visible; waiting for the game's "
                "completion screen."
            )
        else:
            logger.warning(
                "Route geometry is not yet verified. Suppressing further solve "
                "attempts while waiting for the game's completion screen."
            )
        self._transition(ControllerState.WAITING_FOR_COMPLETION)

    @staticmethod
    def _is_reward_screen(frame: np.ndarray) -> bool:
        if frame is None or frame.ndim != 3:
            raise ValueError("Reward-screen detection requires a BGR image frame.")

        height, width = frame.shape[:2]
        lower_start = round(height * 0.6)
        lower_screen = frame[lower_start:, :]
        if lower_screen.size == 0:
            return False
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
            _, y, box_width, box_height, area = stats[label]
            center_y = lower_start + centroids[label][1]
            aspect_ratio = box_width / max(box_height, 1)
            if (
                area >= minimum_area
                and box_width >= width * 0.18
                and box_height >= height * 0.035
                and 1.8 <= aspect_ratio <= 8.0
                and center_y >= height * 0.78
                and y + lower_start + box_height >= height * 0.82
            ):
                return True
        return False

    def _transition(self, state: ControllerState) -> None:
        if state is self.state:
            return
        previous_state = self.state
        self.state = state
        logger.info(
            "Controller state: %s -> %s.",
            previous_state.value,
            state.value,
        )
