"""Lifecycle controller for solving a Terraminer Gold level."""

from enum import Enum
import logging
import random
import time

import cv2
import numpy as np

from game_automation.core.events import PerceptionState

from .agent import TerraminerGoldAgent
from .perception import BoardNotDetectedError, TerraminerGoldPerception

logger = logging.getLogger(__name__)

SETTLING_SECONDS = 3.0
REWARD_CONFIRMATION_FRAMES = 2
BOARD_CONFIRMATION_FRAMES = 3
REWARD_BUTTON_WAIT_SECONDS = 1.0


class ControllerState(Enum):
    WAITING_FOR_BOARD = "WaitingForBoard"
    PLANNING = "Planning"
    EXECUTING = "Executing"
    SETTLING = "Settling"
    VERIFYING = "Verifying"
    WAITING_FOR_COMPLETION = "WaitingForCompletion"
    REWARD_SCREEN = "RewardScreen"
    AWAITING_NEXT_LEVEL = "AwaitingNextLevel"
    FAILED = "Failed"
    STOPPED = "Stopped"


class TerraminerGoldController:
    """Coordinates board solving with the game's animation and reward flow."""

    def __init__(
        self,
        agent: TerraminerGoldAgent,
        perception: TerraminerGoldPerception,
        settling_seconds: float = SETTLING_SECONDS,
        continuous: bool = False,
    ):
        if settling_seconds < 0:
            raise ValueError("Settling duration must not be negative.")
        self.agent = agent
        self.perception = perception
        self.settling_seconds = settling_seconds
        self.continuous = continuous
        self.state = ControllerState.WAITING_FOR_BOARD
        self._settling_started_at: float | None = None
        self._last_detection_error: str | None = None
        self._reward_screen_frames = 0
        self._waiting_for_reward_to_clear = False
        self._candidate_board_signature: tuple | None = None
        self._candidate_board_frames = 0
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

        reward_match = self._detect_reward_screen(frame)
        if self._waiting_for_reward_to_clear:
            if reward_match is not None and reward_match[0] == "play_again":
                return
            self._waiting_for_reward_to_clear = False
        if reward_match is not None:
            self._reward_screen_frames += 1
        else:
            self._reward_screen_frames = 0
        if self._reward_screen_frames >= REWARD_CONFIRMATION_FRAMES:
            self._reset_board_candidate()
            if self.continuous:
                if self.state is not ControllerState.AWAITING_NEXT_LEVEL:
                    self._transition(ControllerState.REWARD_SCREEN)
                    if reward_match[0] == "play_again":
                        self._click_play_again(reward_match[1], frame.shape[:2])
                        self._waiting_for_reward_to_clear = True
                        time.sleep(REWARD_BUTTON_WAIT_SECONDS)
                        self._reset_for_next_level()
                    else:
                        self._transition(ControllerState.AWAITING_NEXT_LEVEL)
                        logger.info(
                            "Ad reward panel confirmed; not tapping the panel. "
                            "Finish any ad, open the next level, then press Enter "
                            "to resume board detection."
                        )
            else:
                self._transition(ControllerState.REWARD_SCREEN)
                logger.info(
                    "Reward screen confirmed; stopping without sending further taps."
                )
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

    def acknowledge_next_level(self) -> None:
        if self.state is not ControllerState.AWAITING_NEXT_LEVEL:
            raise RuntimeError(
                "Can only acknowledge the next level while awaiting it."
            )
        self._waiting_for_reward_to_clear = False
        self._reset_for_next_level()

    def _reset_for_next_level(self) -> None:
        self.agent.reset_for_next_level()
        self._reward_screen_frames = 0
        self._last_detection_error = None
        self._reset_board_candidate()
        self._transition(ControllerState.WAITING_FOR_BOARD)

    def _process_playable_board(self, frame: np.ndarray) -> None:
        try:
            state = self.perception.process(frame)
        except BoardNotDetectedError as error:
            self._reset_board_candidate()
            message = str(error)
            if message != self._last_detection_error:
                logger.info("Waiting for board detection: %s", message)
                self._last_detection_error = message
            return

        self._last_detection_error = None
        board = state.data["board"]
        signature = (
            board.size,
            board.start,
            board.goal,
            tuple(tuple(row) for row in board.tile_masks),
        )
        if signature == self._candidate_board_signature:
            self._candidate_board_frames += 1
        else:
            self._candidate_board_signature = signature
            self._candidate_board_frames = 1
        if self._candidate_board_frames < BOARD_CONFIRMATION_FRAMES:
            logger.debug(
                "Waiting for a stable board: %d/%d matching frames.",
                self._candidate_board_frames,
                BOARD_CONFIRMATION_FRAMES,
            )
            return

        self._reset_board_candidate()
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

    def _reset_board_candidate(self) -> None:
        self._candidate_board_signature = None
        self._candidate_board_frames = 0

    @staticmethod
    def _is_reward_screen(frame: np.ndarray) -> bool:
        return TerraminerGoldController._detect_reward_screen(frame) is not None

    @staticmethod
    def _detect_reward_screen(
        frame: np.ndarray,
    ) -> tuple[str, tuple[int, int, int, int] | None] | None:
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
                logger.debug(
                    "Reward screen matched Play Again button bounding box "
                    "(x=%d, y=%d, width=%d, height=%d).",
                    x,
                    y + lower_start,
                    box_width,
                    box_height,
                )
                return "play_again", (
                    int(x),
                    int(y + lower_start),
                    int(box_width),
                    int(box_height),
                )

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
                    return "level_up_panel", None
        return None

    def _click_play_again(
        self,
        bounds: tuple[int, int, int, int] | None,
        frame_shape: tuple[int, int],
    ) -> None:
        if bounds is None:
            raise RuntimeError("Play Again was detected without button bounds.")
        if self.agent.phone.resolution is None:
            raise RuntimeError("Device resolution is unavailable for reward-button tap.")

        x, y, width, height = bounds
        frame_height, frame_width = frame_shape
        device_width, device_height = self.agent.phone.resolution
        frame_x = random.randint(x, x + width - 1)
        frame_y = random.randint(y, y + height - 1)
        tap_x = round(frame_x * device_width / frame_width)
        tap_y = round(frame_y * device_height / frame_height)
        logger.info(
            "Tapping confirmed Play Again button at device position (%d, %d).",
            tap_x,
            tap_y,
        )
        self.agent.phone.tap(tap_x, tap_y)

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
