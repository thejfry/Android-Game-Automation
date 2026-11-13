"""Lifecycle orchestration for Terraminer Gold."""

from collections.abc import Callable
import logging
import time

import numpy as np

from .agent import SolveResult, TerraminerGoldAgent
from .perception import BoardNotDetectedError, BoardObservation, TerraminerGoldPerception
from .reward_detection import RewardScreenDetector
from .solver import MazeSolution
from game_automation.core.tap_executor import TapExecutor
from game_automation.games.terraminer_common.controller import (
    ControllerState,
    TerraminerController,
)

logger = logging.getLogger(__name__)

SETTLING_SECONDS = 3.0
REWARD_CONFIRMATION_FRAMES = 2
BOARD_CONFIRMATION_FRAMES = 3
REWARD_BUTTON_WAIT_SECONDS = 1.0


class TerraminerGoldController(TerraminerController):
    """Coordinates perception, solving, verification, and level progression."""

    def __init__(
        self,
        agent: TerraminerGoldAgent,
        perception: TerraminerGoldPerception,
        tap_executor: TapExecutor,
        settling_seconds: float = SETTLING_SECONDS,
        continuous: bool = False,
        reward_button_wait_seconds: float = REWARD_BUTTON_WAIT_SECONDS,
        level_progress_callback: Callable[[int, str], None] | None = None,
    ):
        if settling_seconds < 0:
            raise ValueError("Settling duration must not be negative.")
        super().__init__(
            agent,
            perception,
            tap_executor,
            continuous,
            reward_button_wait_seconds,
            board_confirmation_frames=BOARD_CONFIRMATION_FRAMES,
            reward_confirmation_frames=REWARD_CONFIRMATION_FRAMES,
            level_progress_callback=level_progress_callback,
        )
        self.settling_seconds = settling_seconds
        self._settling_started_at: float | None = None
        self._last_detection_error: str | None = None
        self._active_solution: MazeSolution | None = None

    def process_frame(self, frame: np.ndarray) -> None:
        if self.finished:
            return
        if self.state is ControllerState.ADVANCING_LEVEL:
            self._advance_level_transition()
            return

        reward_match = RewardScreenDetector.detect(frame)
        if self._process_reward_frame(reward_match, frame.shape[:2]):
            return

        if self.state is ControllerState.WAITING_FOR_BOARD:
            self._process_playable_board(frame)
        elif self.state is ControllerState.SETTLING:
            self._advance_settling()
        elif self.state is ControllerState.VERIFYING:
            self._verify_board(frame)
        elif self.state is ControllerState.WAITING_FOR_COMPLETION:
            return

    def _reset_for_next_level(self) -> None:
        super()._reset_for_next_level()
        self._last_detection_error = None
        self._active_solution = None

    def _on_level_up_screen(self) -> None:
        logger.info(
            "Ad reward panel confirmed; not tapping the panel. Finish any ad, "
            "open the next puzzle, then press Enter to resume board detection."
        )

    def _process_playable_board(self, frame: np.ndarray) -> None:
        try:
            observation = self.perception.process(frame)
        except BoardNotDetectedError as error:
            self._reset_board_candidate()
            message = str(error)
            if message != self._last_detection_error:
                logger.info("Waiting for board detection: %s", message)
                self._last_detection_error = message
            return

        self._last_detection_error = None
        if not self._confirm_board(observation):
            return

        self._reset_board_candidate()
        self._begin_level_progress()
        self._transition(ControllerState.PLANNING)
        try:
            result = self.agent.solve(
                observation,
                before_execution=lambda: self._transition(ControllerState.EXECUTING),
            )
        except RuntimeError:
            self._transition(ControllerState.FAILED)
            raise
        if result is None:
            self._transition(ControllerState.STOPPED)
            return
        self._active_solution = result.solution
        if not result.executed:
            self._transition(ControllerState.STOPPED)
            return
        self._settling_started_at = time.monotonic()
        self._transition(ControllerState.SETTLING)

    def _confirm_board(self, observation: BoardObservation) -> bool:
        board = observation.board
        signature = (
            board.size,
            board.start,
            board.goal,
            board.tile_masks,
        )
        return self._confirm_candidate(signature)

    def _advance_settling(self) -> None:
        if self._settling_started_at is None:
            raise RuntimeError("Settling state has no start time.")
        if time.monotonic() - self._settling_started_at >= self.settling_seconds:
            self._transition(ControllerState.VERIFYING)

    def _verify_board(self, frame: np.ndarray) -> None:
        try:
            observation = self.perception.process(frame)
        except BoardNotDetectedError as error:
            logger.info(
                "Board is not currently detectable during verification (%s); "
                "waiting for the game's completion screen.",
                error,
            )
            self._transition(ControllerState.WAITING_FOR_COMPLETION)
            return

        solution = self._active_solution
        if solution is None:
            raise RuntimeError("Cannot verify completion because no route was executed.")
        if self.agent.verify_route(observation.board, solution):
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
