"""Lifecycle orchestration for Terraminer Gold."""

from collections.abc import Callable
import logging
import time

import numpy as np

from .agent import TerraminerGoldAgent
from .perception import BoardNotDetectedError, BoardObservation, TerraminerGoldPerception
from .reward_detection import RewardScreenDetector
from .solver import MazeSolution
from game_automation.core.tap_executor import TapExecutor
from game_automation.games.terraminer_common.controller import (
    ControllerState,
    TerraminerController,
)

logger = logging.getLogger(__name__)

SETTLING_SECONDS = 0.5
REWARD_CONFIRMATION_FRAMES = 2
BOARD_CONFIRMATION_FRAMES = 3
REWARD_BUTTON_WAIT_SECONDS = 1.0
NEXT_LEVEL_POLL_INTERVAL_SECONDS = 5.0
NEXT_LEVEL_TIMEOUT_SECONDS = 240.0
NEXT_LEVEL_CONFIRMATION_POLL_RATIO = 0.2
NEXT_LEVEL_CONFIRMATION_POLL_MAX_SECONDS = 1.0


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
        next_level_poll_interval_seconds: float = NEXT_LEVEL_POLL_INTERVAL_SECONDS,
        next_level_timeout_seconds: float = NEXT_LEVEL_TIMEOUT_SECONDS,
    ):
        if settling_seconds < 0:
            raise ValueError("Settling duration must not be negative.")
        if next_level_poll_interval_seconds <= 0:
            raise ValueError("Next-level poll interval must be positive.")
        if next_level_timeout_seconds <= 0:
            raise ValueError("Next-level timeout must be positive.")
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
        self.play_again_description = "confirmed Gold Play Again button"
        self.next_level_poll_interval_seconds = next_level_poll_interval_seconds
        self.next_level_timeout_seconds = next_level_timeout_seconds
        self.next_level_confirmation_poll_interval_seconds = min(
            NEXT_LEVEL_CONFIRMATION_POLL_MAX_SECONDS,
            next_level_poll_interval_seconds * NEXT_LEVEL_CONFIRMATION_POLL_RATIO,
        )
        self._settling_started_at: float | None = None
        self._last_detection_error: str | None = None
        self._active_solution: MazeSolution | None = None
        self._next_level_wait_started_at: float | None = None
        self._last_next_level_poll_at: float | None = None

    def process_frame(self, frame: np.ndarray) -> None:
        if self.finished:
            return
        if self.state is ControllerState.ADVANCING_LEVEL:
            self._advance_level_transition()
            return
        if self.state is ControllerState.AWAITING_NEXT_LEVEL:
            self._poll_for_next_level(frame)
            return

        reward_match = RewardScreenDetector.detect(frame)
        if self._process_reward_frame(reward_match, frame.shape[:2]):
            return

        if self.state is ControllerState.WAITING_FOR_BOARD:
            self._process_playable_board(frame)
        elif self.state is ControllerState.EXECUTING:
            self._execute_next_tap()
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
        self._settling_started_at = None
        self._next_level_wait_started_at = None
        self._last_next_level_poll_at = None

    def _on_level_up_screen(self) -> None:
        self._next_level_wait_started_at = time.monotonic()
        self._last_next_level_poll_at = None
        logger.info(
            "Ad reward panel confirmed; not tapping the panel. Checking for a "
            "stable next board every %.0fs for up to %.0fs.",
            self.next_level_poll_interval_seconds,
            self.next_level_timeout_seconds,
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

        self._start_level(observation)

    def _start_level(self, observation: BoardObservation) -> None:
        self._reset_board_candidate()
        self._begin_level_progress()
        self._transition(ControllerState.PLANNING)
        try:
            solution = self.agent.plan(observation)
        except RuntimeError:
            self._transition(ControllerState.FAILED)
            raise
        if solution is None:
            self._transition(ControllerState.STOPPED)
            return
        self._active_solution = solution
        if self.agent.dry_run:
            self._transition(ControllerState.STOPPED)
            return
        self._transition(ControllerState.EXECUTING)
        self._execute_next_tap()

    def _execute_next_tap(self) -> None:
        try:
            complete = self.agent.execute_next_tap()
        except RuntimeError:
            self._transition(ControllerState.FAILED)
            raise
        if not complete:
            return
        self._settling_started_at = time.monotonic()
        self._transition(ControllerState.SETTLING)

    def _poll_for_next_level(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        if self._next_level_wait_started_at is None:
            raise RuntimeError("AwaitingNextLevel state has no wait start time.")
        if now - self._next_level_wait_started_at >= self.next_level_timeout_seconds:
            logger.warning(
                "No new gold board appeared within %.0fs; stopping automation.",
                self.next_level_timeout_seconds,
            )
            self._transition(ControllerState.STOPPED)
            return
        confirmation_in_progress = self._candidate_board_signature is not None
        poll_interval = (
            self.next_level_confirmation_poll_interval_seconds
            if confirmation_in_progress
            else self.next_level_poll_interval_seconds
        )
        if (
            self._last_next_level_poll_at is not None
            and now - self._last_next_level_poll_at < poll_interval
        ):
            return
        self._last_next_level_poll_at = now

        if RewardScreenDetector.detect(frame) is not None:
            self._reset_board_candidate()
            logger.debug("Reward screen is still visible; waiting for the next board.")
            return
        try:
            observation = self.perception.process(frame)
        except BoardNotDetectedError as error:
            self._reset_board_candidate()
            logger.debug("Waiting for next gold board: %s", error)
            return
        if not self._confirm_board(observation):
            return

        logger.info("Detected a stable gold board after the ad; resuming automatically.")
        self._reset_for_next_level()
        self._start_level(observation)

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
