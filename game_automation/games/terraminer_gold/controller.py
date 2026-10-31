"""Lifecycle orchestration for Terraminer Gold."""

from enum import Enum
import logging
import time

import numpy as np

from .agent import SolveResult, TerraminerGoldAgent
from .perception import BoardNotDetectedError, BoardObservation, TerraminerGoldPerception
from .reward_detection import RewardMatch, RewardMatchKind, RewardScreenDetector
from .solver import MazeSolution
from .tap_executor import TapExecutor

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
    ADVANCING_LEVEL = "AdvancingLevel"
    AWAITING_NEXT_LEVEL = "AwaitingNextLevel"
    FAILED = "Failed"
    STOPPED = "Stopped"


class TerraminerGoldController:
    """Coordinates perception, solving, verification, and level progression."""

    def __init__(
        self,
        agent: TerraminerGoldAgent,
        perception: TerraminerGoldPerception,
        tap_executor: TapExecutor,
        settling_seconds: float = SETTLING_SECONDS,
        continuous: bool = False,
        reward_button_wait_seconds: float = REWARD_BUTTON_WAIT_SECONDS,
    ):
        if settling_seconds < 0:
            raise ValueError("Settling duration must not be negative.")
        if reward_button_wait_seconds < 0:
            raise ValueError("Reward-button wait duration must not be negative.")
        self.agent = agent
        self.perception = perception
        self.tap_executor = tap_executor
        self.settling_seconds = settling_seconds
        self.continuous = continuous
        self.reward_button_wait_seconds = reward_button_wait_seconds
        self.state = ControllerState.WAITING_FOR_BOARD
        self._settling_started_at: float | None = None
        self._advance_level_at: float | None = None
        self._last_detection_error: str | None = None
        self._reward_screen_frames = 0
        self._waiting_for_reward_to_clear = False
        self._candidate_board_signature: tuple | None = None
        self._candidate_board_frames = 0
        self._active_solution: MazeSolution | None = None
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
        if self.state is ControllerState.ADVANCING_LEVEL:
            self._advance_level_transition()
            return

        reward_match = RewardScreenDetector.detect(frame)
        if self._waiting_for_reward_to_clear:
            if reward_match is not None:
                return
            self._waiting_for_reward_to_clear = False
            self._reward_screen_frames = 0
        if reward_match is not None:
            self._reward_screen_frames += 1
        else:
            self._reward_screen_frames = 0
        if self._reward_screen_frames >= REWARD_CONFIRMATION_FRAMES:
            self._handle_confirmed_reward(reward_match, frame.shape[:2])
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
        self._reset_for_next_level()

    def _handle_confirmed_reward(
        self,
        reward_match: RewardMatch | None,
        frame_shape: tuple[int, int],
    ) -> None:
        if reward_match is None:
            raise RuntimeError("Confirmed reward screen has no detected control.")
        self._reset_board_candidate()
        if not self.continuous:
            self._transition(ControllerState.REWARD_SCREEN)
            logger.info("Reward screen confirmed; stopping without further taps.")
            return

        self._transition(ControllerState.REWARD_SCREEN)
        if reward_match.kind is RewardMatchKind.PLAY_AGAIN:
            if reward_match.button_bounds is None:
                raise RuntimeError("Play Again match is missing button bounds.")
            self.tap_executor.tap_bounds(
                reward_match.button_bounds,
                (frame_shape[1], frame_shape[0]),
                description="confirmed Play Again button",
            )
            self._waiting_for_reward_to_clear = True
            self._advance_level_at = (
                time.monotonic() + self.reward_button_wait_seconds
            )
            self._transition(ControllerState.ADVANCING_LEVEL)
            logger.info(
                "Waiting %.1f seconds before resuming board detection.",
                self.reward_button_wait_seconds,
            )
            return

        self._transition(ControllerState.AWAITING_NEXT_LEVEL)
        logger.info(
            "Ad reward panel confirmed; not tapping the panel. Finish any ad, "
            "open the next level, then press Enter to resume board detection."
        )

    def _advance_level_transition(self) -> None:
        if self._advance_level_at is None:
            raise RuntimeError("AdvancingLevel state has no completion deadline.")
        if time.monotonic() < self._advance_level_at:
            return
        self._advance_level_at = None
        self._reset_for_next_level()

    def _reset_for_next_level(self) -> None:
        self.agent.reset_for_next_level()
        self._reward_screen_frames = 0
        self._last_detection_error = None
        self._active_solution = None
        self._reset_board_candidate()
        self._transition(ControllerState.WAITING_FOR_BOARD)

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
            return False
        return True

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

    def _reset_board_candidate(self) -> None:
        self._candidate_board_signature = None
        self._candidate_board_frames = 0

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
