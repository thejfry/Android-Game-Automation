"""Shared lifecycle and level-progression behavior for Terraminer games."""

from enum import Enum
import logging
import time
from typing import Any, Callable

from game_automation.core.tap_executor import TapExecutor

logger = logging.getLogger(__name__)


class ControllerState(Enum):
    WAITING_FOR_BOARD = "WaitingForBoard"
    PLAYING = "Playing"
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


class TerraminerController:
    """Own repeated reward, board-confirmation, and level-transition logic.

    Individual game controllers provide their perception and per-level play
    behavior, while this parent keeps the shared lifecycle consistent.
    """

    def __init__(
        self,
        agent: Any,
        perception: Any,
        tap_executor: TapExecutor,
        continuous: bool,
        reward_button_wait_seconds: float,
        board_confirmation_frames: int = 3,
        reward_confirmation_frames: int = 2,
        level_progress_callback: Callable[[int, str], None] | None = None,
    ):
        if reward_button_wait_seconds < 0:
            raise ValueError("Reward-button wait duration must not be negative.")
        if board_confirmation_frames < 1:
            raise ValueError("Board confirmation frame count must be positive.")
        if reward_confirmation_frames < 1:
            raise ValueError("Reward confirmation frame count must be positive.")
        self.agent = agent
        self.perception = perception
        self.tap_executor = tap_executor
        self.continuous = continuous
        self.reward_button_wait_seconds = reward_button_wait_seconds
        self.play_again_description = "confirmed Play Again button"
        self.board_confirmation_frames = board_confirmation_frames
        self.reward_confirmation_frames = reward_confirmation_frames
        self.level_progress_callback = level_progress_callback
        self.state = ControllerState.WAITING_FOR_BOARD
        self.level_number = 0
        self._advance_level_at: float | None = None
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

    def acknowledge_next_level(self) -> None:
        if self.state is not ControllerState.AWAITING_NEXT_LEVEL:
            raise RuntimeError(
                "Can only acknowledge the next level while awaiting it."
            )
        self._reset_for_next_level()

    def _process_reward_frame(self, match, frame_shape: tuple[int, ...]) -> bool:
        if self._waiting_for_reward_to_clear:
            if match is not None:
                return True
            self._waiting_for_reward_to_clear = False
            self._reward_screen_frames = 0

        if match is not None:
            self._reward_screen_frames += 1
        else:
            self._reward_screen_frames = 0
        if self._reward_screen_frames >= self.reward_confirmation_frames:
            self._handle_confirmed_reward(match, frame_shape)
            return True
        return self._reward_screen_frames > 0

    def _handle_confirmed_reward(
        self,
        match,
        frame_shape: tuple[int, ...],
    ) -> None:
        if match is None:
            raise RuntimeError("Confirmed reward screen has no detected control.")
        self._reset_board_candidate()
        if self.level_number:
            self._report_level_progress("completed")
        if not self.continuous:
            self._transition(ControllerState.REWARD_SCREEN)
            logger.info("Reward screen confirmed; stopping without further taps.")
            return

        self._transition(ControllerState.REWARD_SCREEN)
        if getattr(match.kind, "value", match.kind) == "play_again":
            if match.button_bounds is None:
                raise RuntimeError("Play Again match is missing button bounds.")
            self.tap_executor.tap_bounds(
                match.button_bounds,
                (frame_shape[1], frame_shape[0]),
                description=self.play_again_description,
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
        self._on_level_up_screen()

    def _on_level_up_screen(self) -> None:
        logger.info(
            "Level-up/ad screen confirmed; waiting for the next board."
        )

    def _begin_level_progress(self) -> None:
        self.level_number += 1
        self._report_level_progress("started")

    def _report_level_progress(self, status: str) -> None:
        logger.info("Terraminer level %d %s.", self.level_number, status)
        if self.level_progress_callback is not None:
            self.level_progress_callback(self.level_number, status)

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
        self._reset_board_candidate()
        self._transition(ControllerState.WAITING_FOR_BOARD)

    def _confirm_candidate(self, signature: tuple) -> bool:
        if signature == self._candidate_board_signature:
            self._candidate_board_frames += 1
        else:
            self._candidate_board_signature = signature
            self._candidate_board_frames = 1
        if self._candidate_board_frames < self.board_confirmation_frames:
            logger.debug(
                "Waiting for a stable board: %d/%d matching frames.",
                self._candidate_board_frames,
                self.board_confirmation_frames,
            )
            return False
        return True

    def _reset_board_candidate(self) -> None:
        self._candidate_board_signature = None
        self._candidate_board_frames = 0

    def _transition(self, state: ControllerState) -> None:
        if state is self.state:
            return
        previous_state = self.state
        self.state = state
        self._on_state_transition(previous_state, state)
        logger.info(
            "Controller state: %s -> %s.",
            previous_state.value,
            state.value,
        )

    def _on_state_transition(
        self,
        previous_state: ControllerState,
        next_state: ControllerState,
    ) -> None:
        return
