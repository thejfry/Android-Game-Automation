"""Lifecycle orchestration for Terraminer Rock."""

from enum import Enum
import logging
import time
from typing import Callable

import numpy as np

from .agent import TerraminerRockAgent
from .perception import (
    BoardNotDetectedError,
    BoardObservation,
    TerraminerRockPerception,
)
from .level_monitor import RockLevelMonitor
from .reward_detection import RewardMatch, RewardMatchKind, RewardScreenDetector
from game_automation.games.terraminer_gold.tap_executor import TapExecutor

short_name = ".".join(__name__.split(".")[-2:])
logger = logging.getLogger(short_name)

SETTLING_SECONDS = 0.5
REWARD_CONFIRMATION_FRAMES = 2
BOARD_CONFIRMATION_FRAMES = 3
REWARD_BUTTON_WAIT_SECONDS = 1.0
NEXT_LEVEL_POLL_INTERVAL_SECONDS = 5.0
NEXT_LEVEL_TIMEOUT_SECONDS = 240.0
NEXT_LEVEL_CONFIRMATION_POLL_RATIO = 0.2
NEXT_LEVEL_CONFIRMATION_POLL_MAX_SECONDS = 1.0


class ControllerState(Enum):
    WAITING_FOR_BOARD = "WaitingForBoard"
    PLAYING = "Playing"
    WAITING_FOR_COMPLETION = "WaitingForCompletion"
    REWARD_SCREEN = "RewardScreen"
    ADVANCING_LEVEL = "AdvancingLevel"
    AWAITING_NEXT_LEVEL = "AwaitingNextLevel"
    FAILED = "Failed"
    STOPPED = "Stopped"


class TerraminerRockController:
    """Coordinates card perception, agent actions, and reward progression."""

    def __init__(
        self,
        agent: TerraminerRockAgent,
        perception: TerraminerRockPerception,
        tap_executor: TapExecutor,
        settling_seconds: float = SETTLING_SECONDS,
        continuous: bool = False,
        reward_button_wait_seconds: float = REWARD_BUTTON_WAIT_SECONDS,
        level_time_seconds: float = 90.0,
        next_level_poll_interval_seconds: float = NEXT_LEVEL_POLL_INTERVAL_SECONDS,
        next_level_timeout_seconds: float = NEXT_LEVEL_TIMEOUT_SECONDS,
        progress_callback: Callable[[str | None], None] | None = None,
    ):
        if settling_seconds < 0:
            raise ValueError("Settling duration must not be negative.")
        if reward_button_wait_seconds < 0:
            raise ValueError("Reward-button wait duration must not be negative.")
        if level_time_seconds <= 0:
            raise ValueError("Level timer duration must be positive.")
        if next_level_poll_interval_seconds <= 0:
            raise ValueError("Next-level poll interval must be positive.")
        if next_level_timeout_seconds <= 0:
            raise ValueError("Next-level timeout must be positive.")
        self.agent = agent
        self.perception = perception
        self.tap_executor = tap_executor
        self.settling_seconds = settling_seconds
        self.continuous = continuous
        self.reward_button_wait_seconds = reward_button_wait_seconds
        self.level_monitor = RockLevelMonitor(
            level_time_seconds,
            progress_callback,
        )
        self.next_level_poll_interval_seconds = next_level_poll_interval_seconds
        self.next_level_timeout_seconds = next_level_timeout_seconds
        self.next_level_confirmation_poll_interval_seconds = min(
            NEXT_LEVEL_CONFIRMATION_POLL_MAX_SECONDS,
            next_level_poll_interval_seconds * NEXT_LEVEL_CONFIRMATION_POLL_RATIO,
        )
        self.state = ControllerState.WAITING_FOR_BOARD
        self._settling_started_at: float | None = None
        self._advance_level_at: float | None = None
        self._reward_screen_frames = 0
        self._waiting_for_reward_to_clear = False
        self._next_level_wait_started_at: float | None = None
        self._last_next_level_poll_at: float | None = None
        self._candidate_board_signature: tuple[int, int] | None = None
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
        if self.state is ControllerState.ADVANCING_LEVEL:
            self._advance_level_transition()
            return
        if self.state is ControllerState.AWAITING_NEXT_LEVEL:
            self._poll_for_next_level(frame)
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
            self._process_board(frame)
        elif self.state is ControllerState.PLAYING:
            self._process_board(frame)
        elif self.state is ControllerState.WAITING_FOR_COMPLETION:
            self._advance_completion_wait()

    def acknowledge_next_level(self) -> None:
        if self.state is not ControllerState.AWAITING_NEXT_LEVEL:
            raise RuntimeError("Can only acknowledge the next level while awaiting it.")
        self._reset_for_next_level()

    def _process_board(self, frame: np.ndarray) -> None:
        try:
            observation = self.perception.process(frame)
        except BoardNotDetectedError as error:
            self._reset_board_candidate()
            logger.debug("Waiting for rock board detection: %s", error)
            return

        if self.state is ControllerState.WAITING_FOR_BOARD:
            if not self._confirm_board(observation):
                return
            self._start_level(observation)

        self._process_observation(observation, frame)

    def _start_level(self, observation: BoardObservation) -> None:
        self._reset_board_candidate()
        self.level_monitor.start_level(observation)
        self._transition(ControllerState.PLAYING)

    def _process_observation(
        self,
        observation: BoardObservation,
        frame: np.ndarray,
    ) -> None:
        try:
            complete = self.agent.process(observation)
        except RuntimeError:
            self._transition(ControllerState.FAILED)
            raise
        self.level_monitor.observe(observation, frame, self.agent.tap_count)
        if complete:
            self.level_monitor.record_completion(observation, self.agent.tap_count)
            self._settling_started_at = time.monotonic()
            self._transition(ControllerState.WAITING_FOR_COMPLETION)

    def _poll_for_next_level(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        if self._next_level_wait_started_at is None:
            raise RuntimeError("AwaitingNextLevel state has no wait start time.")
        if now - self._next_level_wait_started_at >= self.next_level_timeout_seconds:
            logger.warning(
                "No new rock board appeared within %.0fs; stopping automation.",
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
            logger.debug("Ad or results screen is still visible; waiting.")
            return
        try:
            observation = self.perception.process(frame)
        except BoardNotDetectedError as error:
            self._reset_board_candidate()
            logger.debug("Waiting for next rock board: %s", error)
            return
        if not self._confirm_board(observation):
            return

        logger.info(
            "Detected a stable %d-card board after the ad; resuming automatically.",
            len(observation.board.cards),
        )
        self._reset_for_next_level()
        self._start_level(observation)
        self._process_observation(observation, frame)

    def _advance_completion_wait(self) -> None:
        if self._settling_started_at is None:
            raise RuntimeError("WaitingForCompletion state has no start time.")
        if time.monotonic() - self._settling_started_at < self.settling_seconds:
            return
        self._settling_started_at = time.monotonic()

    def _confirm_board(self, observation: BoardObservation) -> bool:
        signature = observation.board.signature
        if any(card.face_up for card in observation.board.cards):
            self._reset_board_candidate()
            return False
        if signature == self._candidate_board_signature:
            self._candidate_board_frames += 1
        else:
            self._candidate_board_signature = signature
            self._candidate_board_frames = 1
        if self._candidate_board_frames < BOARD_CONFIRMATION_FRAMES:
            logger.debug(
                "Waiting for stable rock board: %d/%d matching frames.",
                self._candidate_board_frames,
                BOARD_CONFIRMATION_FRAMES,
            )
            return False
        return True

    def _handle_confirmed_reward(
        self,
        reward_match: RewardMatch | None,
        frame_shape: tuple[int, int],
    ) -> None:
        if reward_match is None:
            raise RuntimeError("Confirmed reward screen has no detected control.")
        self._reset_board_candidate()
        if self.level_monitor.has_active_level and not self.level_monitor.recorded:
            self.level_monitor.record_completion(None, self.agent.tap_count)
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
                description="confirmed Rock Play Again button",
            )
            self._waiting_for_reward_to_clear = True
            self._advance_level_at = (
                time.monotonic() + self.reward_button_wait_seconds
            )
            self._transition(ControllerState.ADVANCING_LEVEL)
            return

        self._transition(ControllerState.AWAITING_NEXT_LEVEL)
        self._next_level_wait_started_at = time.monotonic()
        self._last_next_level_poll_at = None
        logger.info(
            "Level-up/ad prompt confirmed; Checking for "
            "a stable board every %.0fs for up to %.0fs.",
            self.next_level_poll_interval_seconds,
            self.next_level_timeout_seconds,
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
        self._settling_started_at = None
        self._next_level_wait_started_at = None
        self._last_next_level_poll_at = None
        self._reset_board_candidate()
        self._transition(ControllerState.WAITING_FOR_BOARD)

    def _reset_board_candidate(self) -> None:
        self._candidate_board_signature = None
        self._candidate_board_frames = 0

    def _transition(self, state: ControllerState) -> None:
        if state is self.state:
            return
        previous_state = self.state
        if previous_state is ControllerState.PLAYING and state is not ControllerState.PLAYING:
            self.level_monitor.clear_progress()
        self.state = state
        logger.info(
            "Controller state: %s -> %s.",
            previous_state.value,
            state.value,
        )
