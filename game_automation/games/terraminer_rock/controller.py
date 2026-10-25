"""Lifecycle orchestration for Terraminer Rock."""

from enum import Enum
from collections import deque
from dataclasses import dataclass, replace
import logging
import time

import numpy as np

from .agent import TerraminerRockAgent
from .perception import (
    BoardNotDetectedError,
    BoardObservation,
    TerraminerRockPerception,
)
from .hud_detection import RockHudDetector, RockHudStats
from .reward_detection import RewardMatch, RewardMatchKind, RewardScreenDetector
from game_automation.games.terraminer_gold.tap_executor import TapExecutor

short_name = ".".join(__name__.split(".")[-2:])
logger = logging.getLogger(short_name)

SETTLING_SECONDS = 0.5
REWARD_CONFIRMATION_FRAMES = 2
BOARD_CONFIRMATION_FRAMES = 3
REWARD_BUTTON_WAIT_SECONDS = 1.0
PROGRESS_BAR_WIDTH = 20
RECENT_LEVELS_COUNT = 5
HUD_STATS_MAX_AGE_SECONDS = 3.0
NEXT_LEVEL_POLL_INTERVAL_SECONDS = 5.0
NEXT_LEVEL_TIMEOUT_SECONDS = 240.0
NEXT_LEVEL_CONFIRMATION_POLL_RATIO = 0.2
NEXT_LEVEL_CONFIRMATION_POLL_MAX_SECONDS = 1.0


@dataclass(frozen=True)
class LevelStats:
    level: int
    pairs: int
    elapsed_seconds: float
    moves: int
    estimated_seconds_left: float


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
        self.level_time_seconds = level_time_seconds
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
        self._level_started_at: float | None = None
        self._level_expected_pairs: int | None = None
        self._level_number = 0
        self._last_progress: tuple[int, int] | None = None
        self._max_matches_seen = 0
        self._max_board_matches_seen = 0
        self._last_hud_stats: RockHudStats | None = None
        self._last_hud_stats_at: float | None = None
        self._last_hud_move_logged: int | None = None
        self._last_hud_time_bucket: int | None = None
        self._recent_levels: deque[LevelStats] = deque(maxlen=RECENT_LEVELS_COUNT)
        self._level_recorded = False
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
        self._level_started_at = time.monotonic()
        self._level_expected_pairs = len(observation.board.cards) // 2
        self._level_recorded = False
        self._last_progress = None
        self._max_matches_seen = 0
        self._max_board_matches_seen = 0
        self._last_hud_stats = None
        self._last_hud_stats_at = None
        self._last_hud_move_logged = None
        self._last_hud_time_bucket = None
        self._level_number += 1
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
        self._log_match_progress(observation, frame)
        if complete:
            self._record_completed_level(observation)
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

    def _log_match_progress(
        self,
        observation: BoardObservation,
        frame: np.ndarray,
    ) -> None:
        expected_total = len(observation.board.cards) // 2
        visible_matches = sum(card.matched for card in observation.board.cards) // 2
        self._max_board_matches_seen = max(
            self._max_board_matches_seen,
            visible_matches,
        )
        stats = RockHudDetector.detect(frame)
        if stats is None:
            return
        if stats.expected_matches != expected_total:
            logger.debug(
                "HUD Matches denominator was read as %d; detected board has %d "
                "pairs. Using the detected board total.",
                stats.expected_matches,
                expected_total,
            )
            stats = replace(stats, expected_matches=expected_total)
        if stats.matches_made > expected_total:
            logger.debug(
                "Ignoring HUD Matches counter %d/%d; detected board has %d pairs.",
                stats.matches_made,
                expected_total,
                expected_total,
            )
            return
        if stats.matches_made > self._max_board_matches_seen:
            logger.debug(
                "Ignoring HUD Matches counter %d/%d; only %d matched pairs "
                "have been confirmed on the board.",
                stats.matches_made,
                stats.expected_matches,
                self._max_board_matches_seen,
            )
            return
        tap_count = getattr(self.agent, "tap_count", None)
        if isinstance(tap_count, int) and stats.moves_used > tap_count:
            logger.debug(
                "Ignoring HUD Moves counter %d/%d; only %d card taps have been sent.",
                stats.moves_used,
                stats.move_limit,
                tap_count,
            )
            return
        previous_stats = self._last_hud_stats
        if previous_stats is not None and (
            stats.moves_used < previous_stats.moves_used
            or stats.matches_made < previous_stats.matches_made
            or stats.seconds_remaining > previous_stats.seconds_remaining
            or stats.move_limit != previous_stats.move_limit
        ):
            logger.debug(
                "Ignoring inconsistent HUD counters %d/%d moves, %ds, %d/%d "
                "matches; last accepted values were %d/%d, %ds, %d/%d.",
                stats.moves_used,
                stats.move_limit,
                stats.seconds_remaining,
                stats.matches_made,
                stats.expected_matches,
                previous_stats.moves_used,
                previous_stats.move_limit,
                previous_stats.seconds_remaining,
                previous_stats.matches_made,
                previous_stats.expected_matches,
            )
            return
        self._last_hud_stats = stats
        self._last_hud_stats_at = time.monotonic()
        self._max_matches_seen = max(self._max_matches_seen, stats.matches_made)
        pairs_matched = min(self._max_matches_seen, stats.expected_matches)
        progress = pairs_matched, stats.expected_matches
        if progress == self._last_progress:
            progress_changed = False
        else:
            progress_changed = True
            self._last_progress = progress

        time_bucket = stats.seconds_remaining // 10
        if (
            progress_changed
            or stats.moves_used != self._last_hud_move_logged
            or time_bucket != self._last_hud_time_bucket
        ):
            filled = round(
                PROGRESS_BAR_WIDTH * pairs_matched / max(stats.expected_matches, 1)
            )
            bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
            logger.info(
                "Level %d progress: [%s] %d/%d matches | Moves %d/%d | "
                "Time %ds.",
                self._level_number,
                bar,
                pairs_matched,
                stats.expected_matches,
                stats.moves_used,
                stats.move_limit,
                stats.seconds_remaining,
            )
            self._last_hud_move_logged = stats.moves_used
            self._last_hud_time_bucket = time_bucket

    def _record_completed_level(
        self,
        observation: BoardObservation | None,
    ) -> None:
        if self._level_recorded:
            return
        if self._level_started_at is None:
            raise RuntimeError("Completed board has no recorded level start time.")
        elapsed = max(0.0, time.monotonic() - self._level_started_at)
        if observation is not None:
            pairs = len(observation.board.cards) // 2
        elif self._level_expected_pairs is not None:
            pairs = self._level_expected_pairs
        else:
            raise RuntimeError("Completed level has no recorded pair count.")
        hud_stats = self._last_hud_stats
        if (
            self._last_hud_stats_at is None
            or time.monotonic() - self._last_hud_stats_at > HUD_STATS_MAX_AGE_SECONDS
        ):
            hud_stats = None
        taps = self.agent.tap_count
        moves = hud_stats.moves_used if hud_stats is not None else (taps + 1) // 2
        estimated_left = (
            float(hud_stats.seconds_remaining)
            if hud_stats is not None
            else max(0.0, self.level_time_seconds - elapsed)
        )
        stats = LevelStats(
            self._level_number,
            pairs,
            elapsed,
            moves,
            estimated_left,
        )
        self._recent_levels.append(stats)
        self._level_recorded = True

        time_source = "estimated" if hud_stats is None else "HUD"
        logger.info(
            "Level %d complete: %s time left %.0fs of %.0fs; "
            "%.1fs elapsed (%.2fs/match); %d moves (%.2f/match).",
            stats.level,
            time_source,
            stats.estimated_seconds_left,
            self.level_time_seconds,
            stats.elapsed_seconds,
            stats.elapsed_seconds / stats.pairs,
            stats.moves,
            stats.moves / stats.pairs,
        )
        average_elapsed = sum(item.elapsed_seconds for item in self._recent_levels) / len(
            self._recent_levels
        )
        total_pairs = sum(item.pairs for item in self._recent_levels)
        average_seconds_per_match = sum(
            item.elapsed_seconds for item in self._recent_levels
        ) / max(total_pairs, 1)
        average_moves_per_match = sum(
            item.moves for item in self._recent_levels
        ) / max(total_pairs, 1)
        logger.info(
            "Recent %d-level stats: avg %.1fs/level, %.2fs/match, "
            "%.2f moves/match.",
            len(self._recent_levels),
            average_elapsed,
            average_seconds_per_match,
            average_moves_per_match,
        )
        logger.info(" Level | Pairs | Time left | Moves")
        logger.info("-------+-------+-----------+------")
        for item in self._recent_levels:
            logger.info(
                " %5d | %5d | %6.0fs   | %5d",
                item.level,
                item.pairs,
                item.estimated_seconds_left,
                item.moves,
            )

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
        if self._level_started_at is not None and not self._level_recorded:
            self._record_completed_level(None)
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
            "Level-up/ad prompt confirmed; not tapping Play Level. Checking for "
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
        self.state = state
        logger.info(
            "Controller state: %s -> %s.",
            previous_state.value,
            state.value,
        )
