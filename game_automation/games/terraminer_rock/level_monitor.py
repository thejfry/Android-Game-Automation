"""Per-level HUD validation, progress display, and completion summaries."""

from collections import deque
from dataclasses import dataclass, replace
import logging
import time
from typing import Callable

import numpy as np

from .hud_detection import RockHudDetector, RockHudStats
from .perception import BoardObservation
from .terminal_progress import format_progress_line

logger = logging.getLogger("terraminer_rock.level_monitor")

RECENT_LEVELS_COUNT = 5
HUD_STATS_MAX_AGE_SECONDS = 3.0
HUD_TIMER_TOLERANCE_SECONDS = 3.0


@dataclass(frozen=True)
class LevelStats:
    level: int
    pairs: int
    elapsed_seconds: float
    moves: int
    estimated_seconds_left: float


class RockLevelMonitor:
    """Track level-local data without owning controller state transitions."""

    def __init__(
        self,
        level_time_seconds: float,
        progress_callback: Callable[[str | None], None] | None = None,
    ):
        self.level_time_seconds = level_time_seconds
        self.progress_callback = (
            progress_callback
            if progress_callback is not None
            else lambda _text: None
        )
        self._level_number = 0
        self._level_started_at: float | None = None
        self._expected_pairs: int | None = None
        self._initial_seconds: int | None = None
        self._recorded = False
        self._live_progress_text: str | None = None
        self._max_matches_seen = 0
        self._max_board_matches_seen = 0
        self._last_hud_stats: RockHudStats | None = None
        self._last_hud_stats_at: float | None = None
        self._recent_levels: deque[LevelStats] = deque(maxlen=RECENT_LEVELS_COUNT)

    @property
    def level_number(self) -> int:
        return self._level_number

    @property
    def has_active_level(self) -> bool:
        return self._level_started_at is not None

    @property
    def recorded(self) -> bool:
        return self._recorded

    @property
    def last_hud_stats(self) -> RockHudStats | None:
        return self._last_hud_stats

    @property
    def recent_levels(self) -> tuple[LevelStats, ...]:
        return tuple(self._recent_levels)

    def start_level(self, observation: BoardObservation) -> None:
        self.clear_progress()
        self._level_number += 1
        self._level_started_at = time.monotonic()
        self._expected_pairs = len(observation.board.cards) // 2
        self._initial_seconds = None
        self._recorded = False
        self._max_matches_seen = 0
        self._max_board_matches_seen = 0
        self._last_hud_stats = None
        self._last_hud_stats_at = None

    def observe(
        self,
        observation: BoardObservation,
        frame: np.ndarray,
        tap_count: int,
    ) -> None:
        if not self.has_active_level:
            raise RuntimeError("Cannot observe progress before a level starts.")

        expected_total = len(observation.board.cards) // 2
        visible_matches = sum(card.matched for card in observation.board.cards) // 2
        self._max_board_matches_seen = max(
            self._max_board_matches_seen,
            visible_matches,
        )
        stats = RockHudDetector.detect(frame)
        if stats is None:
            return
        now = time.monotonic()
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
        if stats.moves_used > tap_count:
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
        if previous_stats is not None and self._last_hud_stats_at is not None:
            elapsed_since_previous = max(0.0, now - self._last_hud_stats_at)
            maximum_timer_drop = elapsed_since_previous + HUD_TIMER_TOLERANCE_SECONDS
            timer_drop = previous_stats.seconds_remaining - stats.seconds_remaining
            if timer_drop > maximum_timer_drop:
                logger.debug(
                    "Ignoring inconsistent HUD timer %ds; it dropped %ds in %.1fs "
                    "(maximum plausible drop %.1fs).",
                    stats.seconds_remaining,
                    timer_drop,
                    elapsed_since_previous,
                    maximum_timer_drop,
                )
                return

        self._last_hud_stats = stats
        self._last_hud_stats_at = now
        if self._initial_seconds is None:
            self._initial_seconds = stats.seconds_remaining
        self._max_matches_seen = max(self._max_matches_seen, stats.matches_made)
        self._update_progress(
            format_progress_line(
                self._level_number,
                stats.moves_used,
                stats.move_limit,
                stats.seconds_remaining,
                self._initial_seconds or stats.seconds_remaining,
                self._max_matches_seen,
                stats.expected_matches,
            )
        )

    def record_completion(
        self,
        observation: BoardObservation | None,
        tap_count: int,
    ) -> None:
        if self._recorded:
            return
        if self._level_started_at is None:
            raise RuntimeError("Completed board has no recorded level start time.")
        self._update_progress(None)
        elapsed = max(0.0, time.monotonic() - self._level_started_at)
        if observation is not None:
            pairs = len(observation.board.cards) // 2
        elif self._expected_pairs is not None:
            pairs = self._expected_pairs
        else:
            raise RuntimeError("Completed level has no recorded pair count.")

        hud_stats = self._last_hud_stats
        if (
            self._last_hud_stats_at is None
            or time.monotonic() - self._last_hud_stats_at > HUD_STATS_MAX_AGE_SECONDS
        ):
            hud_stats = None
        moves = hud_stats.moves_used if hud_stats is not None else (tap_count + 1) // 2
        seconds_left = (
            float(hud_stats.seconds_remaining)
            if hud_stats is not None
            else max(0.0, self.level_time_seconds - elapsed)
        )
        stats = LevelStats(
            self._level_number,
            pairs,
            elapsed,
            moves,
            seconds_left,
        )
        self._recent_levels.append(stats)
        self._recorded = True
        self._log_completion(stats, hud_stats is not None)

    def clear_progress(self) -> None:
        self._update_progress(None)

    def _update_progress(self, text: str | None) -> None:
        if text == self._live_progress_text:
            return
        self._live_progress_text = text
        self.progress_callback(text)

    def _log_completion(self, stats: LevelStats, uses_hud: bool) -> None:
        time_source = "HUD" if uses_hud else "estimated"
        logger.info(
            "Level %d complete: %s time left %.0fs of %.0fs; "
            "%.1fs elapsed (%.2fs/match); %d moves (%.2f/match).",
            stats.level,
            time_source,
            stats.estimated_seconds_left,
            self.level_time_seconds,
            stats.elapsed_seconds,
            stats.elapsed_seconds / max(stats.pairs, 1),
            stats.moves,
            stats.moves / max(stats.pairs, 1),
        )
        average_elapsed = sum(
            item.elapsed_seconds for item in self._recent_levels
        ) / len(self._recent_levels)
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
