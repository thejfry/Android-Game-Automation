"""One-card-at-a-time Android agent for the rock memory minigame."""

import logging
import time

from .perception import BoardObservation
from .solver import MemorySolver
from game_automation.games.terraminer_gold.tap_executor import TapExecutor

logger = logging.getLogger(__name__)


class TerraminerRockAgent:
    def __init__(
        self,
        tap_executor: TapExecutor,
        tap_interval: float = 0.5,
        dry_run: bool = False,
    ):
        if tap_interval < 0:
            raise ValueError("Tap interval must not be negative.")
        self.tap_executor = tap_executor
        self.tap_interval = tap_interval
        self.dry_run = dry_run
        self.solver = MemorySolver()
        self._next_tap_at = 0.0
        self._tapped_cards: set[tuple[int, int]] = set()

    def reset_for_next_level(self) -> None:
        self.solver.reset()
        self._next_tap_at = 0.0
        self._tapped_cards.clear()

    def process(self, observation: BoardObservation) -> bool:
        """Tap at most one card; return whether the board is fully matched."""
        board = observation.board
        if board.complete:
            logger.info("All %d cards are matched.", len(board.cards))
            return True
        if time.monotonic() < self._next_tap_at:
            return False

        position = self.solver.next_card(board)
        if position is None:
            return False
        card = next(card for card in board.cards if card.position == position)
        if self.dry_run:
            logger.info(
                "Dry run: next card is row %d, column %d; no tap sent.",
                *position,
            )
            return False

        self.tap_executor.tap_point(
            card.center,
            board.frame_size,
            description=f"memory card ({position[0]}, {position[1]})",
        )
        self._tapped_cards.add(position)
        self._next_tap_at = time.monotonic() + self.tap_interval
        logger.info(
            "Tapped card (%d, %d); revealed %d distinct cards and confirmed %d matches.",
            position[0],
            position[1],
            len(self._tapped_cards),
            sum(1 for item in board.cards if item.matched),
        )
        return False
