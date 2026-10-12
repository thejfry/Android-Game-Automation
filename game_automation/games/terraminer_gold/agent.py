"""Agent that plans and taps a route through Terraminer Gold."""

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

from .perception import BoardObservation, PORT_CONFIDENCE_THRESHOLD, TerraminerBoard
from .solver import MazeSolution, MazeSolver
from .tap_executor import TapExecutor

logger = logging.getLogger(__name__)

ROUTE_VERIFICATION_CONFIDENCE = 0.875
TAP_POSITION_JITTER_FRACTION = 0.05
TAP_INTERVAL_JITTER_FRACTION = 0.2


@dataclass(frozen=True)
class SolveResult:
    solution: MazeSolution
    executed: bool


class TerraminerGoldAgent:
    def __init__(
        self,
        tap_executor: TapExecutor,
        tap_interval: float = 0.04,
        verification_attempts: int = 3,
        dry_run: bool = False,
    ):
        if tap_interval < 0:
            raise ValueError("Tap interval must not be negative.")
        if verification_attempts < 1:
            raise ValueError("Verification attempts must be positive.")
        self.tap_executor = tap_executor
        self.tap_interval = tap_interval
        self.verification_attempts = verification_attempts
        self.dry_run = dry_run
        self.solver = MazeSolver()
        self._attempts = 0
        self._finished = False
        self._last_solution: MazeSolution | None = None

    @property
    def finished(self) -> bool:
        return self._finished

    def reset_for_next_level(self) -> None:
        self._attempts = 0
        self._finished = False
        self._last_solution = None

    def solve(
        self,
        observation: BoardObservation,
        before_execution: Callable[[], None] | None = None,
    ) -> SolveResult | None:
        if self._finished:
            return None
        board = observation.board

        if self._attempts:
            if (
                self._last_solution is not None
                and self.verify_route(board, self._last_solution)
            ):
                logger.info(
                    "Route geometry verified against the fresh frame; "
                    "game-level success is not independently detected."
                )
                self._finished = True
                return SolveResult(self._last_solution, False)
            if self._attempts >= self.verification_attempts:
                logger.error(
                    "Route did not verify after %d solve attempts.",
                    self._attempts,
                )
                raise RuntimeError("The route did not verify after the allowed solve attempts.")
            logger.warning(
                "Route is not connected yet; retrying solve (%d/%d).",
                self._attempts + 1,
                self.verification_attempts,
            )

        logger.info(
            "Processing detected %dx%d board from %s to %s.",
            board.size,
            board.size,
            board.start,
            board.goal,
        )
        logger.debug(
            "Board bounds %s; tile masks in N=1,E=2,S=4,W=8 bit order:\n%s",
            board.bounds,
            "\n".join(
                " ".join(f"{mask:01X}" for mask in row)
                for row in board.tile_masks
            ),
        )
        if board.tile_port_confidence is not None:
            logger.debug(
                "Tile port confidence (N,E,S,W):\n%s",
                "\n".join(
                    " ".join(
                        "/".join(f"{value:.2f}" for value in ports)
                        for ports in row
                    )
                    for row in board.tile_port_confidence
                ),
            )
        solution = self.solver.solve(board.tile_masks, board.start, board.goal)
        if solution is None:
            logger.error(
                "No route found from %s to %s on the detected %dx%d board.",
                board.start,
                board.goal,
                board.size,
                board.size,
            )
            raise RuntimeError(
                f"No route found from {board.start} to {board.goal} on the detected "
                f"{board.size}x{board.size} board. Check the perception output."
            )

        logger.debug(
            "Selected route %s with tile turns %s.",
            solution.path,
            solution.actions,
        )
        if self.dry_run:
            logger.info(
                "Dry run: found route with %d tiles and %d taps; no device taps sent.",
                len(solution.path),
                solution.tap_count,
            )
            for action in solution.actions:
                logger.info(
                    "Dry run action: rotate tile (%d, %d) by %d quarter-turn(s) "
                    "to mask %X.",
                    action.row,
                    action.column,
                    action.turns,
                    action.target_mask,
                )
            self._finished = True
            self._last_solution = solution
            return SolveResult(solution, False)

        if solution.tap_count == 0 and not self._route_has_confident_ports(board, solution):
            logger.error(
                "Rejecting zero-tap route: the observed tile-port evidence is not "
                "strong enough to confirm every route connection."
            )
            raise RuntimeError(
                "The zero-tap route is not supported by sufficiently confident "
                "tile-port evidence. Check the detected tile masks before continuing."
            )

        if before_execution is not None:
            before_execution()
        self._attempts += 1
        self._last_solution = solution
        logger.info(
            "Executing solve attempt %d/%d: route has %d tiles and %d taps.",
            self._attempts,
            self.verification_attempts,
            len(solution.path),
            solution.tap_count,
        )
        self._execute_solution(board, solution, observation.frame_size)
        return SolveResult(solution, True)

    def _execute_solution(
        self,
        board: TerraminerBoard,
        solution: MazeSolution,
        frame_size: tuple[int, int],
    ):
        left, top, right, bottom = board.bounds
        cell_width = (right - left) / board.size
        cell_height = (bottom - top) / board.size
        for action in solution.actions:
            x, y = board.center(action.row, action.column)
            if action.turns:
                logger.info(
                    "Rotating tile (%d, %d) by %d quarter-turn(s).",
                    action.row,
                    action.column,
                    action.turns,
                )
            for tap_index in range(action.turns):
                self.tap_executor.tap_point(
                    (x, y),
                    frame_size,
                    (
                        cell_width * TAP_POSITION_JITTER_FRACTION,
                        cell_height * TAP_POSITION_JITTER_FRACTION,
                    ),
                    description=f"tile ({action.row}, {action.column})",
                )
                if self.tap_interval and tap_index + 1 < action.turns:
                    interval_jitter = self.tap_interval * TAP_INTERVAL_JITTER_FRACTION
                    time.sleep(
                        random.uniform(
                            self.tap_interval - interval_jitter,
                            self.tap_interval + interval_jitter,
                        )
                    )
        logger.info("Finished sending route taps; waiting for board verification.")

    @staticmethod
    def verify_route(board: TerraminerBoard, solution: MazeSolution) -> bool:
        return TerraminerGoldAgent._is_path_connected(board, solution)

    @staticmethod
    def _route_has_confident_ports(
        board: TerraminerBoard,
        solution: MazeSolution,
        minimum_confidence: float = ROUTE_VERIFICATION_CONFIDENCE,
    ) -> bool:
        if len(solution.path) <= 2:
            return True
        if board.tile_port_confidence is None:
            return False
        for index in range(1, len(solution.path) - 1):
            row, column = solution.path[index]
            previous_row, previous_column = solution.path[index - 1]
            next_row, next_column = solution.path[index + 1]
            incoming = TerraminerGoldAgent._edge_between(
                row, column, previous_row, previous_column
            )
            outgoing = TerraminerGoldAgent._edge_between(
                row, column, next_row, next_column
            )
            mask = board.tile_masks[row][column]
            confidences = board.tile_port_confidence[row][column]
            for edge in (incoming, outgoing):
                edge_index = edge.bit_length() - 1
                if (
                    not mask & edge
                    or confidences[edge_index] < minimum_confidence
                ):
                    return False
        return True

    @staticmethod
    def _is_path_connected(board: TerraminerBoard, solution: MazeSolution) -> bool:
        minimum_confidence = (
            ROUTE_VERIFICATION_CONFIDENCE
            if solution.tap_count == 0
            else PORT_CONFIDENCE_THRESHOLD
        )
        if not TerraminerGoldAgent._route_has_confident_ports(
            board,
            solution,
            minimum_confidence,
        ):
            logger.debug("Route geometry check failed: route ports lack strong evidence.")
            return False
        if board.tile_port_confidence is None:
            return not solution.actions

        for action in solution.actions:
            observed_mask = board.tile_masks[action.row][action.column]
            if observed_mask != action.target_mask:
                logger.debug(
                    "Route geometry check failed: tile (%d, %d) mask was %X, "
                    "expected %X.",
                    action.row,
                    action.column,
                    observed_mask,
                    action.target_mask,
                )
                return False
            confidences = board.tile_port_confidence[action.row][action.column]
            for edge_index, confidence in enumerate(confidences):
                expected_open = bool(action.target_mask & (1 << edge_index))
                if expected_open:
                    if confidence < PORT_CONFIDENCE_THRESHOLD:
                        logger.debug(
                            "Route geometry check failed: tile (%d, %d) target port %d "
                            "was not confidently observed.",
                            action.row,
                            action.column,
                            edge_index,
                        )
                        return False
                elif confidence >= PORT_CONFIDENCE_THRESHOLD:
                    logger.debug(
                        "Route geometry check failed: unexpected port %d observed "
                        "on tile (%d, %d).",
                        edge_index,
                        action.row,
                        action.column,
                    )
                    return False
        return True

    @staticmethod
    def _edge_between(row: int, column: int, other_row: int, other_column: int) -> int:
        delta = (other_row - row, other_column - column)
        edges = {
            (-1, 0): 1,
            (0, 1): 2,
            (1, 0): 4,
            (0, -1): 8,
        }
        if delta not in edges:
            raise ValueError("Route contains non-adjacent tile coordinates.")
        return edges[delta]
