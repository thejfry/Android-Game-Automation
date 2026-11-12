"""Route search for the rotatable path tiles in Terraminer Gold."""

from collections.abc import Sequence
from dataclasses import dataclass
import heapq
import itertools
import logging

logger = logging.getLogger(__name__)

# Bit order is clockwise so rotating left by one position is a clockwise turn.
NORTH, EAST, SOUTH, WEST = 1, 2, 4, 8
DIRECTIONS = (
    (NORTH, -1, 0),
    (EAST, 0, 1),
    (SOUTH, 1, 0),
    (WEST, 0, -1),
)
OPPOSITE = {NORTH: SOUTH, EAST: WEST, SOUTH: NORTH, WEST: EAST}


def rotate_clockwise(mask: int, turns: int = 1) -> int:
    """Rotate a four-port path mask clockwise by ``turns`` quarter turns."""
    turns %= 4
    for _ in range(turns):
        mask = ((mask << 1) & 0b1111) | ((mask >> 3) & 1)
    return mask


@dataclass(frozen=True)
class TileAction:
    row: int
    column: int
    turns: int
    target_mask: int


@dataclass(frozen=True)
class MazeSolution:
    actions: tuple[TileAction, ...]
    path: tuple[tuple[int, int], ...]

    @property
    def tap_count(self) -> int:
        return sum(action.turns for action in self.actions)


class MazeSolver:
    """Find a start-to-goal route, minimizing taps then route length.

    A path mask uses one bit per open tile edge. Zero-mask tiles are treated
    as blocked, while the start and goal cells are endpoints rather than
    rotatable road tiles.
    """

    def solve(
        self,
        tiles: Sequence[Sequence[int]],
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> MazeSolution | None:
        if not tiles or not tiles[0]:
            raise ValueError("The tile grid must not be empty.")
        rows = len(tiles)
        columns = len(tiles[0])
        if any(len(row) != columns for row in tiles):
            raise ValueError("The tile grid must be rectangular.")
        for name, (row, column) in (("start", start), ("goal", goal)):
            if not (0 <= row < rows and 0 <= column < columns):
                raise ValueError(f"The {name} coordinate is outside the tile grid.")
        logger.info(
            "Searching for a route across a %dx%d board from %s to %s.",
            rows,
            columns,
            start,
            goal,
        )
        if start == goal:
            logger.info("Start and goal are the same cell; no tile changes are needed.")
            return MazeSolution((), (start,))

        counter = itertools.count()
        queue = []
        # Cost is lexicographic: fewest taps, then fewest road tiles.
        initial_visited = 1 << (start[0] * columns + start[1])
        heapq.heappush(
            queue,
            (0, 0, next(counter), start, None, initial_visited, (start,), (), 0),
        )
        best: dict[tuple, tuple[int, int]] = {}
        expanded_states = 0

        while queue:
            taps, length, _, cell, incoming, visited, path, actions, current_mask = heapq.heappop(queue)
            row, column = cell
            if cell == goal:
                solution = MazeSolution(actions, path)
                logger.info(
                    "Route found after expanding %d states: %d path cells, %d taps.",
                    expanded_states,
                    len(solution.path),
                    solution.tap_count,
                )
                logger.debug("Route cells: %s", solution.path)
                logger.debug("Tile actions: %s", solution.actions)
                return solution

            state_key = (cell, incoming, visited, current_mask)
            old_cost = best.get(state_key)
            if old_cost is not None and old_cost <= (taps, length):
                continue
            best[state_key] = (taps, length)
            expanded_states += 1
            if expanded_states % 1000 == 0:
                logger.debug(
                    "Route search expanded %d states; %d candidates remain.",
                    expanded_states,
                    len(queue),
                )

            available_edges = (edge for edge, _, _ in DIRECTIONS if current_mask & edge) if cell != start else (edge for edge, _, _ in DIRECTIONS)
            for edge in available_edges:
                dr, dc = next((dr, dc) for candidate, dr, dc in DIRECTIONS if candidate == edge)
                next_cell = (row + dr, column + dc)
                nr, nc = next_cell
                if not (0 <= nr < rows and 0 <= nc < columns):
                    continue
                next_bit = 1 << (nr * columns + nc)
                if visited & next_bit:
                    continue

                if cell == start:
                    if next_cell == goal:
                        heapq.heappush(
                            queue,
                            (
                                taps,
                                length + 1,
                                next(counter),
                                goal,
                                OPPOSITE[edge],
                                visited | next_bit,
                                path + (goal,),
                                actions,
                                0,
                            ),
                        )
                    else:
                        self._push_tile_variants(
                            queue, counter, tiles[nr][nc], nr, nc, OPPOSITE[edge],
                            taps, length, visited | next_bit, path + (next_cell,),
                            actions, columns,
                        )
                    continue

                if next_cell == goal:
                    heapq.heappush(
                        queue,
                        (
                            taps,
                            length + 1,
                            next(counter),
                            goal,
                            OPPOSITE[edge],
                            visited | next_bit,
                            path + (goal,),
                            actions,
                            0,
                        ),
                    )
                else:
                    self._push_tile_variants(
                        queue, counter, tiles[nr][nc], nr, nc, OPPOSITE[edge],
                        taps, length, visited | next_bit, path + (next_cell,),
                        actions, columns,
                    )

        logger.warning(
            "No route found from %s to %s after expanding %d states.",
            start,
            goal,
            expanded_states,
        )
        return None

    @staticmethod
    def _push_tile_variants(
        queue: list,
        counter,
        mask: int,
        row: int,
        column: int,
        incoming: int,
        taps: int,
        length: int,
        visited: int,
        path: tuple[tuple[int, int], ...],
        actions: tuple[TileAction, ...],
        columns: int,
    ) -> None:
        del columns
        if not mask:
            return
        variants: dict[int, int] = {}
        for turns in range(4):
            oriented = rotate_clockwise(mask, turns)
            variants[oriented] = min(turns, variants.get(oriented, 4))

        for oriented, turns in variants.items():
            if not oriented & incoming:
                continue
            action = TileAction(row, column, turns, oriented)
            heapq.heappush(
                queue,
                (
                    taps + turns,
                    length + 1,
                    next(counter),
                    (row, column),
                    incoming,
                    visited,
                    path,
                    actions + (action,),
                    oriented,
                ),
            )
