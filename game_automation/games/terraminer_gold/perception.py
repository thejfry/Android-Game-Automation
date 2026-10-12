"""Computer-vision detection for Terraminer Gold's path board."""

from dataclasses import dataclass
import time

import cv2
import numpy as np

from game_automation.core.events import PerceptionState
from game_automation.core.perception import BasePerception

PORT_BITS = (1, 2, 4, 8)
PORT_CONFIDENCE_THRESHOLD = 0.75


class BoardNotDetectedError(ValueError):
    """Raised when a frame does not yet contain a recognizable game board."""


@dataclass(frozen=True)
class TerraminerBoard:
    tile_masks: list[list[int]]
    start: tuple[int, int]
    goal: tuple[int, int]
    bounds: tuple[int, int, int, int]
    tile_port_confidence: list[list[tuple[float, float, float, float]]] | None = None

    @property
    def size(self) -> int:
        return len(self.tile_masks)

    def center(self, row: int, column: int) -> tuple[int, int]:
        left, top, right, bottom = self.bounds
        step_x = (right - left) / self.size
        step_y = (bottom - top) / self.size
        return round(left + (column + 0.5) * step_x), round(top + (row + 0.5) * step_y)


class TerraminerGoldPerception(BasePerception):
    """Detect board bounds, grid dimensions, endpoint markers, and tile ports."""

    def __init__(
        self,
        bus,
        grid_size: int | None = None,
        start: tuple[int, int] | None = None,
        goal: tuple[int, int] | None = None,
    ):
        super().__init__(bus)
        if grid_size is not None and grid_size < 2:
            raise ValueError("Grid size must be at least 2.")
        self.grid_size = grid_size
        self.start = start
        self.goal = goal

    def process(self, frame: np.ndarray) -> PerceptionState:
        if frame is None or frame.ndim != 3:
            raise ValueError("Terraminer perception requires a BGR image frame.")
        bounds = self._detect_board_bounds(frame)
        x1, y1, x2, y2 = bounds
        size = self.grid_size or self._estimate_grid_size(frame[y1:y2, x1:x2])
        tile_masks, tile_port_confidence = self._detect_tile_masks(frame, bounds, size)
        start = self.start or self._detect_start(frame, bounds, size)
        goal = self.goal or self._detect_goal(frame, bounds, size)
        if start == goal:
            raise ValueError("Detected start and goal on the same tile; configure their coordinates explicitly.")

        board = TerraminerBoard(tile_masks, start, goal, bounds, tile_port_confidence)
        state = PerceptionState(
            timestamp=time.time(),
            data={
                "board": board,
                "grid_size": size,
                "start": start,
                "goal": goal,
                "frame_size": (frame.shape[1], frame.shape[0]),
            },
        )
        self.bus.publish(state)
        return state

    @staticmethod
    def _detect_board_bounds(frame: np.ndarray) -> tuple[int, int, int, int]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        border_mask = cv2.inRange(
            hsv,
            np.array((12, 100, 120), dtype=np.uint8),
            np.array((38, 255, 255), dtype=np.uint8),
        )
        contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = frame.shape[0] * frame.shape[1]
        candidates = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height
            ratio = width / max(height, 1)
            if area >= frame_area * 0.08 and 0.8 <= ratio <= 1.25:
                candidates.append((area, (x, y, x + width, y + height)))
        if not candidates:
            raise BoardNotDetectedError("Could not locate the square board border in the frame.")
        return max(candidates, key=lambda candidate: candidate[0])[1]

    @staticmethod
    def _estimate_grid_size(board_image: np.ndarray) -> int:
        if board_image.size == 0:
            raise ValueError("Detected board region is empty.")
        gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        vertical_strength = np.mean(np.abs(grad_x), axis=0)
        horizontal_strength = np.mean(np.abs(grad_y), axis=1)
        scores = {}
        for count in range(3, 13):
            xs = [round(i * (len(vertical_strength) - 1) / count) for i in range(1, count)]
            ys = [round(i * (len(horizontal_strength) - 1) / count) for i in range(1, count)]
            scores[count] = (
                sum(vertical_strength[x] for x in xs) / len(xs)
                + sum(horizontal_strength[y] for y in ys) / len(ys)
            ) / 2
        return max(scores, key=scores.get)

    @staticmethod
    def _detect_tile_masks(
        frame: np.ndarray,
        bounds: tuple[int, int, int, int],
        size: int,
    ) -> tuple[
        list[list[int]],
        list[list[tuple[float, float, float, float]]],
    ]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_and_value = cv2.inRange(
            hsv,
            np.array((17, 75, 185), dtype=np.uint8),
            np.array((27, 245, 255), dtype=np.uint8),
        )
        bright_gold_mask = cv2.inRange(
            hsv,
            np.array((17, 75, 225), dtype=np.uint8),
            np.array((27, 255, 255), dtype=np.uint8),
        )
        dark_gold_mask = cv2.inRange(
            hsv,
            np.array((17, 245, 160), dtype=np.uint8),
            np.array((23, 255, 224), dtype=np.uint8),
        )
        gold_mask = cv2.bitwise_or(hue_and_value, bright_gold_mask)
        gold_mask = cv2.bitwise_or(gold_mask, dark_gold_mask)
        left, top, right, bottom = bounds
        cell_w = (right - left) / size
        cell_h = (bottom - top) / size
        opening_kernel = np.ones((3, 3), dtype=np.uint8)
        closing_kernel = np.ones((9, 9), dtype=np.uint8)
        masks = []
        confidence_grid = []
        for row in range(size):
            tile_row = []
            confidence_row = []
            for column in range(size):
                x1 = round(left + column * cell_w)
                x2 = round(left + (column + 1) * cell_w)
                y1 = round(top + row * cell_h)
                y2 = round(top + (row + 1) * cell_h)
                tile_mask = cv2.morphologyEx(
                    gold_mask[y1:y2, x1:x2],
                    cv2.MORPH_OPEN,
                    opening_kernel,
                )
                tile_mask = cv2.morphologyEx(
                    tile_mask,
                    cv2.MORPH_CLOSE,
                    closing_kernel,
                )
                gray_tile = cv2.cvtColor(
                    frame[y1:y2, x1:x2],
                    cv2.COLOR_BGR2GRAY,
                )
                gradient_x = cv2.Sobel(gray_tile, cv2.CV_32F, 1, 0, ksize=3)
                gradient_y = cv2.Sobel(gray_tile, cv2.CV_32F, 0, 1, ksize=3)
                edge_features = (
                    gradient_x,
                    gradient_y,
                    cv2.magnitude(gradient_x, gradient_y),
                    cv2.Canny(gray_tile, 40, 120) > 0,
                )
                tile_height, tile_width = tile_mask.shape
                core_bounds = (
                    round(tile_height * 0.4),
                    round(tile_height * 0.6),
                    round(tile_width * 0.4),
                    round(tile_width * 0.6),
                )
                core_y1, core_y2, core_x1, core_x2 = core_bounds
                core = tile_mask[core_y1:core_y2, core_x1:core_x2]
                component = np.zeros_like(tile_mask)
                if core.size and np.any(core):
                    _, labels, _, _ = cv2.connectedComponentsWithStats(
                        tile_mask,
                        connectivity=8,
                    )
                    core_labels, core_counts = np.unique(
                        labels[core_y1:core_y2, core_x1:core_x2],
                        return_counts=True,
                    )
                    seed_labels = [
                        (int(label), int(pixel_count))
                        for label, pixel_count in zip(core_labels, core_counts)
                        if label != 0
                    ]
                    if seed_labels:
                        component_label = max(seed_labels, key=lambda item: item[1])[0]
                        component[labels == component_label] = 255

                mask = 0
                confidences = []
                for side, bit in enumerate(PORT_BITS):
                    confidence = TerraminerGoldPerception._connected_port_confidence(
                        component,
                        side,
                        edge_features,
                    )
                    confidences.append(confidence)
                    if confidence >= PORT_CONFIDENCE_THRESHOLD:
                        mask |= bit
                tile_row.append(mask)
                confidence_row.append(tuple(confidences))
            masks.append(tile_row)
            confidence_grid.append(confidence_row)
        return masks, confidence_grid

    @staticmethod
    def _connected_port_confidence(
        component: np.ndarray,
        side: int,
        edge_features: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> float:
        height, width = component.shape
        cross_size = width if side in (0, 2) else height
        cross_start = round(cross_size * 0.39)
        cross_end = round(cross_size * 0.61)
        longitudinal_size = height if side in (0, 2) else width
        bin_edges = np.linspace(
            round(longitudinal_size * 0.12),
            round(longitudinal_size * 0.48),
            9,
            dtype=int,
        )
        gradient_x, gradient_y, edge_magnitude, edges = edge_features
        if side == 0:
            oriented_component = component
            boundary_edges = (
                (edge_magnitude >= 60)
                & (np.abs(gradient_x) >= 1.5 * np.abs(gradient_y))
                & edges
            )
        elif side == 1:
            oriented_component = np.fliplr(component).T
            boundary_edges = np.fliplr(
                (edge_magnitude >= 60)
                & (np.abs(gradient_y) >= 1.5 * np.abs(gradient_x))
                & edges
            ).T
        elif side == 2:
            oriented_component = np.flipud(component)
            boundary_edges = np.flipud(
                (edge_magnitude >= 60)
                & (np.abs(gradient_x) >= 1.5 * np.abs(gradient_y))
                & edges
            )
        else:
            oriented_component = component.T
            boundary_edges = (
                (edge_magnitude >= 60)
                & (np.abs(gradient_y) >= 1.5 * np.abs(gradient_x))
                & edges
            ).T

        bin_scores = []
        for index in range(8):
            start, end = bin_edges[index], bin_edges[index + 1]
            component_slice = oriented_component[start:end, :]
            color_region = component_slice[:, cross_start:cross_end]
            left_flank = component_slice[
                :,
                round(cross_size * 0.05):round(cross_size * 0.22),
            ]
            right_flank = component_slice[
                :,
                round(cross_size * 0.78):round(cross_size * 0.95),
            ]
            center_coverage = float(np.mean(color_region > 0)) if color_region.size else 0.0
            flank_coverage = max(
                float(np.mean(left_flank > 0)) if left_flank.size else 1.0,
                float(np.mean(right_flank > 0)) if right_flank.size else 1.0,
            )

            edge_slice = boundary_edges[start:end, :]
            first_band = edge_slice[
                :,
                round(cross_size * 0.25):round(cross_size * 0.43),
            ]
            second_band = edge_slice[
                :,
                round(cross_size * 0.57):round(cross_size * 0.75),
            ]
            paired_edge_coverage = float(
                np.mean(
                    np.any(first_band, axis=1)
                    & np.any(second_band, axis=1)
                )
            )
            center_support = float(np.clip(center_coverage / 0.3, 0.0, 1.0))
            shape_support = float(1.0 - np.clip(flank_coverage / 0.3, 0.0, 1.0))
            edge_support = float(np.clip(paired_edge_coverage / 0.35, 0.0, 1.0))
            bin_scores.append(
                center_support * shape_support * (0.7 + 0.3 * edge_support)
            )
        return max(
            sum(bin_scores[start:start + 4]) / 4
            for start in (0, 1)
        )

    @staticmethod
    def _cell_regions(
        frame: np.ndarray,
        bounds: tuple[int, int, int, int],
        size: int,
    ):
        left, top, right, bottom = bounds
        cell_w = (right - left) / size
        cell_h = (bottom - top) / size
        for row in range(size):
            for column in range(size):
                x1 = round(left + (column + 0.2) * cell_w)
                x2 = round(left + (column + 0.8) * cell_w)
                y1 = round(top + (row + 0.2) * cell_h)
                y2 = round(top + (row + 0.8) * cell_h)
                yield (row, column), frame[y1:y2, x1:x2]

    @classmethod
    def _detect_goal(cls, frame, bounds, size) -> tuple[int, int]:
        best_cell = None
        best_score = 0.0
        for cell, region in cls._cell_regions(frame, bounds, size):
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            dark_fraction = np.mean(hsv[:, :, 2] < 65)
            gold = (
                (hsv[:, :, 0] >= 12)
                & (hsv[:, :, 0] <= 38)
                & (hsv[:, :, 1] >= 90)
                & (hsv[:, :, 2] >= 120)
            )
            score = dark_fraction + 0.35 * np.mean(gold)
            if score > best_score:
                best_score, best_cell = score, cell
        if best_cell is None or best_score < 0.12:
            raise BoardNotDetectedError(
                "Could not identify the gold pot; pass goal=(row, column) to perception."
            )
        return best_cell

    @classmethod
    def _detect_start(cls, frame, bounds, size) -> tuple[int, int]:
        best_cell = None
        best_score = 0.0
        for cell, region in cls._cell_regions(frame, bounds, size):
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            skin_or_clothing = (
                (hsv[:, :, 0] >= 3)
                & (hsv[:, :, 0] <= 18)
                & (hsv[:, :, 1] >= 60)
                & (hsv[:, :, 2] >= 65)
            )
            score = float(np.mean(skin_or_clothing))
            if score > best_score:
                best_score, best_cell = score, cell
        if best_cell is None or best_score < 0.04:
            raise BoardNotDetectedError(
                "Could not identify the miner start; pass start=(row, column) to perception."
            )
        return best_cell
