"""Computer-vision perception for Terraminer's rock memory game."""

from dataclasses import dataclass
import time

import cv2
import numpy as np


class BoardNotDetectedError(ValueError):
    """Raised when a frame does not contain a stable card grid."""


@dataclass(frozen=True)
class CardObservation:
    row: int
    column: int
    bounds: tuple[int, int, int, int]
    face_up: bool
    matched: bool
    image: np.ndarray | None

    @property
    def position(self) -> tuple[int, int]:
        return self.row, self.column

    @property
    def center(self) -> tuple[float, float]:
        x, y, width, height = self.bounds
        return x + width / 2, y + height / 2


@dataclass(frozen=True)
class RockBoard:
    cards: tuple[CardObservation, ...]
    rows: int
    columns: int
    frame_size: tuple[int, int]

    @property
    def complete(self) -> bool:
        return bool(self.cards) and all(card.matched for card in self.cards)

    @property
    def signature(self) -> tuple[int, int]:
        return self.rows, self.columns


@dataclass(frozen=True)
class BoardObservation:
    timestamp: float
    board: RockBoard


class TerraminerRockPerception:
    """Detect card rectangles and classify their face-up/matched state."""

    def process(self, frame: np.ndarray) -> BoardObservation:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Rock-game perception requires a BGR image frame.")

        boxes = CardGridDetector.detect(frame)
        centers = [((x + width / 2), (y + height / 2)) for x, y, width, height in boxes]
        widths = [box[2] for box in boxes]
        heights = [box[3] for box in boxes]
        cell_width = float(np.median(widths))
        cell_height = float(np.median(heights))
        x_centers = _cluster_coordinates([center[0] for center in centers], cell_width * 0.45)
        y_centers = _cluster_coordinates([center[1] for center in centers], cell_height * 0.45)
        if len(x_centers) < 2 or len(y_centers) < 2:
            raise BoardNotDetectedError("A rectangular card grid was not detected.")

        cells: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        for box, (center_x, center_y) in zip(boxes, centers):
            column = _nearest_index(x_centers, center_x)
            row = _nearest_index(y_centers, center_y)
            key = row, column
            previous = cells.get(key)
            if previous is None or box[2] * box[3] > previous[2] * previous[3]:
                cells[key] = box

        rows, columns = len(y_centers), len(x_centers)
        if len(cells) != rows * columns or rows * columns < 4:
            raise BoardNotDetectedError(
                f"Incomplete card grid: detected {len(cells)} of {rows * columns} cells."
            )
        if rows * columns % 2:
            raise BoardNotDetectedError("The detected card count is not an even number.")

        cards = []
        for row in range(rows):
            for column in range(columns):
                x, y, width, height = cells[row, column]
                card_image = frame[y:y + height, x:x + width]
                face_up = _is_face_up(card_image)
                matched = face_up and _has_green_match_border(card_image)
                image = _card_image_signature(card_image) if face_up else None
                cards.append(
                    CardObservation(
                        row,
                        column,
                        (x, y, width, height),
                        face_up,
                        matched,
                        image,
                    )
                )

        return BoardObservation(
            timestamp=time.time(),
            board=RockBoard(
                tuple(cards),
                rows,
                columns,
                (frame.shape[1], frame.shape[0]),
            ),
        )


class CardGridDetector:
    """Find the repeated, near-square card outlines on the game board."""

    @staticmethod
    def detect(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 35, 110)
        edges = cv2.dilate(
            edges,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        min_side = max(24, round(min(frame.shape[:2]) * 0.045))
        candidates = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            ratio = width / max(height, 1)
            if (
                width < min_side
                or height < min_side
                or width > frame.shape[1] * 0.35
                or height > frame.shape[0] * 0.2
                or not 0.72 <= ratio <= 1.28
            ):
                continue
            candidates.append((x, y, width, height))

        # Edge maps produce several nested contours per card. Keep the largest
        # rectangle for each overlapping group, then use their centers as cells.
        selected: list[tuple[int, int, int, int]] = []
        for box in sorted(candidates, key=lambda item: item[2] * item[3], reverse=True):
            if not any(_overlap_ratio(box, other) >= 0.45 for other in selected):
                selected.append(box)
        if len(selected) < 4:
            raise BoardNotDetectedError("Fewer than four card outlines were detected.")
        return selected


def _cluster_coordinates(values: list[float], tolerance: float) -> list[float]:
    groups: list[list[float]] = []
    for value in sorted(values):
        if groups and value - float(np.mean(groups[-1])) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [float(np.mean(group)) for group in groups]


def _nearest_index(values: list[float], value: float) -> int:
    return min(range(len(values)), key=lambda index: abs(values[index] - value))


def _overlap_ratio(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    x1, y1, w1, h1 = first
    x2, y2, w2, h2 = second
    intersection_width = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    intersection_height = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection = intersection_width * intersection_height
    return intersection / max(min(w1 * h1, w2 * h2), 1)


def _is_face_up(card: np.ndarray) -> bool:
    height, width = card.shape[:2]
    inner = card[
        round(height * 0.2):round(height * 0.8),
        round(width * 0.2):round(width * 0.8),
    ]
    return float(np.mean(cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY))) > 125


def _has_green_match_border(card: np.ndarray) -> bool:
    height, width = card.shape[:2]
    border = np.concatenate(
        (
            card[:max(1, round(height * 0.12)), :, :].reshape(-1, 3),
            card[-max(1, round(height * 0.12)):, :, :].reshape(-1, 3),
            card[:, :max(1, round(width * 0.12)), :].reshape(-1, 3),
            card[:, -max(1, round(width * 0.12)):, :].reshape(-1, 3),
        )
    )
    hsv = cv2.cvtColor(border.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
    green = cv2.inRange(
        hsv,
        np.array((35, 65, 65), dtype=np.uint8),
        np.array((95, 255, 255), dtype=np.uint8),
    )
    return float(np.mean(green > 0)) >= 0.12


def _card_image_signature(card: np.ndarray) -> np.ndarray:
    height, width = card.shape[:2]
    inner = card[
        round(height * 0.14):round(height * 0.86),
        round(width * 0.14):round(width * 0.86),
    ]
    normalized = cv2.resize(inner, (48, 48), interpolation=cv2.INTER_AREA)
    edge = max(2, round(min(normalized.shape[:2]) * 0.12))
    background_ring = np.concatenate(
        (
            normalized[:edge, :, :].reshape(-1, 3),
            normalized[-edge:, :, :].reshape(-1, 3),
            normalized[:, :edge, :].reshape(-1, 3),
            normalized[:, -edge:, :].reshape(-1, 3),
        )
    )
    background = np.median(background_ring, axis=0)
    foreground_distance = np.linalg.norm(
        normalized.astype(np.float32) - background.astype(np.float32),
        axis=2,
    )
    foreground = (foreground_distance >= 30).astype(np.uint8) * 255
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    image = cv2.resize(normalized, (32, 32), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(foreground, (32, 32), interpolation=cv2.INTER_AREA)
    return np.dstack((image.astype(np.float32), mask.astype(np.float32)))
