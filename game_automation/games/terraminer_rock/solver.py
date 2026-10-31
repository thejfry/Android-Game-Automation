"""Online memory for selecting safe and useful card flips."""

from typing import TypeAlias

import numpy as np

from .perception import CardObservation, RockBoard

CardPosition: TypeAlias = tuple[int, int]
IMAGE_MATCH_THRESHOLD = 20.0


class MemorySolver:
    """Remember revealed images, then take known pairs before exploring."""

    def __init__(self, image_match_threshold: float = IMAGE_MATCH_THRESHOLD):
        if image_match_threshold <= 0:
            raise ValueError("Image-match threshold must be positive.")
        self.image_match_threshold = image_match_threshold
        self._images: dict[CardPosition, np.ndarray] = {}
        self._matched: set[CardPosition] = set()
        self._non_matching_pairs: set[frozenset[CardPosition]] = set()
        self._pending_pair: frozenset[CardPosition] | None = None

    def reset(self) -> None:
        self._images.clear()
        self._matched.clear()
        self._non_matching_pairs.clear()
        self._pending_pair = None

    def next_card(self, board: RockBoard) -> CardPosition | None:
        if board.rows < 2 or board.columns < 2:
            raise ValueError("A memory board must have at least two rows and columns.")
        cards = {card.position: card for card in board.cards}
        expected = board.rows * board.columns
        if len(cards) != expected:
            raise ValueError("The board must contain one observation per grid cell.")

        for card in board.cards:
            if card.matched:
                self._matched.add(card.position)
            if card.face_up and card.image is not None:
                self._images[card.position] = card.image.copy()
        self._resolve_pending_pair(cards)

        unresolved_open = [
            card for card in board.cards
            if card.face_up and not card.matched
        ]
        if len(unresolved_open) >= 2:
            if all(not card.matched for card in unresolved_open[:2]):
                self._pending_pair = frozenset(
                    (unresolved_open[0].position, unresolved_open[1].position)
                )
            return None
        if len(unresolved_open) == 1:
            first = unresolved_open[0]
            if first.image is None:
                raise ValueError("A face-up card is missing its image signature.")
            self._images[first.position] = first.image.copy()
            remaining = self._unmatched_cards(board)
            if len(remaining) == 2:
                return next(card.position for card in remaining if card.position != first.position)
            partner = self._known_partner(first.position, first.image)
            if partner is not None:
                return partner
            unknown = self._first_unknown(cards)
            if unknown is None:
                raise RuntimeError(
                    "A card is face-up, but no matching known card or unknown card "
                    "remains. Check the image-match threshold and card perception."
                )
            return unknown

        remaining = self._unmatched_cards(board)
        if len(remaining) == 2:
            return remaining[0].position

        known_pair = self._first_known_pair()
        if known_pair is not None:
            return known_pair

        unknown = self._first_unknown(cards)
        if unknown is not None:
            return unknown

        if len(self._matched) == expected:
            return None
        raise RuntimeError(
            "No unflipped card or known pair remains, but the board is incomplete."
        )

    @staticmethod
    def _unmatched_cards(board: RockBoard) -> list[CardObservation]:
        return [card for card in board.cards if not card.matched]

    def _first_known_pair(self) -> CardPosition | None:
        for position in sorted(self._images):
            if position in self._matched:
                continue
            partner = self._known_partner(position, self._images[position])
            if (
                partner is not None
                and frozenset((position, partner)) not in self._non_matching_pairs
            ):
                return position
        return None

    def _first_unknown(
        self,
        cards: dict[CardPosition, CardObservation],
    ) -> CardPosition | None:
        for position in sorted(cards):
            card = cards[position]
            if (
                position not in self._matched
                and not card.face_up
                and position not in self._images
            ):
                return position
        return None

    def _known_partner(
        self,
        position: CardPosition,
        image: np.ndarray,
    ) -> CardPosition | None:
        candidates = [
            (self._image_distance(image, other_image), other_position)
            for other_position, other_image in self._images.items()
            if other_position != position and other_position not in self._matched
            and frozenset((position, other_position)) not in self._non_matching_pairs
        ]
        if not candidates:
            return None
        distance, partner = min(candidates, key=lambda item: item[0])
        return partner if distance <= self.image_match_threshold else None

    @staticmethod
    def _image_distance(first: np.ndarray, second: np.ndarray) -> float:
        if first.shape != second.shape:
            raise ValueError("Card image signatures must have equal dimensions.")
        if first.ndim != 3 or first.shape[2] != 4:
            raise ValueError("Card signatures must contain BGR pixels and a foreground mask.")
        first_mask = first[:, :, 3] >= 128
        second_mask = second[:, :, 3] >= 128
        union = np.count_nonzero(first_mask | second_mask)
        if union == 0:
            return float("inf")
        intersection_mask = first_mask & second_mask
        intersection = np.count_nonzero(intersection_mask)
        shape_distance = 1 - intersection / union
        if intersection:
            color_distance = float(
                np.mean(
                    np.abs(
                        first[:, :, :3][intersection_mask]
                        - second[:, :, :3][intersection_mask]
                    )
                )
                / 255
            )
        else:
            color_distance = 1.0
        return 100 * (0.75 * shape_distance + 0.25 * color_distance)

    def _resolve_pending_pair(
        self,
        cards: dict[CardPosition, CardObservation],
    ) -> None:
        if self._pending_pair is None:
            return
        pending_cards = [cards[position] for position in self._pending_pair]
        if any(card.matched for card in pending_cards):
            self._matched.update(self._pending_pair)
            self._pending_pair = None
        elif all(not card.face_up for card in pending_cards):
            self._non_matching_pairs.add(self._pending_pair)
            self._pending_pair = None
