import unittest

import cv2
import numpy as np

from .perception import (
    BoardNotDetectedError,
    TerraminerRockPerception,
    _card_image_signature,
)
from .solver import MemorySolver


class RockPerceptionTests(unittest.TestCase):
    def _frame(self, rows=3, columns=4):
        frame = np.full((500, 600, 3), (35, 43, 57), dtype=np.uint8)
        for row in range(rows):
            for column in range(columns):
                x = 60 + column * 105
                y = 70 + row * 105
                cv2.rectangle(frame, (x, y), (x + 82, y + 82), (75, 59, 40), -1)
                cv2.rectangle(frame, (x, y), (x + 82, y + 82), (101, 81, 55), 4)
                cv2.putText(
                    frame,
                    "?",
                    (x + 27, y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (88, 71, 50),
                    3,
                )
        return frame

    def test_detects_dynamic_three_by_four_backed_grid(self):
        observation = TerraminerRockPerception().process(self._frame())

        self.assertEqual((observation.board.rows, observation.board.columns), (3, 4))
        self.assertEqual(len(observation.board.cards), 12)
        self.assertFalse(any(card.face_up for card in observation.board.cards))

    def test_detects_larger_four_by_six_grid(self):
        observation = TerraminerRockPerception().process(self._frame(4, 6))

        self.assertEqual((observation.board.rows, observation.board.columns), (4, 6))
        self.assertEqual(len(observation.board.cards), 24)

    def test_rejects_frame_without_a_card_grid(self):
        frame = np.zeros((500, 600, 3), dtype=np.uint8)

        with self.assertRaises(BoardNotDetectedError):
            TerraminerRockPerception().process(frame)

    def test_image_identity_ignores_shared_card_background(self):
        first = np.full((80, 80, 3), (240, 245, 244), dtype=np.uint8)
        second = first.copy()
        matching = first.copy()
        cv2.rectangle(first, (26, 29), (54, 50), (25, 35, 45), -1)
        cv2.circle(second, (40, 40), 13, (25, 35, 45), -1)
        cv2.rectangle(matching, (26, 29), (54, 50), (25, 35, 45), -1)

        first_signature = _card_image_signature(first)
        second_signature = _card_image_signature(second)
        matching_signature = _card_image_signature(matching)

        self.assertLess(
            MemorySolver._image_distance(first_signature, matching_signature),
            20,
        )
        self.assertGreater(
            MemorySolver._image_distance(first_signature, second_signature),
            20,
        )


if __name__ == "__main__":
    unittest.main()
