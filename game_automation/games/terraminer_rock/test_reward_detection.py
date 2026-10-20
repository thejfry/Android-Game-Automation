import unittest

import numpy as np

from .reward_detection import RewardMatchKind, RewardScreenDetector


class RockRewardDetectionTests(unittest.TestCase):
    def test_detects_standard_play_again_button(self):
        frame = np.zeros((918, 448, 3), dtype=np.uint8)
        frame[506:554, 102:244] = (0, 215, 255)

        match = RewardScreenDetector.detect(frame)

        self.assertIsNotNone(match)
        self.assertEqual(match.kind, RewardMatchKind.PLAY_AGAIN)
        self.assertEqual(match.button_bounds, (102, 506, 142, 48))

    def test_detects_level_up_prompt_without_exposing_a_tappable_button(self):
        frame = np.zeros((918, 448, 3), dtype=np.uint8)
        frame[442:522, 118:331] = (0, 255, 0)
        frame[480:514, 160:290] = (0, 145, 255)

        match = RewardScreenDetector.detect(frame)

        self.assertIsNotNone(match)
        self.assertEqual(match.kind, RewardMatchKind.LEVEL_UP_PANEL)
        self.assertIsNone(match.button_bounds)

    def test_ignores_regular_board_frame(self):
        frame = np.zeros((918, 448, 3), dtype=np.uint8)

        self.assertIsNone(RewardScreenDetector.detect(frame))


if __name__ == "__main__":
    unittest.main()
