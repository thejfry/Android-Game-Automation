import unittest
from unittest.mock import Mock, patch

from .tap_executor import TapExecutor


class TapExecutorTests(unittest.TestCase):
    def setUp(self):
        self.phone = Mock()
        self.phone.resolution = (648, 1440)
        self.executor = TapExecutor(self.phone)

    def test_tap_bounds_jitters_within_detected_button_before_scaling(self):
        with (
            patch(
                "game_automation.games.terraminer_gold.tap_executor.random.randint",
                side_effect=(120, 205),
            ),
        ):
            position = self.executor.tap_bounds(
                (100, 200, 40, 20),
                (324, 720),
                description="test button",
            )

        self.assertEqual(position, (240, 410))
        self.phone.tap.assert_called_once_with(240, 410)

    def test_tap_bounds_rejects_coordinates_outside_the_frame(self):
        with self.assertRaises(ValueError):
            self.executor.tap_bounds(
                (300, 200, 40, 20),
                (324, 720),
                description="invalid button",
            )
        self.phone.tap.assert_not_called()


if __name__ == "__main__":
    unittest.main()
