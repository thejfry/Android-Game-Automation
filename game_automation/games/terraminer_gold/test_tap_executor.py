import unittest
from unittest.mock import Mock, patch

from game_automation.core.tap_executor import TapExecutor


class TapExecutorTests(unittest.TestCase):
    def setUp(self):
        self.phone = Mock()
        self.phone.resolution = (648, 1440)
        self.executor = TapExecutor(self.phone)

    def test_tap_bounds_jitters_within_detected_button_before_scaling(self):
        with (
            patch(
                "game_automation.core.tap_executor.random.randint",
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

    def test_device_point_uses_bounded_position_jitter(self):
        with patch(
            "game_automation.core.tap_executor.random.uniform",
            side_effect=(3.0, -4.0),
        ):
            position = self.executor.tap_device_point((100, 200), (5, 6))

        self.assertEqual(position, (103, 196))
        self.phone.tap.assert_called_once_with(103, 196)

    def test_timing_jitter_can_be_disabled(self):
        executor = TapExecutor(self.phone, timing_jitter_fraction=0)
        with patch(
            "game_automation.core.tap_executor.random.uniform",
            return_value=0.25,
        ) as uniform:
            duration = executor.randomized_interval(0.25)

        self.assertEqual(duration, 0.25)
        uniform.assert_called_once_with(0.25, 0.25)


if __name__ == "__main__":
    unittest.main()
