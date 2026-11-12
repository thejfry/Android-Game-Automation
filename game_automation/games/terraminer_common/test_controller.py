import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from .controller import ControllerState, TerraminerController


class SharedControllerTests(unittest.TestCase):
    def setUp(self):
        self.agent = Mock()
        self.perception = Mock()
        self.tap_executor = Mock()
        self.progress = Mock()
        self.controller = TerraminerController(
            self.agent,
            self.perception,
            self.tap_executor,
            continuous=True,
            reward_button_wait_seconds=0.5,
            level_progress_callback=self.progress,
        )
        self.match = SimpleNamespace(
            kind=SimpleNamespace(value="play_again"),
            button_bounds=(10, 20, 40, 30),
        )

    def test_confirms_reward_then_advances_with_the_play_again_button(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch(
            "game_automation.games.terraminer_common.controller.time.monotonic",
            return_value=10,
        ):
            self.assertTrue(
                self.controller._process_reward_frame(self.match, frame.shape[:2])
            )
            self.assertTrue(
                self.controller._process_reward_frame(self.match, frame.shape[:2])
            )

        self.assertEqual(self.controller.state, ControllerState.ADVANCING_LEVEL)
        self.tap_executor.tap_bounds.assert_called_once_with(
            (10, 20, 40, 30),
            (100, 100),
            description="confirmed Play Again button",
        )

    def test_reports_level_start_and_completion(self):
        self.controller._begin_level_progress()
        self.controller._report_level_progress("completed")

        self.assertEqual(self.controller.level_number, 1)
        self.progress.assert_any_call(1, "started")
        self.progress.assert_any_call(1, "completed")


if __name__ == "__main__":
    unittest.main()
