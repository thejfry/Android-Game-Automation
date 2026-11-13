import unittest
from unittest.mock import Mock, patch

import numpy as np

from .controller import ControllerState, TerraminerRockController
from .reward_detection import RewardMatch, RewardMatchKind


class RockSharedLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.agent = Mock()
        self.agent.tap_count = 0
        self.perception = Mock()
        self.tap_executor = Mock()
        self.controller = TerraminerRockController(
            self.agent,
            self.perception,
            self.tap_executor,
            continuous=True,
        )
        self.frame = np.zeros((900, 448, 3), dtype=np.uint8)

    def test_play_again_uses_shared_bounded_tap_and_transition(self):
        match = RewardMatch(
            RewardMatchKind.PLAY_AGAIN,
            button_bounds=(100, 500, 40, 40),
        )
        with patch(
            "game_automation.games.terraminer_rock.controller.RewardScreenDetector.detect",
            return_value=match,
        ), patch(
            "game_automation.games.terraminer_rock.controller.time.monotonic",
            return_value=10,
        ):
            self.controller.process_frame(self.frame)
            self.controller.process_frame(self.frame)

        self.assertEqual(self.controller.state, ControllerState.ADVANCING_LEVEL)
        self.tap_executor.tap_bounds.assert_called_once_with(
            (100, 500, 40, 40),
            (448, 900),
            description="confirmed Rock Play Again button",
        )

    def test_level_up_screen_starts_automatic_board_polling(self):
        match = RewardMatch(RewardMatchKind.LEVEL_UP_PANEL)
        with patch(
            "game_automation.games.terraminer_rock.controller.RewardScreenDetector.detect",
            return_value=match,
        ), patch(
            "game_automation.games.terraminer_rock.controller.time.monotonic",
            return_value=10,
        ):
            self.controller.process_frame(self.frame)
            self.controller.process_frame(self.frame)

        self.assertEqual(self.controller.state, ControllerState.AWAITING_NEXT_LEVEL)
        self.assertEqual(self.controller._next_level_wait_started_at, 10)
        self.tap_executor.tap_bounds.assert_not_called()


if __name__ == "__main__":
    unittest.main()
