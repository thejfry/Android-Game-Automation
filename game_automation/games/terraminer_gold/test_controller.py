import unittest
from unittest.mock import Mock, patch

import numpy as np

from game_automation.core.events import PerceptionState

from .controller import ControllerState, TerraminerGoldController


class LifecycleControllerTests(unittest.TestCase):
    def setUp(self):
        self.agent = Mock()
        self.agent.dry_run = False
        self.agent.finished = False
        self.agent._last_solution = object()
        self.agent._is_path_connected.return_value = False
        self.agent.tick.side_effect = (
            lambda state, before_execution=None: before_execution()
            if before_execution is not None
            else None
        )
        self.perception = Mock()
        self.perception.process.return_value = PerceptionState(
            timestamp=0,
            data={"board": object()},
        )
        self.controller = TerraminerGoldController(
            self.agent,
            self.perception,
            settling_seconds=1,
        )
        self.frame = np.zeros((200, 200, 3), dtype=np.uint8)

    def test_executed_plan_is_not_replanned_while_waiting_for_completion(self):
        with patch(
            "game_automation.games.terraminer_gold.controller.time.monotonic",
            side_effect=(0.0, 2.0),
        ):
            self.controller.process_frame(self.frame)
            self.assertEqual(self.controller.state, ControllerState.SETTLING)
            self.controller.process_frame(self.frame)
            self.assertEqual(self.controller.state, ControllerState.VERIFYING)
            self.controller.process_frame(self.frame)
            self.assertEqual(
                self.controller.state,
                ControllerState.WAITING_FOR_COMPLETION,
            )
            self.controller.process_frame(self.frame)

        self.agent.tick.assert_called_once()
        self.agent._is_path_connected.assert_called_once()
        self.assertEqual(self.perception.process.call_count, 2)

    def test_reward_screen_stops_controller_before_board_perception(self):
        frame = np.zeros((720, 324, 3), dtype=np.uint8)
        frame[670:710, 90:230] = (0, 255, 255)

        self.controller.process_frame(frame)
        self.assertEqual(
            self.controller.state,
            ControllerState.WAITING_FOR_BOARD,
        )
        self.perception.process.assert_not_called()
        self.controller.process_frame(frame)

        self.assertEqual(self.controller.state, ControllerState.REWARD_SCREEN)
        self.assertTrue(self.controller.finished)
        self.perception.process.assert_not_called()
        self.agent.tick.assert_not_called()

    def test_gameplay_frame_is_not_misclassified_as_reward_screen(self):
        self.assertFalse(TerraminerGoldController._is_reward_screen(self.frame))

    def test_reward_button_with_text_cutouts_is_detected_near_bottom(self):
        frame = np.zeros((720, 324, 3), dtype=np.uint8)
        frame[662:710, 96:232] = (0, 215, 255)
        frame[683:686, 121:208] = (0, 0, 0)

        self.assertTrue(TerraminerGoldController._is_reward_screen(frame))

    def test_wide_yellow_game_board_tile_above_reward_area_is_not_detected(self):
        frame = np.zeros((720, 324, 3), dtype=np.uint8)
        frame[500:535, 80:250] = (0, 215, 255)

        self.assertFalse(TerraminerGoldController._is_reward_screen(frame))


if __name__ == "__main__":
    unittest.main()
