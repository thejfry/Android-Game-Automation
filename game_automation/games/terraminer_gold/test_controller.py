import unittest
from unittest.mock import Mock, patch

import numpy as np

from game_automation.core.events import PerceptionState

from .controller import ControllerState, TerraminerGoldController
from .perception import TerraminerBoard


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
            data={
                "board": TerraminerBoard(
                    tile_masks=[[0, 0], [0, 0]],
                    start=(0, 0),
                    goal=(1, 1),
                    bounds=(0, 0, 200, 200),
                )
            },
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
            self.controller.process_frame(self.frame)
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

    def test_continuous_mode_waits_for_ack_then_stable_next_board(self):
        controller = TerraminerGoldController(
            self.agent,
            self.perception,
            settling_seconds=0,
            continuous=True,
        )
        reward = np.zeros((900, 448, 3), dtype=np.uint8)
        reward[536:608, 93:355] = (0, 215, 255)
        reward[570:596, 110:338] = (0, 145, 255)

        controller.process_frame(reward)
        controller.process_frame(reward)
        self.assertEqual(controller.state, ControllerState.AWAITING_NEXT_LEVEL)
        self.assertFalse(controller.finished)
        self.agent.phone.tap.assert_not_called()
        self.perception.process.assert_not_called()

        controller.process_frame(self.frame)
        self.assertEqual(controller.state, ControllerState.AWAITING_NEXT_LEVEL)
        controller.acknowledge_next_level()
        self.assertEqual(controller.state, ControllerState.WAITING_FOR_BOARD)
        self.agent.reset_for_next_level.assert_called_once()

        controller.process_frame(self.frame)
        controller.process_frame(self.frame)
        self.assertEqual(controller.state, ControllerState.WAITING_FOR_BOARD)
        controller.process_frame(self.frame)

        self.assertEqual(controller.state, ControllerState.SETTLING)
        self.agent.tick.assert_called_once()

    def test_continuous_mode_taps_confirmed_play_again_then_restarts(self):
        controller = TerraminerGoldController(
            self.agent,
            self.perception,
            continuous=True,
        )
        self.agent.phone.resolution = (324, 720)
        frame = np.zeros((720, 324, 3), dtype=np.uint8)
        frame[662:710, 96:232] = (0, 215, 255)
        bounds = controller._detect_reward_screen(frame)[1]

        with patch(
            "game_automation.games.terraminer_gold.controller.random.randint",
            side_effect=lambda lower, upper: upper,
        ) as random_integer, patch(
            "game_automation.games.terraminer_gold.controller.time.sleep"
        ) as sleep:
            controller.process_frame(frame)
            controller.process_frame(frame)
            controller.process_frame(frame)
            controller.process_frame(frame)

        tap_x, tap_y = self.agent.phone.tap.call_args.args
        x, y, width, height = bounds
        self.assertGreaterEqual(tap_x, x)
        self.assertLessEqual(tap_x, x + width - 1)
        self.assertGreaterEqual(tap_y, y)
        self.assertLessEqual(tap_y, y + height - 1)
        self.agent.phone.tap.assert_called_once_with(tap_x, tap_y)
        self.assertEqual(random_integer.call_count, 2)
        sleep.assert_called_once()
        self.assertEqual(controller.state, ControllerState.WAITING_FOR_BOARD)
        self.agent.reset_for_next_level.assert_called_once()

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
        self.agent.phone.tap.assert_not_called()

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

    def test_level_up_and_watch_ad_reward_panel_is_detected(self):
        frame = np.zeros((900, 448, 3), dtype=np.uint8)
        frame[536:608, 93:355] = (0, 215, 255)
        frame[570:596, 110:338] = (0, 145, 255)

        self.assertTrue(TerraminerGoldController._is_reward_screen(frame))

    def test_yellow_board_region_without_nested_watch_ad_button_is_not_reward(self):
        frame = np.zeros((900, 448, 3), dtype=np.uint8)
        frame[536:608, 93:355] = (0, 215, 255)

        self.assertFalse(TerraminerGoldController._is_reward_screen(frame))


if __name__ == "__main__":
    unittest.main()
