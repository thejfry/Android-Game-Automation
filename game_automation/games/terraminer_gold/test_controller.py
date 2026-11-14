import unittest
from unittest.mock import Mock, patch

import numpy as np

from .controller import ControllerState, TerraminerGoldController
from .perception import BoardObservation, TerraminerBoard
from .reward_detection import RewardMatchKind, RewardScreenDetector
from .solver import MazeSolution


class LifecycleControllerTests(unittest.TestCase):
    def setUp(self):
        self.agent = Mock()
        self.agent.dry_run = False
        self.agent.finished = False
        self.agent.verify_route.return_value = False
        solution = MazeSolution((), ((0, 0), (1, 1)))
        self.agent.plan.return_value = solution
        self.agent.execute_next_tap.return_value = True
        self.perception = Mock()
        self.perception.process.return_value = BoardObservation(
            timestamp=0,
            board=TerraminerBoard(
                tile_masks=((0, 0), (0, 0)),
                start=(0, 0),
                goal=(1, 1),
                bounds=(0, 0, 200, 200),
            ),
            frame_size=(200, 200),
        )
        self.tap_executor = Mock()
        self.controller = TerraminerGoldController(
            self.agent,
            self.perception,
            self.tap_executor,
            settling_seconds=1,
        )
        self.frame = np.zeros((200, 200, 3), dtype=np.uint8)

    def test_route_execution_is_paced_and_not_replanned_while_waiting(self):
        self.agent.execute_next_tap.side_effect = (False, True)
        with patch(
            "game_automation.games.terraminer_gold.controller.time.monotonic",
            side_effect=(0.0, 2.0),
        ):
            self.controller.process_frame(self.frame)
            self.controller.process_frame(self.frame)
            self.controller.process_frame(self.frame)
            self.assertEqual(self.controller.state, ControllerState.EXECUTING)
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

        self.agent.plan.assert_called_once()
        self.assertEqual(self.agent.execute_next_tap.call_count, 2)
        self.agent.verify_route.assert_called_once()
        self.assertEqual(self.perception.process.call_count, 4)

    def test_continuous_mode_automatically_resumes_on_stable_next_board(self):
        controller = TerraminerGoldController(
            self.agent,
            self.perception,
            self.tap_executor,
            settling_seconds=0,
            continuous=True,
            next_level_poll_interval_seconds=1,
            next_level_timeout_seconds=10,
        )
        reward = np.zeros((900, 448, 3), dtype=np.uint8)
        reward[536:608, 93:355] = (0, 215, 255)
        reward[570:596, 110:338] = (0, 145, 255)

        with patch(
            "game_automation.games.terraminer_gold.controller.time.monotonic",
            side_effect=(10.0, 11.0, 11.3, 11.6, 11.7),
        ):
            controller.process_frame(reward)
            controller.process_frame(reward)
            self.assertEqual(controller.state, ControllerState.AWAITING_NEXT_LEVEL)
            self.assertFalse(controller.finished)
            self.tap_executor.tap_bounds.assert_not_called()
            self.perception.process.assert_not_called()

            controller.process_frame(self.frame)
            controller.process_frame(self.frame)
            self.assertEqual(controller.state, ControllerState.AWAITING_NEXT_LEVEL)
            controller.process_frame(self.frame)

        self.assertEqual(controller.state, ControllerState.SETTLING)
        self.agent.reset_for_next_level.assert_called_once()
        self.agent.plan.assert_called_once()

    def test_continuous_mode_stops_when_no_next_board_appears(self):
        controller = TerraminerGoldController(
            self.agent,
            self.perception,
            self.tap_executor,
            continuous=True,
            next_level_timeout_seconds=5,
        )
        controller.state = ControllerState.AWAITING_NEXT_LEVEL
        controller._next_level_wait_started_at = 10

        with patch(
            "game_automation.games.terraminer_gold.controller.time.monotonic",
            return_value=15,
        ):
            controller.process_frame(self.frame)

        self.assertEqual(controller.state, ControllerState.STOPPED)
        self.assertTrue(controller.finished)
        self.perception.process.assert_not_called()

    def test_continuous_mode_taps_confirmed_play_again_then_restarts(self):
        controller = TerraminerGoldController(
            self.agent,
            self.perception,
            self.tap_executor,
            continuous=True,
            reward_button_wait_seconds=1,
        )
        frame = np.zeros((720, 324, 3), dtype=np.uint8)
        frame[662:710, 96:232] = (0, 215, 255)
        match = RewardScreenDetector.detect(frame)
        with patch(
            "game_automation.games.terraminer_gold.controller.time.monotonic",
            side_effect=(100, 100, 101),
        ):
            controller.process_frame(frame)
            controller.process_frame(frame)
            self.assertEqual(controller.state, ControllerState.ADVANCING_LEVEL)
            controller.process_frame(frame)
            self.assertEqual(controller.state, ControllerState.ADVANCING_LEVEL)
            controller.process_frame(frame)

        self.assertEqual(match.kind, RewardMatchKind.PLAY_AGAIN)
        self.tap_executor.tap_bounds.assert_called_once_with(
            match.button_bounds,
            (324, 720),
            description="confirmed Gold Play Again button",
        )
        self.assertEqual(controller.state, ControllerState.WAITING_FOR_BOARD)
        self.agent.reset_for_next_level.assert_called_once()
        self.perception.process.assert_not_called()

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
        self.agent.solve.assert_not_called()
        self.tap_executor.tap_bounds.assert_not_called()

    def test_gameplay_frame_is_not_misclassified_as_reward_screen(self):
        self.assertIsNone(RewardScreenDetector.detect(self.frame))

    def test_reward_button_with_text_cutouts_is_detected_near_bottom(self):
        frame = np.zeros((720, 324, 3), dtype=np.uint8)
        frame[662:710, 96:232] = (0, 215, 255)
        frame[683:686, 121:208] = (0, 0, 0)

        match = RewardScreenDetector.detect(frame)
        self.assertEqual(match.kind, RewardMatchKind.PLAY_AGAIN)

    def test_gold_reward_layout_detects_play_again_below_reward_cards(self):
        frame = np.zeros((866, 422, 3), dtype=np.uint8)
        frame[688:732, 30:392] = (0, 190, 230)
        frame[744:784, 100:228] = (0, 215, 255)
        frame[754:774, 126:144] = (0, 0, 0)
        frame[754:774, 156:210] = (0, 0, 0)
        frame[744:784, 240:323] = (0, 100, 100)

        match = RewardScreenDetector.detect(frame)

        self.assertEqual(match.kind, RewardMatchKind.PLAY_AGAIN)
        self.assertIsNotNone(match.button_bounds)
        x, y, width, height = match.button_bounds
        self.assertAlmostEqual(x + width / 2, 164, delta=3)
        self.assertAlmostEqual(y + height / 2, 764, delta=3)

    def test_rock_positioned_play_again_button_is_not_used_for_gold(self):
        frame = np.zeros((918, 448, 3), dtype=np.uint8)
        frame[506:554, 102:244] = (0, 215, 255)

        self.assertIsNone(RewardScreenDetector.detect(frame))

    def test_wide_yellow_game_board_tile_above_reward_area_is_not_detected(self):
        frame = np.zeros((720, 324, 3), dtype=np.uint8)
        frame[500:535, 80:250] = (0, 215, 255)

        self.assertIsNone(RewardScreenDetector.detect(frame))

    def test_level_up_and_watch_ad_reward_panel_is_detected(self):
        frame = np.zeros((900, 448, 3), dtype=np.uint8)
        frame[536:608, 93:355] = (0, 215, 255)
        frame[570:596, 110:338] = (0, 145, 255)

        match = RewardScreenDetector.detect(frame)
        self.assertEqual(match.kind, RewardMatchKind.LEVEL_UP_PANEL)

    def test_yellow_board_region_without_nested_watch_ad_button_is_not_reward(self):
        frame = np.zeros((900, 448, 3), dtype=np.uint8)
        frame[536:608, 93:355] = (0, 215, 255)

        self.assertIsNone(RewardScreenDetector.detect(frame))


if __name__ == "__main__":
    unittest.main()
