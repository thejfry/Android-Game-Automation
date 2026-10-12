import unittest
from unittest.mock import Mock, patch

import numpy as np

from .controller import ControllerState, TerraminerGoldController
from .agent import SolveResult
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
        self.agent.solve.side_effect = lambda observation, before_execution=None: (
            before_execution(),
            SolveResult(solution, True),
        )[1]
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

        self.agent.solve.assert_called_once()
        self.agent.verify_route.assert_called_once()
        self.assertEqual(self.perception.process.call_count, 2)

    def test_continuous_mode_waits_for_ack_then_stable_next_board(self):
        controller = TerraminerGoldController(
            self.agent,
            self.perception,
            self.tap_executor,
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
        self.tap_executor.tap_bounds.assert_not_called()
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
        self.agent.solve.assert_called_once()

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
            description="confirmed Play Again button",
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
