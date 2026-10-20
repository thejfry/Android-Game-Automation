import unittest
from unittest.mock import Mock, patch

import numpy as np

from .controller import ControllerState, TerraminerRockController
from .perception import BoardObservation, CardObservation, RockBoard


def blank_board():
    cards = tuple(
        CardObservation(row, column, (column * 50, row * 50, 40, 40), False, False, None)
        for row in range(2)
        for column in range(2)
    )
    return RockBoard(cards, 2, 2, (100, 100))


def board_frame():
    return np.zeros((918, 448, 3), dtype=np.uint8)


def play_again_frame():
    frame = board_frame()
    frame[506:554, 102:244] = (0, 215, 255)
    return frame


def level_up_frame():
    frame = board_frame()
    frame[442:522, 118:331] = (0, 255, 0)
    frame[480:514, 160:290] = (0, 145, 255)
    return frame


class RockControllerTests(unittest.TestCase):
    def setUp(self):
        self.agent = Mock()
        self.agent.process.return_value = False
        self.perception = Mock()
        self.perception.process.return_value = BoardObservation(
            timestamp=0,
            board=blank_board(),
        )
        self.tap_executor = Mock()
        self.controller = TerraminerRockController(
            self.agent,
            self.perception,
            self.tap_executor,
        )

    def test_requires_stable_face_down_board_before_play(self):
        frame = board_frame()

        self.controller.process_frame(frame)
        self.controller.process_frame(frame)
        self.assertEqual(self.controller.state, ControllerState.WAITING_FOR_BOARD)
        self.agent.process.assert_not_called()

        self.controller.process_frame(frame)
        self.assertEqual(self.controller.state, ControllerState.PLAYING)
        self.agent.process.assert_called_once()

    def test_non_continuous_mode_stops_on_standard_results(self):
        frame = play_again_frame()

        self.controller.process_frame(frame)
        self.controller.process_frame(frame)

        self.assertEqual(self.controller.state, ControllerState.REWARD_SCREEN)
        self.assertTrue(self.controller.finished)
        self.tap_executor.tap_bounds.assert_not_called()

    def test_continuous_mode_taps_play_again_and_suppresses_repeated_taps(self):
        controller = TerraminerRockController(
            self.agent,
            self.perception,
            self.tap_executor,
            continuous=True,
            reward_button_wait_seconds=0,
        )
        frame = play_again_frame()
        with patch("game_automation.games.terraminer_rock.controller.time.monotonic", return_value=10):
            controller.process_frame(frame)
            controller.process_frame(frame)
            self.assertEqual(controller.state, ControllerState.ADVANCING_LEVEL)
            controller.process_frame(frame)

        self.assertEqual(controller.state, ControllerState.WAITING_FOR_BOARD)
        self.assertTrue(controller._waiting_for_reward_to_clear)
        self.tap_executor.tap_bounds.assert_called_once_with(
            (102, 506, 142, 48),
            (448, 918),
            description="confirmed Rock Play Again button",
        )
        controller.process_frame(frame)
        self.tap_executor.tap_bounds.assert_called_once()

    def test_continuous_mode_does_not_tap_level_up_prompt(self):
        controller = TerraminerRockController(
            self.agent,
            self.perception,
            self.tap_executor,
            continuous=True,
        )
        frame = level_up_frame()

        controller.process_frame(frame)
        controller.process_frame(frame)

        self.assertEqual(controller.state, ControllerState.AWAITING_NEXT_LEVEL)
        self.assertFalse(controller.finished)
        self.tap_executor.tap_bounds.assert_not_called()
        controller.acknowledge_next_level()
        self.assertEqual(controller.state, ControllerState.WAITING_FOR_BOARD)
        self.agent.reset_for_next_level.assert_called_once()


if __name__ == "__main__":
    unittest.main()
