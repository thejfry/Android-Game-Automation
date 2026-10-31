import unittest
from unittest.mock import Mock, patch

import numpy as np

from .controller import ControllerState, TerraminerRockController
from .hud_detection import RockHudStats
from .perception import BoardObservation, CardObservation, RockBoard
from .reward_detection import RewardMatch, RewardMatchKind


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
        self.agent.tap_count = 0
        self.progress_callback = Mock()
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
            progress_callback=self.progress_callback,
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
        self.assertIsNotNone(controller._next_level_wait_started_at)
        controller.acknowledge_next_level()
        self.assertEqual(controller.state, ControllerState.WAITING_FOR_BOARD)
        self.agent.reset_for_next_level.assert_called_once()

    def test_continuous_mode_detects_stable_next_board_without_user_input(self):
        controller = TerraminerRockController(
            self.agent,
            self.perception,
            self.tap_executor,
            continuous=True,
            next_level_poll_interval_seconds=5,
            next_level_timeout_seconds=10,
        )
        controller.state = ControllerState.AWAITING_NEXT_LEVEL
        controller._next_level_wait_started_at = 10
        clock = [10.0]

        with patch(
            "game_automation.games.terraminer_rock.controller.time.monotonic",
            side_effect=lambda: clock[0],
        ), patch(
            "game_automation.games.terraminer_rock.controller.RewardScreenDetector.detect",
            return_value=None,
        ), self.assertLogs(
            "terraminer_rock.controller",
            level="INFO",
        ) as captured:
            controller.process_frame(board_frame())
            clock[0] = 10.5
            controller.process_frame(board_frame())
            self.assertEqual(self.perception.process.call_count, 1)
            clock[0] = 11.0
            controller.process_frame(board_frame())
            clock[0] = 12.0
            controller.process_frame(board_frame())

        self.assertEqual(controller.state, ControllerState.PLAYING)
        self.assertEqual(controller.level_monitor.level_number, 1)
        self.assertEqual(self.perception.process.call_count, 3)
        self.assertEqual(controller.next_level_confirmation_poll_interval_seconds, 1.0)
        self.agent.reset_for_next_level.assert_called_once()
        self.agent.process.assert_called_once()
        self.assertIn("resuming automatically", "\n".join(captured.output))

    def test_stops_after_next_level_wait_timeout(self):
        controller = TerraminerRockController(
            self.agent,
            self.perception,
            self.tap_executor,
            continuous=True,
            next_level_timeout_seconds=5,
        )
        controller.state = ControllerState.AWAITING_NEXT_LEVEL
        controller._next_level_wait_started_at = 10
        with patch(
            "game_automation.games.terraminer_rock.controller.time.monotonic",
            return_value=15,
        ), self.assertLogs(
            "terraminer_rock.controller",
            level="WARNING",
        ) as captured:
            controller.process_frame(board_frame())

        self.assertEqual(controller.state, ControllerState.STOPPED)
        self.assertTrue(controller.finished)
        self.assertIn("No new rock board appeared", "\n".join(captured.output))
        self.perception.process.assert_not_called()

    def test_clears_live_progress_when_controller_leaves_playing(self):
        self.controller._start_level(BoardObservation(0, blank_board()))
        with patch(
            "game_automation.games.terraminer_rock.level_monitor.RockHudDetector.detect",
            return_value=RockHudStats(0, 30, 90, 0, 2),
        ):
            self.controller.level_monitor.observe(
                BoardObservation(0, blank_board()),
                board_frame(),
                0,
            )
        self.progress_callback.reset_mock()

        self.controller._transition(ControllerState.WAITING_FOR_COMPLETION)

        self.progress_callback.assert_called_once_with(None)

    def test_completed_level_stats_include_estimated_time_and_normalized_averages(self):
        matched_cards = tuple(
            CardObservation(
                row,
                column,
                (column * 50, row * 50, 40, 40),
                True,
                True,
                np.zeros((32, 32, 4), dtype=np.float32),
            )
            for row in range(2)
            for column in range(2)
        )
        complete_board = RockBoard(matched_cards, 2, 2, (100, 100))
        self.perception.process.side_effect = [
            BoardObservation(0, blank_board()),
            BoardObservation(0, blank_board()),
            BoardObservation(0, blank_board()),
            BoardObservation(0, blank_board()),
            BoardObservation(0, complete_board),
        ]
        self.agent.process.side_effect = (False, False, True)
        self.agent.tap_count = 8
        ticks = iter((100.0, 112.0, 112.0))

        with patch(
            "game_automation.games.terraminer_rock.controller.time.monotonic",
            side_effect=lambda: next(ticks),
        ), self.assertLogs(
            "terraminer_rock.level_monitor",
            level="INFO",
        ) as captured:
            self.controller.process_frame(board_frame())
            self.controller.process_frame(board_frame())
            self.controller.process_frame(board_frame())
            self.controller.process_frame(board_frame())
            self.controller.process_frame(board_frame())

        summary = "\n".join(captured.output)
        self.assertIn("estimated time left 78s of 90s", summary)
        self.assertIn("12.0s elapsed (6.00s/match)", summary)
        self.assertIn("4 moves (2.00/match)", summary)
        self.assertIn("avg 12.0s/level", summary)
        self.assertIn("Level | Pairs | Time left | Moves", summary)
        self.assertIn("     1 |     2 |     78s", summary)

    def test_reward_screen_records_completion_when_board_disappears(self):
        with patch(
            "game_automation.games.terraminer_rock.level_monitor.time.monotonic",
            return_value=10.0,
        ):
            self.controller._start_level(BoardObservation(0, blank_board()))
        self.agent.tap_count = 26

        with patch(
            "game_automation.games.terraminer_rock.level_monitor.time.monotonic",
            return_value=50.0,
        ), self.assertLogs(
            "terraminer_rock.level_monitor",
            level="INFO",
        ) as captured:
            self.controller._handle_confirmed_reward(
                RewardMatch(RewardMatchKind.PLAY_AGAIN, (1, 1, 10, 10)),
                (918, 448),
            )

        summary = "\n".join(captured.output)
        self.assertIn("Level 1 complete: estimated time left 50s", summary)
        self.assertIn("40.0s elapsed (20.00s/match)", summary)
        self.assertIn("13 moves (6.50/match)", summary)
        self.assertTrue(self.controller.level_monitor.recorded)


if __name__ == "__main__":
    unittest.main()
