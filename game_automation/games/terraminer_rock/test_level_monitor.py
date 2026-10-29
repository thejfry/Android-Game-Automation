import unittest
from unittest.mock import Mock, patch

import numpy as np

from .hud_detection import RockHudStats
from .level_monitor import RockLevelMonitor
from .perception import BoardObservation, CardObservation, RockBoard


def observation_with_matches(match_count):
    cards = tuple(
        CardObservation(
            row,
            column,
            (column * 50, row * 50, 40, 40),
            index < match_count * 2,
            index < match_count * 2,
            None,
        )
        for index, (row, column) in enumerate(
            (divmod(index, 2) for index in range(4))
        )
    )
    return BoardObservation(0, RockBoard(cards, 2, 2, (100, 100)))


def board_frame():
    return np.zeros((918, 448, 3), dtype=np.uint8)


class RockLevelMonitorTests(unittest.TestCase):
    def setUp(self):
        self.progress_callback = Mock()
        self.monitor = RockLevelMonitor(90, self.progress_callback)
        self.monitor.start_level(observation_with_matches(0))

    def test_live_progress_stays_unchanged_when_hud_values_are_unchanged(self):
        with patch(
            "game_automation.games.terraminer_rock.level_monitor.RockHudDetector.detect",
            return_value=RockHudStats(0, 30, 90, 0, 2),
        ):
            for _ in range(4):
                self.monitor.observe(observation_with_matches(0), board_frame(), 0)

        self.progress_callback.assert_called_once()
        line = self.progress_callback.call_args.args[0]
        self.assertIn("Moves used  [" + "-" * 36 + "] 0/30 (0%)", line)
        self.assertIn("Time used   [" + "-" * 36 + "] 0/90s (0%)", line)
        self.assertIn("Matches     [" + "-" * 36 + "] 0/2 (0%)", line)

    def test_progress_does_not_decrease_when_matched_border_is_missed_in_a_frame(self):
        readings = (0, 1, 0, 2, 1)
        with patch(
            "game_automation.games.terraminer_rock.level_monitor.RockHudDetector.detect",
            side_effect=(
                RockHudStats(0, 30, 90, count, 2)
                for count in readings
            ),
        ):
            for count in readings:
                self.monitor.observe(observation_with_matches(count), board_frame(), 0)

        progress_counts = [
            int(
                call.args[0]
                .split("Matches ", 1)[1]
                .split("] ", 1)[1]
                .split("/", 1)[0]
            )
            for call in self.progress_callback.call_args_list
        ]
        self.assertEqual(progress_counts, [0, 1, 2])

    def test_live_progress_refreshes_when_moves_or_time_changes(self):
        self.monitor = RockLevelMonitor(90, self.progress_callback)
        readings = iter(
            (
                RockHudStats(0, 30, 60, 0, 2),
                RockHudStats(1, 30, 59, 0, 2),
                RockHudStats(1, 30, 58, 0, 2),
            )
        )
        with patch(
            "game_automation.games.terraminer_rock.level_monitor.RockHudDetector.detect",
            side_effect=lambda _frame: next(readings),
        ), patch(
            "game_automation.games.terraminer_rock.level_monitor.time.monotonic",
            side_effect=(0.0, 30.0, 31.0, 32.0),
        ):
            self.monitor.start_level(observation_with_matches(0))
            for tap_count in (0, 2, 2):
                self.monitor.observe(
                    observation_with_matches(0),
                    board_frame(),
                    tap_count,
                )

        self.assertEqual(self.progress_callback.call_count, 3)
        first, second, third = (
            call.args[0] for call in self.progress_callback.call_args_list
        )
        self.assertIn("Moves used  [" + "-" * 36 + "] 0/30 (0%)", first)
        self.assertIn("Time used   [" + "-" * 36 + "] 0/60s (0%)", first)
        self.assertIn("Moves used  [" + "#" + "-" * 35 + "] 1/30 (3%)", second)
        self.assertIn("Time used   [" + "#" + "-" * 35 + "] 1/60s (2%)", second)
        self.assertIn("Time used   [" + "#" + "-" * 35 + "] 2/60s (3%)", third)

    def test_rejects_timer_reading_that_drops_from_50_to_5(self):
        readings = iter(
            (
                RockHudStats(0, 30, 50, 0, 2),
                RockHudStats(0, 30, 5, 0, 2),
                RockHudStats(0, 30, 49, 0, 2),
            )
        )
        clock = iter((0.0, 40.0, 40.1, 41.0))
        with patch(
            "game_automation.games.terraminer_rock.level_monitor.RockHudDetector.detect",
            side_effect=lambda _frame: next(readings),
        ), patch(
            "game_automation.games.terraminer_rock.level_monitor.time.monotonic",
            side_effect=lambda: next(clock),
        ), self.assertLogs("terraminer_rock.level_monitor", level="DEBUG") as captured:
            self.monitor.start_level(observation_with_matches(0))
            observation = observation_with_matches(0)
            self.monitor.observe(observation, board_frame(), 0)
            accepted = self.monitor.last_hud_stats
            self.monitor.observe(observation, board_frame(), 0)
            self.monitor.observe(observation, board_frame(), 0)

        self.assertEqual(accepted.seconds_remaining, 50)
        self.assertEqual(self.monitor.last_hud_stats.seconds_remaining, 49)
        self.assertIn("Ignoring inconsistent HUD timer 5s", "\n".join(captured.output))
        self.assertEqual(self.progress_callback.call_count, 2)

    def test_accepts_first_timer_reading_without_assuming_timer_start_time(self):
        with patch(
            "game_automation.games.terraminer_rock.level_monitor.RockHudDetector.detect",
            return_value=RockHudStats(0, 30, 5, 0, 2),
        ), patch(
            "game_automation.games.terraminer_rock.level_monitor.time.monotonic",
            side_effect=(0.0, 0.1),
        ):
            self.monitor.start_level(observation_with_matches(0))
            self.monitor.observe(observation_with_matches(0), board_frame(), 0)

        self.assertEqual(self.monitor.last_hud_stats.seconds_remaining, 5)
        self.progress_callback.assert_called_once()

    def test_ignores_hud_matches_not_confirmed_by_the_board(self):
        with patch(
            "game_automation.games.terraminer_rock.level_monitor.RockHudDetector.detect",
            return_value=RockHudStats(0, 30, 90, 2, 2),
        ), self.assertLogs("terraminer_rock.level_monitor", level="DEBUG") as captured:
            self.monitor.observe(observation_with_matches(1), board_frame(), 0)

        self.assertIsNone(self.monitor.last_hud_stats)
        self.assertIn("only 1 matched pairs", "\n".join(captured.output))

    def test_uses_detected_board_pair_count_when_hud_denominator_is_misread(self):
        with patch(
            "game_automation.games.terraminer_rock.level_monitor.RockHudDetector.detect",
            return_value=RockHudStats(0, 30, 90, 0, 70),
        ):
            self.monitor.observe(observation_with_matches(0), board_frame(), 0)

        self.assertEqual(
            self.monitor.last_hud_stats,
            RockHudStats(0, 30, 90, 0, 2),
        )
        self.progress_callback.assert_called_once()
        line = self.progress_callback.call_args.args[0]
        self.assertIn("Level 1 progress", line)
        self.assertIn("Moves used  [" + "-" * 36 + "] 0/30 (0%)", line)
        self.assertIn("Matches     [" + "-" * 36 + "] 0/2 (0%)", line)

    def test_ignores_hud_values_that_violate_tap_or_time_progress(self):
        with patch(
            "game_automation.games.terraminer_rock.level_monitor.RockHudDetector.detect",
            return_value=RockHudStats(2, 30, 90, 0, 2),
        ), patch(
            "game_automation.games.terraminer_rock.level_monitor.time.monotonic",
            side_effect=(0.0, 0.1),
        ), self.assertLogs("terraminer_rock.level_monitor", level="DEBUG") as captured:
            self.monitor = RockLevelMonitor(90, self.progress_callback)
            self.monitor.start_level(observation_with_matches(0))
            self.monitor.observe(observation_with_matches(0), board_frame(), 1)

        self.assertIsNone(self.monitor.last_hud_stats)
        self.assertIn("only 1 card taps have been sent", "\n".join(captured.output))

        readings = iter(
            (
                RockHudStats(2, 30, 80, 0, 2),
                RockHudStats(1, 30, 82, 0, 2),
            )
        )
        with patch(
            "game_automation.games.terraminer_rock.level_monitor.RockHudDetector.detect",
            side_effect=lambda _frame: next(readings),
        ), patch(
            "game_automation.games.terraminer_rock.level_monitor.time.monotonic",
            side_effect=(0.0, 10.0, 11.0),
        ), self.assertLogs("terraminer_rock.level_monitor", level="DEBUG") as captured:
            self.monitor = RockLevelMonitor(90, self.progress_callback)
            self.monitor.start_level(observation_with_matches(0))
            observation = observation_with_matches(0)
            self.monitor.observe(observation, board_frame(), 10)
            accepted = self.monitor.last_hud_stats
            self.monitor.observe(observation, board_frame(), 10)

        self.assertEqual(accepted, RockHudStats(2, 30, 80, 0, 2))
        self.assertEqual(self.monitor.last_hud_stats, accepted)
        self.assertIn("Ignoring inconsistent HUD counters", "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()
