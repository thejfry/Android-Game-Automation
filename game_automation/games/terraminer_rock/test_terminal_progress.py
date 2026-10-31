import io
import logging
import unittest

from .terminal_progress import (
    LiveProgressDisplay,
    ProgressStreamHandler,
    format_progress_line,
)


class TTYBuffer(io.StringIO):
    def isatty(self):
        return True


class LiveProgressDisplayTests(unittest.TestCase):
    def test_formats_three_separate_hud_progress_bars(self):
        line = format_progress_line(
            level=3,
            moves_used=10,
            move_limit=40,
            seconds_remaining=30,
            initial_seconds=60,
            matches_made=4,
            expected_matches=8,
        )

        self.assertEqual(
            line,
            "\n".join(
                (
                    "Level 3 progress",
                    "Moves used  [" + "#" * 9 + "-" * 27 + "] 10/40 (25%)",
                    "Time used   [" + "#" * 18 + "-" * 18 + "] 30/60s (50%)",
                    "Matches     [" + "#" * 18 + "-" * 18 + "] 4/8 (50%)",
                    "Pace: ON PACE | matches 50% | moves 25% | time 50%",
                )
            ),
        )

    def test_clamps_progress_bar_values_to_valid_range(self):
        line = format_progress_line(1, 50, 40, 0, 60, 9, 8)

        self.assertIn("Moves used  [" + "#" * 36 + "] 50/40 (100%)", line)
        self.assertIn("Time used   [" + "#" * 36 + "] 60/60s (100%)", line)
        self.assertIn("Matches     [" + "#" * 36 + "] 9/8 (100%)", line)
        self.assertIn("Pace: ON PACE", line)

    def test_flags_time_or_move_usage_ahead_of_match_progress(self):
        moves_risk = format_progress_line(1, 20, 40, 45, 60, 2, 8)
        time_risk = format_progress_line(1, 10, 40, 30, 60, 3, 8)
        safe = format_progress_line(1, 10, 40, 45, 60, 5, 8)

        self.assertIn("Pace: AT RISK | matches 25% | moves 50% | time 25%", moves_risk)
        self.assertIn("Pace: AT RISK | matches 38% | moves 25% | time 50%", time_risk)
        self.assertIn("Pace: ON PACE | matches 63% | moves 25% | time 25%", safe)

    def test_updates_and_clears_stacked_terminal_lines(self):
        stream = TTYBuffer()
        progress = LiveProgressDisplay(stream)

        first = "Level 1 progress\nMoves used [##] 1/4\nTime used [##] 1/4\nMatches [##] 1/4\nPace: ON PACE"
        second = "Level 1 progress\nMoves used [###] 2/4\nTime used [###] 2/4\nMatches [###] 2/4\nPace: ON PACE"
        progress.update(first)
        progress.update(second)
        progress.update(None)

        final_text = second.splitlines()
        self.assertEqual(
            stream.getvalue(),
            first
            + "\x1b[4A"
            + "".join("\r\x1b[2K" + line + ("\n" if index < 4 else "")
                      for index, line in enumerate(final_text))
            + "\x1b[4A"
            + "".join("\r\x1b[2K" + ("\n" if index < 4 else "")
                      for index in range(5))
            + "\x1b[4A\r",
        )

    def test_timestamped_logs_temporarily_replace_and_restore_progress(self):
        stream = TTYBuffer()
        progress = LiveProgressDisplay(stream)
        handler = ProgressStreamHandler(progress)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        record = logging.LogRecord(
            "test",
            logging.INFO,
            __file__,
            1,
            "Level 1 complete",
            (),
            None,
        )
        text = "Level 1 progress\nMoves used [##] 2/4\nTime used [##] 2/4\nMatches [##] 2/4\nPace: ON PACE"
        progress.update(text)

        handler.emit(record)

        output = stream.getvalue()
        self.assertIn("INFO: Level 1 complete\n", output)
        self.assertTrue(output.endswith("\n".join(text.splitlines())))

    def test_non_interactive_stream_does_not_emit_live_progress(self):
        stream = io.StringIO()
        progress = LiveProgressDisplay(stream)

        progress.update("Level 1 progress\nMoves used [##] 1/4\nTime used [##] 1/4\nMatches [##] 1/4\nPace: ON PACE")

        self.assertEqual(stream.getvalue(), "")
        self.assertFalse(progress.enabled)


if __name__ == "__main__":
    unittest.main()
