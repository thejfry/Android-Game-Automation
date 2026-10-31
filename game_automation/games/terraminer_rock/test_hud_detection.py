import unittest

import cv2
import numpy as np

from .hud_detection import RockHudDetector, RockHudStats

FRAME_SIZE = (896, 448)


def hud_frame(
    moves="0/40",
    timer="75s",
    matches="0/8",
    labels=True,
    value_y=0.202,
):
    height, width = FRAME_SIZE
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    if labels:
        for label, x in (("Moves", 56), ("Time", 190), ("Matches", 310)):
            cv2.putText(
                frame,
                label,
                (x, round(height * 0.17)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (205, 205, 205),
                1,
                cv2.LINE_AA,
            )
    for text, x, column in (
        (moves, 54, "moves"),
        (timer, 199, "time"),
        (matches, 316, "matches"),
    ):
        if text is None:
            continue
        cv2.putText(
            frame,
            text,
            (x, round(height * value_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


class RockHudDetectorTests(unittest.TestCase):
    def test_reads_move_time_and_match_counters_below_labels(self):
        stats = RockHudDetector.detect(hud_frame())

        self.assertEqual(
            stats,
            RockHudStats(
                moves_used=0,
                move_limit=40,
                seconds_remaining=75,
                matches_made=0,
                expected_matches=8,
            ),
        )

    def test_reads_changed_counters_and_timer_without_seconds_suffix(self):
        stats = RockHudDetector.detect(
            hud_frame(moves="12/40", timer="9", matches="7/8", labels=False)
        )

        self.assertEqual(
            stats,
            RockHudStats(
                moves_used=12,
                move_limit=40,
                seconds_remaining=9,
                matches_made=7,
                expected_matches=8,
            ),
        )

    def test_reads_value_row_when_android_status_bar_is_hidden(self):
        stats = RockHudDetector.detect(hud_frame(value_y=0.16))

        self.assertEqual(
            stats,
            RockHudStats(
                moves_used=0,
                move_limit=40,
                seconds_remaining=75,
                matches_made=0,
                expected_matches=8,
            ),
        )

    def test_returns_no_reading_when_a_counter_is_missing(self):
        self.assertIsNone(RockHudDetector.detect(hud_frame(timer=None)))

    def test_rejects_invalid_counter_ranges(self):
        self.assertIsNone(
            RockHudDetector.detect(
                hud_frame(moves="41/40", timer="75s", matches="0/8")
            )
        )

    def test_corrects_common_leading_one_as_seven_move_ocr_error(self):
        stats = RockHudDetector._parse_readings(
            {"moves": "74/50", "time": "39s", "matches": "4/10"}
        )

        self.assertEqual(
            stats,
            RockHudStats(
                moves_used=14,
                move_limit=50,
                seconds_remaining=39,
                matches_made=4,
                expected_matches=10,
            ),
        )

    def test_accepts_timer_suffix_misread_as_five(self):
        stats = RockHudDetector._parse_readings(
            {"moves": "14/50", "time": "0395", "matches": "4/10"}
        )

        self.assertEqual(
            stats,
            RockHudStats(
                moves_used=14,
                move_limit=50,
                seconds_remaining=39,
                matches_made=4,
                expected_matches=10,
            ),
        )


if __name__ == "__main__":
    unittest.main()
