import unittest

import numpy as np

from .perception import CardObservation, RockBoard
from .solver import MemorySolver


def make_board(states, images, rows=2, columns=2):
    cards = []
    for position in range(rows * columns):
        row, column = divmod(position, columns)
        face_up = position in states
        cards.append(
            CardObservation(
                row,
                column,
                (column * 100, row * 100, 80, 80),
                face_up,
                position in states and states[position] == "matched",
                images.get(position) if face_up else None,
            )
        )
    return RockBoard(tuple(cards), rows, columns, (columns * 100, rows * 100))


class MemorySolverTests(unittest.TestCase):
    def setUp(self):
        self.solver = MemorySolver()
        self.icon_a = np.full((32, 32, 3), 30, dtype=np.float32)
        self.icon_b = np.full((32, 32, 3), 220, dtype=np.float32)

    def test_starts_by_revealing_first_unknown_card(self):
        board = make_board({}, {})

        self.assertEqual(self.solver.next_card(board), (0, 0))

    def test_remembers_first_card_and_reveals_another_unknown(self):
        self.assertEqual(
            self.solver.next_card(make_board({0: "open"}, {0: self.icon_a})),
            (0, 1),
        )

    def test_uses_known_match_when_first_card_is_flipped(self):
        self.solver.next_card(make_board({0: "open"}, {0: self.icon_a}))
        self.solver.next_card(make_board({0: "open", 1: "open"}, {
            0: self.icon_a,
            1: self.icon_b,
        }))

        self.assertEqual(
            self.solver.next_card(make_board({2: "open"}, {2: self.icon_a})),
            (0, 0),
        )

    def test_waits_while_two_unmatched_cards_are_still_visible(self):
        board = make_board(
            {0: "open", 1: "open"},
            {0: self.icon_a, 1: self.icon_b},
        )

        self.assertIsNone(self.solver.next_card(board))

    def test_does_not_retry_a_pair_after_a_mismatch(self):
        self.solver.next_card(make_board({0: "open"}, {0: self.icon_a}))
        self.assertIsNone(
            self.solver.next_card(
                make_board(
                    {0: "open", 1: "open"},
                    {0: self.icon_a, 1: self.icon_b},
                )
            )
        )
        self.solver.next_card(make_board({}, {}))

        self.assertEqual(
            self.solver.next_card(make_board({2: "open"}, {2: self.icon_b})),
            (0, 1),
        )

    def test_matched_card_is_never_selected_again(self):
        board = make_board({0: "matched"}, {0: self.icon_a})

        self.assertEqual(self.solver.next_card(board), (0, 1))

    def test_handles_larger_rectangular_boards(self):
        self.solver = MemorySolver()
        cards = tuple(
            CardObservation(row, column, (column * 50, row * 50, 40, 40), False, False, None)
            for row in range(4)
            for column in range(6)
        )
        board = RockBoard(cards, 4, 6, (300, 200))

        self.assertEqual(self.solver.next_card(board), (0, 0))


if __name__ == "__main__":
    unittest.main()
