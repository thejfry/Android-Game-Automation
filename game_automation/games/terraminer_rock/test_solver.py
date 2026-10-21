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


def make_icon(color, x_start):
    image = np.zeros((32, 32, 4), dtype=np.float32)
    image[6:26, x_start:x_start + 8, :3] = color
    image[6:26, x_start:x_start + 8, 3] = 255
    return image


class MemorySolverTests(unittest.TestCase):
    def setUp(self):
        self.solver = MemorySolver()
        self.icon_a = make_icon(30, 5)
        self.icon_b = make_icon(220, 19)

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

    def test_forces_pair_when_only_two_unmatched_cards_remain(self):
        self.solver.next_card(
            make_board(
                {2: "open", 3: "open"},
                {2: self.icon_a, 3: self.icon_b},
            )
        )
        self.solver.next_card(
            make_board(
                {0: "matched", 1: "matched"},
                {0: self.icon_a, 1: self.icon_a},
            )
        )

        self.assertEqual(
            self.solver.next_card(
                make_board(
                    {0: "matched", 1: "matched"},
                    {0: self.icon_a, 1: self.icon_a},
                )
            ),
            (1, 0),
        )

    def test_forces_remaining_partner_when_one_of_last_two_is_open(self):
        board = make_board(
            {
                0: "matched",
                1: "matched",
                2: "open",
            },
            {
                0: self.icon_a,
                1: self.icon_a,
                2: self.icon_a,
            },
        )

        self.assertEqual(self.solver.next_card(board), (1, 1))


if __name__ == "__main__":
    unittest.main()
