import unittest

from .solver import EAST, NORTH, SOUTH, WEST, MazeSolver, rotate_clockwise


class MazeSolverTests(unittest.TestCase):
    def setUp(self):
        self.solver = MazeSolver()

    def test_clockwise_rotation(self):
        self.assertEqual(rotate_clockwise(NORTH), EAST)
        self.assertEqual(rotate_clockwise(NORTH | SOUTH), EAST | WEST)

    def test_prefers_a_zero_tap_straight_route(self):
        tiles = [
            [0, 0, 0],
            [0, NORTH | SOUTH, 0],
            [0, 0, 0],
        ]
        solution = self.solver.solve(tiles, (1, 0), (1, 2))
        self.assertIsNotNone(solution)
        self.assertEqual(solution.tap_count, 0)
        self.assertEqual(solution.path, ((1, 0), (1, 1), (1, 2)))

    def test_rotates_an_elbow_to_connect_west_and_north(self):
        tiles = [
            [0, 0, 0],
            [0, NORTH | EAST, 0],
            [0, 0, 0],
        ]
        solution = self.solver.solve(tiles, (1, 0), (0, 1))
        self.assertIsNotNone(solution)
        self.assertEqual(solution.tap_count, 3)
        self.assertEqual(solution.actions[0].target_mask, NORTH | WEST)

    def test_supports_connected_t_junctions(self):
        tiles = [
            [0, 0, 0],
            [0, NORTH | EAST | WEST, 0],
            [0, 0, 0],
        ]
        solution = self.solver.solve(tiles, (1, 0), (0, 1))
        self.assertIsNotNone(solution)
        self.assertEqual(solution.tap_count, 0)

    def test_returns_none_when_no_route_exists(self):
        tiles = [[0, 0], [0, 0]]
        self.assertIsNone(self.solver.solve(tiles, (0, 0), (1, 1)))


if __name__ == "__main__":
    unittest.main()
