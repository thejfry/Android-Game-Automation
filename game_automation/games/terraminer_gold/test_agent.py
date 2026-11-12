import unittest
from unittest.mock import Mock, call, patch

from .agent import TerraminerGoldAgent
from .perception import TerraminerBoard
from .perception import BoardObservation
from .solver import EAST, NORTH, SOUTH, WEST, MazeSolution, TileAction
from .tap_executor import TapExecutor


class RouteVerificationTests(unittest.TestCase):
    def setUp(self):
        self.solution = MazeSolution(
            actions=(),
            path=((0, 0), (0, 1), (0, 2)),
        )

    def test_zero_tap_route_requires_strong_port_evidence(self):
        board = TerraminerBoard(
            tile_masks=[[0, EAST | WEST, 0]],
            start=(0, 0),
            goal=(0, 2),
            bounds=(0, 0, 300, 100),
            tile_port_confidence=[[(0.0, 0.0, 0.0, 0.0), (0.0, 0.75, 0.0, 0.75), (0.0, 0.0, 0.0, 0.0)]],
        )

        self.assertFalse(
            TerraminerGoldAgent._route_has_confident_ports(board, self.solution)
        )

    def test_confident_preconnected_route_passes_port_check(self):
        board = TerraminerBoard(
            tile_masks=[[0, EAST | WEST, 0]],
            start=(0, 0),
            goal=(0, 2),
            bounds=(0, 0, 300, 100),
            tile_port_confidence=[[(0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0)]],
        )

        self.assertTrue(
            TerraminerGoldAgent._route_has_confident_ports(board, self.solution)
        )

    def test_verification_requires_observed_target_orientation(self):
        rotated_solution = MazeSolution(
            actions=(TileAction(0, 1, 1, EAST | WEST),),
            path=self.solution.path,
        )
        expected_board = TerraminerBoard(
            tile_masks=[[0, EAST | WEST, 0]],
            start=(0, 0),
            goal=(0, 2),
            bounds=(0, 0, 300, 100),
            tile_port_confidence=[[(0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 0.0, 0.0)]],
        )
        incorrect_orientation = TerraminerBoard(
            tile_masks=[[0, EAST | NORTH | SOUTH | WEST, 0]],
            start=(0, 0),
            goal=(0, 2),
            bounds=(0, 0, 300, 100),
            tile_port_confidence=[[(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0)]],
        )

        self.assertTrue(
            TerraminerGoldAgent.verify_route(expected_board, rotated_solution)
        )
        self.assertFalse(
            TerraminerGoldAgent.verify_route(incorrect_orientation, rotated_solution)
        )

    def test_post_action_verification_accepts_ports_at_detection_threshold(self):
        rotated_solution = MazeSolution(
            actions=(TileAction(0, 1, 1, EAST | WEST),),
            path=self.solution.path,
        )
        board = TerraminerBoard(
            tile_masks=[[0, EAST | WEST, 0]],
            start=(0, 0),
            goal=(0, 2),
            bounds=(0, 0, 300, 100),
            tile_port_confidence=[
                [(0.0, 0.0, 0.0, 0.0), (0.0, 0.75, 0.0, 0.75), (0.0, 0.0, 0.0, 0.0)]
            ],
        )

        self.assertTrue(TerraminerGoldAgent.verify_route(board, rotated_solution))

    def test_dry_run_reports_route_without_sending_taps(self):
        phone = Mock()
        phone.resolution = None
        agent = TerraminerGoldAgent(TapExecutor(phone), dry_run=True)
        board = TerraminerBoard(
            tile_masks=[[0, NORTH | SOUTH, 0]],
            start=(0, 0),
            goal=(0, 2),
            bounds=(0, 0, 300, 100),
            tile_port_confidence=[
                [
                    (0.0, 0.0, 0.0, 0.0),
                    (1.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 0.0),
                ]
            ],
        )
        observation = BoardObservation(
            timestamp=0,
            board=board,
            frame_size=(300, 100),
        )

        result = agent.solve(observation)

        self.assertTrue(agent.finished)
        self.assertEqual(result.solution.tap_count, 1)
        phone.tap.assert_not_called()

    def test_execute_solution_jitters_tap_position_and_interval_within_bounds(self):
        phone = Mock()
        phone.resolution = (300, 300)
        agent = TerraminerGoldAgent(TapExecutor(phone), tap_interval=0.04)
        board = TerraminerBoard(
            tile_masks=[[0, 0, 0], [0, NORTH | SOUTH, 0], [0, 0, 0]],
            start=(1, 0),
            goal=(1, 2),
            bounds=(0, 0, 300, 300),
        )
        solution = MazeSolution(
            actions=(TileAction(1, 1, 2, EAST | WEST),),
            path=((1, 0), (1, 1), (1, 2)),
        )
        jitter_values = (4, -3, 0.032, -5, 2)

        with (
            patch(
                "game_automation.games.terraminer_gold.tap_executor.random.uniform",
                side_effect=jitter_values,
            ),
            patch("game_automation.games.terraminer_gold.agent.time.sleep") as sleep,
        ):
            agent._execute_solution(board, solution, (300, 300))

        self.assertEqual(phone.tap.call_args_list, [call(154, 147), call(145, 152)])
        sleep.assert_called_once_with(0.032)


if __name__ == "__main__":
    unittest.main()
