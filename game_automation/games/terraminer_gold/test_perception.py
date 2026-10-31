import unittest

import cv2
import numpy as np

from .perception import TilePortDetector


class TileMaskPerceptionTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((400, 400, 3), (5, 193, 206), dtype=np.uint8)
        self.gold = (46, 202, 246)

    def _draw_vertical(self, row, column):
        center_x = column * 100 + 50
        x1, x2 = center_x - 13, center_x + 13
        y1, y2 = row * 100 + 10, row * 100 + 90
        cv2.rectangle(self.frame, (x1, y1), (x2, y2), self.gold, thickness=-1)

    def _detect(self):
        return TilePortDetector._detect_tile_masks(
            self.frame,
            (0, 0, 400, 400),
            4,
        )

    def test_vertical_straight_tile_has_only_north_and_south_ports(self):
        self._draw_vertical(1, 1)

        masks, confidence = self._detect()

        self.assertEqual(masks[1][1], 1 | 4)
        self.assertEqual(confidence[1][1][0], 1.0)
        self.assertEqual(confidence[1][1][2], 1.0)
        self.assertLess(confidence[1][1][1], 0.75)
        self.assertLess(confidence[1][1][3], 0.75)
        self.assertEqual(confidence[1][1][1], 0.0)
        self.assertEqual(confidence[1][1][3], 0.0)

    def test_three_contiguous_supported_samples_retain_port_confidence(self):
        component = np.zeros((80, 80), dtype=np.uint8)
        gradient_x = np.zeros((80, 80), dtype=np.float32)
        gradient_y = np.zeros((80, 80), dtype=np.float32)
        edge_magnitude = np.zeros((80, 80), dtype=np.float32)
        edges = np.zeros((80, 80), dtype=bool)
        bin_edges = np.linspace(10, 38, 9, dtype=int)
        sample_start, sample_end = bin_edges[1], bin_edges[4]
        component[sample_start:sample_end, 31:49] = 255
        for edge_x in (25, 50):
            gradient_x[sample_start:sample_end, edge_x] = 100
            edge_magnitude[sample_start:sample_end, edge_x] = 100
            edges[sample_start:sample_end, edge_x] = True

        confidence = TilePortDetector._connected_port_confidence(
            component,
            0,
            (gradient_x, gradient_y, edge_magnitude, edges),
        )

        self.assertAlmostEqual(confidence, 0.75)

    def test_two_isolated_supported_samples_do_not_form_a_port(self):
        component = np.zeros((80, 80), dtype=np.uint8)
        gradient_x = np.zeros((80, 80), dtype=np.float32)
        gradient_y = np.zeros((80, 80), dtype=np.float32)
        edge_magnitude = np.zeros((80, 80), dtype=np.float32)
        edges = np.zeros((80, 80), dtype=bool)
        bin_edges = np.linspace(10, 38, 9, dtype=int)
        for sample_index in (1, 5):
            sample_start, sample_end = bin_edges[sample_index:sample_index + 2]
            component[sample_start:sample_end, 31:49] = 255
            for edge_x in (25, 50):
                gradient_x[sample_start:sample_end, edge_x] = 100
                edge_magnitude[sample_start:sample_end, edge_x] = 100
                edges[sample_start:sample_end, edge_x] = True

        confidence = TilePortDetector._connected_port_confidence(
            component,
            0,
            (gradient_x, gradient_y, edge_magnitude, edges),
        )

        self.assertLess(confidence, 0.75)

    def test_disconnected_gold_background_marks_no_ports(self):
        self.frame[145:155, 170:180] = (4, 140, 170)
        self.frame[145:155, 220:230] = (4, 140, 170)

        masks, _ = self._detect()

        self.assertEqual(masks[1][1], 0)
        self.assertEqual(masks[1][2], 0)

    def test_connected_elbow_exposes_only_its_two_ports(self):
        center_x, center_y = 150, 150
        cv2.rectangle(self.frame, (center_x - 13, 110), (center_x + 13, center_y), self.gold, -1)
        cv2.rectangle(self.frame, (center_x, center_y - 13), (190, center_y + 13), self.gold, -1)

        masks, _ = self._detect()

        self.assertEqual(masks[1][1], 1 | 2)

    def test_brick_seams_do_not_break_straight_or_elbow_ports(self):
        center_x, center_y = 150, 150
        cv2.rectangle(
            self.frame,
            (center_x - 13, 110),
            (center_x + 13, center_y),
            self.gold,
            thickness=-1,
        )
        cv2.rectangle(
            self.frame,
            (center_x, center_y - 13),
            (center_x + 40, center_y + 13),
            self.gold,
            thickness=-1,
        )
        for y in range(120, 150, 12):
            self.frame[y:y + 4, center_x - 13:center_x + 14] = (40, 180, 40)
        for x in range(160, 190, 12):
            self.frame[center_y - 13:center_y + 14, x:x + 4] = (40, 180, 40)

        masks, _ = self._detect()

        self.assertEqual(masks[1][1], 1 | 2)

    def test_horizontal_straight_with_brick_seams_exposes_east_and_west(self):
        center_y = 150
        cv2.rectangle(
            self.frame,
            (110, center_y - 13),
            (190, center_y + 13),
            self.gold,
            thickness=-1,
        )
        for x in range(120, 190, 12):
            self.frame[center_y - 13:center_y + 14, x:x + 4] = (40, 180, 40)

        masks, _ = self._detect()

        self.assertEqual(masks[1][1], 2 | 8)

    def test_screenshot_like_yellow_terrain_does_not_become_four_way_ports(self):
        self.frame[:] = (5, 193, 206)
        self.frame[::4, ::4] = (1, 168, 167)
        self.frame[2::5, 3::5] = (0, 140, 160)

        masks, confidence = self._detect()

        for row in range(4):
            for column in range(4):
                self.assertEqual(masks[row][column], 0)
                self.assertTrue(all(value < 0.75 for value in confidence[row][column]))

    def test_broad_gold_texture_with_parallel_edges_is_not_a_port(self):
        self.frame[100:200, 100:200] = self.gold
        for offset in (25, 43, 57, 75):
            cv2.line(
                self.frame,
                (100 + offset, 100),
                (100 + offset, 199),
                (40, 180, 40),
                thickness=3,
            )
            cv2.line(
                self.frame,
                (100, 100 + offset),
                (199, 100 + offset),
                (40, 180, 40),
                thickness=3,
            )

        masks, confidence = self._detect()

        self.assertEqual(masks[1][1], 0)
        self.assertTrue(all(value < 0.75 for value in confidence[1][1]))

    def test_bright_borders_without_gold_interior_do_not_form_ports(self):
        self.frame[137:140, 110:190] = (255, 255, 255)
        self.frame[160:163, 110:190] = (255, 255, 255)

        masks, _ = self._detect()

        self.assertEqual(masks[1][1], 0)


if __name__ == "__main__":
    unittest.main()
