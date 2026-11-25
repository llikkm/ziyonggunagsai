import importlib.util
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

spec = importlib.util.find_spec('capture3d')
if spec:
    import capture3d as c


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(spec, 'Multi-camera capture entry missing')

    def test_relative_angle_allows_oblique_camera(self):
        ref = {'rvec': [.6, -.3, .1], 'reprojection_rms_px': .4}
        good, angle, reason = c.pose_quality(ref, ref, 3, 2)
        self.assertTrue(good)
        self.assertAlmostEqual(angle, 0, places=5)
        tilted = {'rvec': [.8, -.3, .1], 'reprojection_rms_px': .4}
        good, angle, reason = c.pose_quality(tilted, ref, 3, 2)
        self.assertFalse(good)
        self.assertGreater(angle, 2)

    def test_reprojection_gate(self):
        pose = {'rvec': [0, 0, 0], 'reprojection_rms_px': 4}
        self.assertFalse(c.pose_quality(pose, None, 3, 2)[0])

    def test_save_raw_frame_without_overwriting(self):
        frame = np.full((12, 16, 3), 100, np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'frame.png'
            c.save_frame(path, frame)
            np.testing.assert_array_equal(cv2.imdecode(np.fromfile(path, np.uint8), 1), frame)
            with self.assertRaises(FileExistsError):
                c.save_frame(path, frame)

    def test_calibration_grid_positions_and_overlay(self):
        self.assertTrue(hasattr(c, 'grid_cell'), 'Calibration grid is missing')
        frame = np.zeros((720, 1280, 3), np.uint8)
        counts = np.zeros((3, 3), dtype=int)
        for row in range(3):
            for col in range(3):
                corners = np.array([[[1280*(col+.5)/3, 720*(row+.5)/3]]], np.float32)
                self.assertEqual(c.grid_cell(corners, frame.shape), (row, col))
                counts[c.grid_cell(corners, frame.shape)] += 1
        np.testing.assert_array_equal(counts, np.ones((3, 3)))
        preview = c.draw_calibration_grid(frame, counts, corners)
        self.assertTrue(np.any(preview[:, 1280//3]))
        self.assertTrue(np.any(preview[720//3]))
        self.assertFalse(frame.any())


if __name__ == '__main__':
    unittest.main(verbosity=2)
