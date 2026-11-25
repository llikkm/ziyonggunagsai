import importlib.util
import unittest

import cv2
import numpy as np

spec = importlib.util.find_spec('refine_calibration')
if spec:
    import refine_calibration as r


class RefinementTests(unittest.TestCase):
    def test_recovers_focal_length_from_poor_initial_estimate(self):
        self.assertIsNotNone(spec, 'Calibration refinement entry missing')
        true_k = np.array([[800., 0, 640], [0, 802., 360], [0, 0, 1]])
        distortion = np.array([.2, -.4, .001, -.002, .3])
        obj = np.zeros((40, 3), np.float32)
        obj[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2)*19
        objects, pixels = [], []
        for i in range(12):
            rv = np.array([.03+.008*i, -.08+.01*i, .15*i])
            tv = np.array([-65.+i, -40.+i, 450.+10*i])
            image = cv2.projectPoints(obj, rv, tv, true_k, distortion)[0]
            objects.append(obj)
            pixels.append(image)
        initial = true_k.copy()
        initial[0, 0] = 1110.
        initial[1, 1] = 1112.
        solved = r.refine_intrinsics(objects, pixels, (1280, 720), initial, np.zeros(5))
        self.assertLess(solved['rms_px'], 1e-3)
        np.testing.assert_allclose(solved['camera_matrix'], true_k, atol=.1)
        self.assertEqual(solved['max_iterations'], 300)


if __name__ == '__main__':
    unittest.main(verbosity=2)
