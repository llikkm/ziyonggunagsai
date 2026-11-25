"""数学验证；所有由 projectPoints 生成的观测均为合成数据。"""
import importlib.util
import unittest

import cv2
import numpy as np

spec = importlib.util.find_spec("measure3d")
if spec:
    import measure3d as m


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(spec, "独立三维测量模块尚未实现")
        self.K = np.array([[980., 0, 640], [0, 974., 360], [0, 0, 1]])
        self.dist = np.array([-.12, .025, .001, -.002, 0.])

    def test_distorted_oblique_plane_roundtrip(self):
        xy = np.array([[-80., -40.], [0, 0], [70, 50], [120, -50]])
        xyz = np.column_stack([xy, (500 - .2*xy[:, 0] + .1*xy[:, 1])])
        uv, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), self.K, self.dist)
        points, valid = m.intersect_pixels(uv.reshape(-1, 2), self.K, self.dist, [.2, -.1, 1, -500])
        self.assertTrue(valid.all())
        np.testing.assert_allclose(points, xyz, atol=1e-5)
        scaled, mask = m.intersect_pixels(uv.reshape(-1, 2), self.K, self.dist, [-2e-9, 1e-9, -1e-8, 5e-6])
        np.testing.assert_allclose(scaled, points, atol=1e-6)
        self.assertTrue(mask.all())

    def test_invalid_intersections(self):
        uv = [[640, 360], [738, 360]]
        pts, valid = m.intersect_pixels(uv, self.K, np.zeros(5), [1, 0, 0, -10])
        self.assertEqual(valid.tolist(), [False, True])
        self.assertTrue(np.isnan(pts[0]).all())
        _, valid = m.intersect_pixels(uv, self.K, np.zeros(5), [0, 0, 1, 10])
        self.assertFalse(valid.any())
        for plane in ([0, 0, 0, 1], [1, 0, np.nan, 1]):
            with self.assertRaises(ValueError):
                m.intersect_pixels(uv, self.K, self.dist, plane)

    def test_pnp_and_height_along_tilted_normal(self):
        rv = np.array([.3, -.2, .08])
        R = cv2.Rodrigues(rv)[0]
        ref = {"rvec": rv.tolist(), "tvec_mm": [-60, -40, 600]}
        samples = []
        obj = m.cl.object_points((8, 5), 20).astype(float)
        for height in (19., 20., 21.):
            t = np.array(ref["tvec_mm"]) + R[:, 2]*height + R[:, 0]*30
            uv, _ = cv2.projectPoints(obj, rv, t, self.K, self.dist)
            pose = m.pose_from_corners(uv, (8, 5), 20, self.K, self.dist)
            np.testing.assert_allclose(pose["tvec_mm"], t, atol=1e-3)
            samples.append(pose)
        report = m.height_statistics(ref, samples, 20, 2)
        self.assertAlmostEqual(report["mean_mm"], 20., places=3)
        self.assertAlmostEqual(report["absolute_error_mm"], 0., places=3)
        self.assertAlmostEqual(report["std_mm"], 1., places=3)
        self.assertAlmostEqual(report["mean_absolute_error_mm"], 2/3, places=3)

    def test_height_rejects_tilt_and_insufficient_samples(self):
        ref = {"rvec": [0, 0, 0], "tvec_mm": [0, 0, 600]}
        bad = {"rvec": [.2, 0, 0], "tvec_mm": [0, 0, 620]}
        with self.assertRaises(ValueError):
            m.height_statistics(ref, [bad, bad], 20, 2)
        with self.assertRaises(ValueError):
            m.height_statistics(ref, [ref], 20, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
