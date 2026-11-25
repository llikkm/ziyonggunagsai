"""laser_lib / simulate_step 的单元测试。

在 01_camera_calib/03_laser_triangulation 下运行：
    python -m unittest test_laser_lib -v
"""

from __future__ import annotations

import unittest

import numpy as np

import laser_lib as ll
import simulate_step as sim

K = sim.K_TRUE.copy()
CAMERA = ll.CameraModel(K, np.zeros(5), sim.IMAGE_SIZE, source="TEST")
PLANE = ll.LaserPlane.from_abcd([0.0, 1.0, 0.2, -120.0], source="TEST")
SCENE = ll.StepScene(z_far=500.0, z_near=469.0, x_min=-120.0, x_max=120.0)


def _truth_and_pixels():
    truth, roles = SCENE.curve_points(PLANE)
    uv = CAMERA.project(truth)
    return truth, roles, uv


class TestGeometry(unittest.TestCase):
    def test_roundtrip_no_noise(self):
        truth, roles, uv = _truth_and_pixels()
        rec = ll.reconstruct(uv, CAMERA, PLANE)
        self.assertTrue(rec.valid.all())
        self.assertLess(float(np.max(np.linalg.norm(rec.points_mm - truth, axis=1))), 1e-9)
        est = ll.robust_step_height(rec.points_mm)
        self.assertTrue(est.has_step)
        self.assertAlmostEqual(est.height_mm, 31.0, places=9)
        self.assertAlmostEqual(ll.rmse_3d(rec.points_mm, truth), 0.0, places=9)

    def test_intersect_pixels_wrapper(self):
        truth, _, uv = _truth_and_pixels()
        pts, valid = ll.intersect_pixels(uv, K, np.zeros(5), PLANE.abcd)
        self.assertTrue(valid.all())
        self.assertLess(float(np.max(np.linalg.norm(pts - truth, axis=1))), 1e-8)

    def test_project_behind_camera(self):
        uv = CAMERA.project(np.array([[0.0, 0.0, -10.0]]))
        self.assertTrue(np.isnan(uv).all())

    def test_wrong_focal_length(self):
        truth, _, uv = _truth_and_pixels()
        for f in (800.0, 820.0):
            cam = sim.build_camera(f, f)
            rec = ll.reconstruct(uv, cam, PLANE)
            est = ll.robust_step_height(rec.points_mm)
            self.assertTrue(est.has_step)
            self.assertGreater(abs(est.height_mm - 31.0), 0.05,
                               f"f={f} 应产生可见的系统性误差")
        # 控制组
        rec = ll.reconstruct(uv, sim.build_camera(806.0, 806.0), PLANE)
        est = ll.robust_step_height(rec.points_mm)
        self.assertAlmostEqual(est.height_mm, 31.0, places=9)

    def test_plane_normal_error(self):
        truth, _, uv = _truth_and_pixels()
        pivot = np.array([0.0,
                          0.5 * (float(PLANE.y_at_xz(0.0, SCENE.z_far))
                                 + float(PLANE.y_at_xz(0.0, SCENE.z_near))),
                          0.5 * (SCENE.z_far + SCENE.z_near)])
        for nz in (0.198, 0.205):
            p = ll.LaserPlane.from_point_normal(pivot, [0.0, 1.0, nz])
            rec = ll.reconstruct(uv, CAMERA, p)
            est = ll.robust_step_height(rec.points_mm)
            self.assertGreater(abs(est.height_mm - 31.0), 0.05)
        p = ll.LaserPlane.from_point_normal(pivot, [0.0, 1.0, 0.2])
        rec = ll.reconstruct(uv, CAMERA, p)
        self.assertAlmostEqual(ll.robust_step_height(rec.points_mm).height_mm, 31.0, places=6)


class TestPlaneFit(unittest.TestCase):
    def test_fit_plane_svd(self):
        rng = np.random.default_rng(3)
        x, y = np.meshgrid(np.linspace(-50, 50, 12), np.linspace(-30, 30, 9))
        z = 500.0 - 0.2 * x + 0.1 * y
        pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        pts += rng.normal(0.0, 0.01, pts.shape)
        fitted = ll.fit_plane_svd(pts)
        ref = ll.LaserPlane.from_abcd([0.2, -0.1, 1.0, -500.0])
        self.assertLess(fitted.angle_to(ref), 0.1)
        self.assertLess(float(np.sqrt(np.mean(fitted.distance(pts) ** 2))), 0.05)

    def test_three_points(self):
        p = ll.plane_from_three_points([0, 0, 500], [1, 0, 500], [0, 1, 500])
        self.assertAlmostEqual(float(p.distance(np.array([[5.0, 6.0, 500.0]]))[0]), 0.0, places=12)

    def test_collinear_points(self):
        with self.assertRaises(ValueError):
            ll.plane_from_three_points([0, 0, 0], [1, 0, 0], [2, 0, 0])


class TestRobustHeight(unittest.TestCase):
    def test_two_level_split(self):
        x = np.r_[np.full(100, 469.0), np.full(100, 500.0)]
        labels, lo, hi = ll.two_level_split(x)
        self.assertEqual(int(labels.sum()), 100)
        self.assertAlmostEqual(lo, 469.0)
        self.assertAlmostEqual(hi, 500.0)

    def test_outliers_do_not_move_height(self):
        rng = np.random.default_rng(4)
        far = np.column_stack([np.linspace(-100, -1, 200),
                               np.full(200, 20.0), np.full(200, 500.0)])
        near = np.column_stack([np.linspace(1, 100, 200),
                                np.full(200, 26.2), np.full(200, 469.0)])
        outliers = np.column_stack([rng.uniform(-100, 100, 40),
                                    rng.uniform(20, 27, 40),
                                    rng.uniform(475, 495, 40)])
        pts = np.vstack([far, near, outliers])
        est = ll.robust_step_height(pts)
        self.assertTrue(est.has_step)
        self.assertLess(abs(est.height_mm - 31.0), 0.02)
        self.assertGreater(est.n_outliers, 0)

    def test_flat_surface_is_not_a_step(self):
        rng = np.random.default_rng(5)
        pts = np.column_stack([rng.uniform(-100, 100, 400),
                               rng.uniform(20, 26, 400),
                               500.0 + rng.normal(0, 1.0, 400)])
        est = ll.robust_step_height(pts, min_separation_mm=2.0)
        self.assertFalse(est.has_step)


class TestExtraction(unittest.TestCase):
    def _render_and_extract(self, dropout_spans=(), hotspots=0, seed=11):
        truth, roles, uv = _truth_and_pixels()
        img = ll.render_laser_image(
            uv, sim.IMAGE_SIZE, sigma_px=1.3, peak=190.0, background=9.0,
            gradient=(4.0, 2.0), read_sigma=1.5, shot_gain=0.2,
            dropout_spans=dropout_spans, hotspot_count=hotspots,
            rng=np.random.default_rng(seed),
        )
        u_min, u_max = uv[:, 0].min(), uv[:, 0].max()
        v_min, v_max = uv[:, 1].min(), uv[:, 1].max()
        roi = (int(np.ceil(u_min)), int(np.floor(u_max)) + 1,
               int(np.floor(v_min - 10)), int(np.ceil(v_max + 10)))
        ext = ll.extract_laser_centers(
            img, axis="auto", roi=roi, smooth_sigma=1.0,
            threshold_sigma=4.0, min_peak=10.0, centroid_radius=4, max_width=9.0,
        )
        return truth, uv, img, ext

    def test_clean_extraction(self):
        truth, uv, img, ext = self._render_and_extract()
        self.assertGreater(len(ext.uv), 300)
        err = ll.distance_to_polyline(ext.uv, uv)
        self.assertLess(float(np.mean(err)), 0.1)
        rec = ll.reconstruct(ext.uv, CAMERA, PLANE)
        est = ll.robust_step_height(rec.points_mm[rec.valid])
        self.assertTrue(est.has_step)
        self.assertLess(abs(est.height_mm - 31.0), 0.2)

    def test_dropout_extraction(self):
        truth, uv, img, ext = self._render_and_extract(
            dropout_spans=[(560.0, 590.0), (700.0, 720.0)], hotspots=4)
        self.assertLess(ext.valid_fraction, 1.0)
        self.assertGreater(len(ext.uv), 250)
        rec = ll.reconstruct(ext.uv, CAMERA, PLANE)
        est = ll.robust_step_height(rec.points_mm[rec.valid])
        self.assertTrue(est.has_step)
        self.assertLess(abs(est.height_mm - 31.0), 0.5)

    def test_row_mode_on_vertical_line(self):
        # 人为构造一条近竖直亮线，验证 row 模式也能工作。
        img = np.full((200, 160), 8.0)
        for v in range(20, 180):
            u = 80.0 + 0.05 * (v - 100)
            uu = np.arange(img.shape[1])
            img[v, :] += 160.0 * np.exp(-((uu - u) ** 2) / (2 * 1.2 ** 2))
        ext = ll.extract_laser_centers(img, axis="row", min_peak=10.0)
        self.assertEqual(ext.axis, "row")
        self.assertGreater(len(ext.uv), 140)
        true_u = 80.0 + 0.05 * (ext.uv[:, 1] - 100.0)
        self.assertLess(float(np.mean(np.abs(ext.uv[:, 0] - true_u))), 0.2)


class TestDistortion(unittest.TestCase):
    def test_distorted_roundtrip(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV 不可用")
        dist = np.array([-0.08, 0.03, 0.0005, -0.0004, 0.0])
        cam = ll.CameraModel(K, dist, sim.IMAGE_SIZE, source="TEST_DIST")
        truth, _, _ = _truth_and_pixels()
        uv = cam.project(truth)
        self.assertTrue(np.isfinite(uv).all())
        rec = ll.reconstruct(uv, cam, PLANE)
        self.assertTrue(rec.valid.all())
        err = float(np.max(np.linalg.norm(rec.points_mm - truth, axis=1)))
        self.assertLess(err, 1e-6)


class TestValidation(unittest.TestCase):
    def test_zero_plane_normal(self):
        with self.assertRaises(ValueError):
            ll.LaserPlane([0.0, 0.0, 0.0], -1.0)

    def test_bad_camera_matrix(self):
        with self.assertRaises(ValueError):
            ll.CameraModel(np.zeros((3, 3)), np.zeros(5))

    def test_non_2d_image(self):
        with self.assertRaises(ValueError):
            ll.extract_laser_centers(np.zeros(10))

    def test_roi_out_of_bounds(self):
        with self.assertRaises(ValueError):
            ll.extract_laser_centers(np.zeros((10, 10)), roi=(0, 11, 0, 10))


if __name__ == "__main__":
    unittest.main(verbosity=2)
