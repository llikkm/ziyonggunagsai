"""激光平面标定仿真单元测试。

在 01_camera_calib/04_laser_plane_calib 下运行：
    python -m unittest test_plane_calib -v
"""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

import plane_calib_lib as pc
import laser_lib as ll
import calib_lib as cl


K = np.array([[806.0, 0.0, 640.0], [0.0, 806.0, 360.0], [0.0, 0.0, 1.0]])
CAM = ll.CameraModel(K, np.zeros(5), (1280, 720), source="TEST_806")
TRUE = ll.LaserPlane.from_abcd([1.0, 0.0, 0.2, -100.0], source="TEST_TRUE")
BOARD = pc.BoardSpec(square_mm=20.0)


def _poses(n=8, seed=7):
    ranges = pc.PoseRanges(
        z=(450, 560), x=(-45, 45), y=(-25, 25),
        rx=(-18, 18), ry=(-18, 18), rz=(-12, 12),
    )
    return pc.sample_poses(n, CAM, TRUE, ranges, BOARD, np.random.default_rng(seed))


class TestBoardSpec(unittest.TestCase):
    def test_20mm_geometry(self):
        self.assertEqual(BOARD.inner, (8, 5))
        self.assertAlmostEqual(BOARD.physical_width_mm, 204.0)
        self.assertAlmostEqual(BOARD.physical_height_mm, 144.0)
        np.testing.assert_allclose(BOARD.center_obj, [70.0, 40.0, 0.0])
        self.assertEqual(BOARD.bounds, (-32.0, 172.0, -32.0, 112.0))

    def test_19mm_print_scaled(self):
        b19 = pc.BoardSpec(square_mm=19.0)
        self.assertAlmostEqual(b19.physical_width_mm, 195.0)
        self.assertAlmostEqual(b19.physical_height_mm, 138.0)
        np.testing.assert_allclose(b19.center_obj, [66.5, 38.0, 0.0])


class TestPlaneFit(unittest.TestCase):
    def test_fit_known_plane(self):
        rng = np.random.default_rng(1)
        x, y = np.meshgrid(np.linspace(-40, 40, 12), np.linspace(-30, 30, 10))
        z = 500.0 - 0.2 * x + 0.1 * y
        pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        pts += rng.normal(0, 0.01, pts.shape)
        fit = pc.fit_plane_points(pts)
        ref = ll.LaserPlane.from_abcd([0.2, -0.1, 1.0, -500.0])
        self.assertLess(pc.angle_between_normals(fit.normal, ref.unit_normal), 0.02)
        self.assertLess(fit.residual_rms_mm, 0.05)
        self.assertFalse(fit.degenerate)

    def test_degenerate_line(self):
        s = np.linspace(-50, 50, 100)
        pts = np.column_stack([0.1 * s, 0.2 * s, 500.0 + s])
        fit = pc.fit_plane_points(pts)
        self.assertTrue(fit.degenerate)
        self.assertGreater(fit.condition, 50.0)

    def test_plane_json_roundtrip(self):
        x, y = np.meshgrid(np.linspace(-20, 20, 6), np.linspace(-10, 10, 5))
        pts = np.column_stack([x.ravel(), y.ravel(), 500.0 - 0.2 * x.ravel()])
        fit = pc.fit_plane_points(pts)
        data = fit.to_plane().to_dict()
        parsed = ll.LaserPlane.from_dict(data)
        self.assertLess(pc.angle_between_normals(parsed.normal, fit.normal), 1e-9)
        self.assertAlmostEqual(parsed.unit_d, fit.d, places=9)


class TestBoardRender(unittest.TestCase):
    def test_render_and_detect(self):
        board = pc.BoardSpec(square_mm=20.0)
        R = np.eye(3)
        t = np.array([0.0, 0.0, 500.0]) - R @ board.center_obj
        rvec = np.zeros(3)
        rng = np.random.default_rng(2)
        tex = pc.make_board_texture(board)
        img = pc.render_board_image(rvec, t, tex, board, CAM, rng)
        ok, corners = cl.find_corners(img, board.inner)
        self.assertTrue(ok, "渲染的棋盘格应能检出角点")
        rvec2, tvec2, rms = cl.solve_pose(
            corners.reshape(-1, 1, 2), board.inner, board.square_mm,
            CAM.K, CAM.dist,
        )
        self.assertLess(rms, 1.0)
        self.assertLess(float(np.linalg.norm(tvec2.ravel() - t)), 2.0)
        self.assertLess(float(np.linalg.norm(rvec2.ravel())), 0.02)


class TestIdealCalibration(unittest.TestCase):
    def test_exact_geometry(self):
        poses = _poses(8)
        meas = pc.build_ideal_measurements(poses, CAM, BOARD)
        calib = pc.calibrate_plane(meas, CAM, BOARD)
        ev = pc.evaluate_plane(calib.fit, TRUE)
        self.assertLess(ev["normal_angle_error_deg"], 1e-3)
        self.assertLess(calib.fit.residual_rms_mm, 1e-3)
        self.assertFalse(calib.fit.degenerate)

    def test_measurements_have_no_truth(self):
        poses = _poses(4)
        meas = pc.build_ideal_measurements(poses, CAM, BOARD)
        for m in meas:
            self.assertNotIn("true_plane", m)
            self.assertNotIn("rvec", m)
            self.assertNotIn("tvec", m)
        params = inspect.signature(pc.calibrate_plane).parameters
        self.assertNotIn("true_plane", params)
        calib = pc.calibrate_plane(meas, CAM, BOARD)
        self.assertTrue(np.isfinite(calib.fit.normal).all())


class TestNoisyCalibration(unittest.TestCase):
    def test_noisy_calibration_and_validation(self):
        poses = _poses(8)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rng = np.random.default_rng(3)
            ds = pc.build_dataset(
                poses, CAM, TRUE, BOARD, tmp / "data", rng,
                sigma_corner=0.3, sigma_laser=0.3, save_images=False,
            )
            calib = pc.calibrate_plane(ds.measurements, CAM, BOARD)
            ev = pc.evaluate_plane(calib.fit, TRUE)
            self.assertLess(ev["normal_angle_error_deg"], 1.0)
            cases = pc.build_validation_cases(
                CAM, TRUE, tmp / "validation", rng,
                sigma_laser=0.3, save_images=False,
            )
            measured = pc.measure_validation(calib.fit, [c.measurement for c in cases], CAM)
            _, summary = pc.evaluate_validation(measured, cases)
            self.assertLess(summary["mean_abs_error_mm"], 1.0)


class TestOrientation(unittest.TestCase):
    def test_vertical_stripe_needs_row_scan(self):
        v = np.linspace(280, 520, 180)
        u = np.full_like(v, 640.0)
        uv = np.column_stack([u, v])
        img = ll.render_laser_image(
            uv, (1280, 720), sigma_px=1.4, peak=220.0, background=6.0,
            gradient=(2.0, 2.0), read_sigma=1.0, shot_gain=0.1,
            rng=np.random.default_rng(4),
        )
        ext_row = ll.extract_laser_centers(
            img.astype(float), axis="row", min_peak=12.0,
            min_peak_fraction=0.1,
        )
        ext_col = ll.extract_laser_centers(
            img.astype(float), axis="column", min_peak=12.0,
            min_peak_fraction=0.1,
        )
        self.assertGreater(len(ext_row.uv), 5 * max(1, len(ext_col.uv)))
        row_err = ll.distance_to_polyline(ext_row.uv, uv)
        col_err = ll.distance_to_polyline(ext_col.uv, uv)
        self.assertLess(float(np.mean(row_err)), float(np.mean(col_err)))

    def test_auto_with_tight_roi(self):
        v = np.linspace(280, 520, 180)
        uv = np.column_stack([np.full_like(v, 640.0), v])
        img = ll.render_laser_image(
            uv, (1280, 720), sigma_px=1.4, peak=220.0, background=6.0,
            read_sigma=1.0, shot_gain=0.1, rng=np.random.default_rng(5),
        )
        ext = ll.extract_laser_centers(
            img.astype(float), axis="auto", roi=(610, 670, 250, 550),
            min_peak=12.0, min_peak_fraction=0.1,
        )
        self.assertEqual(ext.axis, "row")


if __name__ == "__main__":
    unittest.main(verbosity=2)
