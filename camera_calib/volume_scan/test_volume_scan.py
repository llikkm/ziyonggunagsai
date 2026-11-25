"""多帧扫描 / 高度图体积的单元测试。

在 camera_calib/volume_scan 下运行：
    python -m unittest test_volume_scan -v
"""

from __future__ import annotations

import inspect
import unittest

import numpy as np

import scan_lib as sc
import laser_lib as ll


CAM = ll.CameraModel(
    np.array([[806.0, 0.0, 640.0], [0.0, 806.0, 360.0], [0.0, 0.0, 1.0]]),
    np.zeros(5), (1280, 720), source="TEST_806")


def one_scan(obj, step=2.0, span=50.0, seed=0):
    setup = sc.ScanSetup(camera=CAM,
                         scan_x_mm=np.arange(-span, span + step * 0.5, step))
    scan = sc.simulate_scan(obj, setup, seed=seed, read_sigma=0.1, shot_gain=0.0)
    return setup, scan


def true_base():
    return sc.BasePlane(np.array([0.0, 0.0, 1.0]), 0.0, 0.0, 1.0, 0)


def volume_of(scan, obj, plane=None, pixel_noise=0.0, motion=0.0,
              fill=6, cell=0.5, known_base=True):
    plane = scan.setup.laser_plane_c if plane is None else plane
    pc = sc.reconstruct_scan(scan, plane, pixel_noise_px=pixel_noise,
                             motion_sigma_mm=motion, seed=3)
    base = true_base() if known_base else sc.estimate_base_plane(
        pc.points_world, np.random.default_rng(0))
    foot = sc.known_footprint_in_base(obj, base)
    hm = sc.height_map_volume(pc.points_world, base, cell_mm=cell,
                              fill_iterations=fill, footprint=foot)
    return hm.volume_mm3 / 1000.0, base, pc, hm


class TestObjects(unittest.TestCase):
    def test_true_volumes(self):
        self.assertEqual(sc.make_object("A").true_volume_cm3, 400.0)
        self.assertEqual(sc.make_object("B").true_volume_cm3, 396.0)
        self.assertEqual(sc.make_object("C").true_volume_cm3, 368.0)

    def test_slot_and_hole_geometry(self):
        b = sc.make_object("B")
        self.assertEqual(sum(q.name.startswith("slot_") for q in b.quads), 5)
        self.assertTrue(any(q.name == "slot_bottom" for q in b.quads))
        c = sc.make_object("C")
        self.assertTrue(any(q.name.startswith("hole_") for q in c.quads))
        self.assertEqual(sum(q.name.startswith("hole_") for q in c.quads), 4)


class TestGeometry(unittest.TestCase):
    def test_plane_quad_intersection(self):
        box = sc.make_object("A")
        top = [q for q in box.quads if q.name == "box_zmax"][0]
        plane = ll.LaserPlane.from_abcd([1.0, 0.0, 0.0, -50.0])  # X=50
        pts = sc.plane_quad_segment(plane, top, point_spacing_mm=1.0)
        self.assertIsNotNone(pts)
        self.assertGreater(len(pts), 10)
        np.testing.assert_allclose(pts[:, 0], 50.0, atol=1e-9)
        np.testing.assert_allclose(pts[:, 2], 50.0, atol=1e-9)
        self.assertAlmostEqual(float(pts[:, 1].min()), 0.0, places=6)
        self.assertAlmostEqual(float(pts[:, 1].max()), 80.0, places=6)

    def test_ray_blocking(self):
        box = sc.make_object("A")
        quads = list(box.quads) + [sc.ground_quad()]
        inside = np.array([50.0, 40.0, 40.0])
        above = np.array([50.0, 40.0, 100.0])
        self.assertTrue(sc.ray_blocked(inside, above, quads, skip_index=-1))
        outside = np.array([50.0, 40.0, 60.0])
        self.assertFalse(sc.ray_blocked(outside, above, quads, skip_index=-1))

    def test_visibility_sees_top_and_ground(self):
        setup = sc.ScanSetup(camera=CAM, scan_x_mm=np.array([0.0]))
        obj = sc.make_object("A")
        vis = sc.visible_laser_points(list(obj.quads) + [sc.ground_quad()],
                                      setup, setup.pose(0))
        self.assertGreater(vis.n_visible, 100)
        self.assertIn("box_zmax", vis.visible_by_quad)
        self.assertIn("ground", vis.visible_by_quad)


class TestAnalyticVolume(unittest.TestCase):
    def test_analytic_a_b_c(self):
        self.assertAlmostEqual(sc.analytic_height_map_volume("A", 0.1), 400.0, places=6)
        self.assertAlmostEqual(sc.analytic_height_map_volume("B", 0.1), 396.0, places=6)
        # 高度图只看到 C 的完整顶面；真实 368，差 32 是隐藏侧孔
        self.assertAlmostEqual(sc.analytic_height_map_volume("C", 0.1), 400.0, places=6)
        self.assertAlmostEqual(sc.make_object("C").true_volume_cm3, 368.0, places=6)

    def test_dense_height_map_a_b(self):
        for name, target in (("A", 400.0), ("B", 396.0)):
            x = np.arange(0.25, 100.0, 0.5)
            y = np.arange(0.25, 80.0, 0.5)
            xg, yg = np.meshgrid(x, y)
            z = sc.true_height_at(name, xg, yg)
            pts = np.column_stack([xg.ravel(), yg.ravel(), z.ravel()])
            base = sc.BasePlane(np.array([0.0, 0.0, 1.0]), 0.0, 0.0, 1.0, len(pts))
            foot = (0.0, 100.0, 0.0, 80.0)
            hm = sc.height_map_volume(pts, base, cell_mm=0.5, footprint=foot)
            # 槽边界格取 max 会把半个格算成顶面；B 允许 ~0.6 cm³ 的格边界误差
            self.assertAlmostEqual(hm.volume_mm3 / 1000.0, target,
                                   delta=0.02 if name == "A" else 0.8)


class TestScanReconstruction(unittest.TestCase):
    def test_a_scan_volume(self):
        obj = sc.make_object("A")
        setup, scan = one_scan(obj)
        vol, base, pc, hm = volume_of(scan, obj)
        self.assertLess(abs(vol - 400.0), 2.0)
        self.assertGreater(base.normal[2], 0.999)
        self.assertGreater(len(pc.points_world), 1000)
        self.assertGreater(hm.coverage, 0.15)

    def test_b_slot_visibility_and_volume(self):
        obj = sc.make_object("B")
        setup, scan = one_scan(obj)
        vis = sc.visible_fraction(scan, ["slot_bottom"])
        self.assertIsNotNone(vis)
        self.assertGreater(vis, 0.5)
        vol, _, _, _ = volume_of(scan, obj)
        self.assertLess(abs(vol - 396.0), 2.0)

    def test_c_height_map_limitation(self):
        c = sc.make_object("C")
        self.assertGreater(sc.analytic_height_map_volume("C") - 368.0, 20.0)
        setup, scan = one_scan(c)
        self.assertLess(sc.visible_fraction(scan, ["hole_x0", "hole_x1",
                                                   "hole_z0", "hole_z1"]) or 0.0, 0.01)

    def test_motion_plane_and_noise_change_volume(self):
        obj = sc.make_object("A")
        setup, scan = one_scan(obj)
        base_vol, _, _, _ = volume_of(scan, obj)
        noise_vol, _, _, _ = volume_of(scan, obj, pixel_noise=1.0)
        motion_vol, _, _, _ = volume_of(scan, obj, motion=1.0)
        plane = ll.LaserPlane.from_abcd([1.0, 0.0, 0.205, -100.0])
        plane_vol, _, _, _ = volume_of(scan, obj, plane=plane)
        self.assertGreater(abs(noise_vol - base_vol), 0.5)
        self.assertGreater(abs(motion_vol - base_vol), 0.1)
        self.assertGreater(abs(plane_vol - base_vol), 2.0)

    def test_truth_isolation(self):
        params = inspect.signature(sc.reconstruct_scan).parameters
        self.assertNotIn("true_plane", params)
        self.assertNotIn("true_points", params)
        fields = sc.ScanFrame.__dataclass_fields__
        self.assertNotIn("true_points", fields)
        self.assertNotIn("surface_points", fields)
        self.assertIn("uv", fields)
        self.assertIn("pose", fields)


if __name__ == "__main__":
    unittest.main(verbosity=2)
