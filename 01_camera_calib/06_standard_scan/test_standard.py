import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from models import build
from simulate import make_scene, first_hits, camera_visible, in_frame


class StandardGeometry(unittest.TestCase):
    def test_exported_solids(self):
        cases = [('A', 80000*np.pi, [80,80,50]), ('B', 318000-500*np.pi, [100,80,40]), ('C', 300000, [100,100,38])]
        with tempfile.TemporaryDirectory() as directory:
            for name, volume, extent in cases:
                with self.subTest(name=name):
                    path = Path(directory)/f'{name}.ply'
                    self.assertTrue(o3d.io.write_triangle_mesh(str(path), build(name)))
                    mesh = o3d.io.read_triangle_mesh(str(path))
                    self.assertTrue(mesh.is_watertight())
                    self.assertTrue(mesh.is_orientable())
                    self.assertAlmostEqual(mesh.get_volume()/volume, 1, delta=0.0002)
                    np.testing.assert_allclose(mesh.get_axis_aligned_bounding_box().get_extent(), extent, atol=0.02)

    def test_recesses_and_wave(self):
        for name, locations in [('B',[(-25,0,35),(-16,9,35),(-14,0,40),(25,0,35),(33,8,40),(0,0,40)]),('C',[(25,25,38),(-25,25,22),(0,0,30)])]:
            scene = make_scene(build(name))
            origins = np.array([[xx,yy,100] for xx,yy,_ in locations])
            hits = first_hits(scene, origins, np.tile([0,0,-1],(len(origins),1)))
            np.testing.assert_allclose(hits[:,2], [item[2] for item in locations], atol=0.02)


class Visibility(unittest.TestCase):
    def setUp(self):
        self.scene = make_scene(o3d.geometry.TriangleMesh.create_box(2,2,2))

    def test_laser_shadow(self):
        hit = first_hits(self.scene, [[1,1,10]], [[0,0,-1]])
        np.testing.assert_allclose(hit, [[1,1,2]], atol=1e-4)

    def test_camera_occlusion(self):
        np.testing.assert_array_equal(camera_visible(self.scene, [-5,1,1], [[0,1,1],[2,1,1]]), [True,False])

    def test_image_bounds(self):
        np.testing.assert_array_equal(in_frame(np.array([[0,0],[19,9],[20,4],[-1,0],[4,10]]),20,10),[1,1,0,0,0])
