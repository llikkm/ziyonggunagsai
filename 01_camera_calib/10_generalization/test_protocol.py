import unittest
import numpy as np
from solids import specs, solid
from protocol import check_lock


class ProtocolTests(unittest.TestCase):
    def test_lock(self):
        check_lock()

    def test_disjoint_reproducible(self):
        dev, held = specs('development'), specs('holdout')
        self.assertEqual(dev,specs('development'))
        self.assertNotEqual(dev,held)
        for split in (dev,held):
            for category in 'ABC':
                self.assertGreaterEqual(len({s['kind'] for s in split if s['category']==category}),3)

    def test_closed_in_bounds(self):
        for split in ('development','holdout'):
            for spec in specs(split):
                with self.subTest(spec=spec):
                    mesh = solid(spec)
                    self.assertTrue(mesh.is_watertight())
                    extent = mesh.get_axis_aligned_bounding_box().get_extent()
                    self.assertTrue(np.all((extent>=50-1e-7)&(extent<=120+1e-7)))
                    self.assertTrue(50000<=mesh.get_volume()<=800000)
                    self.assertAlmostEqual(mesh.get_min_bound()[2],0)
