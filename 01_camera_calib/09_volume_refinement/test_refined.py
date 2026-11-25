import unittest
from pathlib import Path
import numpy as np
import open3d as o3d

from refined import refine
from estimator import measure


SOURCE = Path(__file__).resolve().parents[1]/'06_standard_scan/runs/baseline_20261118'


class Refinement(unittest.TestCase):
    def pair(self,case):
        path = SOURCE/case/'reconstruction/cloud.ply'
        points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
        baseline = measure(points)
        original = baseline['height'].copy()
        result = refine(points,baseline)
        np.testing.assert_array_equal(original,baseline['height'])
        return baseline,result

    def test_cylinder_candidate(self):
        base,result = self.pair('A_ideal')
        self.assertEqual(result['method'],'cylinder')
        self.assertLess(abs(result['volume_mm3']-251327.4),abs(base['volume_mm3']-251327.4))

    def test_recesses(self):
        _,result = self.pair('B_ideal')
        xx,yy=np.meshgrid(result['x'],result['y'])
        for center in [-25,25]:
            self.assertAlmostEqual(np.nanmedian(result['height'][np.hypot(xx-center,yy)<7]),35,delta=.3)
        self.assertGreater(result['step_fills'],0)

    def test_curved_outline(self):
        base,result = self.pair('C_jitter')
        self.assertEqual(result['method'],'strip')
        self.assertLess(result['domain'].sum(),base['domain'].sum())
        self.assertTrue(np.isnan(result['height'][~result['domain']]).all())
