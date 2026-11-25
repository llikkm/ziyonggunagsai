import unittest
from pathlib import Path

import numpy as np

from estimator import measure, integrate, fit_table


ARCHIVE = Path(__file__).resolve().parents[1]/'06_standard_scan/runs/baseline_20261118'


def cloud(case):
    with np.load(ARCHIVE/case/'reconstruction/points.npz') as data:
        return data['xyz_mm']


class HeightmapTests(unittest.TestCase):
    def test_ideal_integral_and_missing(self):
        self.assertEqual(integrate(np.array([[40.,35.],[np.nan,35.]]),0.5),27.5)

    def test_reference_plane(self):
        points = cloud('C_ideal')
        points[:,2] += 4+0.015*points[:,0]-0.02*points[:,1]
        coeff = fit_table(points)
        np.testing.assert_allclose(coeff,[0.015,-0.02,4.116],atol=0.002)

    def test_archive_cloud(self):
        result = measure(cloud('C_jitter'))
        self.assertTrue(280000 < result['volume_mm3'] < 310000)
        self.assertTrue((result['status']==1).any() and (result['status']==2).any())

    def test_outline_and_external(self):
        result = measure(cloud('A_ideal'))
        xx,yy = np.meshgrid(result['x'],result['y'])
        self.assertFalse(result['domain'][np.hypot(xx,yy)>42].any())
        self.assertTrue(result['domain'][np.hypot(xx,yy)<38].all())

    def test_recess_depth_deducted(self):
        result = measure(cloud('B_ideal'))
        xx,yy = np.meshgrid(result['x'],result['y'])
        for center in [-25,25]:
            floor = result['height'][np.hypot(xx-center,yy)<7]
            self.assertAlmostEqual(np.nanmedian(floor),35,delta=0.3)
        self.assertGreater(np.nansum(np.maximum(0,39.8-result['height']))*0.25,2800)

    def test_units_and_grid_size(self):
        for pitch in (0.25,0.5,1.):
            self.assertEqual(integrate(np.full((int(80/pitch),int(100/pitch)),40.),pitch),320000)
        with self.assertRaises(ValueError):
            measure(cloud('B_ideal'),units='cm')

    def test_missing_region_stays_missing(self):
        points = cloud('B_ideal')
        keep = np.hypot(points[:,0],points[:,1])>5
        result = measure(points[keep])
        xx,yy = np.meshgrid(result['x'],result['y'])
        self.assertTrue((result['status'][np.hypot(xx,yy)<2]==3).all())
