import unittest
import numpy as np
from gallery import compare_cells


class GalleryChecks(unittest.TestCase):
    def test_missing_extra_and_common(self):
        height = np.array([[3., np.nan, 7., np.nan]])
        truth = np.array([[4., 5., np.nan, np.nan]])
        residual, delta = compare_cells(height, truth, .5)
        np.testing.assert_allclose(residual, [[-1., np.nan, np.nan, np.nan]], equal_nan=True)
        np.testing.assert_allclose(delta, [[-.25, -1.25, 1.75, np.nan]], equal_nan=True)
        self.assertAlmostEqual(np.nansum(delta), .25*(np.nansum(height)-np.nansum(truth)))
