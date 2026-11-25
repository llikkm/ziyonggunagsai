import unittest
import numpy as np

from diagnose import ledger, experiment, score, circle_fit, feature_stats


class Diagnostics(unittest.TestCase):
    def setUp(self):
        self.height = np.array([[49.,50.,np.nan,10.]])
        self.status = np.array([[1,2,3,4]])
        self.domain = np.array([[True,True,True,False]])
        self.truth = np.full((1,4),50.)

    def test_ledger_closure(self):
        row = ledger(self.height,self.status,self.domain,self.truth,1,150.)
        self.assertEqual(row['boundary'],10)
        self.assertEqual(row['height'],-1)
        self.assertEqual(row['missing'],-50)
        self.assertEqual(row['grid'],0)
        self.assertEqual(row['closure'],0)

    def test_oracle_preserves_unknown(self):
        original = self.height.copy()
        mask = self.domain
        row = experiment(self.height,self.status,mask,mask,self.truth,1,False,False)
        self.assertEqual(row['volume'],99)
        self.assertEqual(row['unresolved_pct'],100/3)
        np.testing.assert_equal(original,self.height)
        filled = experiment(self.height,self.status,mask,mask,self.truth,1,False,True)
        self.assertEqual(filled['volume'],149)
        self.assertEqual(filled['oracle_fill_pct'],100/3)

    def test_circle(self):
        angle = np.linspace(0,2*np.pi,40,endpoint=False)
        np.testing.assert_allclose(circle_fit(np.column_stack((3+40*np.cos(angle),-4+40*np.sin(angle)))),[3,-4,40],atol=1e-8)

    def test_weighted_score(self):
        self.assertAlmostEqual(score(dict(A=-1.,B=2.,C=-3.)),0.023)

    def test_blind_cavity(self):
        xx,yy = np.meshgrid(np.arange(-51,52),np.arange(-42,43))
        status = np.ones(xx.shape,int)
        regions = feature_stats(xx,yy,status,np.full(xx.shape,35.),1)
        for row in regions:
            self.assertEqual(row['missing_cells'],0)
            self.assertEqual(row['direct_cells'],row['cells'])
            if row['region']=='opening':
                self.assertEqual(row['void_above_floor_mm3'],5*row['cells'])
