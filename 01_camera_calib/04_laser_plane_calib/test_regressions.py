import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import plane_calib_lib as pc
import laser_lib as ll


class TestRepairs(unittest.TestCase):
    def setUp(self):
        self.camera = ll.CameraModel(np.array([[806.,0,640],[0,806.,360],[0,0,1]]), image_size=(1280,720))
        self.plane = ll.LaserPlane.from_abcd([1,0,.2,-100])
        self.board = pc.BoardSpec()
        x,y = np.meshgrid(np.linspace(-10,10,5),np.linspace(-10,10,5))
        self.fit = pc.fit_plane_points(np.column_stack([x.ravel(),y.ravel(),500+0*x.ravel()]))

    def test_degenerate_fit_cannot_be_exported(self):
        s = np.arange(10.)
        fit = pc.fit_plane_points(np.column_stack([s,0*s,500+0*s]))
        with self.assertRaises(ValueError):
            fit.to_plane()

    def test_degenerate_calibration_is_rejected(self):
        s = np.arange(10.)
        pts = np.column_stack([s,0*s,500+0*s])
        prepared = pc.PreparedPose(0,pts,self.plane,0.,10,10)
        with self.assertRaises(ValueError):
            pc.calibrate_from_prepared([prepared])

    def test_validation_takes_only_measurement_dicts(self):
        rows = pc.measure_validation(self.fit,[{'case':'empty','laser_uv':[]}],self.camera)
        self.assertFalse(rows[0]['has_step'])
        self.assertFalse(np.isfinite(rows[0]['measured_height_mm']))

    def test_omitted_validation_case_is_counted_as_failed(self):
        cases=[SimpleNamespace(name=n,height_mm=10.,z_far_mm=500.,y_split_mm=0.) for n in ('one','two')]
        rows,summary=pc.evaluate_validation([{'case':'one','measured_height_mm':10.,'has_step':True}],cases)
        self.assertEqual(summary['n_cases'],2)
        self.assertEqual(summary['n_failed'],1)
        self.assertFalse(summary['all_valid'])

    def test_invalid_flag_does_not_count_finite_height(self):
        case=SimpleNamespace(name='one',height_mm=10.,z_far_mm=500.,y_split_mm=0.)
        _,summary=pc.evaluate_validation([{'case':'one','measured_height_mm':10.,'has_step':False}],[case])
        self.assertEqual(summary['n_valid'],0)

    def test_validation_rejects_truth_fields(self):
        with self.assertRaises(ValueError):
            pc.measure_validation(self.fit,[{'case':'empty','laser_uv':[], 'true_height_mm':10.}],self.camera)

    def test_degenerate_diagnostic_does_not_measure(self):
        s=np.arange(10.)
        pts=np.column_stack([s,0*s,500+0*s])
        prepared=pc.PreparedPose(0,pts,self.plane,0.,10,10)
        case=SimpleNamespace(name='one',height_mm=10.,z_far_mm=500.,y_split_mm=0.,measurement={'case':'one','laser_uv':[]})
        metric=pc._subset_fit_metrics([prepared],self.plane,[case],self.camera)
        self.assertTrue(metric['degenerate'])
        self.assertEqual(metric['validation_n_failed'],1)
        self.assertTrue(np.isnan(metric['validation_mean_abs_mm']))

    def test_board_subset_archive_retains_indices(self):
        ranges=pc.PoseRanges((450,560),(-45,45),(-25,25),(-18,18),(-18,18),(-12,12))
        poses=pc.sample_poses(8,self.camera,self.plane,ranges,self.board,np.random.default_rng(7))
        meas=pc.build_ideal_measurements(poses,self.camera,self.board)
        prepared=pc.prepare_all(meas,self.camera,self.board)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pc.run_board_count(prepared,self.plane,[],self.camera,counts=(5,),repeats=1,audit_dir=root)
            import json
            saved=json.loads(next((root/'data/experiments/board_count').rglob('*.json')).read_text(encoding='utf-8'))
            self.assertEqual(len(saved['indices']),5)

    def test_experiment_archive_can_replay_noise_trial(self):
        ranges=pc.PoseRanges((450,560),(-45,45),(-25,25),(-18,18),(-18,18),(-12,12))
        poses=pc.sample_poses(8,self.camera,self.plane,ranges,self.board,np.random.default_rng(7))
        meas=pc.build_ideal_measurements(poses,self.camera,self.board)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pc.run_noise(meas,self.camera,self.board,self.plane,[],corner_sigmas=(.2,),laser_sigmas=(),repeats=1,audit_dir=root)
            paths=list((root/'data/experiments/noise').rglob('*.json'))
            self.assertEqual(len(paths),1)
            import json
            saved=json.loads(paths[0].read_text(encoding='utf-8'))
            replay=pc.calibrate_plane(saved['measurements'],self.camera,self.board)
            result=json.loads((root/'results/experiments/noise'/paths[0].relative_to(root/'data/experiments/noise')).read_text(encoding='utf-8'))
            np.testing.assert_allclose(replay.fit.normal,result['fit']['normal'])


if __name__=='__main__':
    unittest.main(verbosity=2)
