import tempfile
import unittest
from pathlib import Path
import numpy as np
import scan_lib as sc
import laser_lib as ll


CAM=ll.CameraModel(np.array([[201.5,0,160],[0,201.5,120],[0,0,1.]]),image_size=(320,240))


class TestScanRepairs(unittest.TestCase):
    def test_unbiased_base_with_zero_mean_noise(self):
        rng=np.random.default_rng(17)
        ground=np.column_stack([rng.uniform(-50,150,10000),rng.uniform(-40,120,10000),rng.normal(0,1,10000)])
        top=np.column_stack([rng.uniform(0,100,40000),rng.uniform(0,80,40000),50+rng.normal(0,1,40000)])
        base=sc.estimate_base_plane(np.vstack([ground,top]),np.random.default_rng(0))
        self.assertLess(abs(base.d),.08)

    def test_base_refinement_keeps_tilt(self):
        rng=np.random.default_rng(18)
        x,y=rng.uniform(-50,150,12000),rng.uniform(-40,120,12000)
        ground=np.column_stack([x,y,.02*x-.01*y+rng.normal(0,.5,len(x))])
        top=ground[:9000]+[0,0,50]
        base=sc.estimate_base_plane(np.vstack([ground,top]))
        expected=np.array([-.02,.01,1.]); expected/=np.linalg.norm(expected)
        self.assertGreater(float(base.normal@expected),.99999)
        self.assertLess(abs(base.d),.08)

    def test_sparse_ground_is_not_replaced_by_top(self):
        rng=np.random.default_rng(9)
        ground=np.column_stack([rng.uniform(-50,150,2000),rng.uniform(-40,120,2000),rng.normal(0,.05,2000)])
        top=np.column_stack([rng.uniform(0,100,40000),rng.uniform(0,80,40000),50+rng.normal(0,.05,40000)])
        base=sc.estimate_base_plane(np.vstack([ground,top]))
        self.assertLess(abs(base.d),.08)

    def test_sparse_ground_with_slot_and_top(self):
        rng=np.random.default_rng(19)
        levels=np.r_[np.zeros(1000),np.full(1232,40.),np.full(18246,50.)]
        points=np.column_stack([rng.uniform(0,100,len(levels)),rng.uniform(0,80,len(levels)),
                                levels+rng.normal(0,.01,len(levels))])
        base=sc.estimate_base_plane(points)
        self.assertLess(abs(base.d),.08)

    def test_roi_extraction_uses_original_pixel_coordinates(self):
        image=np.full((100,120),6.)
        x=np.arange(120)
        image[20:80]+=200*np.exp(-((x-70.2)**2)/(2*1.4**2))
        ext=sc.extract_centers(image,roi=(60,85,30,70))
        self.assertGreater(len(ext.uv),30)
        self.assertLess(abs(np.median(ext.uv[:,0])-70.2),.1)
        self.assertTrue(np.all((ext.uv[:,1]>=30)&(ext.uv[:,1]<70)))

    def test_all_images_and_measurements_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            setup=sc.ScanSetup(CAM,np.array([-45.,0.,45.]))
            scan=sc.simulate_scan(sc.make_box(),setup,image_dir=folder/'images',read_sigma=0,shot_gain=0)
            self.assertEqual(len(list((folder/'images').glob('*.png'))),3)
            sc.save_scan_measurements(folder/'measurements',scan)
            restored=sc.load_scan_measurements(folder/'measurements',setup.laser_plane_c)
            a=sc.reconstruct_scan(scan,setup.laser_plane_c)
            b=sc.reconstruct_scan(restored,setup.laser_plane_c)
            np.testing.assert_allclose(a.points_world,b.points_world)

    def test_actual_perturbed_inputs_replay(self):
        setup=sc.ScanSetup(CAM,np.arange(-40.,41.,20.))
        scan=sc.simulate_scan(sc.make_box(),setup,read_sigma=0,shot_gain=0)
        cloud=sc.reconstruct_scan(scan,setup.laser_plane_c,pixel_noise_px=.3,motion_sigma_mm=.5,omission=.25,seed=27)
        data=cloud.measurements
        pts=[]
        for i in range(len(data['pose_t'])):
            uv=data['uv'][data['offsets'][i]:data['offsets'][i+1]]
            rec=ll.reconstruct(uv,CAM,setup.laser_plane_c)
            pts.append(rec.points_mm[rec.valid]@data['pose_R'][i].T+data['pose_t'][i])
        np.testing.assert_allclose(np.vstack(pts),cloud.points_world)

    def test_image_fov_preserves_frame_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup=sc.ScanSetup(CAM,np.array([-45.,0.,45.]))
            scan=sc.simulate_scan(sc.make_box(),setup,image_dir=Path(tmp),read_sigma=0,shot_gain=0)
            cropped=sc.reextract_scan(scan,roi=(0,320,100,140))
            self.assertEqual(len(cropped.frames),len(scan.frames))
            for original,frame in zip(scan.frames,cropped.frames):
                np.testing.assert_array_equal(original.pose.t,frame.pose.t)
                self.assertTrue(np.all((frame.uv[:,1]>=100)&(frame.uv[:,1]<140)))

    def test_no_noise_image_chain_with_estimated_base(self):
        for obj,target in ((sc.make_box(),400.),(sc.make_slot_box(),396.)):
            setup=sc.ScanSetup(CAM,np.arange(-55.,56.,1.))
            scan=sc.simulate_scan(obj,setup,read_sigma=0,shot_gain=0)
            cloud=sc.reconstruct_scan(scan,setup.laser_plane_c)
            base=sc.estimate_base_plane(cloud.points_world)
            hm=sc.height_map_volume(cloud.points_world,base,footprint=sc.known_footprint_in_base(obj,base))
            self.assertLess(abs(base.d),.15)
            self.assertLess(abs(hm.volume_mm3/1000-target),2.)


if __name__=='__main__':
    unittest.main(verbosity=2)
