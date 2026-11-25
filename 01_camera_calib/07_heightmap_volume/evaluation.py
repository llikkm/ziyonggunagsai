"""Archive audit and scoring only. The estimator never imports this module."""
import json

import numpy as np
from scipy.spatial import cKDTree


REFERENCE = dict(A=251327.4, B=316429.2, C=300000.0)
WEIGHTS = dict(A=0.2, B=0.3, C=0.5)


def reference_surface(name, xx, yy):
    if name == 'A':
        return xx**2+yy**2 <= 40**2, np.full(xx.shape, 50.)
    if name == 'B':
        groove = (abs(xx+25)<=10) & (abs(yy)<=10)
        bore = (xx-25)**2+yy**2 <= 100
        return (abs(xx)<=50)&(abs(yy)<=40), 40.-5*(groove|bore)
    if name == 'C':
        return (abs(xx)<=50)&(abs(yy)<=50), 30+8*np.sin(np.pi*xx/50)*np.sin(np.pi*yy/50)
    raise ValueError(name)


def limits(values):
    return [values.min(axis=0).tolist(), values.max(axis=0).tolist()]


def audit(case):
    config = json.loads((case/'measurements/config.json').read_text())
    log = json.loads((case/'render_log.json').read_text())
    frames = config['frames']
    pose = np.array([frame['actual_camera_mm'] for frame in log['frames']])
    nominal = np.array([frame['t_mm'] for frame in frames])
    step = np.diff(pose,axis=0)
    with np.load(case/'reconstruction/points.npz') as record:
        points, offsets, uv = (record[key] for key in ('xyz_mm','offsets','uv'))
    rotation = np.array(config['R_camera_to_world'])
    np.testing.assert_array_equal(rotation, np.diag([1,-1,-1]))
    if len(frames)!=63 or pose.shape!=(63,3) or len(offsets)!=64 or offsets[0]!=0 or offsets[-1]!=len(points) or min(np.diff(offsets))<=0:
        raise ValueError(f'{case}: invalid frames')
    if len(set(frame['image'] for frame in frames))!=63 or any(not (case/'measurements'/frame['image']).is_file() for frame in frames):
        raise ValueError(f'{case}: missing frame image')
    cam = (points-np.repeat(nominal,np.diff(offsets),axis=0)) @ rotation
    optical = cam @ np.array(config['K']).T
    np.testing.assert_allclose(optical[:,:2]/optical[:,2:],uv,atol=1e-6)
    plane = np.array(config['plane_camera'])
    np.testing.assert_allclose(cam @ plane[:3]+plane[3],0,atol=1e-6)
    axis = np.arange(-50,50,0.5)+0.25
    xx, yy = np.meshgrid(axis,axis)
    domain, zz = reference_surface(case.name[0],xx,yy)
    targets = np.column_stack((xx[domain],yy[domain],zz[domain]))
    rays = (targets[:,None,:]-pose) @ rotation
    signed = rays @ plane[:3]+plane[3]
    bracketed = (signed.min(axis=1)<=0)&(signed.max(axis=1)>=0)
    closest = rays[np.arange(len(targets)),np.argmin(abs(signed),axis=1)]
    optical = closest @ np.array(config['K']).T
    pixels = optical[:,:2]/optical[:,2:]
    fov = ((pixels>=0)&(pixels<config['size'])).all(axis=1)
    fan = abs(closest[:,1]/closest[:,2])<=0.47
    surface = points[points[:,2]>2]
    tree = cKDTree(surface[:,:2])
    dist, near = tree.query(targets[:,:2],k=16)
    matched = ((dist<=1.5)&(abs(surface[near,2]-targets[:,2,None])<=1)).any(axis=1)
    correct = reference_surface(case.name[0],surface[:,0],surface[:,1])
    inside = surface[correct[0] & (abs(surface[:,2]-correct[1])<=1)]
    cells = np.unique(np.floor((inside[:,:2]+50)/0.5).astype(int),axis=0)
    centers = -50+(cells+0.5)*0.5
    direct = reference_surface(case.name[0],centers[:,0],centers[:,1])[0].sum()/domain.sum()
    per_frame = [dict(frame=index,actual_xyz_mm=pose[index].tolist(),nominal_xyz_mm=nominal[index].tolist(),
                      xy_bounds_mm=limits(points[start:end,:2]), surface_points=int((points[start:end,2]>2).sum()))
                 for index,(start,end) in enumerate(zip(offsets[:-1],offsets[1:]))]
    return dict(frames=len(frames), nominal_step_x_mm=limits(np.diff(nominal[:,0])),actual_step_x_mm=limits(step[:,0]),
                mean_step_x_mm=float(step[:,0].mean()), actual_step_3d_mm=limits(np.linalg.norm(step,axis=1)),
                nominal_travel_mm=float(np.ptp(nominal[:,0])),x_travel_mm=float(np.ptp(pose[:,0])),
                trajectory_length_mm=float(np.linalg.norm(step,axis=1).sum()), xy_mm=limits(points[:,:2]),
                top_xy_mm=limits(surface[:,:2]), itinerary_fraction=float(bracketed.mean()), fov_fraction=float(fov.mean()),
                fan_fraction=float(fan.mean()), direct_top_fraction=float(direct),near_top_fraction=float(matched.mean()),
                worst_nearest_xy_mm=float(dist[:,0].max()),per_frame=per_frame)
