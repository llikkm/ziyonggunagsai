import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
import cv2
import numpy as np
import open3d as o3d

from protocol import HERE,ROOT,POSES,FROZEN,check_lock
from solids import SEEDS,specs,solid
sys.path[:0]=[str(ROOT/'06_standard_scan'),str(ROOT/'09_volume_refinement')]
from simulate import make_scene,render
from replay import CameraModel,replay
from refined import refine


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_grid(mesh,grid,poses,spec):
    pitch=.5
    lower=np.minimum(np.floor(mesh.get_min_bound()[:2]/pitch)*pitch,[grid['x'][0]-.25,grid['y'][0]-.25])
    upper=np.maximum(mesh.get_max_bound()[:2],[grid['x'][-1],grid['y'][-1]])
    xx,yy=np.meshgrid(np.arange(lower[0]+.25,upper[0]+pitch,pitch),np.arange(lower[1]+.25,upper[1]+pitch,pitch))
    rays=np.stack((xx,yy,np.full_like(xx,160),xx*0,yy*0,np.full_like(xx,-1)),axis=-1).astype('float32')
    dist=make_scene(mesh).cast_rays(o3d.core.Tensor(rays),nthreads=1)['t_hit'].numpy()
    target=np.isfinite(dist)
    truth=160-dist.astype(float)
    height=np.full(xx.shape,np.nan)
    status=np.zeros(xx.shape,int)
    col,row=np.rint(([grid['x'][0],grid['y'][0]]-lower-.25)/pitch).astype(int)
    view=np.s_[row:row+len(grid['y']),col:col+len(grid['x'])]
    height[view]=grid['height']
    status[view]=grid['status']
    valid=np.isfinite(height)
    common=valid&target
    gap=target&~valid
    depth=210-truth[target]
    ideal_origin=xx[target]+62-.18*depth
    best=np.abs(ideal_origin[:,None]-poses[:,0]).argmin(axis=1)
    depth=poses[best,2]-truth[target]
    offset=xx[target]-poses[best,0]
    stripe=ideal_origin
    clipped=(stripe<poses[:,0].min())|(stripe>poses[:,0].max())|(abs((yy[target]-poses[best,1])/depth)>0.47)
    uu=350*offset/depth+159.5
    vv=350*(poses[best,1]-yy[target])/depth+119.5
    clipped|=(uu<0)|(uu>=320)|(vv<0)|(vv>=240)
    cutoff=np.zeros(target.shape,bool)
    cutoff[target]=clipped
    recess=gap&(truth<spec['height']-.5)&(truth>spec['height']-5.1) if spec['category']=='B' else np.zeros_like(gap)
    terms=dict(scan=-truth[gap&cutoff].sum()*.25,recess=-truth[recess&~cutoff].sum()*.25,
               boundary=(height[valid&~target].sum()-truth[gap&~cutoff&~recess&(status==0)].sum())*.25,
               missing=-truth[gap&~cutoff&~recess&(status!=0)].sum()*.25,
               surface=(height[common]-truth[common]).sum()*.25)
    count=int((status>0).sum())
    return dict(direct_pct=100*(status==1).sum()/count,estimated_pct=100*((status==2)|(status==4)|(status==5)).sum()/count,
                valid_pct=100*valid.sum()/count,true_direct_pct=100*((status==1)&target).sum()/target.sum(),
                range_loss_pct=100*clipped.mean(),method=grid['method'],dominant=max(terms,key=lambda key:abs(terms[key])),
                **{key+'_mm3':float(val) for key,val in terms.items()})


def worker(task):
    splitroot,spec,pose_id,mode=task
    key=spec['id']
    folder=splitroot/f'{key}_p{pose_id}_{mode}'
    inputs=folder/'measurements'
    inputs.mkdir(parents=True)
    mesh=o3d.io.read_triangle_mesh(str(splitroot/'models'/f'{key}.ply'))
    dx,dy,angle=POSES[pose_id]
    mesh.rotate(mesh.get_rotation_matrix_from_xyz([0,0,np.deg2rad(angle)]),center=[0,0,0]).translate([dx,dy,0])
    scene=make_scene(mesh+o3d.geometry.TriangleMesh.create_box(170,150,2).translate([-85,-75,-2]))
    camera=CameraModel([[350,0,159.5],[0,350,119.5],[0,0,1]],[],(320,240))
    rotation=np.diag([1,-1,-1])
    fan=np.column_stack([np.full(1201,.18),np.linspace(-.47,.47,1201),-np.ones(1201)])
    seed=spec['seed']+1000*(ord(spec['category'])-65)+100*spec['index']+pose_id
    rng=np.random.default_rng(seed)
    frames,poses=[],[]
    record=dict(model=key,category=spec['category'],kind=spec['kind'],pose=pose_id,mode=mode,reference_mm3=spec['volume_mm3'],
                failure='',render_s=None,replay_s=None,measure_s=None)
    start=perf_counter()
    phase='render'
    try:
        for number,position in enumerate(np.arange(-62.,63.,2.)):
            nominal=np.array([position+32,0,210])
            actual=nominal+(rng.normal(0,[.25,.12,.10]) if mode=='jitter' else 0)
            image,_=render(scene,camera,rotation,actual,fan,62.)
            filename=f'{number:04}.png'
            ok,encoded=cv2.imencode('.png',image)
            if not ok:
                raise IOError(filename)
            encoded.tofile(inputs/filename)
            frames.append(dict(image=filename,t_mm=nominal.tolist()))
            poses.append(actual.tolist())
        dump(inputs/'config.json',dict(K=camera.K.tolist(),dist=camera.dist.tolist(),size=camera.image_size,plane_camera=[1,0,-.18,62],R_camera_to_world=rotation.tolist(),frames=frames))
        dump(folder/'render_log.json',dict(seed=seed,actual_camera_mm=poses))
        record['render_s']=perf_counter()-start
        phase='replay'
        clock=perf_counter()
        replay(inputs,folder/'reconstruction')
        record['replay_s']=perf_counter()-clock
        phase='measure'
        clock=perf_counter()
        points=np.asarray(o3d.io.read_point_cloud(str(folder/'reconstruction/cloud.ply')).points)
        grid=refine(points)
        record['measure_s']=perf_counter()-clock
        record['runtime_s']=perf_counter()-start
        phase='evaluation'
        value=round(grid['volume_mm3'],1)
        error=value-spec['volume_mm3']
        record.update(volume_mm3=value,signed_pct=100*error/spec['volume_mm3'],abs_pct=100*abs(error)/spec['volume_mm3'],abs_mm3=abs(error))
        np.savez_compressed(folder/'grid.npz',**{key:grid[key] for key in ('x','y','height','status')})
        record.update(score_grid(mesh,grid,np.array(poses),spec))
    except Exception as exc:
        record.update(failure=f'{phase}: {type(exc).__name__}: {exc}',runtime_s=perf_counter()-start)
        record['dominant']='geometry_assumption' if 'Ordered X scans' in str(exc) else phase+'_failure'
    dump(folder/'result.json',record)
    return record


def prepare(root):
    check_lock()
    root.mkdir(parents=True,exist_ok=False)
    manifest={}
    for split,seed in SEEDS.items():
        folder=root/split/'models'
        folder.mkdir(parents=True)
        records=[]
        for spec in specs(split):
            mesh=solid(spec)
            extent=mesh.get_axis_aligned_bounding_box().get_extent()
            volume=mesh.get_volume()
            if not (mesh.is_watertight() and np.all((extent>=50)&(extent<=120)) and 50000<=volume<=800000):
                raise ValueError(spec)
            key=spec['category']+str(spec['index']+1)
            path=folder/f'{key}.ply'
            if not o3d.io.write_triangle_mesh(str(path),mesh):
                raise IOError(path)
            records.append(dict(**spec,id=key,seed=seed,volume_mm3=volume,extent_mm=extent.tolist(),sha256=digest(path)))
        dump(folder/'manifest.json',records)
        manifest[split]=digest(folder/'manifest.json')
    dump(root/'lock.json',dict(baseline='0b82f3b5940ad9a715ba3428376cd0d2104bd5ac',algorithms=FROZEN,poses=POSES,seeds=SEEDS,
                              manifests=manifest,scripts={file:digest(HERE/file) for file in ['benchmark.py','solids.py','protocol.py','summarize.py']}))


def execute(root,split,jobs):
    check_lock()
    lock=json.loads((root/'lock.json').read_text())
    for file,sha in lock['scripts'].items():
        if digest(HERE/file)!=sha: raise ValueError(file)
    folder=root/split
    manifest=folder/'models/manifest.json'
    if digest(manifest)!=lock['manifests'][split]: raise ValueError(manifest)
    records=json.loads(manifest.read_text())
    for spec in records:
        if digest(folder/'models'/(spec['id']+'.ply'))!=spec['sha256']: raise ValueError(spec['id'])
    tasks=[(folder,spec,pose,mode) for spec in records for pose in range(3) for mode in ('ideal','jitter')]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        rows=list(pool.map(worker,tasks))
    with (folder/'results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    from summarize import summarize
    dump(folder/'summary.json',summarize(rows,SEEDS[split]))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare',*SEEDS])
    parser.add_argument('root',type=Path)
    parser.add_argument('--jobs',type=int,default=3)
    args=parser.parse_args()
    if args.stage=='prepare': prepare(args.root)
    else: execute(args.root,args.stage,args.jobs)
