import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from models import export_models
from replay import CameraModel, replay


def make_scene(mesh):
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def distances(scene, origins, directions):
    rays = np.column_stack([np.broadcast_to(origins,np.shape(directions)),directions]).astype('float32')
    return scene.cast_rays(o3d.core.Tensor(rays),nthreads=1)['t_hit'].numpy()


def first_hits(scene, origins, directions):
    lengths = distances(scene,origins,directions)
    valid = np.isfinite(lengths)
    return np.broadcast_to(origins,np.shape(directions))[valid]+np.asarray(directions)[valid]*lengths[valid,None]


def camera_visible(scene, origin, points):
    vectors = np.asarray(points)-origin
    lengths = np.linalg.norm(vectors,axis=1)
    return distances(scene,origin,vectors/lengths[:,None]) >= lengths-0.002


def in_frame(uv,width,height):
    return (uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)


def render(scene, camera, rotation, origin, directions, baseline):
    width,height = camera.image_size
    hit = first_hits(scene,origin+[-baseline,0,0],directions)
    visible = hit[camera_visible(scene,origin,hit)] if len(hit) else hit
    uv = camera.project((visible-origin) @ rotation)
    uv = uv[in_frame(uv,width,height)]
    raster = np.zeros((height,width),np.float32)
    if len(uv):
        left, row = np.floor(uv).astype(int).T
        frac = uv[:,0]-left
        for shift, weight in [(0,1-frac),(1,frac)]:
            valid = left+shift < width
            np.add.at(raster,(row[valid],left[valid]+shift),weight[valid])
    raster = cv2.GaussianBlur(raster,(5,3),0.75,borderType=cv2.BORDER_CONSTANT)
    return np.clip(10+95*raster,0,255).astype('uint8'), [len(hit),len(visible),len(uv)]


def run(folder,seed):
    folder.mkdir(parents=True,exist_ok=False)
    export_models(folder/'models')
    camera = CameraModel([[350,0,159.5],[0,350,119.5],[0,0,1]],[],(320,240))
    rotation = np.diag([1,-1,-1])
    baseline, slope = 62.,0.18
    directions = np.column_stack([np.full(1201,slope),np.linspace(-0.47,0.47,1201),-np.ones(1201)])
    report = {}
    for index,name in enumerate('ABC'):
        mesh = o3d.io.read_triangle_mesh(str(folder/'models'/f'{name}.ply'))
        platform = o3d.geometry.TriangleMesh.create_box(170,150,2).translate([-85,-75,-2])
        scene = make_scene(mesh+platform)
        for mode in ['ideal','jitter']:
            case = folder/f'{name}_{mode}'
            inputs = case/'measurements'
            inputs.mkdir(parents=True)
            rng = np.random.default_rng(seed+index)
            frames, truth = [], []
            for number, position in enumerate(np.arange(-62.,63.,2.)):
                nominal = np.array([position+32,0,210])
                actual = nominal+(rng.normal(0,[0.25,0.12,0.10]) if mode=='jitter' else 0)
                image, counts = render(scene,camera,rotation,actual,directions,baseline)
                filename = f'{number:04}.png'
                ok, encoded = cv2.imencode('.png',image)
                if not ok:
                    raise IOError(filename)
                encoded.tofile(inputs/filename)
                frames.append({'image':filename,'t_mm':nominal.tolist()})
                truth.append({'frame':number,'actual_camera_mm':actual.tolist(),'hits_visible_in_frame':counts})
            config = dict(K=camera.K.tolist(),dist=camera.dist.tolist(),size=camera.image_size,plane_camera=[1,0,-slope,baseline],R_camera_to_world=rotation.tolist(),frames=frames)
            (inputs/'config.json').write_text(json.dumps(config,indent=2))
            (case/'render_log.json').write_text(json.dumps(dict(seed=seed+index,mode=mode,frames=truth),indent=2))
            cloud = replay(inputs,case/'reconstruction')
            result = dict(frames=len(frames),points=len(cloud))
            if name=='B':
                for label,center in [('slot',-25),('hole',25)]:
                    radius = np.hypot(cloud[:,0]-center,cloud[:,1])
                    bottom = cloud[(radius<7)&(abs(cloud[:,2]-35)<1),2]
                    if not len(bottom):
                        raise ValueError(f'{name}/{label}: missing bottom')
                    result[label] = dict(points=len(bottom),median_z_mm=float(np.median(bottom)))
            report[case.name] = result
            print(case.name,result,flush=True)
    (folder/'summary.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    parser.add_argument('--seed',type=int,default=42)
    args = parser.parse_args()
    run(args.out,args.seed)
