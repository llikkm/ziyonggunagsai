"""Additional read-only diagnostics on archived grids; no reconstruction changes."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import open3d as o3d
from protocol import POSES,check_lock


def inspect(root):
    check_lock()
    for split in ('development','holdout'):
        folder=root/split
        specs={s['id']:s for s in json.loads((folder/'models/manifest.json').read_text())}
        rows=[]
        for record in csv.DictReader((folder/'results.csv').open(encoding='utf-8-sig')):
            if record['failure']: continue
            key=record['model']
            case=folder/f"{key}_p{record['pose']}_{record['mode']}"
            spec=specs[key]
            grid=np.load(case/'grid.npz')
            xx,yy=np.meshgrid(grid['x'],grid['y'])
            mesh=o3d.io.read_triangle_mesh(str(folder/'models'/f'{key}.ply'))
            dx,dy,angle=POSES[int(record['pose'])]
            mesh.rotate(mesh.get_rotation_matrix_from_xyz([0,0,np.deg2rad(angle)]),center=[0,0,0]).translate([dx,dy,0])
            scene=o3d.t.geometry.RaycastingScene(nthreads=1)
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            rays=np.stack((xx,yy,xx*0+160,xx*0,yy*0,xx*0-1),axis=-1).astype('float32')
            truth=160-scene.cast_rays(o3d.core.Tensor(rays),nthreads=1)['t_hit'].numpy()
            inside=np.isfinite(truth)
            transitions=np.diff(np.pad(inside,((1,1),(0,0))).astype(int),axis=0)
            multi=int(((transitions==1).sum(axis=0)>1).sum())
            cap=np.isfinite(grid['height'])
            recess=inside&(truth<spec['height']-.5) if spec['category']=='B' else np.zeros_like(inside)
            visits=int(recess.sum())
            abs_bias=float(np.mean(np.abs(grid['height'][inside&cap]-truth[inside&cap])))
            rows.append(dict(model=key,mode=record['mode'],pose=record['pose'],chosen_method=record['method'],
                             multisegment_columns=multi,recess_cells=visits,recess_missing_cells=int((recess&~cap).sum()),
                             recess_missing_pct=100*(recess&~cap).sum()/visits if visits else None,
                             common_mean_abs_height_mm=abs_bias,range_loss_pct=record['range_loss_pct']))
            grid.close()
        with (folder/'mechanisms.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__=='__main__':
    inspect(Path(sys.argv[1]))
