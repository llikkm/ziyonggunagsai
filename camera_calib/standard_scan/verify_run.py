import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d

from replay import replay


def verify(folder):
    report = {}
    for name in 'ABC':
        for mode in ('ideal','jitter'):
            case = folder/f'{name}_{mode}'
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                shutil.copytree(case/'measurements',root/'inputs')
                points = replay(root/'inputs',root/'replay')
                with np.load(case/'reconstruction'/'points.npz') as old, np.load(root/'replay'/'points.npz') as new:
                    for key in ('xyz_mm','uv','offsets'):
                        np.testing.assert_array_equal(new[key],old[key])
                    counts = np.diff(new['offsets'])
                    if len(counts)!=63 or min(counts)<=0:
                        raise ValueError(f'{case.name}: missing frame')
                np.testing.assert_array_equal(points,np.asarray(o3d.io.read_point_cloud(str(case/'reconstruction'/'cloud.ply')).points))
                xx,yy,zz = points.T
                if name=='A':
                    sections = [(np.hypot(xx,yy)<35,50)]
                elif name=='B':
                    sections = [(np.hypot(xx-center,yy)<7,35) for center in (-25,25)]
                    sections.append(((abs(xx)<10)&(abs(yy)<30),40))
                else:
                    sections = [((abs(xx)<45)&(abs(yy)<45),30+8*np.sin(xx*np.pi/50)*np.sin(yy*np.pi/50))]
                checks = []
                for region, target in sections:
                    errors = abs(zz-target)[region]
                    if len(errors)<100 or np.quantile(errors,0.95)>1:
                        raise ValueError(f'{case.name}: interior surface mismatch')
                    checks.append(dict(samples=len(errors),p95_abs_z_mm=float(np.quantile(errors,0.95))))
                report[case.name] = dict(frames=len(counts),points=len(points),surfaces=checks)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.folder),indent=2))
