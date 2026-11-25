import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d

from refined import refine, measure
from evaluation import REFERENCE, WEIGHTS, reference_surface


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'06_standard_scan/runs/baseline_20261118'


def compare(folder):
    folder.mkdir(parents=True,exist_ok=False)
    rows = []
    scores = {}
    for mode in ('ideal','jitter'):
        scores[mode] = dict(before=0.,after=0.)
        for name in 'ABC':
            path = SOURCE/f'{name}_{mode}/reconstruction/cloud.ply'
            points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
            before = measure(points)
            after = refine(points,before)
            row = dict(case=f'{name}_{mode}',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),method=after['method'],
                       edge_residual_p95_mm=after['edge_residual_p95_mm'],step_fills=after['step_fills'],local_fills=after['local_fills'])
            xx,yy = np.meshgrid(after['x'],after['y'])
            target, truth = reference_surface(name,xx,yy)
            for label,data in [('before',before),('after',after)]:
                valid = np.isfinite(data['height'])
                volume = round(data['volume_mm3'],1)
                error = (volume-REFERENCE[name])/REFERENCE[name]
                scores[mode][label] += WEIGHTS[name]*abs(error)
                row.update({label+'_mm3':volume,label+'_signed_pct':100*error,label+'_valid_pct':100*valid.sum()/data['domain'].sum(),
                            label+'_direct_pct':100*(data['status']==1).sum()/data['domain'].sum(),
                            label+'_area_mm2':0.25*data['domain'].sum(),label+'_height_bias_mm':np.mean((data['height']-truth)[valid&target])})
            row['regression'] = bool(abs(row['after_signed_pct'])>abs(row['before_signed_pct']))
            rows.append(row)
            np.savez_compressed(folder/(row['case']+'.npz'),**{k:after[k] for k in ('x','y','height','status','domain')})
    with (folder/'volumes.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (folder/'scores.json').write_text(json.dumps(scores,indent=2))
    print(json.dumps(scores,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=HERE/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    compare(parser.parse_args().out)
