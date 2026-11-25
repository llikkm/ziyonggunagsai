import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from estimator import measure
from evaluation import REFERENCE, WEIGHTS, audit, reference_surface


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent/'06_standard_scan/runs/baseline_20261118'


def dump(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def inspect_height(name, grid, path):
    xx, yy = np.meshgrid(grid['x'],grid['y'])
    domain, truth = reference_surface(name,xx,yy)
    estimated = grid['domain']
    height = grid['height']
    valid = np.isfinite(height)
    area = grid['pitch_mm']**2
    common = valid & domain
    delta = np.where(common,height-truth,np.nan)
    report = dict(reference_area_mm2=float(area*domain.sum()),auto_area_mm2=float(area*estimated.sum()),
                  excess_outline_mm2=float(area*(estimated&~domain).sum()),lost_outline_mm2=float(area*(domain&~estimated).sum()),
                  reference_unintegrated_mm2=float(area*(domain&~valid).sum()),
                  valid_height_bias_mm=float(np.mean(delta[common])),abs_height_p95_mm=float(np.quantile(abs(delta[common]),0.95)),
                  missed_volume_mm3=float(area*truth[domain&~valid].sum()),
                  extra_volume_mm3=float(area*height[valid&~domain].sum()),
                  common_height_error_mm3=float(area*delta[common].sum()))
    if name=='B':
        for key, region in [('groove',(abs(xx+25)<=10)&(abs(yy)<=10)),('blind_hole',(xx-25)**2+yy**2<=100)]:
            floor = height[region & valid]
            report[key] = dict(valid_fraction=float(len(floor)/region.sum()),floor_median_mm=float(np.median(floor)),
                               observed_fraction=float((grid['status'][region]==1).mean()),
                               supported_recess_mm3=float(area*np.maximum(0,40-floor).sum()))
    pitch = grid['pitch_mm']
    extent = [xx.min()-pitch/2,xx.max()+pitch/2,yy.min()-pitch/2,yy.max()+pitch/2]
    fig, axes = plt.subplots(1,3,figsize=(13,4),layout='constrained')
    specs = [(height,'Estimated height (mm)','viridis',{}),
             (grid['status'],'Grid support',ListedColormap(['#eeeeee','#074e85','#8cc9df','#eb723d','#62a66e']),{}),
             (delta,'Height error inside reference (mm)','coolwarm',dict(vmin=-1,vmax=1))]
    for index,(ax,(data,title,cmap,options)) in enumerate(zip(axes,specs)):
        if index==1:
            options['norm'] = BoundaryNorm(np.arange(-0.5,5.5),5)
        im = ax.imshow(data,origin='lower',extent=extent,cmap=cmap,interpolation='nearest',**options)
        bar = fig.colorbar(im,ax=ax,shrink=0.68,extend='both' if index==2 else 'neither')
        if index==1:
            bar.set_ticks(range(5),labels=['outside','hit','triangle','missing','local'])
        if index==2:
            ax.contour(xx,yy,domain.astype(int),levels=[0.5],colors='black',linewidths=0.6)
        ax.set(xlabel='X (mm)',ylabel='Y (mm)',title=title)
    fig.suptitle(path.stem)
    fig.savefig(path,dpi=130)
    plt.close(fig)
    return report


def run(source, output, pitch):
    output.mkdir(parents=True,exist_ok=False)
    diagnostics = output/'diagnostics'
    diagnostics.mkdir()
    cases = [source/f'{name}_{mode}' for mode in ('ideal','jitter') for name in 'ABC']
    audits = {case.name:audit(case) for case in cases}
    dump(diagnostics/'archive_audit.json',audits)
    if any(min(row[key] for key in ('itinerary_fraction','fov_fraction','fan_fraction'))<0.999 for row in audits.values()):
        raise ValueError('Coverage audit failed; revise simulation scan range before measuring.')
    summary, details, tables = [], {}, []
    for case in cases:
        folder = output/case.name
        folder.mkdir()
        start = perf_counter()
        with np.load(case/'reconstruction/points.npz') as record:
            points = record['xyz_mm']
        config = json.loads((case/'measurements/config.json').read_text())
        nominal = np.array([frame['t_mm'] for frame in config['frames']])
        scan_step = float(np.median(abs(np.diff(nominal[:,0]))))
        loaded = perf_counter()
        grid = measure(points,pitch,scan_step=scan_step)
        np.savez_compressed(folder/'grid.npz',**{key:grid[key] for key in ('height','status','domain','ground','x','y','plane','pitch_mm')})
        counts = np.bincount(grid['status'].ravel(),minlength=5)
        area_cells = counts[1:].sum()
        row = dict(case=case.name,reference_mm3=REFERENCE[case.name[0]],volume_mm3=round(grid['volume_mm3'],1))
        row['relative_error_pct'] = abs(row['volume_mm3']-row['reference_mm3'])/row['reference_mm3']*100
        row.update(area_mm2=float(area_cells*pitch**2),direct_pct=100*counts[1]/area_cells,
                   interpolated_pct=100*counts[2]/area_cells,local_pct=100*counts[4]/area_cells,
                   valid_pct=100*(1-counts[3]/area_cells),missing_pct=100*counts[3]/area_cells,
                   load_s=loaded-start,**grid['timing'])
        row['measure_s'] = sum(row[key] for key in ('load_s','plane_s','occupancy_s','grid_s','integration_s'))
        summary.append(row)
        details[case.name] = inspect_height(case.name[0],grid,diagnostics/f'{case.name}.png')
        record = audits[case.name]
        tables.append(f"| {case.name} | "+' | '.join(f'{val:.3f}' for val in [*record['actual_step_x_mm'],record['x_travel_mm'],
                      record['xy_mm'][0][0],record['xy_mm'][1][0],record['xy_mm'][0][1],record['xy_mm'][1][1],100*record['direct_top_fraction'],100*record['near_top_fraction']])+' |')
    scores = {}
    for mode in ('ideal','jitter'):
        raw = sum(WEIGHTS[row['case'][0]]*row['relative_error_pct']/100 for row in summary if row['case'].endswith(mode))
        scores[mode] = dict(raw=raw,competition=f'{raw:.2f}')
    dump(diagnostics/'reference_comparison.json',details)
    dump(output/'scores.json',scores)
    with (output/'volumes.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)
    report = ['# 六组扫描体积验收', '',f'网格 {pitch:g} mm，数据源：{source.name}。详细计时见 volumes.csv。','',
              '| 组别 | 理论 mm³ | 积分 mm³ | 绝对相对误差 % | 直接命中 % | 有效 % | 缺测 % | 计算 s |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        report.append('| {case} | {reference_mm3:.1f} | {volume_mm3:.1f} | {relative_error_pct:.3f} | {direct_pct:.2f} | {valid_pct:.2f} | {missing_pct:.2f} | {measure_s:.3f} |'.format(**row))
    report.extend(['',*(f"- {mode}：P = {score['raw']:.6f}，规则保留两位为 {score['competition']}。" for mode,score in scores.items()),
                   '', '## 归档覆盖审计（独立评价）','',
                   '| 组别 | ΔX 最小 | ΔX 最大 | X 行程 | Xmin | Xmax | Ymin | Ymax | 顶面直接 % | 邻域同层 % |',
                   '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',*tables,'',
                   '坐标及步长单位 mm。逐帧位置和顶面 XY 范围见 diagnostics/archive_audit.json；六张图及轮廓误差分解见 diagnostics/。'])
    (output/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ARCHIVE)
    parser.add_argument('--out',type=Path,default=HERE/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    parser.add_argument('--pitch',type=float,default=0.5)
    args = parser.parse_args()
    run(args.source,args.out,args.pitch)
