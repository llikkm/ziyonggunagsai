"""Post-measurement diagnostics for the three archived ideal scans."""
import argparse
import csv
import hashlib
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0,str(PROJECT/'heightmap_volume'))
from estimator import measure
from evaluation import reference_surface, REFERENCE, WEIGHTS

EXACT = dict(A=80000*np.pi,B=318000-500*np.pi,C=300000.)
AREA = dict(A=1600*np.pi,B=8000.,C=10000.)


def ledger(height,status,target,truth,pitch,reference):
    area = pitch**2
    observed = np.isfinite(height)
    auto = status!=0
    common = target & observed
    row = dict(boundary=area*(height[observed & ~target].sum()-truth[target & ~auto].sum()),
               height=area*(height[common]-truth[common]).sum(),
               missing=-area*truth[target & auto & ~observed].sum(),
               grid=area*truth[target].sum()-reference)
    row['closure'] = area*height[observed].sum()-reference-sum(row.values())
    return row


def experiment(height,status,mask,target,truth,pitch,oracle_height,oracle_fill):
    result = height.copy()
    available = np.isfinite(result)
    corrected = available & target & mask & oracle_height
    filled = ~available & target & mask & oracle_fill
    result[corrected|filled] = truth[corrected|filled]
    valid = mask & np.isfinite(result)
    def pct(selection):
        return 100*int(np.sum(mask & selection))/int(mask.sum())
    return dict(volume=float(result[valid].sum()*pitch**2),area_mm2=float(mask.sum()*pitch**2),
                direct_pct=pct(status==1),estimated_pct=pct((status==2)|(status==4)),
                oracle_replaced_pct=pct(corrected),oracle_fill_pct=pct(filled),unresolved_pct=pct(~valid))


def score(errors_pct):
    return sum(WEIGHTS[name]*abs(errors_pct[name])/100 for name in 'ABC')


def circle_fit(edge):
    def residual(params):
        return np.linalg.norm(edge-params[:2],axis=1)-params[2]
    center = edge.mean(axis=0)
    return least_squares(residual,[*center,np.linalg.norm(edge-center,axis=1).mean()],loss='soft_l1',f_scale=0.5).x


def feature_stats(xx,yy,status,height,pitch):
    rows = []
    for name,signed in [('slot',np.maximum(abs(xx+25),abs(yy))-10),('bore',np.hypot(xx-25,yy)-10)]:
        for region,mask in [('opening',signed<=0),('inner_edge',(signed<=0)&(signed>=-1)),('outer_edge',(signed>0)&(signed<=1)),('core',signed < -1)]:
            subset = status[mask]
            raw = height[mask]
            direct = subset==1
            valid = np.isfinite(raw)
            expected = 40. if region=='outer_edge' else 35.
            rows.append(dict(feature=name,region=region,cells=int(mask.sum()),area_mm2=float(mask.sum()*pitch**2),
                             direct_cells=int(direct.sum()),triangle_cells=int((subset==2).sum()),local_cells=int((subset==4).sum()),
                             missing_cells=int((subset==3).sum()),auto_outside_cells=int((subset==0).sum()),
                             correct_layer_direct_cells=int((direct & (abs(raw-expected)<=1)).sum()),
                             correct_layer_valid_cells=int((valid & (abs(raw-expected)<=1)).sum()),
                             valid_median_mm=float(np.median(raw[valid])) if valid.any() else None,
                             void_above_floor_mm3=float((40-expected)*mask.sum()*pitch**2)))
    return rows


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_case(folder,name,grid,target,truth,fit):
    height,status = grid['height'],grid['status']
    valid = np.isfinite(height)
    error = np.where(target & valid,height-truth,np.nan)
    delta = np.full(height.shape,np.nan)
    delta[target & ~valid] = -truth[target & ~valid]
    delta[valid & ~target] = height[valid & ~target]
    delta[target & valid] = error[target & valid]
    xx,yy = np.meshgrid(grid['x'],grid['y'])
    fig,axes = plt.subplots(1,3,figsize=(13,4),layout='constrained')
    extent = [xx.min()-.25,xx.max()+.25,yy.min()-.25,yy.max()+.25]
    for ax,(data,title,options) in zip(axes,[(status,'Sampling status (0..4)',dict(cmap='viridis',vmin=0,vmax=4)),
                                            (error,'Common height residual (mm)',dict(cmap='coolwarm',vmin=-0.5,vmax=0.5)),
                                            (delta*0.25,'Volume delta per cell (mm³)',dict(cmap='coolwarm',vmin=-12.5,vmax=12.5))]):
        im = ax.imshow(data,origin='lower',extent=extent,interpolation='nearest',**options)
        ax.contour(xx,yy,target.astype(float),levels=[0.5],colors='black',linewidths=0.6)
        fig.colorbar(im,ax=ax,shrink=.65)
        ax.set(xlabel='X (mm)',ylabel='Y (mm)',title=title)
    axes[0].contour(xx,yy,(status>0).astype(float),levels=[0.5],colors='red',linewidths=.7)
    if fit is not None:
        axes[0].add_patch(plt.Circle(fit[:2],fit[2],fill=False,color='orange',linestyle='--'))
    if name=='B':
        axes[0].contour(xx,yy,truth,levels=[37.5],colors='white',linewidths=.8)
    np.savez_compressed(folder/f'{name}_spatial.npz',x=grid['x'],y=grid['y'],height_mm=height,status=status,
                        oracle_domain=target,oracle_height_mm=np.where(target,truth,np.nan),height_error_mm=error,volume_delta_mm3=delta*.25)
    fig.savefig(folder/f'{name}_errors.png',dpi=150)
    plt.close(fig)


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    archive = PROJECT/'heightmap_volume/runs/baseline_20261125'
    source = PROJECT/'standard_scan/runs/baseline_20261118'
    results, entries, manifest, geometry = [], [], {}, {}
    for name in 'ABC':
        path = source/f'{name}_ideal/reconstruction/points.npz'
        manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as cloud:
            points,offsets = cloud['xyz_mm'],cloud['offsets']
        for pitch in [0.25,0.5,1.]:
            grid = measure(points,pitch)
            height,status = grid['height'],grid['status']
            xx,yy = np.meshgrid(grid['x'],grid['y'])
            target,truth = reference_surface(name,xx,yy)
            auto = grid['domain']
            variants = [(f'grid_{pitch:g}',auto,False,False)]
            if pitch==0.5:
                with np.load(archive/f'{name}_ideal/grid.npz') as fixed:
                    for key in ('height','status','domain','x','y','plane'):
                        np.testing.assert_array_equal(grid[key],fixed[key])
                for boundary, surface, missing in itertools.product([False,True],repeat=3):
                    variants.append((f'factor_{int(boundary)}{int(surface)}{int(missing)}',target if boundary else auto,surface,missing))
                fit = None
                circle = auto
                if name=='A':
                    endpoints = []
                    for start,end in zip(offsets[:-1],offsets[1:]):
                        frame = points[start:end]
                        top = frame[frame[:,2]>frame[:,0]*grid['plane'][0]+frame[:,1]*grid['plane'][1]+grid['plane'][2]+1,:2]
                        if len(top)>1:
                            endpoints.extend([top[top[:,1].argmin()],top[top[:,1].argmax()]])
                    fit = circle_fit(np.array(endpoints))
                    circle = np.hypot(xx-fit[0],yy-fit[1])<=fit[2]
                    geometry['circle'] = dict(center_mm=fit[:2].tolist(),edge_points=len(endpoints),fitted_radius_mm=fit[2],
                        fitted_radius_bias_mm=fit[2]-40,auto_equivalent_radius_mm=np.sqrt(auto.sum()*pitch**2/np.pi),
                        true_radius_mm=40,fit_continuous_area_error_mm2=np.pi*fit[2]**2-AREA[name])
                if name=='B':
                    write_csv(output/'B_features.csv',feature_stats(xx,yy,status,height,pitch))
                valid = np.isfinite(height)
                common = valid & target
                geometry[name] = dict(excess_area_mm2=float((auto & ~target).sum()*pitch**2),
                    lost_area_mm2=float((target & ~auto).sum()*pitch**2),outside_target_cells=int((~target).sum()),
                    common_mean_height_error_mm=float((height[common]-truth[common]).mean()),
                    common_p95_abs_height_error_mm=float(np.quantile(abs(height[common]-truth[common]),0.95)))
                variants.append(('circle_A_only',circle,False,False))
                plot_case(output,name,grid,target,truth,fit)
            for label,mask,replace,fill in variants:
                row = dict(case=name,experiment=label,pitch_mm=pitch,**experiment(height,status,mask,target,truth,pitch,replace,fill))
                row['area_error_mm2'] = row['area_mm2']-AREA[name]
                row['signed_pct'] = 100*(row['volume']-REFERENCE[name])/REFERENCE[name]
                results.append(row)
            entry = ledger(height,status,target,truth,pitch,EXACT[name])
            entries.append(dict(case=name,pitch_mm=pitch,**entry,reference_rounding_mm3=EXACT[name]-REFERENCE[name]))
            if abs(entry['closure'])>1e-6:
                raise ValueError(entry)
    scores = {}
    for label in dict.fromkeys(row['experiment'] for row in results):
        group = [row for row in results if row['experiment']==label]
        scores[label] = score({row['case']:100*(round(row['volume'],1)-REFERENCE[row['case']])/REFERENCE[row['case']] for row in group})
        for row in group:
            row['P_unrounded'] = scores[label]
    write_csv(output/'experiments.csv',results)
    write_csv(output/'ledger.csv',entries)
    (output/'details.json').write_text(json.dumps(dict(inputs_sha256=manifest,geometry=geometry,scores=scores),indent=2,allow_nan=False))
    fig,axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    base = [row for row in entries if row['pitch_mm']==0.5]
    for idx,key in enumerate(('boundary','height','missing','grid')):
        axes[0].bar(np.arange(3)+(idx-1.5)*0.18,[row[key] for row in base],width=.18,label=key)
    axes[0].set(xticks=range(3),xticklabels=list('ABC'),ylabel='Signed volume (mm³)')
    axes[0].legend()
    for name in 'ABC':
        rows = [row for row in results if row['case']==name and row['experiment'].startswith('grid_')]
        axes[1].plot([row['pitch_mm'] for row in rows],[row['signed_pct'] for row in rows],'-o',label=name)
    axes[1].set(xlabel='Pitch (mm)',ylabel='Signed volume error (%)')
    axes[1].legend()
    fig.savefig(output/'overview.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=HERE/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run(parser.parse_args().out)
