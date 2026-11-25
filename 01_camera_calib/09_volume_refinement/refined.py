"""Contour and local surface candidates for ordered, X-travel scan clouds (mm)."""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'07_heightmap_volume'))
from estimator import measure, integrate


def refine(points, baseline=None, pitch=0.5, step=2.):
    base = measure(points,pitch,scan_step=step) if baseline is None else baseline
    if base['pitch_mm']!=pitch:
        raise ValueError('Grid pitch mismatch')
    coeff = base['plane']
    relative = points[:,2]-points[:,:2]@coeff[:2]-coeff[2]
    positive = relative>1
    xy = points[positive,:2]
    heights = relative[positive]
    indices = np.arange(len(points))
    packs = np.split(indices,np.flatnonzero(np.diff(points[:,1])>2.5*step)+1)
    strips = [pack[positive[pack]] for pack in packs if positive[pack].sum()>2]
    locations = np.array([np.median(points[strip,0]) for strip in strips])
    if len(strips)<4 or np.any(np.diff(locations)<=0):
        raise ValueError('Ordered X scans required; Y must reset between profiles')
    low = np.array([points[strip[np.argmin(points[strip,1])],:2] for strip in strips])
    high = np.array([points[strip[np.argmax(points[strip,1])],:2] for strip in strips])
    edges = np.vstack((low,high))
    center = edges.mean(axis=0)
    def residual(params):
        return np.linalg.norm(edges-params[:2],axis=1)-params[2]
    fit = least_squares(residual,[*center,np.linalg.norm(edges-center,axis=1).mean()],loss='soft_l1',f_scale=pitch).x
    xx,yy = np.meshgrid(base['x'],base['y'])
    circle_residual = np.quantile(abs(residual(fit)),0.95)
    flat = np.ptp(np.quantile(heights,[.02,.98]))<=1
    row_pitch = np.median([np.median(abs(np.diff(points[strip,1]))) for strip in strips])
    band = (xx>=locations[0]-step/2)&(xx<=locations[-1]+step/2)
    band &= (yy>=np.interp(xx,locations,low[:,1])-row_pitch/2)&(yy<=np.interp(xx,locations,high[:,1])+row_pitch/2)
    method = 'strip'
    if flat and circle_residual<=step/2:
        band = np.hypot(xx-fit[0],yy-fit[1])<=fit[2]
        method = 'cylinder'
    domain = band & ~base['ground']
    height = base['height'].copy()
    status = base['status'].copy()
    height[~domain] = np.nan
    status[~domain] = 0
    missing = domain & ~np.isfinite(height)
    status[missing] = 3
    query = np.column_stack((xx[missing],yy[missing]))
    distance, neighbors = cKDTree(xy).query(query,k=12)
    levels = heights[neighbors]
    ordered = np.sort(levels,axis=1)
    gaps = np.diff(ordered,axis=1)
    split = np.argmax(gaps,axis=1)
    jump = gaps.max(axis=1)>1.6
    threshold = (ordered[np.arange(len(split)),split]+ordered[np.arange(len(split)),split+1])/2
    same = (levels<=threshold[:,None])==(levels[:,:1]<=threshold[:,None])
    same |= ~jump[:,None]
    supports = same & (distance<=1.5*step)
    competitor = np.where(same,np.inf,distance).min(axis=1)
    unambiguous = (~jump)|(distance[:,0]+pitch/4<competitor)
    ground_distance = cKDTree(points[abs(relative)<0.6,:2]).query(query)[0]
    valid = (distance[:,0]<=step)&(supports.sum(axis=1)>=3)&unambiguous&(ground_distance>distance[:,0])
    weights = supports/np.maximum(distance,0.1*pitch)**2
    values = (weights*levels).sum(axis=1)/np.maximum(weights.sum(axis=1),1e-12)
    selected = np.flatnonzero(missing)[valid]
    height.flat[selected] = values[valid]
    status.flat[selected] = 5
    return dict(height=height,status=status,domain=domain,x=base['x'],y=base['y'],volume_mm3=integrate(height,pitch),
                method=method,fitted_radius_mm=float(fit[2]),edge_residual_p95_mm=float(circle_residual),
                step_fills=int((valid&jump).sum()),local_fills=len(selected))
