"""Point measurements in world millimetres -> XY height integral in mm³."""
from time import perf_counter

import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay, cKDTree


def fit_table(points):
    xy1 = np.column_stack((points[:,:2],np.ones(len(points))))
    zz = points[:,2]
    support = zz <= np.quantile(zz,0.15)
    for _ in range(5):
        coeff, _, rank, _ = np.linalg.lstsq(xy1[support],zz[support],rcond=None)
        if rank<3:
            raise ValueError('Insufficient 2D table support')
        residual = zz-xy1@coeff
        threshold = max(0.45,3*1.4826*np.median(abs(residual[support]-np.median(residual[support]))))
        support = abs(residual)<min(threshold,1.0)
    return coeff


def integrate(height, pitch):
    if not np.isfinite(pitch) or pitch<=0:
        raise ValueError('grid pitch must be positive mm')
    return float(np.nansum(height)*pitch**2)


def measure(points, pitch=0.5, units='mm', scan_step=2.):
    start = perf_counter()
    points = np.asarray(points,dtype=float)
    if units!='mm' or not np.isfinite([pitch,scan_step]).all() or min(pitch,scan_step)<=0:
        raise ValueError('expected positive mm dimensions')
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<6 or not np.isfinite(points).all():
        raise ValueError('invalid XYZ cloud')
    coeff = fit_table(points)
    heights = points[:,2]-(points[:,:2]@coeff[:2]+coeff[2])
    table = abs(heights)<0.6
    top = heights>1.0
    if top.sum()<3 or table.sum()<3:
        raise ValueError('missing surface or table')
    plane_end = perf_counter()
    bins = np.floor(points[:,:2]/pitch).astype(int)
    origin = bins.min(axis=0)-2
    bins -= origin
    width, rows = bins.max(axis=0)+3
    shape = (rows,width)
    x_axis = (origin[0]+np.arange(width)+0.5)*pitch
    y_axis = (origin[1]+np.arange(rows)+0.5)*pitch
    ground = np.zeros(shape,bool)
    ground[bins[table,1],bins[table,0]] = True
    flat = np.ravel_multi_index((bins[top,1],bins[top,0]),shape)
    order = np.lexsort((heights[top],flat))
    ids, first, counts = np.unique(flat[order],return_index=True,return_counts=True)
    samples = heights[top][order]
    height = np.full(shape,np.nan)
    height.flat[ids] = (samples[first+(counts-1)//2]+samples[first+counts//2])/2
    observed = np.isfinite(height)
    ground &= ~observed
    top_xy = points[top,:2]
    top_z = heights[top]
    tree = cKDTree(top_xy)
    xx, yy = np.meshgrid(x_axis,y_axis)
    query = np.column_stack((xx.ravel(),yy.ravel()))
    support_axes = np.array([0.5*scan_step+0.5*pitch,0.40+0.5*pitch])
    normalized = cKDTree(top_xy/support_axes)
    shell = normalized.query(query/support_axes)[0].reshape(shape)<=1
    domain = ndimage.binary_fill_holes(shell|observed) & ~ground
    status = np.zeros(shape,np.uint8)
    status[domain] = 3
    status[observed] = 1
    boundary_end = perf_counter()
    candidates = np.flatnonzero(domain & ~observed)
    local = query[candidates]
    tri = Delaunay(top_xy)
    simplex = tri.find_simplex(local)
    valid = simplex>=0
    vertices = tri.simplices[simplex]
    corners = top_xy[vertices]
    edge = np.linalg.norm(corners-np.roll(corners,1,axis=1),axis=2).max(axis=1)
    valid &= (edge<=1.45*scan_step)&(np.ptp(top_z[vertices],axis=1)<=1.6)
    table_dist = cKDTree(points[table,:2]).query(local)[0]
    valid &= table_dist>pitch
    transforms = tri.transform[simplex[valid]]
    bary = np.einsum('nij,nj->ni',transforms[:,:2],local[valid]-transforms[:,2])
    weights = np.column_stack((bary,1-bary.sum(axis=1)))
    height.flat[candidates[valid]] = (weights*top_z[vertices[valid]]).sum(axis=1)
    status.flat[candidates[valid]] = 2
    outer = domain & ~ndimage.binary_erosion(domain,iterations=max(1,int(np.ceil(scan_step/(2*pitch)))))
    border = (~valid)&outer.ravel()[candidates]&(simplex<0)&(table_dist>pitch)
    near_dist, near_ids = tree.query(local[border],k=3)
    ok = (near_dist[:,0]<=np.linalg.norm(support_axes)) & (near_dist[:,-1]<=scan_step) & (np.ptp(top_z[near_ids],axis=1)<=1.6)
    selected = candidates[border][ok]
    height.flat[selected] = top_z[near_ids[ok,0]]
    status.flat[selected] = 4
    transition = (~valid)&(simplex>=0)&(edge<=1.45*scan_step)&(np.ptp(top_z[vertices],axis=1)>1.6)&(table_dist>scan_step)
    distance, neighbors = tree.query(local[transition],k=8)
    levels = top_z[neighbors]
    same = abs(levels-levels[:,:1])<=0.5
    other_distance = np.where(same,np.inf,distance).min(axis=1)
    supported = (same & (distance<scan_step)).sum(axis=1)>=3
    coherent = supported & (distance[:,0]<0.5*scan_step) & (distance[:,0]+pitch/2<other_distance)
    selected = candidates[transition][coherent]
    height.flat[selected] = levels[coherent,0]
    status.flat[selected] = 4
    grid_end = perf_counter()
    volume = integrate(height,pitch)
    stages = dict(plane_s=plane_end-start,occupancy_s=boundary_end-plane_end,grid_s=grid_end-boundary_end,integration_s=perf_counter()-grid_end)
    return dict(height=height,status=status,domain=domain,ground=ground,x=x_axis,y=y_axis,plane=coeff,
                pitch_mm=pitch,volume_mm3=volume,timing=stages)
