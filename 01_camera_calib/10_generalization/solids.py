"""Seeded closed test solids, in millimetres. No measurement dependencies."""
import numpy as np
import open3d as o3d


SEEDS = {'development':17291,'holdout':84317}


def specs(split):
    rng = np.random.default_rng(SEEDS[split])
    groups = {'A':['cone','square_pyramid','hex_prism'] if split=='development' else ['cap','tri_pyramid','hex_prism'],
              'B':['slot_bore','slot_ramp','step_bore'] if split=='development' else ['ramp_bore','slot_step','slot_bore'],
              'C':['gaussian','twin_peaks','saddle'] if split=='development' else ['skew_peak','three_peaks','ridge']}
    results = []
    for category,kinds in groups.items():
        for index,kind in enumerate(kinds):
            results.append(dict(category=category,kind=kind,index=index,width=float(rng.uniform(83,95)),depth=float(rng.uniform(69,80)),
                                height=float(rng.uniform(51,60)),amplitude=float(rng.uniform(9,16)),shift=float(rng.uniform(-.18,.18)),
                                opening_mm=15. if index==0 else float(rng.uniform(16,21)),recess_mm=5. if index==0 else float(rng.uniform(3.2,5))))
            if kind=='tri_pyramid':
                results[-1]['width']+=16
    return results


def solid(spec):
    kind = spec['kind']
    width, depth, height = [spec[key] for key in ('width','depth','height')]
    vertices,faces,lookup = [],[],{}
    def vert(xx,yy,zz):
        key = tuple(round(float(v),8) for v in (xx,yy,zz))
        if key not in lookup:
            lookup[key]=len(vertices)
            vertices.append(key)
        return lookup[key]
    def band(outer,inner):
        for idx,start in enumerate(outer):
            nxt=(idx+1)%len(outer)
            faces.extend([(start,outer[nxt],inner[nxt]),(start,inner[nxt],inner[idx])])
    def cap(loop,center):
        faces.extend((start,loop[(idx+1)%len(loop)],center) for idx,start in enumerate(loop))
    if spec['category']=='A':
        count={'hex_prism':6,'tri_pyramid':3,'square_pyramid':4,'cone':160,'cap':160}[kind]
        angles=np.arange(count)*2*np.pi/count
        radius=width/2 if kind!='cap' else (60+height)/2
        def ring(radius,zz):
            return [vert(radius*np.cos(angle),radius*np.sin(angle),zz) for angle in angles]
        rim=ring(radius,0)
        cap(rim[::-1],vert(0,0,0))
        if kind=='hex_prism':
            upper=ring(radius,height)
            band(rim,upper)
            cap(upper,vert(0,0,height))
        else:
            if kind=='cap':
                sphere=(height**2+radius**2)/(2*height)
                for angle in np.linspace(np.arccos((sphere-height)/sphere),0,41)[1:-1]:
                    upper=ring(sphere*np.sin(angle),height-sphere+sphere*np.cos(angle))
                    band(rim,upper)
                    rim=upper
            cap(rim,vert(0,0,height))
    elif spec['category']=='B':
        for center,feature in zip((-width/4,width/4),kind.split('_')):
            outer,base,rim,inner=[],[],[],[]
            for angle in np.arange(96)*np.pi/48:
                unit=np.array([np.cos(angle),np.sin(angle)])
                square=unit/max(abs(unit))
                border=square*[width/4,depth/2]+[center,0]
                local=unit*spec['opening_mm']/2 if feature=='bore' else square*[spec['opening_mm']*.64,spec['opening_mm']/2]
                opening=local+[center,0]
                offset={'bore':-spec['recess_mm'],'slot':-spec['recess_mm'],'step':6.,'ramp':4.+3*local[0]/(spec['opening_mm']*.64)}[feature]
                for loop,xy,zz in [(outer,border,height),(base,border,0),(rim,opening,height),(inner,opening,height+offset)]:
                    loop.append(vert(*xy,zz))
            band(outer,rim)
            band(rim,inner)
            cap(inner,vert(center,0,np.mean([vertices[idx][2] for idx in inner])))
            cap(base[::-1],vert(center,0,0))
            for idx,start in enumerate(base):
                nxt=(idx+1)%len(base)
                if vertices[start][0]==vertices[base[nxt]][0]==0:
                    continue
                faces.extend([(start,base[nxt],outer[nxt]),(start,outer[nxt],outer[idx])])
    else:
        axis=np.linspace(-1,1,65)
        xx,yy=np.meshgrid(axis,axis)
        shift=spec['shift']
        def bump(cx,cy,sx,sy):
            return np.exp(-((xx-cx)/sx)**2-((yy-cy)/sy)**2)
        waves={'gaussian':bump(.24,shift,.48,.67),'twin_peaks':bump(-.46,.22,.32,.45)+.8*bump(.43,-.31,.4,.34),
               'saddle':.4*(xx**2-yy**2)+.23*xx*yy,'skew_peak':bump(shift,.29,.52,.38)*(1+.3*xx),
               'three_peaks':bump(-.55,.3,.35,.4)+.7*bump(.42,.42,.4,.33)+.87*bump(.15,-.52,.45,.31),
               'ridge':np.exp(-((yy-.4*xx**2-shift)/.43)**2)*(.72+.22*xx)}
        zz=height+spec['amplitude']*waves[kind]
        grid=np.array([vert(xval*width/2,yval*depth/2,zval) for xval,yval,zval in zip(xx.flat,yy.flat,zz.flat)]).reshape(xx.shape)
        for row in range(len(axis)-1):
            for col in range(len(axis)-1):
                left,right=grid[row,col:col+2]
                upper,diagonal=grid[row+1,col:col+2]
                faces.extend([(left,right,diagonal),(left,diagonal,upper)])
        rim=np.r_[grid[0,:-1],grid[:-1,-1],grid[-1,:0:-1],grid[:0:-1,0]]
        base=[vert(*vertices[idx][:2],0) for idx in rim]
        band(base,rim)
        cap(base[::-1],vert(0,0,0))
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces))
