"""Read-only visualization of a completed frozen benchmark."""
import argparse
import copy
import csv
import hashlib
import json
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import open3d as o3d


SPLITS = {'development':'开发验证集','holdout':'最终保留集'}
NAMES = dict(cone='圆锥', square_pyramid='四棱锥', hex_prism='六棱柱', cap='球冠',
             tri_pyramid='三棱锥', slot_bore='槽＋盲孔', slot_ramp='槽＋斜台', step_bore='台阶＋盲孔',
             ramp_bore='斜台＋盲孔', slot_step='槽＋台阶', gaussian='偏心单峰', twin_peaks='双峰',
             saddle='鞍面', skew_peak='非对称单峰', three_peaks='三峰', ridge='弯脊')
COLORS = ['#f2f1ee','#104660','#57b9c7','#ed913c','#accc99','#7768aa']
LABELS = ['足迹外','直接命中','三角估计','缺测','单侧估计','局部保边']
MODES = {'ideal':'理想','jitter':'扰动'}


def compare_cells(height, truth, pitch):
    valid, target = np.isfinite(height), np.isfinite(truth)
    common = valid & target
    residual = np.where(common,height-truth,np.nan)
    delta = residual.copy()
    delta[target & ~valid] = -truth[target & ~valid]
    delta[valid & ~target] = height[valid & ~target]
    return residual,delta*pitch**2


def spatial(folder,mesh,pose,row,output,title):
    mesh = copy.deepcopy(mesh)
    dx,dy,angle = pose
    mesh.rotate(mesh.get_rotation_matrix_from_xyz([0,0,np.deg2rad(angle)]),center=[0,0,0]).translate([dx,dy,0])
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    with np.load(folder/'grid.npz') as grid:
        pitch=float(np.diff(grid['x'])[0])
        np.testing.assert_allclose(np.diff(grid['y']),pitch)
        first=np.array([grid['x'][0],grid['y'][0]])
        last=np.array([grid['x'][-1],grid['y'][-1]])
        pad=np.maximum(0,np.ceil((first-mesh.get_min_bound()[:2])/pitch)).astype(int)+2
        end=np.maximum(0,np.ceil((mesh.get_max_bound()[:2]-last)/pitch)).astype(int)+2
        cols,rows=np.array([len(grid['x']),len(grid['y'])])+pad+end
        xx,yy=np.meshgrid(first[0]+pitch*(np.arange(cols)-pad[0]), first[1]+pitch*(np.arange(rows)-pad[1]))
        height=np.full(xx.shape,np.nan)
        status=np.zeros(xx.shape,int)
        window=np.s_[pad[1]:pad[1]+len(grid['y']),pad[0]:pad[0]+len(grid['x'])]
        height[window],status[window]=grid['height'],grid['status']
    rays=np.stack((xx,yy,xx*0+160,xx*0,xx*0,xx*0-1),axis=-1).astype('float32')
    truth=160.-scene.cast_rays(o3d.core.Tensor(rays),nthreads=1)['t_hit'].numpy().astype(float)
    truth[~np.isfinite(truth)]=np.nan
    residual,delta=compare_cells(height,truth,pitch)
    np.testing.assert_allclose(np.nansum(height)*pitch**2,float(row['volume_mm3']),atol=.051,rtol=0)
    np.testing.assert_allclose(np.nansum(delta),pitch**2*(np.nansum(height)-np.nansum(truth)),atol=1e-7,rtol=0)
    fig,axes=plt.subplots(1,3,figsize=(14,4.9),layout='constrained')
    extent=[xx.min()-pitch/2,xx.max()+pitch/2,yy.min()-pitch/2,yy.max()+pitch/2]
    palettes=[ListedColormap(COLORS),'RdBu_r','RdBu_r']
    target=np.isfinite(truth)
    for index,(ax,data,label,palette) in enumerate(zip(axes,[status,residual,delta],
            ['采样状态与轮廓','共同有效区 · 高度误差 (mm)','每格体积误差贡献 (mm³)'],palettes)):
        options=dict(norm=BoundaryNorm(np.arange(-.5,6.5),6)) if index==0 else dict(vmin=-[0,.5,12.5][index],vmax=[0,.5,12.5][index])
        im=ax.imshow(data,origin='lower',extent=extent,cmap=palette,interpolation='nearest',**options)
        ax.contour(xx,yy,target.astype(int),levels=[.5],colors='#202020',linewidths=.75)
        if index==0:
            ax.contour(xx,yy,(status>0).astype(int),levels=[.5],colors='#c72837',linewidths=.8,linestyles='dashed')
        bar=fig.colorbar(im,ax=ax,shrink=.70,extend='neither' if index==0 else 'both')
        if index==0:
            bar.set_ticks(range(6),labels=LABELS)
        ax.set(xlabel='X (mm)',ylabel='Y (mm)',title=label)
    fig.suptitle(f"{title}   |   p{row['pose']} · {MODES[row['mode']]}   |   体积 {float(row['volume_mm3']):,.1f} mm³   |   偏差 {float(row['signed_pct']):+.2f}%")
    fig.savefig(output,dpi=125)
    plt.close(fig)


def preview(mesh,rows,spec,title,output):
    fig=plt.figure(figsize=(12,6),layout='constrained')
    layout=fig.add_gridspec(2,2,width_ratios=[1.15,1])
    ax=fig.add_subplot(layout[:,0],projection='3d')
    verts=np.asarray(mesh.vertices)
    ax.plot_trisurf(*verts.T,triangles=np.asarray(mesh.triangles),color='#77aeb5',linewidth=0,antialiased=False,shade=True)
    ax.view_init(elev=34,azim=-58)
    ax.set_box_aspect(np.ptp(verts,axis=0))
    ax.set(xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',title=f"参考几何 · 实体体积 {spec['volume_mm3']:,.1f} mm³")
    positions=np.arange(len(rows))
    labels=[f"p{r['pose']}\n{MODES[r['mode']]}" for r in rows]
    values=np.array([float(r['signed_pct']) if not r['failure'] else np.nan for r in rows])
    ax=fig.add_subplot(layout[0,1])
    ax.bar(positions,values,color=['#236079' if r['mode']=='ideal' else '#d77e40' for r in rows],width=.65)
    ax.axhline(0,color='black',lw=.8)
    ax.set(xticks=positions,xticklabels=labels,ylabel='有符号误差 (%)',title='六次扫描 · 体积误差')
    for idx,val in enumerate(values):
        if np.isfinite(val):
            ax.annotate(f'{val:+.2f}',(idx,val),xytext=(0,2 if val>=0 else -2),textcoords='offset points',ha='center',va='bottom' if val>=0 else 'top',fontsize=9)
        else:
            ax.text(idx,0,'失败',ha='center',color='red')
    ax.margins(y=.23)
    ax=fig.add_subplot(layout[1,1])
    bottom=np.zeros(len(rows))
    direct=np.array([float(r['direct_pct']) if not r['failure'] else np.nan for r in rows])
    estimated=np.array([float(r['estimated_pct']) if not r['failure'] else np.nan for r in rows])
    for label,portion,color in [('直接命中',direct,COLORS[1]),('局部估计',estimated,COLORS[2]),('缺测',100-direct-estimated,COLORS[3])]:
        ax.bar(positions,portion,bottom=bottom,color=color,label=label,width=.65)
        bottom+=portion
    ax.set(xticks=positions,xticklabels=labels,ylim=(0,105),ylabel='自动足迹面积 (%)',title='有效面积的来源')
    ax.legend(ncol=3,loc='lower center',bbox_to_anchor=(.5,-.43),frameon=False)
    fig.suptitle(title,fontsize=16)
    fig.savefig(output,dpi=125)
    plt.close(fig)


def generate(root,output):
    output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','Noto Sans CJK SC','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
    poses=json.loads((root/'lock.json').read_text())['poses']
    html=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>泛化扫描图集</title>',
          '<style>body{font:16px/1.6 system-ui;margin:auto;padding:2rem;max-width:1220px;color:#153345;background:#f6f8f7}article{background:white;margin:2rem 0;padding:1rem 2rem;border-radius:.8rem}img{max-width:100%;height:auto}a{color:#165e79}summary{cursor:pointer;padding:.4rem}nav{display:flex;flex-wrap:wrap;gap:.6em}h2{border-bottom:2px solid #377c89}</style><main>',
          '<h1>泛化测试 · 模型与测量图集</h1><p>每件先看三维参考几何与六次扫描指标，再展开姿态查看空间误差；图片可点开原尺寸。p0 / p1 / p2 分别为 '+ '、'.join(f'({dx}, {dy}, {ang}°)' for dx,dy,ang in poses)+'，平移单位 mm。</p>',
          '<p>黑线：真实轮廓；红虚线：测量轮廓。高度误差仅显示共同有效格，红为偏高、蓝为偏低；体积贡献包含外扩和漏测。两种误差图分别在 ±0.5 mm 和 ±12.5 mm³ 处饱和。白色为无对应值。覆盖率分母为自动足迹。</p>']
    contents=[]
    for split,caption in SPLITS.items():
        folder=root/split
        specs=json.loads((folder/'models/manifest.json').read_text())
        with (folder/'results.csv').open(encoding='utf-8-sig') as stream:
            records=list(csv.DictReader(stream))
        html.append(f'<h2>{caption}</h2><nav>'+ ' '.join(f'<a href="#{split}-{s["id"]}">{s["id"]} {NAMES[s["kind"]]}</a>' for s in specs)+'</nav>')
        contents.extend([spec|{'split':split,'records':records} for spec in specs])
    for spec in contents:
        split,key=spec['split'],spec['id']
        folder=root/split
        model=folder/'models'/f'{key}.ply'
        if hashlib.sha256(model.read_bytes()).hexdigest()!=spec['sha256']:
            raise ValueError(model)
        mesh=o3d.io.read_triangle_mesh(str(model))
        rows=sorted((r for r in spec['records'] if r['model']==key),key=lambda r:(int(r['pose']),r['mode']))
        if {(int(r['pose']),r['mode']) for r in rows}!={(i,m) for i in range(len(poses)) for m in MODES} or len(rows)!=2*len(poses):
            raise ValueError(key)
        path=output/split/key
        path.mkdir(parents=True)
        title=f'{SPLITS[split]} · {key}   {NAMES[spec["kind"]]}'
        preview(mesh,rows,spec,title,path/'preview.png')
        prefix=f'{split}/{key}/'
        def embed(filename,desc):
            url=prefix+filename
            return f'<a href="{url}"><img loading="lazy" src="{url}" alt="{desc}"></a>'
        html.append(f'<article id="{split}-{key}"><h2>{title}</h2>'+embed('preview.png',title))
        for row in rows:
            suffix=f"p{row['pose']}_{row['mode']}"
            if row['failure']:
                html.append(f"<p>{suffix} 失败：{escape(row['failure'])}</p>")
                continue
            name=suffix+'.png'
            spatial(folder/f'{key}_{suffix}',mesh,poses[int(row['pose'])],row,path/name,title)
            html.append(f"<details><summary>p{row['pose']} · {MODES[row['mode']]}　{float(row['signed_pct']):+.3f}%　查看空间误差</summary>"+embed(name,suffix)+'</details>')
        html.append('<a href="#">回到索引</a></article>')
    (output/'index.html').write_text('\n'.join(html)+'</main></html>',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    generate(args.root,args.out or args.root/'gallery')
