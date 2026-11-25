import json
import numpy as np
import open3d as o3d


VOLUMES = {'A': 80000*np.pi, 'B': 318000-500*np.pi, 'C': 300000.0}


def build(name):
    if name == 'A':
        mesh = o3d.geometry.TriangleMesh.create_cylinder(40, 50, resolution=256, split=1)
        return mesh.translate([0,0,25]).compute_vertex_normals()
    verts, faces, indices = [], [], {}

    def vertex(point):
        key = tuple(np.round(point, 8))
        if key not in indices:
            indices[key] = len(verts)
            verts.append(key)
        return indices[key]

    def band(outer, inner):
        for idx in range(len(outer)):
            nxt = (idx+1) % len(outer)
            faces.extend([(outer[idx],outer[nxt],inner[nxt]), (outer[idx],inner[nxt],inner[idx])])

    def cap(loop, center):
        for idx, start in enumerate(loop):
            faces.append((start,loop[(idx+1)%len(loop)],center))

    if name == 'B':
        for center in [-25,25]:
            outer, rim, floor, base = [], [], [], []
            for angle in np.arange(128)*np.pi/64:
                unit = np.array([np.cos(angle), np.sin(angle)])
                square = unit/max(abs(unit))
                border = square*[25,40]+[center,0]
                opening = (square if center < 0 else unit)*10+[center,0]
                for loop, xy, height in [(outer,border,40),(rim,opening,40),(floor,opening,35),(base,border,0)]:
                    loop.append(vertex([*xy,height]))
            band(outer,rim)
            band(rim,floor)
            cap(floor,vertex([center,0,35]))
            cap(base[::-1],vertex([center,0,0]))
            for idx in range(len(base)):
                nxt = (idx+1)%len(base)
                if verts[base[idx]][0] == verts[base[nxt]][0] == 0:
                    continue
                faces.extend([(base[idx],base[nxt],outer[nxt]),(base[idx],outer[nxt],outer[idx])])
    elif name == 'C':
        axis = np.linspace(-50,50,81)
        grid = np.array([[vertex([xx,yy,30+8*np.sin(xx*np.pi/50)*np.sin(yy*np.pi/50)]) for xx in axis] for yy in axis])
        for row in range(80):
            for col in range(80):
                left,right = grid[row,col:col+2]
                upper,diagonal = grid[row+1,col:col+2]
                faces.extend([(left,right,diagonal),(left,diagonal,upper)])
        rim = np.r_[grid[0,:-1],grid[:-1,-1],grid[-1,:0:-1],grid[:0:-1,0]]
        base = [vertex([*verts[idx][:2],0]) for idx in rim]
        band(base,rim)
        cap(base[::-1],vertex([0,0,0]))
    else:
        raise ValueError(name)
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)).compute_vertex_normals()


def export_models(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    folder.mkdir()
    report = {}
    for name, exact in VOLUMES.items():
        mesh = build(name)
        if not o3d.io.write_triangle_mesh(str(folder/f'{name}.ply'),mesh):
            raise IOError(name)
        vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        figure = plt.figure(figsize=(6,5))
        axes = figure.add_subplot(projection='3d', computed_zorder=False)
        axes.plot_trisurf(*vertices.T, triangles=triangles, color='#88bace', lightsource=LightSource(310,40), linewidth=0, antialiased=False)
        axes.set(xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',title=f'{name}   |   {exact:.1f} mm³')
        axes.set_box_aspect(np.ptp(vertices,axis=0))
        figure.savefig(folder/f'{name}.png',dpi=125)
        plt.close(figure)
        report[name] = {'analytic_mm3':exact, 'mesh_mm3':mesh.get_volume()}
    (folder/'volumes.json').write_text(json.dumps(report,indent=2))
