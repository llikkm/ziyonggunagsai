import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'laser_triangulation'))
from laser_lib import CameraModel, LaserPlane, extract_laser_centers, reconstruct


def replay(inputs, output):
    config = json.loads((inputs/'config.json').read_text())
    camera = CameraModel(config['K'],config['dist'],config['size'])
    plane = LaserPlane.from_abcd(config['plane_camera'])
    rotation = np.array(config['R_camera_to_world'])
    output.mkdir(parents=True,exist_ok=False)
    pixels, points, offsets = [], [], [0]
    for frame in config['frames']:
        image = cv2.imdecode(np.fromfile(inputs/frame['image'],dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape[::-1] != tuple(config['size']):
            raise ValueError(frame['image'])
        centers = extract_laser_centers(image,axis='row').uv
        result = reconstruct(centers,camera,plane)
        pixels.append(centers[result.valid])
        points.append(result.finite_points() @ rotation.T + frame['t_mm'])
        offsets.append(offsets[-1]+len(points[-1]))
    cloud = np.vstack(points)
    if not len(cloud):
        raise ValueError('扫描未提取到点')
    np.savez_compressed(output/'points.npz',xyz_mm=cloud,uv=np.vstack(pixels),offsets=offsets)
    geometry = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud))
    if not o3d.io.write_point_cloud(str(output/'cloud.ply'),geometry):
        raise IOError(output/'cloud.ply')
    return cloud


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    replay(args.inputs,args.output)
