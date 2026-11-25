"""Refine existing camera calibration with a measured square size and 300 iterations."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np

import measure3d as m


def refine_intrinsics(objects, pixels, image_size, initial_k, initial_dist, model='standard'):
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    if model == 'rational':
        flags |= cv2.CALIB_RATIONAL_MODEL
    elif model != 'standard':
        raise ValueError(f'Unsupported calibration model: {model}')
    rms, k, dist, rotations, translations = cv2.calibrateCamera(
        objects, pixels, tuple(image_size), np.array(initial_k, dtype=float, copy=True),
        np.array(initial_dist, dtype=float, copy=True), flags=flags,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 300, 1e-12))
    if not np.isfinite(rms) or not np.isfinite(k).all() or not np.isfinite(dist).all():
        raise ValueError('Calibration returned nonfinite parameters')
    return {'camera_matrix': k, 'dist_coeffs': dist, 'rms_px': float(rms),
            'rvecs': rotations, 'tvecs': translations, 'max_iterations': 300, 'epsilon': 1e-12}


def main():
    cv2.ocl.setUseOpenCL(False)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calib', required=True, type=Path, help='Existing calibration file')
    p.add_argument('--dir', required=True, type=Path, help='Original multi-pose calibration images')
    p.add_argument('--square', required=True, type=float, help='Measured square edge length in mm')
    p.add_argument('--out', required=True, type=Path, help='New calibration JSON; existing files are preserved')
    args = p.parse_args()
    if not np.isfinite(args.square) or args.square <= 0:
        p.error('--square must be positive and finite')
    if args.out.resolve() == args.calib.resolve() or args.out.exists():
        raise ValueError('Choose a new output file to preserve existing calibration')
    base = m.cl.load_calib(args.calib)
    pattern = base['pattern']['inner_cols'], base['pattern']['inner_rows']
    obj = m.cl.object_points(pattern, args.square)
    objects, pixels, paths = [], [], m.cl.list_images(args.dir)
    for path in paths:
        gray = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None or list(gray.shape[::-1]) != list(base['image_size']):
            raise ValueError(f'Unreadable image or wrong image size: {path}')
        found, corners = m.cl.find_corners(gray, pattern)
        if not found:
            raise ValueError(f'Chessboard not detected: {path}')
        objects.append(obj)
        pixels.append(corners)
    if len(paths) < 6:
        raise ValueError('At least 6 calibration images required')
    solved = refine_intrinsics(objects, pixels, base['image_size'], base['camera_matrix'],
                               base['dist_coeffs'], base.get('model', 'standard'))
    per_view = []
    for path, corners, rv, tv in zip(paths, pixels, solved['rvecs'], solved['tvecs']):
        projected = cv2.projectPoints(obj, rv, tv, solved['camera_matrix'], solved['dist_coeffs'])[0]
        error = np.linalg.norm(projected.reshape(-1, 2)-corners.reshape(-1, 2), axis=1)
        per_view.append({'image': path.name, 'rms_px': float(np.sqrt(np.mean(error**2))),
                         'mean_px': float(error.mean()), 'max_px': float(error.max())})
    result = dict(base)
    result.update(created=datetime.now().astimezone().isoformat(timespec='seconds'),
                  camera_matrix=solved['camera_matrix'], dist_coeffs=solved['dist_coeffs'],
                  rms_px=solved['rms_px'], n_views=len(paths), per_view=per_view, skipped=[],
                  pattern={**base['pattern'], 'square_mm': args.square},
                  refined_from=str(args.calib.resolve()),
                  calibration_image_directory=str(args.dir.resolve()),
                  optimizer={'max_iterations': solved['max_iterations'], 'epsilon': solved['epsilon']})
    m.cl.save_calib(args.out, result)
    print(f'Saved {args.out}: {len(paths)} views, square={args.square} mm, RMS={solved["rms_px"]:.6f} px')
    print('Focal lengths:', solved['camera_matrix'][0, 0], solved['camera_matrix'][1, 1])


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, KeyError, cv2.error) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
