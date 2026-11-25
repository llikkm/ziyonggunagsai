"""Manual capture from built-in, USB or OBS virtual cameras, with live pose checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

import measure3d as m


def pose_quality(pose, reference, max_rms, max_angle):
    """Compare two board normals in camera coordinates (degrees)."""
    rms = pose['reprojection_rms_px']
    if not np.isfinite(rms) or rms > max_rms:
        return False, None, f'RMS {rms:.2f} > {max_rms:.2f} px'
    if reference is None:
        return True, None, 'ready'
    normal = cv2.Rodrigues(np.asarray(pose['rvec'], dtype=float))[0][:, 2]
    ref_normal = cv2.Rodrigues(np.asarray(reference['rvec'], dtype=float))[0][:, 2]
    angle = float(np.degrees(np.arccos(np.clip(abs(normal @ ref_normal), 0, 1))))
    if not np.isfinite(angle) or angle > max_angle:
        return False, angle, f'board angle {angle:.2f} > {max_angle:.2f} deg'
    return True, angle, 'ready'


def save_frame(path, frame):
    """Save raw camera pixels, excluding all preview overlays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode('.png', frame)
    if not ok:
        raise RuntimeError('PNG encoding failed')
    with path.open('xb') as stream:
        stream.write(encoded.tobytes())


def grid_cell(corners, image_shape):
    """Assign a board to the cell containing the mean of its inner corners."""
    height, width = image_shape[:2]
    x, y = np.asarray(corners).reshape(-1, 2).mean(axis=0)
    return int(np.clip(y*3/height, 0, 2)), int(np.clip(x*3/width, 0, 2))


def draw_calibration_grid(frame, counts, corners=None):
    preview = frame.copy()
    height, width = preview.shape[:2]
    for i in (1, 2):
        cv2.line(preview, (width*i//3, 0), (width*i//3, height-1), (110, 110, 110), 1)
        cv2.line(preview, (0, height*i//3), (width-1, height*i//3), (110, 110, 110), 1)
    if corners is not None:
        row, col = grid_cell(corners, preview.shape)
        cv2.rectangle(preview, (width*col//3+2, height*row//3+2),
                      (width*(col+1)//3-2, height*(row+1)//3-2), (0, 220, 220), 2)
    for row in range(3):
        for col in range(3):
            count = int(counts[row, col])
            color = (0, 210, 0) if count >= 2 else (0, 170, 255) if count else (190, 190, 190)
            y = height*row//3 + (85 if row == 0 else 30)
            cv2.putText(preview, f'{count} saved', (width*col//3+12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
    return preview


def open_camera(index, backend, width, height):
    backends = {'dshow': [cv2.CAP_DSHOW], 'msmf': [cv2.CAP_MSMF],
                'auto': [cv2.CAP_DSHOW, cv2.CAP_MSMF]}
    for api in backends[backend]:
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, frame = cap.read()
            if ok:
                return cap, frame
        cap.release()
    raise RuntimeError(f'Camera {index} could not provide frames using {backend}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', choices=['builtin', 'usb', 'obs'], default='builtin', help='Label for saved metadata; device is selected by --camera')
    p.add_argument('--camera', type=int, default=0, help='OpenCV device index for the chosen backend')
    p.add_argument('--backend', choices=['auto', 'dshow', 'msmf'], default='dshow')
    p.add_argument('--list-cameras', action='store_true')
    p.add_argument('--preview', action='store_true', help='Preview chosen camera without saving or loading calibration')
    p.add_argument('--max-index', type=int, default=5)
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--out', type=Path)
    p.add_argument('--count', type=int, default=10)
    p.add_argument('--calib', type=Path, default=m.ROOT/'out/calib.json')
    p.add_argument('--reference', type=Path, help='Baseline RAW image from this same camera and imaging setup')
    p.add_argument('--calibration', action='store_true', help='Collect varied board poses for a new camera calibration')
    p.add_argument('--cols', type=int, default=9, help='Number of squares when collecting calibration images')
    p.add_argument('--rows', type=int, default=6)
    p.add_argument('--max-rms', type=float, default=3.)
    p.add_argument('--max-angle', type=float, default=2.)
    args = p.parse_args()
    if min(args.width, args.height, args.count) <= 0 or args.camera < 0 or args.max_index < 0:
        p.error('Positive resolution/count and nonnegative camera indices required')
    if not np.isfinite(args.max_rms) or args.max_rms <= 0 or not 0 < args.max_angle < 90:
        p.error('max-rms > 0 and 0 < max-angle < 90 required')
    if args.list_cameras:
        for index in range(args.max_index+1):
            try:
                cap, frame = open_camera(index, args.backend, args.width, args.height)
                print(f'camera {index}: {frame.shape[1]}x{frame.shape[0]}, backend={cap.getBackendName()}')
                cap.release()
            except RuntimeError:
                print(f'camera {index}: unavailable')
        return
    if args.preview:
        cap, frame = open_camera(args.camera, args.backend, args.width, args.height)
        title = f'Preview camera {args.camera}'
        try:
            print(f'camera {args.camera}: {frame.shape[1]}x{frame.shape[0]}, backend={cap.getBackendName()}; Q/ESC to close')
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            while True:
                cv2.imshow(title, frame)
                if cv2.waitKey(1) & 0xff in (27, ord('q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera stopped providing frames')
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return
    if args.out is None:
        p.error('--out is required for capture')
    if args.calibration and args.reference:
        p.error('Use --reference for height capture; use --calibration for new camera intrinsics')
    if args.cols < 3 or args.rows < 3:
        p.error('Board needs at least 3x3 squares')
    if args.out.exists() and (m.cl.list_images(args.out) or (args.out/'capture.json').exists()):
        raise ValueError(f'Output already contains a capture: {args.out}. Choose a new folder.')
    calib, reference = None, None
    pattern = (args.cols-1, args.rows-1)
    if not args.calibration:
        calib = m.cl.load_calib(args.calib)
        if list(calib['image_size']) != [args.width, args.height]:
            raise ValueError(f'Calibration size is {calib["image_size"]}; match --width and --height')
        pattern = calib['pattern']['inner_cols'], calib['pattern']['inner_rows']
        if args.reference:
            reference = m.pose_image(args.reference, calib, args.max_rms)
    cap, frame = open_camera(args.camera, args.backend, args.width, args.height)
    records = []
    grid_counts = np.zeros((3, 3), dtype=int)
    actual_backend = None
    window = f'3D capture | {args.source} camera {args.camera}'
    try:
        actual_backend = cap.getBackendName()
        print(f'Using camera {args.camera}, backend={actual_backend}')
        args.out.mkdir(parents=True, exist_ok=True)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        if [frame.shape[1], frame.shape[0]] != [args.width, args.height]:
            raise ValueError(f'Camera returned {frame.shape[1]}x{frame.shape[0]}; requested {args.width}x{args.height}')
        print('SPACE/S = save current frame; Q/ESC = quit. Capture ends automatically at --count.')
        while True:
            if [frame.shape[1], frame.shape[0]] != [args.width, args.height]:
                raise ValueError('Camera resolution changed during capture')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = m.cl.find_corners(gray, pattern)
            good, angle, status, pose = False, None, 'searching for full chessboard', None
            if found:
                good, status = True, 'ready: vary board pose for calibration' if args.calibration else 'ready'
                if calib is not None:
                    try:
                        pose = m.pose_from_corners(corners, pattern, calib['pattern']['square_mm'], calib['camera_matrix'], calib['dist_coeffs'])
                        good, angle, status = pose_quality(pose, reference, args.max_rms, args.max_angle)
                    except (ValueError, cv2.error) as exc:
                        good, status = False, str(exc)
            preview = frame.copy()
            if args.calibration:
                preview = draw_calibration_grid(frame, grid_counts, corners if found else None)
            if found:
                cv2.drawChessboardCorners(preview, pattern, corners, True)
            color = (0, 210, 0) if good else (0, 0, 240)
            cv2.putText(preview, f'{len(records)}/{args.count} | {status}', (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
            if pose is not None:
                label = f'RMS: {pose["reprojection_rms_px"]:.2f} px'
                if angle is not None:
                    label += f' | relative board angle: {angle:.2f} deg'
                cv2.putText(preview, label, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xff
            if key in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (32, ord('s')):
                if good:
                    path = args.out / f'frame_{len(records)+1:02d}.png'
                    save_frame(path, frame)
                    record = {'image': path.name, 'relative_board_angle_deg': angle,
                              'reprojection_rms_px': pose['reprojection_rms_px'] if pose else None}
                    if args.calibration:
                        row, col = grid_cell(corners, frame.shape)
                        grid_counts[row, col] += 1
                        record['grid_cell_row_col'] = [row, col]
                    records.append(record)
                    print(f'Saved {path} ({len(records)}/{args.count})')
                    if len(records) >= args.count:
                        break
                else:
                    print(f'Frame not saved: {status}')
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera stopped providing frames')
    finally:
        cap.release()
        cv2.destroyAllWindows()
        metadata = {'source': args.source, 'camera': args.camera, 'backend': actual_backend,
                    'requested_backend': args.backend,
                    'image_size': [args.width, args.height], 'mode': 'calibration' if args.calibration else 'measurement',
                    'calibration': str(args.calib.resolve()) if calib else None,
                    'reference': str(args.reference.resolve()) if args.reference else None,
                    'frames': records}
        if args.calibration:
            metadata['grid_counts'] = grid_counts.tolist()
        if args.out.is_dir():
            m.write_json(args.out/'capture.json', metadata)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, cv2.error) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
