"""Independent 3D experiments; all lengths are mm, points in camera frame."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

# Load the existing helper without writing __pycache__ into the original project.
ROOT = Path(os.environ.get("CAMERA_CALIB_ROOT", Path(__file__).resolve().parent.parent))
helper = ROOT / "calib_lib.py"
if not helper.is_file():
    raise ImportError("Place 02_measurement_3d inside 01_camera_calib, or set CAMERA_CALIB_ROOT")
cl = importlib.util.module_from_spec(importlib.util.spec_from_file_location("camera_calib_existing", helper))
exec(compile(helper.read_bytes(), str(helper), "exec"), cl.__dict__)


def intersect_pixels(pixels, K, dist, plane, eps=1e-10):
    """Raw distorted pixels -> (Nx3 camera points in mm, N valid mask).

    plane = [a,b,c,d], aX+bY+cZ+d=0 in CAMERA coordinates.
    Invalid/parallel/behind-camera intersections are NaN, mask=False.
    """
    uv = np.asarray(pixels, dtype=float)
    K = np.asarray(K, dtype=float)
    dist = np.asarray(dist, dtype=float)
    p = np.asarray(plane, dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
        raise ValueError("pixels must be finite Nx2 raw pixel coordinates")
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid camera matrix")
    if not np.isfinite(dist).all() or p.shape != (4,) or not np.isfinite(p).all():
        raise ValueError("Invalid distortion coefficients or plane")
    norm = np.linalg.norm(p[:3])
    if norm == 0 or not np.isfinite(norm) or not 0 < eps < 1:
        raise ValueError("Plane normal must be nonzero; eps must lie in (0,1)")
    p = p / norm  # scale-invariant parallel threshold
    if len(uv) == 0:
        return np.empty((0, 3)), np.empty(0, dtype=bool)
    xy = cv2.undistortPointsIter(
        uv.reshape(-1, 1, 2), K, dist, None, None,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12),
    ).reshape(-1, 2)
    rays = np.column_stack([xy, np.ones(len(xy))])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    denom = rays @ p[:3]
    valid = np.isfinite(rays).all(axis=1) & (np.abs(denom) > eps)
    scale = np.full(len(uv), np.nan)
    np.divide(-p[3], denom, out=scale, where=valid)
    valid &= np.isfinite(scale) & (scale > 0)
    xyz = rays * scale[:, None]
    xyz[~valid] = np.nan
    return xyz, valid


def pose_from_corners(corners, pattern, square_mm, K, dist):
    """Reuse existing solvePnP interface: X_camera = R X_board + t."""
    rvec, tvec, rms = cl.solve_pose(corners, pattern, square_mm, K, dist)
    if rvec is None or not np.isfinite(tvec).all() or not np.isfinite(rvec).all():
        raise ValueError("solvePnP failed")
    R = cv2.Rodrigues(rvec)[0]
    xyz = cl.object_points(pattern, square_mm).astype(float) @ R.T + tvec.reshape(3)
    if np.any(xyz[:, 2] <= 0) or not np.isfinite(rms):
        raise ValueError("Invalid pose: board behind camera or nonfinite RMS")
    return {"rvec": rvec.ravel().tolist(), "tvec_mm": tvec.ravel().tolist(),
            "rotation_matrix": R.tolist(), "normal_camera": R[:, 2].tolist(),
            "reprojection_rms_px": rms, "points_camera_mm": xyz.tolist()}


def pose_image(path, calib, max_rms=3.0):
    """Use original, unscaled images and original K/dist; reject wrong resolution."""
    path = Path(path)
    gray = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Cannot read image: {path}")
    if list(gray.shape[::-1]) != list(calib["image_size"]):
        raise ValueError(f"{path}: image size must be {calib['image_size']}")
    pat = calib["pattern"]
    pattern = (pat["inner_cols"], pat["inner_rows"])
    ok, corners = cl.find_corners(gray, pattern)
    if not ok:
        raise ValueError(f"{path}: chessboard not detected")
    result = pose_from_corners(corners, pattern, pat["square_mm"],
                               calib["camera_matrix"], calib["dist_coeffs"])
    if result["reprojection_rms_px"] > max_rms:
        raise ValueError(f"{path}: RMS {result['reprojection_rms_px']:.3f} > {max_rms} px")
    result["image"] = str(path.resolve())
    return result


def height_statistics(reference, samples, known_mm, max_angle_deg=2.0):
    """Unsigned plane separation along reference normal; sample std (ddof=1)."""
    if len(samples) < 2:
        raise ValueError("At least two repeated target images are required for sample std")
    if not np.isfinite(known_mm) or known_mm < 0 or not 0 < max_angle_deg < 90:
        raise ValueError("known_mm >= 0; max_angle_deg must lie in (0,90)")
    n = cv2.Rodrigues(np.asarray(reference["rvec"], dtype=float))[0][:, 2]
    # Stabilize sign when chessboard ordering flips by 180 degrees.
    if n[2] < 0:
        n = -n
    t0 = np.asarray(reference["tvec_mm"], dtype=float)
    rows = []
    for sample in samples:
        ni = cv2.Rodrigues(np.asarray(sample["rvec"], dtype=float))[0][:, 2]
        angle = float(np.degrees(np.arccos(np.clip(abs(n @ ni), 0, 1))))
        if angle > max_angle_deg:
            raise ValueError(f"Nonparallel board: {angle:.3f} > {max_angle_deg} deg")
        signed = float(n @ (np.asarray(sample["tvec_mm"]) - t0))
        if not np.isfinite(signed):
            raise ValueError("Nonfinite height")
        height = abs(signed)
        rows.append({"image": sample.get("image", "synthetic"), "height_mm": height,
                     "signed_displacement_mm": signed, "parallel_angle_deg": angle,
                     "absolute_error_mm": abs(height-known_mm),
                     "reprojection_rms_px": sample.get("reprojection_rms_px")})
    h = np.array([r["height_mm"] for r in rows])
    return {"unit": "mm", "count": len(h), "known_mm": float(known_mm),
            "mean_mm": float(h.mean()), "absolute_error_mm": float(abs(h.mean()-known_mm)),
            "mean_absolute_error_mm": float(np.mean(abs(h-known_mm))),
            "std_mm": float(h.std(ddof=1)), "std_ddof": 1,
            "reference_normal_camera": n.tolist(), "samples": rows}


def make_point_cloud(points):
    """Open3D is optional for all numerical commands."""
    import open3d as o3d
    xyz = np.asarray(points, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Point cloud must contain finite Nx3 points")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.paint_uniform_color([.1, .7, .95])
    return cloud


def view_points(points, title):
    import open3d as o3d
    cloud = make_point_cloud(points)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
    o3d.visualization.draw_geometries([cloud, axis], window_name=title,
                                    width=1000, height=720)


def synthetic_check(calib, out):
    """Analytic 3D truth -> distorted pixels -> ray-plane reconstruction."""
    K, dist = calib["camera_matrix"], calib["dist_coeffs"]
    x, y = np.meshgrid(np.linspace(-90, 90, 21), np.linspace(-60, 60, 15))
    xyz = np.column_stack([x.ravel(), y.ravel(), (550-.2*x+.1*y).ravel()])
    plane = [.2, -.1, 1., -550.]
    uv = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
    recovered, valid = intersect_pixels(uv, K, dist, plane)
    error = np.linalg.norm(recovered-xyz, axis=1)
    max_error = float(np.max(error))
    if not valid.all() or not max_error < 1e-5:
        raise AssertionError(f"Synthetic intersection check failed: max error {max_error}")
    pat = calib["pattern"]
    pattern = pat["inner_cols"], pat["inner_rows"]
    obj = cl.object_points(pattern, pat["square_mm"]).astype(float)
    rv, t0 = np.array([.2, -.15, .05]), np.array([-60., -40., 650.])
    normal = cv2.Rodrigues(rv)[0][:, 2]
    poses = []
    for h in (0., 19.8, 20., 20.2):
        pixels = cv2.projectPoints(obj, rv, t0+h*normal, K, dist)[0]
        poses.append(pose_from_corners(pixels, pattern, pat["square_mm"], K, dist))
    heights = height_statistics(poses[0], poses[1:], 20.)
    if abs(heights["mean_mm"]-20) > 1e-3 or abs(heights["std_mm"]-.2) > 1e-3:
        raise AssertionError("Synthetic pose/height check failed")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "synthetic_points.npz", points_mm=recovered, truth_mm=xyz,
             pixels=uv, valid=valid, plane_camera=plane, source="SYNTHETIC_MATH_VALIDATION")
    report = {"source": "SYNTHETIC_MATH_VALIDATION", "unit": "mm", "count": len(xyz),
              "max_3d_error_mm": max_error, "tolerance_mm": 1e-5,
              "max_plane_residual_mm": float(np.max(abs(recovered @ np.array(plane[:3])+plane[3])) / np.linalg.norm(plane[:3])),
              "height_check": heights}
    write_json(out / "synthetic_report.json", report)
    return report


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib", type=Path, default=ROOT / "out/calib.json")
    parser.add_argument("--max-rms", type=float, default=3., help="PnP RMS limit in pixels")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pose", help="Chessboard pose from one raw image")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results/pose.json")
    p.add_argument("--view", action="store_true")
    p = sub.add_parser("height", help="Fixed camera, parallel board repeated observations")
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--targets", required=True, type=Path, help="Folder of repeated images at ONE height")
    p.add_argument("--known-mm", required=True, type=float)
    p.add_argument("--max-angle", type=float, default=2.)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results/height.json")
    p = sub.add_parser("intersect", help="Raw pixels CSV u,v and known camera-frame plane")
    p.add_argument("--pixels", required=True, type=Path)
    p.add_argument("--plane", required=True, type=float, nargs=4, metavar=("A", "B", "C", "D"))
    p.add_argument("--source", required=True, help="Describe actual input provenance, e.g. SYNTHETIC")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results/intersections.npz")
    p = sub.add_parser("synthetic", help="Synthetic mathematical validation; no laser measurements")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results/synthetic")
    p.add_argument("--view", action="store_true")
    p = sub.add_parser("view", help="View saved camera-frame points with Open3D")
    p.add_argument("--points", type=Path, required=True)
    p.add_argument("--export-ply", type=Path)
    p.add_argument("--no-window", action="store_true", help="Validate/export point cloud without GUI")
    args = parser.parse_args()
    if args.command == "view":
        with np.load(args.points, allow_pickle=False) as data:
            source = str(data["source"].item())
            points = data["points_mm"]
            points = points[np.isfinite(points).all(axis=1)]
        cloud = make_point_cloud(points)
        print(f"{source}: {len(points)} points in camera frame (mm)")
        if args.export_ply:
            import open3d as o3d
            args.export_ply.parent.mkdir(parents=True, exist_ok=True)
            if not o3d.io.write_point_cloud(str(args.export_ply), cloud):
                raise RuntimeError("Could not write PLY")
        if not args.no_window:
            view_points(points, source + " | camera frame (mm)")
        return
    calib = cl.load_calib(args.calib)
    if not np.isfinite(args.max_rms) or args.max_rms <= 0:
        raise ValueError("max-rms must be positive")
    if args.command == "pose":
        result = pose_image(args.image, calib, args.max_rms)
        result.update(source="CHESSBOARD_IMAGE_PNP", calibration=str(args.calib.resolve()), unit="mm")
        write_json(args.out, result)
        if args.view:
            view_points(result["points_camera_mm"], "Chessboard PnP | camera frame (mm)")
    elif args.command == "height":
        ref = pose_image(args.reference, calib, args.max_rms)
        paths = cl.list_images(args.targets)
        if args.reference.resolve() in [p.resolve() for p in paths]:
            raise ValueError("Reference image must be outside targets folder")
        samples = [pose_image(p, calib, args.max_rms) for p in paths]
        result = height_statistics(ref, samples, args.known_mm, args.max_angle)
        result.update(source="CHESSBOARD_IMAGE_HEIGHT", reference=ref,
                      calibration=str(args.calib.resolve()), max_angle_deg=args.max_angle)
        write_json(args.out, result)
        with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(result["samples"][0]))
            writer.writeheader()
            writer.writerows(result["samples"])
    elif args.command == "intersect":
        with args.pixels.open(encoding="utf-8-sig", newline="") as f:
            pixels = np.array([[float(row["u"]), float(row["v"])] for row in csv.DictReader(f)]).reshape(-1, 2)
        xyz, valid = intersect_pixels(pixels, calib["camera_matrix"], calib["dist_coeffs"], args.plane)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("wb") as f:
            np.savez(f, points_mm=xyz, valid=valid, pixels=pixels,
                     plane_camera=args.plane, source=args.source)
        result = {"source": args.source, "valid_count": int(valid.sum()),
                  "invalid_count": int((~valid).sum()), "points_file": str(args.out)}
    else:
        result = synthetic_check(calib, args.out)
        if args.view:
            with np.load(args.out / "synthetic_points.npz") as data:
                view_points(data["points_mm"], "SYNTHETIC mathematical validation | mm")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, KeyError, ImportError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
