"""用真实或合成激光图像做三角测量；为 webcam 与实测激光平面预留接口。

常用命令（在 camera_calib/laser_triangulation 下执行）：

1) 处理一张图（先跑仿真生成的图验证整条链路）：
   python laser_measure.py image --image out/laser_clean.png \
       --calib out/synthetic_camera.json --plane out/synthetic_plane.json --plot

2) 接真实相机（相机内参来自 camera_calib/out/calib.json）：
   python laser_measure.py live --camera 0 --calib ../out/calib.json \
       --plane my_laser_plane.json

3) 由相机坐标系中的激光平面点云拟合平面（真实标定用）：
   python laser_measure.py fit-plane --points plane_points.npz --out my_laser_plane.json

4) 由“平面上一点 + 法向”手写平面：
   python laser_measure.py make-plane --point 0 20 500 --normal 0 1 0.2 \
       --out my_laser_plane.json

真实测量前必须确认：
* 相机内参对应实际分辨率/焦距，标定后没有改过；
* 激光平面 n、d 与相机内参在**同一个相机坐标系**，长度单位都是 mm；
* 真实图像有畸变时，本程序会自动用 OpenCV undistortPointsIter 去畸变。
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import laser_lib as ll

HERE = Path(__file__).resolve().parent
DEFAULT_CALIB = HERE.parent / "out" / "calib.json"
DEFAULT_PLANE = HERE / "laser_plane.example.json"


# ==========================================================================
# 输入/输出
# ==========================================================================
def _load_camera(path: Path) -> ll.CameraModel:
    return ll.CameraModel.from_calib_json(path)


def _load_plane(path: Path) -> ll.LaserPlane:
    plane = ll.LaserPlane.from_json(path)
    if plane.source and "SYNTHETIC" in str(plane.source).upper():
        print("警告: 当前激光平面来自仿真/示例，不是实测标定值；"
              "真实测量请用 fit-plane 或 make-plane 生成。", file=sys.stderr)
    return plane


def _check_image_size(gray: np.ndarray, camera: ll.CameraModel, path: Path) -> None:
    if camera.image_size is None:
        return
    h, w = gray.shape
    if (w, h) != tuple(camera.image_size):
        raise ValueError(
            f"{path.name}: 图像分辨率 {w}x{h} 与标定的 "
            f"{camera.image_size[0]}x{camera.image_size[1]} 不一致。"
            "真实测量必须保持标定时的分辨率、焦距和对焦状态。"
        )


def _load_points(path: Path) -> np.ndarray:
    """读 N×3 点云：支持 .npz / .npy / .csv。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到点云文件: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            for key in ("points_mm", "points", "xyz", "arr_0"):
                if key in data:
                    pts = data[key]
                    break
            else:
                arrays = [data[k] for k in data.files]
                if len(arrays) != 1:
                    raise ValueError(f"{path.name} 里找不到 points_mm/points/xyz")
                pts = arrays[0]
    elif suffix == ".npy":
        pts = np.load(path, allow_pickle=False)
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            lower = {name.lower(): name for name in fields}
            if all(k in lower for k in ("x", "y", "z")):
                rows = [[float(row[lower["x"]]), float(row[lower["y"]]),
                         float(row[lower["z"]])] for row in reader]
                pts = np.asarray(rows, dtype=np.float64)
            else:
                pts = np.loadtxt(path, delimiter=",", dtype=np.float64)
    else:
        raise ValueError(f"不支持的点云格式: {path.suffix}（用 .npz/.npy/.csv）")

    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"点云必须是 N×3，当前 shape={pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError("点云含有 NaN/Inf")
    if len(pts) < 3:
        raise ValueError("至少需要 3 个点才能拟合平面")
    return pts


def _get_plt(show: bool):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.28})
    return plt


# ==========================================================================
# image 子命令
# ==========================================================================
def cmd_image(args) -> None:
    camera = _load_camera(args.calib)
    plane = _load_plane(args.plane)
    gray = ll.load_gray_image(args.image)
    _check_image_size(gray, camera, args.image)

    ext = ll.extract_laser_centers(
        gray, axis=args.axis, roi=args.roi, smooth_sigma=args.smooth,
        threshold_sigma=args.threshold_sigma, min_peak=args.min_peak,
        min_peak_fraction=args.min_peak_fraction,
        centroid_radius=args.centroid_radius, max_width=args.max_width,
    )
    rec = ll.reconstruct(ext.uv, camera, plane)
    est = ll.robust_step_height(rec.points_mm[rec.valid],
                                min_separation_mm=args.min_step)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": "LASER_IMAGE",
        "unit": "mm",
        "image": str(args.image.resolve()),
        "calibration": str(args.calib.resolve()),
        "laser_plane": plane.to_dict(),
        "parameters": {
            "axis": args.axis, "roi_xy": args.roi, "smooth_sigma": args.smooth,
            "threshold_sigma": args.threshold_sigma, "min_peak": args.min_peak,
            "min_peak_fraction": args.min_peak_fraction,
            "centroid_radius": args.centroid_radius, "max_width": args.max_width,
            "min_step_mm": args.min_step,
        },
        "extraction": ext.summary(),
        "reconstruction": {
            "valid_count": int(rec.valid.sum()),
            "invalid_count": int((~rec.valid).sum()),
        },
        "height": est.summary(),
    }
    ll.write_json(out / "measure.json", ll.clean_json(result))

    if len(ext.uv) and rec.valid.any():
        ext.to_csv(out / "extraction.csv")
        rows = []
        for (u, v), point in zip(ext.uv, rec.points_mm[rec.valid]):
            rows.append({"u": float(u), "v": float(v),
                         "X_mm": float(point[0]), "Y_mm": float(point[1]),
                         "Z_mm": float(point[2])})
        with (out / "points.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (out / "points.npz").open("wb") as f:
            np.savez(f, points_mm=rec.points_mm, uv=ext.uv,
                     valid=rec.valid, plane_abcd=plane.abcd,
                     source="LASER_IMAGE")

    if args.plot:
        try:
            plt = _get_plt(args.show)
            _plot_measure(plt, gray, ext, est, args.image.name,
                          out / "measure.png", args.show)
        except ImportError as exc:
            print(f"警告: 画图跳过（{exc}）", file=sys.stderr)

    print("=" * 64)
    print(f"图像       : {args.image.resolve()}")
    print(f"分辨率     : {gray.shape[1]}x{gray.shape[0]}")
    print(f"相机       : fx={camera.fx:.3f}, fy={camera.fy:.3f}, "
          f"cx={camera.cx:.3f}, cy={camera.cy:.3f}, "
          f"{'有畸变' if camera.has_distortion else '无畸变'}")
    print(f"激光平面   : n={np.round(plane.normal, 6).tolist()}, d={plane.d:.4f} "
          f"({plane.source or '-'})")
    print(f"提取       : axis={ext.axis}, {len(ext.uv)}/{len(ext.scan)} 有效 "
          f"({ext.valid_fraction * 100:.1f}%), "
          f"SNR 中位数={np.nanmedian(ext.snr[ext.valid]) if ext.valid.any() else float('nan'):.2f}")
    print(f"重建       : {int(rec.valid.sum())}/{len(rec)} 有效点")
    if est.has_step:
        print(f"台阶高度   : {est.height_mm:.4f} mm "
              f"(远面 {est.level_hi_mm:.3f}, 近面 {est.level_lo_mm:.3f}, "
              f"标准误 {est.height_se_mm:.4f})")
        print(f"参与估计   : 内点 {est.n_inliers}, 剔除 {est.n_outliers}")
    else:
        print("台阶高度   : 未检出明显台阶（可减小 --min-step 或检查图像/平面）")
    print(f"结果       : {out.resolve()}")
    print("=" * 64)


def _plot_measure(plt, gray, ext, est, title, path, show):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    ax = axes[0]
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255, aspect="auto")
    if len(ext.uv):
        ax.scatter(ext.uv[:, 0], ext.uv[:, 1], s=2, color="red", label="extracted")
        # 自动把视野收到激光线附近；整幅画像大部分是背景，收窄后台阶更清楚。
        x0 = max(ext.roi[0], int(np.floor(ext.uv[:, 0].min())) - 20)
        x1 = min(ext.roi[1], int(np.ceil(ext.uv[:, 0].max())) + 20)
        y0 = max(ext.roi[2], int(np.floor(ext.uv[:, 1].min())) - 20)
        y1 = min(ext.roi[3], int(np.ceil(ext.uv[:, 1].max())) + 20)
    else:
        x0, x1, y0, y1 = ext.roi
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xlabel("u (pixel)")
    ax.set_ylabel("v (pixel)")
    ax.set_title(f"{title}: {len(ext.uv)} centers")
    ax.legend(loc="upper right", fontsize=7, markerscale=2)

    ax = axes[1]
    ax.plot(ext.scan, ext.center, ".", ms=3, color="red")
    ax.invert_yaxis()
    ax.set_xlabel("scan coordinate (pixel)")
    ax.set_ylabel("center coordinate (pixel)")
    text = (f"H = {est.height_mm:.3f} mm" if est.has_step
            else "no step detected")
    ax.set_title(text)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ==========================================================================
# live 子命令
# ==========================================================================
def cmd_live(args) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("live 模式需要 OpenCV：pip install opencv-python") from exc

    camera = _load_camera(args.calib)
    plane = _load_plane(args.plane)
    out_dir = Path(args.out_dir) / datetime.now().strftime("live_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else 0
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        raise RuntimeError(
            f"打不开相机 {args.camera}。检查是否被微信/腾讯会议/浏览器占用，"
            "或换 --camera 1。"
        )
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    frames = 0
    saved = 0
    last = None
    size_warned = False
    print("live 模式：q 退出，s 保存当前帧和结果。")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("读不到相机画面，退出。", file=sys.stderr)
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if camera.image_size is not None and not size_warned:
                h, w = gray.shape
                if (w, h) != tuple(camera.image_size):
                    print(f"警告: 画面 {w}x{h} 与标定 {camera.image_size[0]}x"
                          f"{camera.image_size[1]} 不一致，内参可能失效。",
                          file=sys.stderr)
                    size_warned = True

            ext = ll.extract_laser_centers(
                gray, axis=args.axis, roi=args.roi, smooth_sigma=args.smooth,
                threshold_sigma=args.threshold_sigma, min_peak=args.min_peak,
                min_peak_fraction=args.min_peak_fraction,
                centroid_radius=args.centroid_radius, max_width=args.max_width,
            )
            rec = ll.reconstruct(ext.uv, camera, plane)
            est = ll.robust_step_height(rec.points_mm[rec.valid],
                                        min_separation_mm=args.min_step)
            last = {"extraction": ext.summary(), "height": est.summary()}

            vis = frame.copy()
            for u, v in ext.uv:
                cv2.circle(vis, (int(round(u)), int(round(v))), 1, (0, 0, 255), -1)
            if est.has_step:
                text = f"H = {est.height_mm:.2f} mm   N = {len(ext.uv)}"
            else:
                text = f"H = --   N = {len(ext.uv)}"
            cv2.putText(vis, text, (18, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("laser triangulation", vis)
            frames += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                saved += 1
                cv2.imwrite(str(out_dir / f"frame_{saved:03d}.png"), frame)
                ll.write_json(out_dir / f"frame_{saved:03d}.json",
                              ll.clean_json({"frame": saved, **last}))
                print(f"已保存 {out_dir / f'frame_{saved:03d}.png'}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    ll.write_json(out_dir / "live_summary.json", ll.clean_json({
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": "WEBCAM_LIVE",
        "camera": str(args.calib.resolve()),
        "laser_plane": plane.to_dict(),
        "frames_processed": frames,
        "frames_saved": saved,
        "last_result": last,
    }))
    print(f"共处理 {frames} 帧，保存 {saved} 帧；结果目录 {out_dir.resolve()}")


# ==========================================================================
# fit-plane / make-plane 子命令
# ==========================================================================
def cmd_fit_plane(args) -> None:
    pts = _load_points(args.points)
    plane = ll.fit_plane_svd(pts)
    plane.source = args.source
    residual = plane.distance(pts)
    data = plane.to_dict()
    data["fit"] = {
        "n_points": int(len(pts)),
        "rms_mm": float(np.sqrt(np.mean(residual ** 2))),
        "max_abs_mm": float(np.max(np.abs(residual))),
        "points": str(args.points.resolve()),
    }
    ll.write_json(args.out, data)
    print("=" * 64)
    print(f"已拟合激光平面: n={np.round(plane.normal, 8).tolist()}, d={plane.d:.6f}")
    print(f"平面方程: {plane.unit_normal[0]:+.6f} X "
          f"{plane.unit_normal[1]:+.6f} Y {plane.unit_normal[2]:+.6f} Z "
          f"{plane.unit_d:+.6f} = 0")
    print(f"拟合残差: RMS={data['fit']['rms_mm']:.6f} mm, "
          f"max={data['fit']['max_abs_mm']:.6f} mm, 点数={len(pts)}")
    print(f"已写入: {Path(args.out).resolve()}")
    print("=" * 64)


def cmd_make_plane(args) -> None:
    plane = ll.LaserPlane.from_point_normal(args.point, args.normal,
                                            source=args.source)
    ll.write_json(args.out, plane.to_dict())
    print(f"已写入激光平面: {Path(args.out).resolve()}")
    print(f"n={plane.normal.tolist()}, d={plane.d:.6f}")


# ==========================================================================
# 命令行
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="激光三角测量：真实图像 / webcam / 平面配置")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("image", help="处理一张激光图像")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB,
                   help="camera_calib 的 calib.json（默认 ../out/calib.json）")
    p.add_argument("--plane", type=Path, default=DEFAULT_PLANE,
                   help="激光平面 JSON（默认示例，真实测量必须替换）")
    p.add_argument("--roi", type=int, nargs=4, metavar=("X0", "X1", "Y0", "Y1"),
                   help="只处理这个矩形区域；不填则全图")
    p.add_argument("--axis", choices=["auto", "column", "row"], default="auto")
    p.add_argument("--smooth", type=float, default=1.0, help="沿扫描方向的高斯平滑 sigma")
    p.add_argument("--threshold-sigma", type=float, default=4.0)
    p.add_argument("--min-peak", type=float, default=8.0)
    p.add_argument("--min-peak-fraction", type=float, default=0.10,
                   help="相对全图条纹峰值的下限，默认 10%%，用于去掉背景假峰")
    p.add_argument("--centroid-radius", type=int, default=4)
    p.add_argument("--max-width", type=float, default=9.0)
    p.add_argument("--min-step", type=float, default=2.0, help="判定台阶存在的最小间距 mm")
    p.add_argument("--out", type=Path, default=Path("out/measure"))
    p.add_argument("--plot", action="store_true")
    p.add_argument("--show", action="store_true")
    p.set_defaults(func=cmd_image)

    p = sub.add_parser("live", help="webcam 实时预览 + 高度估计")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    p.add_argument("--plane", type=Path, default=DEFAULT_PLANE)
    p.add_argument("--roi", type=int, nargs=4, metavar=("X0", "X1", "Y0", "Y1"))
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--axis", choices=["auto", "column", "row"], default="auto")
    p.add_argument("--smooth", type=float, default=1.0)
    p.add_argument("--threshold-sigma", type=float, default=4.0)
    p.add_argument("--min-peak", type=float, default=8.0)
    p.add_argument("--min-peak-fraction", type=float, default=0.10,
                   help="相对全图条纹峰值的下限，默认 10%%，用于去掉背景假峰")
    p.add_argument("--centroid-radius", type=int, default=4)
    p.add_argument("--max-width", type=float, default=9.0)
    p.add_argument("--min-step", type=float, default=2.0)
    p.add_argument("--out-dir", type=Path, default=Path("out/live"))
    p.set_defaults(func=cmd_live)

    p = sub.add_parser("fit-plane", help="由相机坐标系点云拟合激光平面")
    p.add_argument("--points", type=Path, required=True, help=".npz/.npy/.csv，N×3 mm")
    p.add_argument("--out", type=Path, default=Path("out/laser_plane.json"))
    p.add_argument("--source", default="FIT_FROM_POINTS")
    p.set_defaults(func=cmd_fit_plane)

    p = sub.add_parser("make-plane", help="由一点+法向生成激光平面 JSON")
    p.add_argument("--point", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    p.add_argument("--normal", type=float, nargs=3, required=True, metavar=("A", "B", "C"))
    p.add_argument("--out", type=Path, default=Path("out/laser_plane.json"))
    p.add_argument("--source", default="MANUAL")
    p.set_defaults(func=cmd_make_plane)
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(2)
