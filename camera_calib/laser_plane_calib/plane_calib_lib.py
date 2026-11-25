"""激光平面标定仿真核心库。

这个模块是**独立新增**的，不修改 `laser_lib.py` / `simulate_step.py` 的基础算法。
它只做三件事：

1. 用已知的模拟相机、棋盘格位姿和真实激光平面，渲染“棋盘格图”和“激光条纹图”；
2. 从**测量到的**角点和激光中心出发，做 PnP、相机射线与棋盘格平面求交、SVD 拟合；
3. 用未参与标定的台阶做独立验证。

数据隔离约定
------------
* ``build_dataset()`` 知道真实平面，因为它要渲染图像、生成真值线；
* ``calibrate_plane()`` / ``prepare_measurement()`` / ``fit_plane_points()``
  只接收 **measured corners + measured laser pixels + camera K/dist + board spec**，
  函数签名里没有 true_plane；
* 真值只在 ``evaluate_plane()``、``evaluate_validation()`` 和画图时使用。

棋盘格
------
默认 9×6 个方格、8×5 个内部角点、**20 mm 标称方格**、12 mm 白边。
实物标定板打印后实测 19 mm（打印缩放），真实实验应把 ``--square`` 改成实测值；
本仿真按 20 mm 标称值生成，避免把一次打印缩放误当成算法误差。

激光平面
--------
默认真实平面 ``X + 0.2 Z - 100 = 0``。法向主要在 X 方向，所以激光条纹在图像里
近似竖直。这个方向是有意选的：用来检验现有 ``extract_laser_centers`` 的按列模式
是否适合，以及按行模式是否才正确。
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

# 复用已有工程，但不修改它们。
ROOT = Path(__file__).resolve().parent.parent  # camera_calib
for _path in (str(ROOT), str(ROOT / "laser_triangulation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import calib_lib as cl  # noqa: E402
import laser_lib as ll  # noqa: E402


# ==========================================================================
# 棋盘格规格
# ==========================================================================
@dataclass(frozen=True)
class BoardSpec:
    """9×6 方格棋盘格；坐标原点在第一个内部角点，单位 mm。"""

    cols: int = 9
    rows: int = 6
    square_mm: float = 20.0
    margin_mm: float = 12.0
    ppm: float = 12.0

    @property
    def inner(self) -> tuple[int, int]:
        return self.cols - 1, self.rows - 1

    @property
    def object_points(self) -> np.ndarray:
        return cl.object_points(self.inner, self.square_mm).astype(np.float64)

    @property
    def center_obj(self) -> np.ndarray:
        return self.object_points.mean(axis=0)

    @property
    def x_min(self) -> float:
        return -(self.margin_mm + self.square_mm)

    @property
    def x_max(self) -> float:
        return (self.cols - 1) * self.square_mm + self.margin_mm

    @property
    def y_min(self) -> float:
        return -(self.margin_mm + self.square_mm)

    @property
    def y_max(self) -> float:
        return (self.rows - 1) * self.square_mm + self.margin_mm

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.x_min, self.x_max, self.y_min, self.y_max

    @property
    def physical_corners(self) -> np.ndarray:
        return np.array([
            [self.x_min, self.y_min, 0.0],
            [self.x_max, self.y_min, 0.0],
            [self.x_max, self.y_max, 0.0],
            [self.x_min, self.y_max, 0.0],
        ], dtype=np.float64)

    @property
    def physical_width_mm(self) -> float:
        return self.x_max - self.x_min

    @property
    def physical_height_mm(self) -> float:
        return self.y_max - self.y_min

    def to_dict(self) -> dict:
        return {
            "cols": self.cols,
            "rows": self.rows,
            "inner_cols": self.inner[0],
            "inner_rows": self.inner[1],
            "square_mm": self.square_mm,
            "margin_mm": self.margin_mm,
            "physical_width_mm": self.physical_width_mm,
            "physical_height_mm": self.physical_height_mm,
            "object_origin": "第一个内部角点",
            "note": f"本次方格 {self.square_mm:g} mm；实物打印缩放后若实测 19mm，"
                    "真实标定请传 --square 19（或实测值）",
        }


DEFAULT_BOARD = BoardSpec(square_mm=20.0)
DEFAULT_TRUE_PLANE_ABCD = (1.0, 0.0, 0.2, -100.0)
DEFAULT_IMAGE_SIZE = (1280, 720)


def make_board_texture(board: BoardSpec = DEFAULT_BOARD) -> np.ndarray:
    """生成棋盘格纹理；纹理坐标与物点坐标的对应由 render_board_image() 负责。"""
    w_mm = board.cols * board.square_mm + 2 * board.margin_mm
    h_mm = board.rows * board.square_mm + 2 * board.margin_mm
    width = int(round(w_mm * board.ppm))
    height = int(round(h_mm * board.ppm))
    tex = np.full((height, width), 255, dtype=np.uint8)
    for j in range(board.rows):
        for i in range(board.cols):
            if (i + j) % 2 == 1:
                x0 = int(round((board.margin_mm + i * board.square_mm) * board.ppm))
                y0 = int(round((board.margin_mm + j * board.square_mm) * board.ppm))
                x1 = int(round((board.margin_mm + (i + 1) * board.square_mm) * board.ppm))
                y1 = int(round((board.margin_mm + (j + 1) * board.square_mm) * board.ppm))
                tex[y0:y1, x0:x1] = 0
    return tex


# ==========================================================================
# 基本几何
# ==========================================================================
def project_points(points_mm, camera: ll.CameraModel) -> np.ndarray:
    """相机坐标系中的三维点 -> 原始（含畸变）像素。"""
    pts = np.asarray(points_mm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_mm 必须是 N×3")
    return cv2.projectPoints(
        pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), camera.K, camera.dist
    )[0].reshape(-1, 2)


def project_object_points(points_obj, rvec, tvec,
                          camera: ll.CameraModel) -> np.ndarray:
    """棋盘格坐标系中的点 -> 像素（先做位姿变换，再投影）。"""
    pts = np.asarray(points_obj, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_obj 必须是 N×3")
    return cv2.projectPoints(
        pts.reshape(-1, 1, 3), np.asarray(rvec, dtype=np.float64).reshape(3),
        np.asarray(tvec, dtype=np.float64).reshape(3), camera.K, camera.dist,
    )[0].reshape(-1, 2)


def board_plane_from_pose(R: np.ndarray, t: np.ndarray) -> ll.LaserPlane:
    """棋盘格平面 z=0 在相机坐标系中的平面方程。"""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    n = R[:, 2]
    return ll.LaserPlane(n, float(-n @ t), source="BOARD_PLANE")


def angle_between_normals(n1, n2) -> float:
    a = np.asarray(n1, dtype=np.float64).ravel()
    b = np.asarray(n2, dtype=np.float64).ravel()
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), 0.0, 1.0))))


def _axis_interval(p0: float, d: float, lo: float, hi: float, eps: float = 1e-12):
    """直线 p0 + s*d 落在 [lo,hi] 内的 s 区间；无约束返回 (-inf, inf)。"""
    if abs(d) < eps:
        return (-np.inf, np.inf) if lo <= p0 <= hi else None
    a = (lo - p0) / d
    b = (hi - p0) / d
    return (min(a, b), max(a, b))


def line_on_plane(plane_a: ll.LaserPlane, plane_b: ll.LaserPlane,
                  R: np.ndarray, t: np.ndarray,
                  bounds: tuple[float, float, float, float],
                  samples: int = 240):
    """两个平面在棋盘格矩形内的交线。

    返回 (points_camera, direction, length_mm) 或 None。
    ``R,t`` 把棋盘格坐标映射到相机：X_cam = R X_board + t。
    """
    na, da = plane_a.unit_normal, plane_a.unit_d
    nb, db = plane_b.unit_normal, plane_b.unit_d
    direction = np.cross(na, nb)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-10:
        return None
    direction = direction / norm

    A = np.vstack([na, nb])
    b = np.array([-da, -db], dtype=np.float64)
    p0 = np.linalg.lstsq(A, b, rcond=None)[0]
    q0 = R.T @ (p0 - t)
    qd = R.T @ direction

    xmin, xmax, ymin, ymax = bounds
    sx = _axis_interval(float(q0[0]), float(qd[0]), xmin, xmax)
    sy = _axis_interval(float(q0[1]), float(qd[1]), ymin, ymax)
    if sx is None or sy is None:
        return None
    lo = max(sx[0], sy[0])
    hi = min(sx[1], sy[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    s = np.linspace(lo, hi, int(samples))
    points = p0[None, :] + s[:, None] * direction[None, :]
    return points, direction, float(hi - lo)


def step_curve(plane: ll.LaserPlane, z_far: float, z_near: float,
               y_split: float, y_half: float = 60.0,
               n_surface: int = 121, n_wall: int = 41):
    """两阶台阶上的真实激光曲线：远面 -> 台阶侧面 -> 近面。

    台阶面是 Z=z_far / Z=z_near 的水平面，沿 Y=y_split 分界。
    激光平面与每个水平面的交线是一条空间直线；这里按 Y 采样。
    """
    n = plane.normal
    d = plane.d
    if abs(n[0]) < 1e-12:
        raise ValueError("该激光平面没有 X 分量，不能由 Z/Y 反求 X")
    if z_far <= z_near:
        raise ValueError("要求 z_far > z_near")

    def x_at(y, z):
        return -(n[1] * y + n[2] * z + d) / n[0]

    y_low = np.linspace(y_split - y_half, y_split, n_surface, endpoint=False)
    p_low = np.column_stack([x_at(y_low, z_far), y_low, np.full_like(y_low, z_far)])

    z_wall = np.linspace(z_far, z_near, n_wall)
    p_wall = np.column_stack([x_at(y_split, z_wall), np.full_like(z_wall, y_split), z_wall])

    y_high = np.linspace(y_split, y_split + y_half, n_surface)
    p_high = np.column_stack([x_at(y_high, z_near), y_high, np.full_like(y_high, z_near)])

    points = np.vstack([p_low, p_wall, p_high])
    roles = np.r_[
        np.zeros(len(p_low), dtype=int),
        np.full(len(p_wall), 2, dtype=int),
        np.ones(len(p_high), dtype=int),
    ]
    return points, roles


# ==========================================================================
# 棋盘格图像渲染
# ==========================================================================
def render_board_image(rvec, tvec, texture: np.ndarray, board: BoardSpec,
                       camera: ll.CameraModel, rng: np.random.Generator,
                       read_sigma: float = 1.5) -> np.ndarray:
    """按真实位姿渲染棋盘格图（含镜头畸变）。

    物点坐标 X_obj 映射到纹理坐标 (X_obj + margin + square) * ppm，
    因此纹理上的第一个内部角点正好对应物点 (0,0)。
    """
    h, w = camera.image_size[1], camera.image_size[0]
    # 注意：np.mgrid 的第一个输出沿行变化，第二个沿列变化；
    # 这里要的是 gx=列坐标、gy=行坐标，顺序不能写反。
    gy, gx = np.mgrid[0:h, 0:w]
    pixels = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
    ideal = cv2.undistortPoints(pixels, camera.K, camera.dist).reshape(-1, 2)

    R = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    n = R[:, 2]
    d = float(-n @ t)

    rays = np.column_stack([ideal, np.ones(len(ideal), dtype=np.float64)])
    denom = rays @ n
    lam = np.full(len(rays), np.nan, dtype=np.float64)
    ok = np.abs(denom) > 1e-12
    np.divide(-d, denom, out=lam, where=ok)
    ok &= np.isfinite(lam) & (lam > 0)

    cam = rays * lam[:, None]
    q = (cam - t) @ R  # = R.T @ (cam - t)
    X = q[:, 0]
    Y = q[:, 1]
    offset = board.margin_mm + board.square_mm
    mapx = (X + offset) * board.ppm - 0.5
    mapy = (Y + offset) * board.ppm - 0.5
    mapx[~ok] = -1e6
    mapy[~ok] = -1e6
    mapx = mapx.reshape(h, w).astype(np.float32)
    mapy = mapy.reshape(h, w).astype(np.float32)

    img = cv2.remap(texture, mapx, mapy, cv2.INTER_AREA,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    img = img.astype(np.float32)
    ramp = np.linspace(rng.uniform(0.92, 0.98), 1.0, w, dtype=np.float32)[None, :]
    img *= ramp
    if read_sigma > 0:
        img += rng.normal(0.0, read_sigma, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0.5)


# ==========================================================================
# 位姿采样
# ==========================================================================
@dataclass(frozen=True)
class PoseRanges:
    z: tuple[float, float]
    x: tuple[float, float]
    y: tuple[float, float]
    rx: tuple[float, float]
    ry: tuple[float, float]
    rz: tuple[float, float]
    min_board_px: tuple[float, float] = (220.0, 150.0)
    max_board_px: tuple[float, float] = (700.0, 500.0)
    margin_px: float = 20.0
    min_line_mm: float = 40.0


@dataclass
class Pose:
    index: int
    rvec: np.ndarray
    tvec: np.ndarray
    R: np.ndarray
    center_cam: np.ndarray
    board_plane: ll.LaserPlane
    line_points: np.ndarray
    line_uv: np.ndarray
    line_length_mm: float
    features: dict = field(default_factory=dict)


def _rot(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def sample_poses(n: int, camera: ll.CameraModel, laser_plane: ll.LaserPlane,
                 ranges: PoseRanges, board: BoardSpec,
                 rng: np.random.Generator, max_attempts: int = 20000) -> list[Pose]:
    """按给定范围采样 n 个“整块板在画面内、激光线足够长”的位姿。"""
    poses: list[Pose] = []
    w, h = camera.image_size
    for _ in range(int(max_attempts)):
        if len(poses) >= n:
            break
        Z = float(rng.uniform(*ranges.z))
        X = float(rng.uniform(*ranges.x))
        Y = float(rng.uniform(*ranges.y))
        rx = float(rng.uniform(*ranges.rx))
        ry = float(rng.uniform(*ranges.ry))
        rz = float(rng.uniform(*ranges.rz))
        R = _rot(rx, ry, rz)
        rvec = cv2.Rodrigues(R)[0].ravel()
        t = np.array([X, Y, Z], dtype=np.float64) - R @ board.center_obj

        uv_board = project_object_points(board.physical_corners, rvec, t, camera)
        uv_corners = project_object_points(board.object_points, rvec, t, camera)
        if not (np.isfinite(uv_board).all() and np.isfinite(uv_corners).all()):
            continue
        if (uv_board[:, 0].min() < ranges.margin_px or
                uv_board[:, 0].max() > w - ranges.margin_px or
                uv_board[:, 1].min() < ranges.margin_px or
                uv_board[:, 1].max() > h - ranges.margin_px):
            continue
        wpx = float(uv_board[:, 0].max() - uv_board[:, 0].min())
        hpx = float(uv_board[:, 1].max() - uv_board[:, 1].min())
        if not (ranges.min_board_px[0] <= wpx <= ranges.max_board_px[0] and
                ranges.min_board_px[1] <= hpx <= ranges.max_board_px[1]):
            continue

        bp = board_plane_from_pose(R, t)
        seg = line_on_plane(laser_plane, bp, R, t, board.bounds)
        if seg is None:
            continue
        line_points, _, line_len = seg
        if line_len < ranges.min_line_mm:
            continue
        line_uv = project_points(line_points, camera)
        if not np.isfinite(line_uv).all():
            continue
        if (line_uv[:, 0].min() < 5 or line_uv[:, 0].max() > w - 5 or
                line_uv[:, 1].min() < 5 or line_uv[:, 1].max() > h - 5):
            continue

        poses.append(Pose(
            index=-1, rvec=rvec, tvec=t, R=R, center_cam=np.array([X, Y, Z]),
            board_plane=bp, line_points=line_points, line_uv=line_uv,
            line_length_mm=float(line_len),
            features={"z_mm": Z, "x_mm": X, "y_mm": Y,
                      "rx_deg": rx, "ry_deg": ry, "rz_deg": rz,
                      "board_width_px": wpx, "board_height_px": hpx,
                      "line_length_mm": float(line_len)},
        ))
    if len(poses) < n:
        raise RuntimeError(
            f"只采到 {len(poses)}/{n} 个可行位姿；请放宽范围或检查相机/平面参数。"
        )
    for i, pose in enumerate(poses):
        pose.index = i
    return poses


# ==========================================================================
# 数据集：图像 + 测量值 + 真值（真值单独存放）
# ==========================================================================
@dataclass
class Dataset:
    measurements: list[dict]
    truth: dict
    poses: list[Pose]
    orientation_rows: list[dict]


def save_png(path, img) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", np.asarray(img, dtype=np.uint8))
    if not ok:
        raise RuntimeError(f"PNG 编码失败: {path}")
    buf.tofile(str(path))


def _extract_mode(image: np.ndarray, axis: str,
                  roi: tuple[int, int, int, int] | None = None) -> ll.LaserExtraction:
    return ll.extract_laser_centers(
        image.astype(np.float64), axis=axis, roi=roi, smooth_sigma=1.0,
        threshold_sigma=4.0, min_peak=12.0, min_peak_fraction=0.10,
        centroid_radius=4, max_width=9.0,
    )


def build_dataset(poses: list[Pose], camera: ll.CameraModel,
                  laser_plane: ll.LaserPlane, board: BoardSpec,
                  data_dir: Path, rng: np.random.Generator,
                  sigma_corner: float = 0.2, sigma_laser: float = 0.2,
                  save_images: bool = True) -> Dataset:
    """渲染每个位姿的棋盘格图和激光条纹图，并保存测量值/真值。

    注意：``laser_plane`` 只用于渲染和生成真值线；返回的 ``measurements``
    里没有真实平面，也没有真实位姿。真值单独放在 ``truth``。
    """
    texture = make_board_texture(board)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    measurements = []
    truth_poses = []
    orientation_rows = []

    for i, pose in enumerate(poses):
        pose.index = i
        board_img = render_board_image(pose.rvec, pose.tvec, texture, board, camera, rng)
        ok, corners = cl.find_corners(board_img, board.inner)
        if not ok:
            raise RuntimeError(f"第 {i} 个位姿的棋盘格角点检测失败；请调整位姿范围。")
        corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        corners_meas = corners + rng.normal(0.0, sigma_corner, corners.shape)

        laser_img_float = ll.render_laser_image(
            pose.line_uv, camera.image_size, sigma_px=1.4, peak=220.0,
            background=6.0, gradient=(3.0, 2.0), read_sigma=1.5,
            shot_gain=0.20, rng=rng,
        )
        laser_img = ll.to_uint8(laser_img_float)
        laser_for_extract = laser_img.astype(np.float64)

        ext_row = _extract_mode(laser_for_extract, "row")
        ext_col = _extract_mode(laser_for_extract, "column")
        ext_auto = _extract_mode(laser_for_extract, "auto")

        # 全图 auto 对“细线只占不到 1% 像素”不敏感；再做一个紧 ROI 的 auto 对照。
        w, h = camera.image_size
        lu = pose.line_uv
        auto_roi = (
            max(0, int(np.floor(lu[:, 0].min())) - 20),
            min(w, int(np.ceil(lu[:, 0].max())) + 20),
            max(0, int(np.floor(lu[:, 1].min())) - 20),
            min(h, int(np.ceil(lu[:, 1].max())) + 20),
        )
        ext_auto_roi = _extract_mode(laser_for_extract, "auto", roi=auto_roi)

        corners_raw = corners.copy()
        row_uv_raw = ext_row.uv.copy()
        row_uv = row_uv_raw + rng.normal(0.0, sigma_laser, row_uv_raw.shape)
        col_uv = ext_col.uv
        auto_uv = ext_auto.uv
        auto_roi_uv = ext_auto_roi.uv

        def center_errors(uv):
            if len(uv) == 0:
                return float("nan"), float("nan")
            d = ll.distance_to_polyline(uv, pose.line_uv)
            return float(np.mean(d)), float(np.max(d))

        row_mean, row_max = center_errors(row_uv_raw)
        col_mean, col_max = center_errors(col_uv)
        auto_mean, auto_max = center_errors(auto_uv)
        auto_roi_mean, auto_roi_max = center_errors(auto_roi_uv)

        board_name = f"board_{i + 1:02d}.png"
        laser_name = f"laser_{i + 1:02d}.png"
        if save_images:
            save_png(data_dir / board_name, board_img)
            save_png(data_dir / laser_name, laser_img)

        measurements.append({
            "index": i,
            "corners": corners_meas.tolist(),
            "corners_raw": corners_raw.tolist(),
            "laser_uv_row": row_uv.tolist(),
            "laser_uv_row_raw": row_uv_raw.tolist(),
            "laser_uv_column": col_uv.tolist(),
            "laser_uv_auto": auto_uv.tolist(),
            "auto_axis": ext_auto.axis,
            "board_image": board_name if save_images else None,
            "laser_image": laser_name if save_images else None,
            "n_row": int(len(ext_row.uv)),
            "n_column": int(len(ext_col.uv)),
            "n_auto": int(len(ext_auto.uv)),
        })
        truth_poses.append({
            "index": i,
            "rvec": pose.rvec.tolist(),
            "tvec": pose.tvec.tolist(),
            "center_cam_mm": pose.center_cam.tolist(),
            "board_plane": pose.board_plane.to_dict(),
            "line_uv": pose.line_uv.tolist(),
            "line_length_mm": float(pose.line_length_mm),
            "features": pose.features,
        })
        orientation_rows.append({
            "index": i,
            "row_count": int(len(ext_row.uv)),
            "column_count": int(len(ext_col.uv)),
            "auto_count": int(len(ext_auto.uv)),
            "auto_roi_count": int(len(ext_auto_roi.uv)),
            "auto_axis": ext_auto.axis,
            "auto_roi_axis": ext_auto_roi.axis,
            "auto_roi_xy": list(auto_roi),
            "row_center_error_mean_px": row_mean,
            "row_center_error_max_px": row_max,
            "column_center_error_mean_px": col_mean,
            "column_center_error_max_px": col_max,
            "auto_center_error_mean_px": auto_mean,
            "auto_center_error_max_px": auto_max,
            "auto_roi_center_error_mean_px": auto_roi_mean,
            "auto_roi_center_error_max_px": auto_roi_max,
            "true_line_length_mm": float(pose.line_length_mm),
        })

    truth = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "unit": "mm",
        "frame": "camera: X right, Y down, Z forward",
        "laser_plane": laser_plane.to_dict(),
        "board": board.to_dict(),
        "poses": truth_poses,
    }
    return Dataset(measurements=measurements, truth=truth, poses=poses,
                   orientation_rows=orientation_rows)


def build_ideal_measurements(poses: list[Pose], camera: ll.CameraModel,
                             board: BoardSpec,
                             sigma_corner: float = 0.0,
                             sigma_laser: float = 0.0,
                             rng: np.random.Generator | None = None) -> list[dict]:
    """理想测量：直接用真实位姿投影角点和激光线，不渲染、不检测。

    这是一个“几何上限”对照；仍然只把测量值交给标定函数，不传真实平面。
    """
    rng = np.random.default_rng(0) if rng is None else rng
    measurements = []
    for i, pose in enumerate(poses):
        corners = project_object_points(board.object_points, pose.rvec, pose.tvec, camera)
        corners = corners + rng.normal(0.0, sigma_corner, corners.shape)
        laser = pose.line_uv + rng.normal(0.0, sigma_laser, pose.line_uv.shape)
        measurements.append({
            "index": i,
            "corners": corners.tolist(),
            "laser_uv_row": laser.tolist(),
            "laser_uv_column": [],
            "laser_uv_auto": [],
            "auto_axis": "projected",
            "board_image": None,
            "laser_image": None,
            "n_row": int(len(laser)),
            "n_column": 0,
            "n_auto": 0,
        })
    return measurements


# ==========================================================================
# 标定：只接收测量值
# ==========================================================================
@dataclass
class PreparedPose:
    index: int
    points: np.ndarray
    board_plane: ll.LaserPlane
    pnp_rms_px: float
    n_laser: int
    n_points: int


@dataclass
class PlaneFit:
    normal: np.ndarray
    d: float
    residual_rms_mm: float
    residual_max_mm: float
    singular_values: np.ndarray
    condition: float
    degenerate: bool
    n_points: int

    def to_plane(self) -> ll.LaserPlane:
        if self.degenerate:
            raise ValueError("标定点云退化，不能导出或使用激光平面")
        return ll.LaserPlane(self.normal, self.d, source="CALIBRATED")

    def to_dict(self) -> dict:
        n = np.asarray(self.normal, dtype=np.float64)
        return {
            "normal": n.tolist(),
            "d": float(self.d),
            "equation": "n . X + d = 0",
            "unit": "mm",
            "frame": "camera: X right, Y down, Z forward",
            "normalized": True,
            "residual_rms_mm": float(self.residual_rms_mm),
            "residual_max_mm": float(self.residual_max_mm),
            "singular_values": np.asarray(self.singular_values).tolist(),
            "condition": float(self.condition),
            "degenerate": bool(self.degenerate),
            "n_points": int(self.n_points),
        }


def prepare_measurement(measurement: dict, camera: ll.CameraModel,
                        board: BoardSpec) -> PreparedPose:
    """单组测量 -> 棋盘格位姿 -> 平面上的激光三维点。

    输入只有 measured corners、measured laser UV、相机参数、棋盘格规格；
    没有真实位姿或真实平面。
    """
    corners = np.asarray(measurement["corners"], dtype=np.float64).reshape(-1, 2)
    laser_uv = np.asarray(measurement["laser_uv_row"], dtype=np.float64).reshape(-1, 2)
    if len(corners) != int(np.prod(board.inner)):
        raise ValueError(f"角点数 {len(corners)} != {board.inner} 的角点数")
    if len(laser_uv) == 0:
        raise ValueError(f"第 {measurement.get('index')} 组没有激光中心")

    rvec, tvec, rms = cl.solve_pose(
        corners.reshape(-1, 1, 2), board.inner, board.square_mm,
        camera.K, camera.dist,
    )
    if rvec is None or not np.isfinite(tvec).all():
        raise ValueError(f"第 {measurement.get('index')} 组 solvePnP 失败")

    R = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))[0]
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    board_plane = board_plane_from_pose(R, t)

    rec = ll.reconstruct(laser_uv, camera, board_plane)
    valid = rec.valid.copy()
    if valid.any():
        q = (rec.points_mm[valid] - t) @ R
        xmin, xmax, ymin, ymax = board.bounds
        inside = (
            (q[:, 0] >= xmin - 8.0) & (q[:, 0] <= xmax + 8.0) &
            (q[:, 1] >= ymin - 8.0) & (q[:, 1] <= ymax + 8.0)
        )
        idx = np.flatnonzero(valid)
        valid[idx[~inside]] = False

    points = rec.points_mm[valid]
    return PreparedPose(
        index=int(measurement["index"]),
        points=points,
        board_plane=board_plane,
        pnp_rms_px=float(rms),
        n_laser=int(len(laser_uv)),
        n_points=int(len(points)),
    )


def prepare_all(measurements: Iterable[dict], camera: ll.CameraModel,
                board: BoardSpec) -> list[PreparedPose]:
    return [prepare_measurement(m, camera, board) for m in measurements]


def fit_plane_points(points_mm) -> PlaneFit:
    """对相机坐标系中的点做 SVD 平面拟合，输出归一化平面。"""
    pts = np.asarray(points_mm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_mm 必须是 N×3")
    if len(pts) < 3:
        raise ValueError("至少需要 3 个点才能拟合平面")
    if not np.isfinite(pts).all():
        raise ValueError("点云含有 NaN/Inf")

    center = pts.mean(axis=0)
    U, s, Vt = np.linalg.svd(pts - center, full_matrices=False)
    normal = Vt[-1]
    d = float(-normal @ center)
    if d > 0:
        normal = -normal
        d = -d
    residual = pts @ normal + d
    rms = float(np.sqrt(np.mean(residual ** 2)))
    max_abs = float(np.max(np.abs(residual)))
    condition = float(s[0] / s[1]) if len(s) > 1 and s[1] > 0 else float("inf")
    # s0/s1 很大说明点云接近一条线；此时法向绕这条线旋转是不确定的。
    degenerate = bool(condition > 50.0 or len(s) < 2 or s[1] < 1e-9 * max(s[0], 1e-12))
    return PlaneFit(
        normal=normal, d=d, residual_rms_mm=rms, residual_max_mm=max_abs,
        singular_values=np.asarray(s, dtype=np.float64), condition=condition,
        degenerate=degenerate, n_points=int(len(pts)),
    )


@dataclass
class Calibration:
    fit: PlaneFit
    prepared: list[PreparedPose]
    per_pose: list[dict]


def calibrate_from_prepared(prepared: Sequence[PreparedPose], *,
                            allow_degenerate: bool = False) -> Calibration:
    points = [p.points for p in prepared if len(p.points) > 0]
    if not points:
        raise ValueError("没有可用于拟合的激光三维点")
    all_points = np.vstack(points)
    fit = fit_plane_points(all_points)
    if fit.degenerate and not allow_degenerate:
        raise ValueError("标定点云退化；请增加不同距离及倾角的棋盘格位姿")
    per_pose = []
    for p in prepared:
        if len(p.points) == 0:
            per_pose.append({"index": p.index, "pnp_rms_px": p.pnp_rms_px,
                             "n_laser": p.n_laser, "n_points": 0,
                             "point_residual_rms_mm": None,
                             "point_residual_max_mm": None})
            continue
        res = p.points @ fit.normal + fit.d
        per_pose.append({
            "index": p.index,
            "pnp_rms_px": float(p.pnp_rms_px),
            "n_laser": int(p.n_laser),
            "n_points": int(len(p.points)),
            "point_residual_rms_mm": float(np.sqrt(np.mean(res ** 2))),
            "point_residual_max_mm": float(np.max(np.abs(res))),
        })
    return Calibration(fit=fit, prepared=list(prepared), per_pose=per_pose)


def calibrate_plane(measurements: Iterable[dict], camera: ll.CameraModel,
                    board: BoardSpec) -> Calibration:
    """完整的激光平面标定入口。

    签名里没有 true_plane；所有输入都是测量值。真值只在 evaluate_* 使用。
    """
    prepared = prepare_all(measurements, camera, board)
    return calibrate_from_prepared(prepared)


# ==========================================================================
# 评价：这里才使用真值
# ==========================================================================
def evaluate_plane(fit: PlaneFit, true_plane: ll.LaserPlane) -> dict:
    n_t = true_plane.unit_normal
    d_t = true_plane.unit_d
    dot = float(fit.normal @ n_t)
    if dot < 0:
        n_f = -fit.normal
        d_f = -fit.d
    else:
        n_f = fit.normal
        d_f = fit.d
    angle = float(np.degrees(np.arccos(np.clip(abs(dot), 0.0, 1.0))))
    return {
        "normal_angle_error_deg": angle,
        "d_error_mm": float(d_f - d_t),
        "normal_fit_aligned": n_f.tolist(),
        "d_fit_aligned": float(d_f),
        "normal_true": n_t.tolist(),
        "d_true": float(d_t),
    }


def evaluate_board_planes(calibration: Calibration,
                          poses: Sequence[Pose]) -> list[dict]:
    """每组棋盘格由 PnP 恢复的板面 vs 真实板面（仅评价用）。"""
    truth_by_index = {p.index: p for p in poses}
    rows = []
    for prepared in calibration.prepared:
        pose = truth_by_index.get(prepared.index)
        if pose is None:
            continue
        angle = angle_between_normals(prepared.board_plane.normal,
                                      pose.board_plane.normal)
        n_rec = prepared.board_plane.unit_normal
        n_true = pose.board_plane.unit_normal
        d_rec = prepared.board_plane.unit_d
        d_true = pose.board_plane.unit_d
        if float(n_rec @ n_true) < 0:
            d_rec = -d_rec
        rows.append({
            "index": prepared.index,
            "board_plane_angle_error_deg": angle,
            "board_plane_d_error_mm": float(d_rec - d_true),
            "pnp_rms_px": float(prepared.pnp_rms_px),
            "n_points": int(prepared.n_points),
        })
    return rows


# ==========================================================================
# 独立验证：10 / 31 / 50 mm 台阶
# ==========================================================================
@dataclass
class ValidationCase:
    name: str
    height_mm: float
    z_far_mm: float
    z_near_mm: float
    y_split_mm: float
    measurement: dict
    truth: dict


def build_validation_cases(camera: ll.CameraModel, true_plane: ll.LaserPlane,
                           out_dir: Path, rng: np.random.Generator,
                           sigma_laser: float = 0.2,
                           save_images: bool = True) -> list[ValidationCase]:
    """生成独立验证数据；真实平面只用于渲染，不参与反算。"""
    cases_spec = [
        ("h10_center", 10.0, 500.0, 0.0),
        ("h31_center", 31.0, 500.0, 0.0),
        ("h50_center", 50.0, 500.0, 0.0),
        ("h31_near_left", 31.0, 450.0, -25.0),
        ("h31_far_right", 31.0, 550.0, 25.0),
        ("h31_outside", 31.0, 650.0, 0.0),
        ("h10_near", 10.0, 450.0, 20.0),
        ("h10_far", 10.0, 550.0, -20.0),
        ("h50_near", 50.0, 450.0, 20.0),
        ("h50_far", 50.0, 550.0, -20.0),
    ]
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    cases = []
    for name, height, z_far, y_split in cases_spec:
        z_near = z_far - height
        curve, roles = step_curve(true_plane, z_far, z_near, y_split)
        uv_true = project_points(curve, camera)
        if not np.isfinite(uv_true).all():
            raise RuntimeError(f"验证场景 {name} 投影出现 NaN")
        img_float = ll.render_laser_image(
            uv_true, camera.image_size, sigma_px=1.4, peak=220.0,
            background=7.0, gradient=(2.0, 2.0), read_sigma=1.5,
            shot_gain=0.20, rng=rng,
        )
        img = ll.to_uint8(img_float)
        ext = _extract_mode(img.astype(np.float64), "row")
        uv_meas = ext.uv + rng.normal(0.0, sigma_laser, ext.uv.shape)
        img_name = f"{name}.png"
        if save_images:
            save_png(out_dir / "images" / img_name, img)
        measurement = {
            "case": name,
            "laser_uv": uv_meas.tolist(),
            "n_extracted": int(len(ext.uv)),
            "image": f"images/{img_name}" if save_images else None,
        }
        truth = {
            "case": name,
            "true_height_mm": float(height),
            "z_far_mm": float(z_far),
            "z_near_mm": float(z_near),
            "y_split_mm": float(y_split),
            "true_uv": uv_true.tolist(),
            "roles": roles.tolist(),
            "position": {"z_far_mm": float(z_far), "y_split_mm": float(y_split)},
        }
        cases.append(ValidationCase(
            name=name, height_mm=height, z_far_mm=z_far, z_near_mm=z_near,
            y_split_mm=y_split, measurement=measurement, truth=truth,
        ))
    return cases


def measure_validation(fit: PlaneFit, measurements: Sequence[dict],
                       camera: ll.CameraModel,
                       min_step_mm: float = 1.0) -> list[dict]:
    """用**拟合平面**反算验证台阶；真值只在返回后用于算误差。"""
    plane = fit.to_plane()
    rows = []
    for measurement in measurements:
        if set(measurement) - {"case", "laser_uv", "n_extracted", "image"}:
            raise ValueError("验证反算只接受测量记录，禁止传入真值字段")
        uv = np.asarray(measurement["laser_uv"], dtype=np.float64).reshape(-1, 2)
        rec = ll.reconstruct(uv, camera, plane)
        est = ll.robust_step_height(rec.points_mm[rec.valid],
                                    min_separation_mm=min_step_mm)
        height = est.height_mm if est.has_step else float("nan")
        rows.append({
            "case": measurement["case"],
            "measured_height_mm": float(height),
            "has_step": bool(est.has_step),
            "n_inliers": int(est.n_inliers),
            "n_outliers": int(est.n_outliers),
            "height_se_mm": float(est.height_se_mm),
        })
    return rows


def evaluate_validation(measured_rows: Sequence[dict],
                        cases: Sequence[ValidationCase]) -> tuple[list[dict], dict]:
    truth = {c.name: c for c in cases}
    measured_by_name = {r["case"]: r for r in measured_rows}
    if len(measured_by_name) != len(measured_rows) or set(measured_by_name) - set(truth):
        raise ValueError("验证测量包含重复或未知场景")
    rows = []
    for name, case in truth.items():
        row = measured_by_name.get(name, {"case": name, "has_step": False,
                                         "measured_height_mm": float("nan")})
        true_h = case.height_mm
        measured = float(row["measured_height_mm"])
        err = measured - true_h if row.get("has_step", False) and np.isfinite(measured) else float("nan")
        rows.append({
            **row,
            "true_height_mm": float(true_h),
            "z_far_mm": float(case.z_far_mm),
            "y_split_mm": float(case.y_split_mm),
            "error_mm": float(err),
            "abs_error_mm": float(abs(err)) if np.isfinite(err) else float("nan"),
        })
    abs_errors = np.array([r["abs_error_mm"] for r in rows], dtype=np.float64)
    valid = abs_errors[np.isfinite(abs_errors)]
    summary = {
        "n_cases": len(rows),
        "n_valid": int(len(valid)),
        "n_failed": len(rows) - int(len(valid)),
        "all_valid": bool(len(rows) > 0 and len(valid) == len(rows)),
        "mean_abs_error_mm": float(np.mean(valid)) if len(valid) else float("nan"),
        "max_abs_error_mm": float(np.max(valid)) if len(valid) else float("nan"),
    }
    return rows, summary


# ==========================================================================
# 多组实验：板数 / 覆盖 / 噪声
# ==========================================================================
def _subset_fit_metrics(prepared: Sequence[PreparedPose], true_plane: ll.LaserPlane,
                        cases: Sequence[ValidationCase], camera: ll.CameraModel) -> dict:
    # 退化覆盖实验保留诊断，但不把无效平面交给测量函数。
    calib = calibrate_from_prepared(prepared, allow_degenerate=True)
    ev = evaluate_plane(calib.fit, true_plane)
    val_rows = [] if calib.fit.degenerate else measure_validation(
        calib.fit, [c.measurement for c in cases], camera)
    evaluated_rows, val_summary = evaluate_validation(val_rows, cases)
    return {
        "fit": calib.fit,
        "angle_deg": ev["normal_angle_error_deg"],
        "d_error_mm": ev["d_error_mm"],
        "residual_rms_mm": calib.fit.residual_rms_mm,
        "residual_max_mm": calib.fit.residual_max_mm,
        "condition": calib.fit.condition,
        "degenerate": calib.fit.degenerate,
        "n_points": calib.fit.n_points,
        "validation_mean_abs_mm": val_summary["mean_abs_error_mm"] if val_summary["all_valid"] else float("nan"),
        "validation_max_abs_mm": val_summary["max_abs_error_mm"] if val_summary["all_valid"] else float("nan"),
        "validation_all_valid": val_summary["all_valid"],
        "validation_n_failed": val_summary["n_failed"],
        "validation_rows": evaluated_rows,
    }


def _save_trial(root, tag: str, inputs: dict, metrics: dict) -> None:
    if root is None:
        return
    root = Path(root)
    write_json(root / "data" / "experiments" / f"{tag}.json", inputs)
    write_json(root / "results" / "experiments" / f"{tag}.json", {
        "fit": metrics["fit"].to_dict(),
        **{k: v for k, v in metrics.items() if k not in ("fit", "validation_rows")},
    })
    write_json(root / "validation" / "experiments" / f"{tag}.json",
               metrics["validation_rows"])


def run_board_count(prepared: Sequence[PreparedPose], true_plane: ll.LaserPlane,
                    cases: Sequence[ValidationCase], camera: ll.CameraModel,
                    counts=(5, 8, 10, 12, 15), repeats: int = 20,
                    rng: np.random.Generator | None = None, audit_dir=None) -> list[dict]:
    rng = np.random.default_rng(123) if rng is None else rng
    prepared = list(prepared)
    n_all = len(prepared)
    rows = []
    for n in counts:
        if n > n_all:
            continue
        if n == n_all:
            subsets = [list(range(n_all))]
        else:
            subsets = []
            seen = set()
            for _ in range(int(repeats) * 5):
                if len(subsets) >= int(repeats):
                    break
                idx = tuple(sorted(rng.choice(n_all, size=n, replace=False).tolist()))
                if idx not in seen:
                    seen.add(idx)
                    subsets.append(list(idx))
        metrics = []
        degenerate = 0
        for trial, subset in enumerate(subsets):
            m = _subset_fit_metrics([prepared[i] for i in subset],
                                    true_plane, cases, camera)
            metrics.append(m)
            _save_trial(audit_dir, f"board_count/n{n}/trial_{trial:03d}",
                        {"source": "data/measurements.json",
                         "indices": [prepared[i].index for i in subset]}, m)
            degenerate += int(m["degenerate"])
        angles = np.array([m["angle_deg"] for m in metrics], dtype=np.float64)
        rms = np.array([m["residual_rms_mm"] for m in metrics], dtype=np.float64)
        vals = np.array([m["validation_mean_abs_mm"] for m in metrics], dtype=np.float64)
        conditions = np.array([m["condition"] for m in metrics], dtype=np.float64)
        rows.append({
            "n_boards": int(n),
            "n_subsets": int(len(subsets)),
            "n_degenerate": int(degenerate),
            "n_failed_validation": sum(not m["validation_all_valid"] for m in metrics),
            "angle_mean_deg": float(np.nanmean(angles)),
            "angle_std_deg": float(np.nanstd(angles, ddof=1)) if len(angles) > 1 else 0.0,
            "rms_mean_mm": float(np.nanmean(rms)),
            "rms_std_mm": float(np.nanstd(rms, ddof=1)) if len(rms) > 1 else 0.0,
            "validation_mean_abs_mean_mm": float(np.mean(vals)) if all(m["validation_all_valid"] for m in metrics) else float("nan"),
            "validation_mean_abs_std_mm": float(np.std(vals, ddof=1)) if len(vals) > 1 and all(m["validation_all_valid"] for m in metrics) else float("nan"),
            "condition_median": float(np.nanmedian(conditions)),
        })
    return rows


def run_coverage(coverage_sets: dict[str, Sequence[PreparedPose]],
                 true_plane: ll.LaserPlane, cases: Sequence[ValidationCase],
                 camera: ll.CameraModel, audit_dir=None) -> list[dict]:
    rows = []
    for name, prepared in coverage_sets.items():
        m = _subset_fit_metrics(prepared, true_plane, cases, camera)
        _save_trial(audit_dir, f"coverage/{name}",
                    {"source": "data/measurements.json" if name == "good" else f"data/coverage/{name}/measurements.json",
                     "indices": [p.index for p in prepared]}, m)
        rows.append({
            "coverage": name,
            "n_boards": int(len(prepared)),
            "angle_deg": m["angle_deg"],
            "d_error_mm": m["d_error_mm"],
            "residual_rms_mm": m["residual_rms_mm"],
            "condition": m["condition"],
            "degenerate": m["degenerate"],
            "validation_n_failed": m["validation_n_failed"],
            "validation_mean_abs_mm": m["validation_mean_abs_mm"],
            "validation_max_abs_mm": m["validation_max_abs_mm"],
        })
    return rows


def run_noise(main_measurements: Sequence[dict], camera: ll.CameraModel,
              board: BoardSpec, true_plane: ll.LaserPlane,
              cases: Sequence[ValidationCase],
              corner_sigmas=(0.0, 0.1, 0.2, 0.3, 0.5, 1.0),
              laser_sigmas=(0.0, 0.1, 0.2, 0.3, 0.5, 1.0),
              repeats: int = 10, rng: np.random.Generator | None = None,
              audit_dir=None) -> list[dict]:
    """在已测角点/中心上再加高斯噪声，看标定和验证误差。"""
    rng = np.random.default_rng(321) if rng is None else rng
    rows = []

    def add_noise(measurements, sigma_corner, sigma_laser):
        out = []
        for m in measurements:
            corners = np.asarray(m.get("corners_raw", m["corners"]), dtype=np.float64)
            laser = np.asarray(m.get("laser_uv_row_raw", m["laser_uv_row"]), dtype=np.float64)
            out.append({
                **m,
                "corners": (corners + rng.normal(0, sigma_corner, corners.shape)).tolist(),
                "laser_uv_row": (laser + rng.normal(0, sigma_laser, laser.shape)).tolist(),
            })
        return out

    for kind, sigmas in (("corner", corner_sigmas), ("laser", laser_sigmas)):
        for sigma in sigmas:
            metrics = []
            for trial in range(int(repeats)):
                if kind == "corner":
                    meas = add_noise(main_measurements, sigma, 0.0)
                else:
                    meas = add_noise(main_measurements, 0.0, sigma)
                prepared = prepare_all(meas, camera, board)
                metric = _subset_fit_metrics(prepared, true_plane, cases, camera)
                metrics.append(metric)
                _save_trial(audit_dir, f"noise/{kind}_{sigma:g}/trial_{trial:03d}",
                            {"measurements": meas, "sigma_corner": sigma if kind == "corner" else 0.0,
                             "sigma_laser": sigma if kind == "laser" else 0.0}, metric)
            angles = np.array([m["angle_deg"] for m in metrics], dtype=np.float64)
            rms = np.array([m["residual_rms_mm"] for m in metrics], dtype=np.float64)
            vals = np.array([m["validation_mean_abs_mm"] for m in metrics], dtype=np.float64)
            rows.append({
                "noise_type": kind,
                "sigma_px": float(sigma),
                "n_repeats": int(repeats),
                "n_failed_validation": sum(not m["validation_all_valid"] for m in metrics),
                "angle_mean_deg": float(np.nanmean(angles)),
                "angle_std_deg": float(np.nanstd(angles, ddof=1)) if len(angles) > 1 else 0.0,
                "rms_mean_mm": float(np.nanmean(rms)),
                "rms_std_mm": float(np.nanstd(rms, ddof=1)) if len(rms) > 1 else 0.0,
                "validation_mean_abs_mean_mm": float(np.mean(vals)) if all(m["validation_all_valid"] for m in metrics) else float("nan"),
                "validation_mean_abs_std_mm": float(np.std(vals, ddof=1)) if len(vals) > 1 and all(m["validation_all_valid"] for m in metrics) else float("nan"),
            })
    return rows


# ==========================================================================
# IO / 报告小工具
# ==========================================================================
def write_json(path, value) -> None:
    ll.write_json(path, ll.clean_json(value))


def write_csv(path, rows: Sequence[dict]) -> None:
    path = Path(path)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_orientation(rows: Sequence[dict]) -> dict:
    def arr(key):
        return np.array([r[key] for r in rows], dtype=np.float64)

    row_counts = arr("row_count")
    col_counts = arr("column_count")
    auto_counts = arr("auto_count")
    auto_roi_counts = arr("auto_roi_count")
    row_err = arr("row_center_error_mean_px")
    col_err = arr("column_center_error_mean_px")
    auto_err = arr("auto_center_error_mean_px")
    auto_roi_err = arr("auto_roi_center_error_mean_px")
    auto_row = sum(1 for r in rows if r["auto_axis"] == "row")
    auto_roi_row = sum(1 for r in rows if r["auto_roi_axis"] == "row")
    return {
        "n_poses": len(rows),
        "row_count_mean": float(np.nanmean(row_counts)),
        "column_count_mean": float(np.nanmean(col_counts)),
        "auto_count_mean": float(np.nanmean(auto_counts)),
        "auto_roi_count_mean": float(np.nanmean(auto_roi_counts)),
        "row_error_mean_px": float(np.nanmean(row_err)),
        "column_error_mean_px": float(np.nanmean(col_err)),
        "auto_error_mean_px": float(np.nanmean(auto_err)),
        "auto_roi_error_mean_px": float(np.nanmean(auto_roi_err)),
        "auto_axis_row_fraction": float(auto_row / len(rows)) if rows else 0.0,
        "auto_roi_axis_row_fraction": float(auto_roi_row / len(rows)) if rows else 0.0,
    }
