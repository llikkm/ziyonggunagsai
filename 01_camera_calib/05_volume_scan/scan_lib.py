"""多帧线激光扫描 + 高度图体积仿真核心库。

单位与坐标约定（与已有工程一致）：
* 世界坐标系：X 向右（物体长 100 mm），Y 向里（宽 80 mm），Z 向上（高 50 mm）。
* 相机坐标系：X 右、Y 下、Z 前。
* 机构把相机坐标系变换到世界：P_w = R_i @ P_c + t_i。
* 本仿真里相机和激光器刚性连接，R_i 固定，t_i 沿世界 X 平移。

数据隔离：
* 本库知道物体的真实四边形表面，因为它要计算激光平面与可见表面的交线、渲染图像；
* ``reconstruct_scan()`` 只接收像素、相机内参、激光平面、机构位姿；
* 真值点云/真实体积只在 ``evaluate_*`` 和画图时使用。
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # 01_camera_calib
for _p in (str(ROOT), str(ROOT / "03_laser_triangulation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import laser_lib as ll  # noqa: E402

X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])


# ==========================================================================
# 平面四边形 / 物体
# ==========================================================================
@dataclass(frozen=True)
class Quad:
    """矩形表面：P = origin + s*e1 + t*e2，s∈[0,l1]，t∈[0,l2]。"""

    name: str
    origin: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    length1: float
    length2: float
    normal: np.ndarray

    def __post_init__(self) -> None:
        o = np.asarray(self.origin, dtype=np.float64).reshape(3)
        e1 = np.asarray(self.e1, dtype=np.float64).reshape(3)
        e2 = np.asarray(self.e2, dtype=np.float64).reshape(3)
        n = np.asarray(self.normal, dtype=np.float64).reshape(3)
        for name, v in (("origin", o), ("e1", e1), ("e2", e2), ("normal", n)):
            if not np.isfinite(v).all():
                raise ValueError(f"{self.name}: {name} 含 NaN/Inf")
        if abs(np.linalg.norm(e1) - 1.0) > 1e-6 or abs(np.linalg.norm(e2) - 1.0) > 1e-6:
            raise ValueError(f"{self.name}: e1/e2 必须是单位向量")
        if abs(float(e1 @ e2)) > 1e-6:
            raise ValueError(f"{self.name}: e1/e2 必须正交")
        if abs(np.linalg.norm(n) - 1.0) > 1e-6:
            raise ValueError(f"{self.name}: normal 必须是单位向量")
        object.__setattr__(self, "origin", o)
        object.__setattr__(self, "e1", e1)
        object.__setattr__(self, "e2", e2)
        object.__setattr__(self, "normal", n)
        object.__setattr__(self, "length1", float(self.length1))
        object.__setattr__(self, "length2", float(self.length2))

    @property
    def d(self) -> float:
        return float(-self.normal @ self.origin)

    @property
    def area_mm2(self) -> float:
        return float(self.length1 * self.length2)

    def point(self, s, t):
        return self.origin + np.asarray(s)[..., None] * self.e1 + np.asarray(t)[..., None] * self.e2

    def corners(self) -> np.ndarray:
        return np.array([
            self.origin,
            self.origin + self.length1 * self.e1,
            self.origin + self.length1 * self.e1 + self.length2 * self.e2,
            self.origin + self.length2 * self.e2,
        ])

    def contains(self, points, eps: float = 1e-6) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        s = (p - self.origin) @ self.e1
        t = (p - self.origin) @ self.e2
        return ((s >= -eps) & (s <= self.length1 + eps) &
                (t >= -eps) & (t <= self.length2 + eps))


def _q(name, origin, e1, e2, l1, l2, normal) -> Quad:
    return Quad(name, np.asarray(origin, float), np.asarray(e1, float),
                np.asarray(e2, float), float(l1), float(l2), np.asarray(normal, float))


def _box_faces(x0=0.0, x1=100.0, y0=0.0, y1=80.0, z0=0.0, z1=50.0, prefix="box"):
    return [
        _q(f"{prefix}_xmin", [x0, y0, z0], Y_AXIS, Z_AXIS, y1 - y0, z1 - z0, -X_AXIS),
        _q(f"{prefix}_xmax", [x1, y0, z0], Y_AXIS, Z_AXIS, y1 - y0, z1 - z0, X_AXIS),
        _q(f"{prefix}_ymin", [x0, y0, z0], X_AXIS, Z_AXIS, x1 - x0, z1 - z0, -Y_AXIS),
        _q(f"{prefix}_ymax", [x0, y1, z0], X_AXIS, Z_AXIS, x1 - x0, z1 - z0, Y_AXIS),
        _q(f"{prefix}_zmin", [x0, y0, z0], X_AXIS, Y_AXIS, x1 - x0, y1 - y0, -Z_AXIS),
        _q(f"{prefix}_zmax", [x0, y0, z1], X_AXIS, Y_AXIS, x1 - x0, y1 - y0, Z_AXIS),
    ]


@dataclass
class ScanObject:
    name: str
    quads: list[Quad]
    true_volume_cm3: float
    top_height_mm: float
    description: str
    hidden_volume_cm3: float = 0.0
    footprint: tuple[float, float, float, float] = (0.0, 100.0, 0.0, 80.0)


def make_box() -> ScanObject:
    return ScanObject(
        name="A_box",
        quads=_box_faces(),
        true_volume_cm3=400.0,
        top_height_mm=50.0,
        description="100 x 80 x 50 mm 规则长方体；真实体积 400 cm^3",
        footprint=(0.0, 100.0, 0.0, 80.0),
    )


def make_slot_box() -> ScanObject:
    """顶部 20x20x10 mm 盲槽；槽底 z=40，槽口 x 40..60, y 30..50。"""
    quads = [q for q in _box_faces() if q.name != "box_zmax"]
    # 顶部四块（中间挖掉 20x20）
    quads += [
        _q("top_x0", [0, 0, 50], X_AXIS, Y_AXIS, 40, 80, Z_AXIS),
        _q("top_x2", [60, 0, 50], X_AXIS, Y_AXIS, 40, 80, Z_AXIS),
        _q("top_y0", [40, 0, 50], X_AXIS, Y_AXIS, 20, 30, Z_AXIS),
        _q("top_y1", [40, 50, 50], X_AXIS, Y_AXIS, 20, 30, Z_AXIS),
        # 槽底
        _q("slot_bottom", [40, 30, 40], X_AXIS, Y_AXIS, 20, 20, Z_AXIS),
        # 四个槽壁：法向指向槽内
        _q("slot_wall_x0", [40, 30, 40], Y_AXIS, Z_AXIS, 20, 10, X_AXIS),
        _q("slot_wall_x1", [60, 30, 40], Y_AXIS, Z_AXIS, 20, 10, -X_AXIS),
        _q("slot_wall_y0", [40, 30, 40], X_AXIS, Z_AXIS, 20, 10, Y_AXIS),
        _q("slot_wall_y1", [40, 50, 40], X_AXIS, Z_AXIS, 20, 10, -Y_AXIS),
    ]
    return ScanObject(
        name="B_slot",
        quads=quads,
        true_volume_cm3=396.0,
        top_height_mm=50.0,
        description="长方体顶部开 20 x 20 x 10 mm 盲槽；真实体积 396 cm^3",
        hidden_volume_cm3=4.0,
        footprint=(0.0, 100.0, 0.0, 80.0),
    )


def make_side_hole_box() -> ScanObject:
    """沿 Y 方向贯穿的 20x20 mm 方孔：X 40..60, Z 20..40；真实体积 368 cm^3。"""
    quads = []
    # x=0 / x=100 完整
    quads += [
        _q("box_xmin", [0, 0, 0], Y_AXIS, Z_AXIS, 80, 50, -X_AXIS),
        _q("box_xmax", [100, 0, 0], Y_AXIS, Z_AXIS, 80, 50, X_AXIS),
    ]
    # y=0 / y=80：挖掉 x 40..60, z 20..40
    for yy, nrm, tag in ((0.0, -Y_AXIS, "ymin"), (80.0, Y_AXIS, "ymax")):
        quads += [
            _q(f"box_{tag}_x0", [0, yy, 0], X_AXIS, Z_AXIS, 40, 50, nrm),
            _q(f"box_{tag}_x2", [60, yy, 0], X_AXIS, Z_AXIS, 40, 50, nrm),
            _q(f"box_{tag}_z0", [40, yy, 0], X_AXIS, Z_AXIS, 20, 20, nrm),
            _q(f"box_{tag}_z1", [40, yy, 40], X_AXIS, Z_AXIS, 20, 10, nrm),
        ]
    # 底面、顶面完整
    quads += [
        _q("box_zmin", [0, 0, 0], X_AXIS, Y_AXIS, 100, 80, -Z_AXIS),
        _q("box_zmax", [0, 0, 50], X_AXIS, Y_AXIS, 100, 80, Z_AXIS),
    ]
    # 孔内壁：法向指向孔内
    quads += [
        _q("hole_x0", [40, 0, 20], Y_AXIS, Z_AXIS, 80, 20, X_AXIS),
        _q("hole_x1", [60, 0, 20], Y_AXIS, Z_AXIS, 80, 20, -X_AXIS),
        _q("hole_z0", [40, 0, 20], X_AXIS, Y_AXIS, 20, 80, Z_AXIS),
        _q("hole_z1", [40, 0, 40], X_AXIS, Y_AXIS, 20, 80, -Z_AXIS),
    ]
    return ScanObject(
        name="C_side_hole",
        quads=quads,
        true_volume_cm3=368.0,
        top_height_mm=50.0,
        description="长方体侧面开 20 x 20 mm 贯穿方孔；真实体积 368 cm^3",
        hidden_volume_cm3=32.0,
        footprint=(0.0, 100.0, 0.0, 80.0),
    )


def make_object(name: str) -> ScanObject:
    key = name.lower()
    if key in ("a", "a_box", "box"):
        return make_box()
    if key in ("b", "b_slot", "slot"):
        return make_slot_box()
    if key in ("c", "c_side_hole", "hole"):
        return make_side_hole_box()
    raise ValueError(f"未知物体: {name}")


def ground_quad() -> Quad:
    return _q("ground", [-50, -40, 0], X_AXIS, Y_AXIS, 200, 160, Z_AXIS)


# ==========================================================================
# 扫描机构 / 相机 / 激光
# ==========================================================================
@dataclass(frozen=True)
class ScanPose:
    index: int
    R: np.ndarray       # camera -> world
    t: np.ndarray       # camera origin in world


@dataclass
class ScanSetup:
    camera: ll.CameraModel
    scan_x_mm: np.ndarray
    camera_y_mm: float = 40.0
    camera_z_mm: float = 300.0
    laser_plane_c: ll.LaserPlane = field(
        default_factory=lambda: ll.LaserPlane.from_abcd([1.0, 0.0, 0.2, -100.0],
                                                       source="SCAN_TRUE"))
    laser_source_c: np.ndarray = field(default_factory=lambda: np.array([100.0, 0.0, 0.0]))
    R_c2w: np.ndarray = field(default_factory=lambda: np.diag([1.0, -1.0, -1.0]))

    def __post_init__(self) -> None:
        self.scan_x_mm = np.asarray(self.scan_x_mm, dtype=np.float64).ravel()
        self.R_c2w = np.asarray(self.R_c2w, dtype=np.float64)
        self.laser_source_c = np.asarray(self.laser_source_c, dtype=np.float64).reshape(3)
        if not np.isfinite(self.scan_x_mm).all() or not np.isfinite(self.R_c2w).all():
            raise ValueError("扫描位置/旋转含有 NaN/Inf")
        if abs(np.linalg.det(self.R_c2w) - 1.0) > 1e-6:
            raise ValueError("R_c2w 必须是旋转矩阵")
        dist = float(self.laser_plane_c.distance(self.laser_source_c.reshape(1, 3))[0])
        if abs(dist) > 1e-6:
            raise ValueError(f"激光源不在激光平面上: {dist:.3e} mm")

    @property
    def image_size(self) -> tuple[int, int]:
        return self.camera.image_size

    def pose(self, i: int) -> ScanPose:
        x = float(self.scan_x_mm[i])
        t = np.array([x, self.camera_y_mm, self.camera_z_mm], dtype=np.float64)
        return ScanPose(index=int(i), R=self.R_c2w.copy(), t=t)

    def world_plane(self, pose: ScanPose) -> ll.LaserPlane:
        n = pose.R @ self.laser_plane_c.unit_normal
        d = float(self.laser_plane_c.unit_d - n @ pose.t)
        return ll.LaserPlane(n, d, source="LASER_PLANE_WORLD")

    def world_source(self, pose: ScanPose) -> np.ndarray:
        return pose.R @ self.laser_source_c + pose.t

    def project_world(self, points_world, pose: ScanPose) -> np.ndarray:
        p = np.asarray(points_world, dtype=np.float64)
        pc = (p - pose.t) @ pose.R  # = R.T @ (p - t)
        return self.camera.project(pc)


# ==========================================================================
# 几何求交 / 遮挡
# ==========================================================================
def plane_quad_segment(plane: ll.LaserPlane, quad: Quad,
                       point_spacing_mm: float = 0.5) -> np.ndarray | None:
    """激光平面与矩形表面的交线段（相机/世界坐标均可，统一用同一坐标）。"""
    n_p, d_p = plane.unit_normal, plane.unit_d
    n_q = quad.normal
    direction = np.cross(n_p, n_q)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-10:
        # 两平面平行；没有唯一交线
        return None
    direction /= norm
    # 两平面交线上离原点最近的点
    A = np.vstack([n_p, n_q])
    b = np.array([-d_p, -quad.d], dtype=np.float64)
    p0 = np.linalg.lstsq(A, b, rcond=None)[0]

    s0 = float((p0 - quad.origin) @ quad.e1)
    t0 = float((p0 - quad.origin) @ quad.e2)
    sd = float(direction @ quad.e1)
    td = float(direction @ quad.e2)

    def interval(p, d, lo, hi):
        if abs(d) < 1e-12:
            return (-np.inf, np.inf) if lo <= p <= hi else None
        a = (lo - p) / d
        b = (hi - p) / d
        return (min(a, b), max(a, b))

    i1 = interval(s0, sd, 0.0, quad.length1)
    i2 = interval(t0, td, 0.0, quad.length2)
    if i1 is None or i2 is None:
        return None
    lo = max(i1[0], i2[0])
    hi = min(i1[1], i2[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        return None
    n = max(2, int(np.ceil((hi - lo) / max(point_spacing_mm, 1e-6))) + 1)
    us = np.linspace(lo, hi, n)
    return p0[None, :] + us[:, None] * direction[None, :]


def ray_blocked(p: np.ndarray, target: np.ndarray, quads: list[Quad],
                skip_index: int, eps: float = 1e-6) -> bool:
    """线段 p->target 是否被任一面片挡住（跳过 p 所在面片）。"""
    p = np.asarray(p, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    delta = target - p
    dist = float(np.linalg.norm(delta))
    if dist < eps:
        return False
    direction = delta / dist
    for j, q in enumerate(quads):
        if j == skip_index:
            continue
        den = float(q.normal @ direction)
        if abs(den) < 1e-12:
            continue
        t = -float(q.normal @ p + q.d) / den
        if t <= eps or t >= dist - eps:
            continue
        hit = p + t * direction
        s = float((hit - q.origin) @ q.e1)
        u = float((hit - q.origin) @ q.e2)
        if -eps <= s <= q.length1 + eps and -eps <= u <= q.length2 + eps:
            return True
    return False


@dataclass
class VisibleSegment:
    quad: Quad
    points_world: np.ndarray
    uv: np.ndarray


@dataclass
class VisibilityResult:
    segments: list[VisibleSegment]
    visible_by_quad: dict[str, int]
    total_by_quad: dict[str, int]
    n_visible: int
    n_candidates: int


def visible_laser_points(quads: list[Quad], setup: ScanSetup, pose: ScanPose,
                         point_spacing_mm: float = 0.5) -> VisibilityResult:
    """计算当前位姿下激光平面与所有表面的可见交线。"""
    plane_w = setup.world_plane(pose)
    source_w = setup.world_source(pose)
    cam_w = pose.t
    segments: list[VisibleSegment] = []
    visible_by_quad: dict[str, int] = {}
    total_by_quad: dict[str, int] = {}
    n_visible = 0
    n_candidates = 0

    for qi, quad in enumerate(quads):
        pts = plane_quad_segment(plane_w, quad, point_spacing_mm)
        if pts is None or len(pts) < 2:
            continue
        uv_all = setup.project_world(pts, pose)
        vis = np.isfinite(uv_all).all(axis=1)
        h, w = setup.image_size[1], setup.image_size[0]
        vis &= (uv_all[:, 0] >= 0) & (uv_all[:, 0] < w) & \
               (uv_all[:, 1] >= 0) & (uv_all[:, 1] < h)

        # 激光入射面 / 相机可见面
        front_l = ((source_w - pts) @ quad.normal) > 1e-6
        front_c = ((cam_w - pts) @ quad.normal) > 1e-6
        vis &= front_l & front_c

        # 遮挡：逐点射线测试（先跳过已失效点，节省时间）
        for k in np.flatnonzero(vis):
            p = pts[k]
            if ray_blocked(p, source_w, quads, qi):
                vis[k] = False
                continue
            if ray_blocked(p, cam_w, quads, qi):
                vis[k] = False

        total_by_quad[quad.name] = total_by_quad.get(quad.name, 0) + len(pts)
        n_candidates += len(pts)
        if not vis.any():
            visible_by_quad[quad.name] = visible_by_quad.get(quad.name, 0) + 0
            continue
        visible_by_quad[quad.name] = visible_by_quad.get(quad.name, 0) + int(vis.sum())
        n_visible += int(vis.sum())

        # 遮挡/阴影可能把可见点切成多段；按连续 True 分组，避免渲染时跨阴影连线
        idx = np.flatnonzero(vis)
        splits = np.flatnonzero(np.diff(idx) > 1) + 1
        for run in np.split(idx, splits):
            if len(run) < 1:
                continue
            segments.append(VisibleSegment(quad=quad, points_world=pts[run],
                                           uv=uv_all[run]))

    return VisibilityResult(segments=segments, visible_by_quad=visible_by_quad,
                            total_by_quad=total_by_quad, n_visible=n_visible,
                            n_candidates=n_candidates)


# ==========================================================================
# 条纹图像渲染 / 提取
# ==========================================================================
def render_stripe_image(segments: list[VisibleSegment], setup: ScanSetup,
                        sigma_px: float = 1.4, peak: float = 220.0,
                        background: float = 6.0, gradient=(2.0, 2.0),
                        read_sigma: float = 1.0, shot_gain: float = 0.1,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """多段可见条纹 -> 一张灰度图；每段单独 splat，再取 max，避免跨阴影连线。"""
    rng = np.random.default_rng() if rng is None else rng
    w, h = setup.image_size
    layer = np.zeros((h, w), dtype=np.float64)
    for seg in segments:
        if len(seg.uv) < 2:
            continue
        img = ll.render_laser_image(
            seg.uv, (w, h), sigma_px=sigma_px, peak=peak,
            background=0.0, gradient=(0.0, 0.0), read_sigma=0.0,
            shot_gain=0.0, rng=rng,
        )
        np.maximum(layer, img, out=layer)
    yy, xx = np.mgrid[0:h, 0:w]
    img = (float(background)
           + float(gradient[0]) * (xx / max(w - 1, 1))
           + float(gradient[1]) * (yy / max(h - 1, 1))
           + layer)
    if read_sigma > 0:
        img = img + rng.normal(0.0, read_sigma, img.shape)
    if shot_gain > 0:
        img = img + rng.normal(0.0, np.sqrt(np.clip(img, 0, None)) * shot_gain, img.shape)
    return np.clip(img, 0.0, 255.0)


def extract_centers(image: np.ndarray, roi=None) -> ll.LaserExtraction:
    """条纹近似竖直，按行提取；和真实图像路径共用同一个提取器。"""
    return ll.extract_laser_centers(
        np.asarray(image, dtype=np.float64), axis="row", roi=roi, smooth_sigma=1.0,
        threshold_sigma=4.0, min_peak=12.0, min_peak_fraction=0.10,
        centroid_radius=4, max_width=9.0,
    )


# ==========================================================================
# 单次扫描：只保存像素测量和机构位姿，不保存真实点云到反算路径
# ==========================================================================
@dataclass
class ScanFrame:
    index: int
    pose: ScanPose
    uv: np.ndarray
    n_visible: int
    n_extracted: int
    visible_by_quad: dict[str, int]
    total_by_quad: dict[str, int]
    image_path: str | None = None


@dataclass
class ScanData:
    object_name: str
    setup: ScanSetup
    frames: list[ScanFrame]
    source_label: str
    timings_s: dict[str, float]
    total_candidates: int
    total_visible: int


def simulate_scan(obj: ScanObject, setup: ScanSetup, source_label: str = "high",
                  save_image_indices: set[int] | None = None,
                  image_dir: Path | None = None, seed: int = 0,
                  read_sigma: float = 1.0, shot_gain: float = 0.1) -> ScanData:
    """逐帧：真实求交 -> 投影 -> 渲染条纹 -> 现有提取器 -> 像素中心。"""
    rng = np.random.default_rng(seed)
    quads = list(obj.quads) + [ground_quad()]
    frames: list[ScanFrame] = []
    t_vis = t_render = t_extract = 0.0
    total_candidates = total_visible = 0
    save_image_indices = (set(range(len(setup.scan_x_mm))) if image_dir is not None else set()) \
        if save_image_indices is None else set(save_image_indices)

    for i in range(len(setup.scan_x_mm)):
        pose = setup.pose(i)
        t0 = time.perf_counter()
        vis = visible_laser_points(quads, setup, pose)
        t1 = time.perf_counter()
        t_vis += t1 - t0
        img = render_stripe_image(vis.segments, setup, read_sigma=read_sigma,
                                  shot_gain=shot_gain, rng=rng)
        t2 = time.perf_counter()
        t_render += t2 - t1
        image_u8 = ll.to_uint8(img)
        ext = extract_centers(image_u8)
        t3 = time.perf_counter()
        t_extract += t3 - t2

        path = None
        if image_dir is not None and i in save_image_indices:
            image_dir = Path(image_dir)
            image_dir.mkdir(parents=True, exist_ok=True)
            path = str(image_dir / f"frame_{i:03d}.png")
            save_png(path, image_u8)

        frames.append(ScanFrame(
            index=i, pose=pose, uv=ext.uv, n_visible=vis.n_visible,
            n_extracted=int(len(ext.uv)), visible_by_quad=vis.visible_by_quad,
            total_by_quad=vis.total_by_quad, image_path=path,
        ))
        total_candidates += vis.n_candidates
        total_visible += vis.n_visible

    return ScanData(
        object_name=obj.name, setup=setup, frames=frames, source_label=source_label,
        timings_s={"visibility": t_vis, "render": t_render, "extract": t_extract},
        total_candidates=total_candidates, total_visible=total_visible,
    )


def visible_fraction(scan: "ScanData", names) -> float | None:
    """指定面片集合的可见点/候选点比例（只用于评价，不进反算）。"""
    vis = sum(f.visible_by_quad.get(n, 0) for f in scan.frames for n in names)
    tot = sum(f.total_by_quad.get(n, 0) for f in scan.frames for n in names)
    return float(vis / tot) if tot else None


def reextract_scan(scan: ScanData, roi) -> ScanData:
    """在原图的有效成像窗口内重新提取；保持所有扫描位姿不变。"""
    frames = []
    elapsed = 0.0
    for frame in scan.frames:
        if frame.image_path is None:
            raise ValueError("视场对照需要每帧原图")
        image = ll.load_gray_image(frame.image_path)
        start = time.perf_counter()
        ext = extract_centers(image, roi=roi)
        elapsed += time.perf_counter() - start
        frames.append(ScanFrame(frame.index, frame.pose, ext.uv, 0, len(ext.uv),
                                {}, {}, frame.image_path))
    # 面片可见率不从裁剪后的提取结果推测，保留为空。
    return ScanData(scan.object_name, scan.setup, frames, "image_roi",
                    {"extract": elapsed}, 0, 0)


def save_scan_measurements(folder, scan: ScanData) -> None:
    """保存原始像素及机构位姿；不保存真实表面点云。"""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    counts = [len(f.uv) for f in scan.frames]
    np.savez_compressed(folder / "measurements.npz",
        uv=np.vstack([f.uv for f in scan.frames]) if counts else np.empty((0, 2)),
        offsets=np.r_[0, np.cumsum(counts)],
        pose_R=np.array([f.pose.R for f in scan.frames]),
        pose_t=np.array([f.pose.t for f in scan.frames]),
        frame_ids=np.array([f.index for f in scan.frames], dtype=int))
    write_json(folder / "measurement_config.json", {
        "camera": scan.setup.camera.to_dict(), "object_name": scan.object_name,
        "image_paths": [f.image_path for f in scan.frames],
        "source_label": scan.source_label,
        "note": "pixels and recorded poses only; reconstruction plane supplied separately",
    })


def load_scan_measurements(folder, plane_c: ll.LaserPlane) -> ScanData:
    import json
    folder = Path(folder)
    config = json.loads((folder / "measurement_config.json").read_text(encoding="utf-8"))
    camera = ll.CameraModel.from_dict(config["camera"])
    with np.load(folder / "measurements.npz", allow_pickle=False) as data:
        setup = ScanSetup(camera, data["pose_t"][:, 0], laser_plane_c=plane_c,
                          laser_source_c=-plane_c.unit_d * plane_c.unit_normal)
        frames = [ScanFrame(int(index), ScanPose(int(index), R.copy(), t.copy()),
                    data["uv"][data["offsets"][i]:data["offsets"][i+1]].copy(),
                    0, int(data["offsets"][i+1]-data["offsets"][i]), {}, {},
                    config["image_paths"][i])
                  for i, (index, R, t) in enumerate(zip(data["frame_ids"], data["pose_R"], data["pose_t"]))]
    return ScanData(config["object_name"], setup, frames, config["source_label"], {}, 0, 0)


# ==========================================================================
# 多帧反算：像素 + K/dist + 平面 + 机构位姿
# ==========================================================================
@dataclass
class PointCloudResult:
    points_world: np.ndarray
    frame_indices: np.ndarray
    n_frames: int
    timings_s: dict[str, float]
    measurements: dict = field(default_factory=dict)


def reconstruct_scan(scan: ScanData, plane_c: ll.LaserPlane,
                     pixel_noise_px: float = 0.0,
                     motion_sigma_mm: float = 0.0,
                     omission: float = 0.0,
                     frame_mask: np.ndarray | None = None,
                     pose_errors: np.ndarray | None = None,
                     seed: int = 1) -> PointCloudResult:
    """用测量像素反算，再按机构位姿统一到世界系。

    ``motion_sigma_mm`` 在变换阶段给 t_i 加高斯误差；这是“编码器/定位误差”
    对坐标统一的影响，和图像渲染分开，便于单独定量。
    """
    rng = np.random.default_rng(seed)
    n = len(scan.frames)
    if frame_mask is None:
        frame_mask = np.ones(n, dtype=bool)
    else:
        frame_mask = np.asarray(frame_mask, dtype=bool)
    if pose_errors is None:
        pose_errors = np.zeros((n, 3), dtype=np.float64)
    else:
        pose_errors = np.asarray(pose_errors, dtype=np.float64)
        if pose_errors.shape != (n, 3):
            raise ValueError("pose_errors 必须是 N×3")

    keep = frame_mask.copy()
    if omission > 0:
        keep &= rng.random(n) >= float(omission)

    pts_all = []
    frame_idx = []
    used_uv, used_R, used_t, used_ids = [], [], [], []
    t_rec = t_trans = 0.0
    for i, frame in enumerate(scan.frames):
        if not keep[i] or len(frame.uv) == 0:
            continue
        uv = np.asarray(frame.uv, dtype=np.float64)
        if pixel_noise_px > 0:
            uv = uv + rng.normal(0.0, pixel_noise_px, uv.shape)
        t0 = time.perf_counter()
        rec = ll.reconstruct(uv, scan.setup.camera, plane_c)
        t1 = time.perf_counter()
        t_rec += t1 - t0
        if not rec.valid.any():
            continue
        pc = rec.points_mm[rec.valid]
        t_err = frame.pose.t + pose_errors[i] if motion_sigma_mm == 0 else \
            frame.pose.t + pose_errors[i] + rng.normal(0.0, motion_sigma_mm, 3)
        pw = pc @ frame.pose.R.T + t_err
        t2 = time.perf_counter()
        t_trans += t2 - t1
        pts_all.append(pw)
        frame_idx.append(np.full(len(pw), i, dtype=int))
        used_uv.append(uv.copy())
        used_R.append(frame.pose.R.copy())
        used_t.append(t_err.copy())
        used_ids.append(frame.index)

    measurements = {
        "uv": np.vstack(used_uv) if used_uv else np.empty((0, 2)),
        "offsets": np.r_[0, np.cumsum([len(uv) for uv in used_uv])],
        "pose_R": np.array(used_R).reshape(-1, 3, 3),
        "pose_t": np.array(used_t).reshape(-1, 3),
        "frame_ids": np.array(used_ids, dtype=int),
        "selected_frames": np.flatnonzero(keep), "seed": seed,
        "camera_matrix": scan.setup.camera.K,
        "dist_coeffs": scan.setup.camera.dist,
        "plane_abcd": plane_c.abcd,
    }

    if not pts_all:
        return PointCloudResult(np.empty((0, 3)), np.empty(0, dtype=int), 0,
                                {"reconstruct": t_rec, "transform": t_trans}, measurements)
    return PointCloudResult(np.vstack(pts_all), np.concatenate(frame_idx),
                            int(len(pts_all)), {"reconstruct": t_rec, "transform": t_trans}, measurements)


# ==========================================================================
# 基准面估计 / 高度图 / 体积
# ==========================================================================
@dataclass
class BasePlane:
    normal: np.ndarray
    d: float
    rms_mm: float
    inlier_ratio: float
    n_inliers: int

    def height(self, points) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        return p @ self.normal + self.d


def estimate_base_plane(points_world, rng: np.random.Generator | None = None,
                        threshold_mm: float = 1.0, iterations: int = 300,
                        low_percentile: float = 15.0) -> BasePlane:
    """低分位点寻找初值，完整点云中重新选择地面并细化。"""
    pts = np.asarray(points_world, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 10:
        raise ValueError("基准面估计需要至少 10 个点")
    rng = np.random.default_rng(0) if rng is None else rng
    zcut = np.percentile(pts[:, 2], low_percentile)
    low = pts[pts[:, 2] <= zcut]
    if len(low) < 10:
        low = pts[np.argsort(pts[:, 2])[:max(10, len(pts) // 3)]]

    best = None
    best_count = -1
    for _ in range(int(iterations)):
        idx = rng.choice(len(low), size=3, replace=False)
        p0, p1, p2 = low[idx]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if n[2] < 0:
            n = -n
        d = float(-n @ p0)
        count = int(np.sum(np.abs(low @ n + d) < threshold_mm))
        if count > best_count:
            best_count = count
            best = (n, d)
    if best is None:
        raise ValueError("基准面 RANSAC 失败")
    n, d = best
    residual = pts @ n + d
    # 地面点很少时，两簇中位数迭代可能把两簇都放在顶面。
    # 从完整残差中找有足够支持点的明显层间空隙，保留较低的一层。
    ordered = np.sort(residual)
    gaps = np.diff(ordered)
    split = 9 + int(np.argmax(gaps[9:-9])) if len(ordered) >= 20 else 0
    groups = (ordered[:split+1], ordered[split+1:])
    sigmas = [1.4826 * np.median(np.abs(group - np.median(group)))
              for group in groups if len(group)]
    separated = len(ordered) >= 20 and gaps[split] > max(2 * threshold_mm, 6 * max(sigmas))
    pool = pts[residual <= ordered[split]] if separated else pts
    if len(pool) < 10:
        raise ValueError("基准面支持点不足")
    for _ in range(4):
        res = pool @ n + d
        center = np.median(res)
        sigma = 1.4826 * np.median(np.abs(res - center))
        inliers = np.abs(res - center) <= max(threshold_mm, 4 * sigma)
        refine = pool[inliers]
        if len(refine) < 3:
            raise ValueError("基准面细化支持点不足")
        c = refine.mean(axis=0)
        _, _, vt = np.linalg.svd(refine - c, full_matrices=False)
        n = vt[-1]
        if n[2] < 0:
            n = -n
        d = float(-n @ c)
    res = refine @ n + d
    rms = float(np.sqrt(np.mean(res ** 2)))
    return BasePlane(normal=n, d=d, rms_mm=rms,
                     inlier_ratio=float(len(refine) / len(pts)),
                     n_inliers=int(inliers.sum()))


@dataclass
class HeightMapResult:
    height: np.ndarray
    u_edges: np.ndarray
    v_edges: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    surface_height: np.ndarray   # 去掉 NaN 后的高度图（未填）
    observed: np.ndarray
    coverage: float
    volume_mm3: float
    base: BasePlane
    footprint: tuple[float, float, float, float]


def base_axes(base: BasePlane) -> tuple[np.ndarray, np.ndarray]:
    """基准面内的正交基；u 尽量沿世界 X，v = up × u。"""
    up = base.normal
    ref = X_AXIS - float(X_AXIS @ up) * up
    if np.linalg.norm(ref) < 0.1:
        ref = Y_AXIS - float(Y_AXIS @ up) * up
    u_axis = ref / np.linalg.norm(ref)
    v_axis = np.cross(up, u_axis)
    v_axis /= np.linalg.norm(v_axis)
    return u_axis, v_axis


def known_footprint_in_base(obj: ScanObject, base: BasePlane) -> tuple[float, float, float, float]:
    """把已知物体足迹（世界 XY 的矩形，z=0）投到基准面坐标。"""
    x0, x1, y0, y1 = obj.footprint
    corners = np.array([
        [x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0],
    ])
    u_axis, v_axis = base_axes(base)
    uv = corners @ np.column_stack([u_axis, v_axis])
    return (float(uv[:, 0].min()), float(uv[:, 0].max()),
            float(uv[:, 1].min()), float(uv[:, 1].max()))


def height_map_volume(points_world, base: BasePlane,
                      cell_mm: float = 0.5,
                      min_height_mm: float = 1.0,
                      fill_iterations: int = 3,
                      footprint: tuple[float, float, float, float] | None = None
                      ) -> HeightMapResult:
    """高度图积分：V = sum(h_ij) * du * dv。

    只使用基准面以上的点；格内取最大高度；空格用邻域均值做有限次填充
    （只填扫描线之间的窄缝，不做大范围外推，避免把 FOV/遗漏的空洞
    用邻近顶面“补”成实体）。``footprint`` 给定时使用已知 D；
    否则用物体点的最外框。
    """
    pts = np.asarray(points_world, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 10:
        raise ValueError("高度图需要至少 10 个点")
    h = base.height(pts)
    obj = pts[h > min_height_mm]
    if len(obj) < 10:
        raise ValueError("没有足够的物体表面点（全部低于基准面阈值）")

    u_axis, v_axis = base_axes(base)
    coords = obj @ np.column_stack([u_axis, v_axis])  # N×2
    if footprint is None:
        u0, u1 = float(coords[:, 0].min()), float(coords[:, 0].max())
        v0, v1 = float(coords[:, 1].min()), float(coords[:, 1].max())
    else:
        u0, u1, v0, v1 = (float(x) for x in footprint)
    if u1 - u0 < 1e-6 or v1 - v0 < 1e-6:
        raise ValueError("物体足迹退化")
    nx = max(2, int(np.ceil((u1 - u0) / cell_mm)))
    ny = max(2, int(np.ceil((v1 - v0) / cell_mm)))
    du = (u1 - u0) / nx
    dv = (v1 - v0) / ny
    u_edges = np.linspace(u0, u1, nx + 1)
    v_edges = np.linspace(v0, v1, ny + 1)

    inside = ((coords[:, 0] >= u0) & (coords[:, 0] <= u1) &
              (coords[:, 1] >= v0) & (coords[:, 1] <= v1))
    iu = np.clip(np.floor((coords[:, 0] - u0) / du).astype(int), 0, nx - 1)
    iv = np.clip(np.floor((coords[:, 1] - v0) / dv).astype(int), 0, ny - 1)
    hobj = h[h > min_height_mm]
    hmap = np.full((ny, nx), np.nan, dtype=np.float64)
    for a, b, hh, ok in zip(iv, iu, hobj, inside):
        if not ok:
            continue
        if np.isnan(hmap[a, b]) or hh > hmap[a, b]:
            hmap[a, b] = hh
    observed = np.isfinite(hmap)
    coverage = float(observed.mean())
    filled = np.array(hmap, copy=True)

    for _ in range(int(fill_iterations)):
        if not np.isnan(filled).any():
            break
        pad = np.pad(filled, 1, constant_values=np.nan)
        stack = np.stack([
            pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]
        ])
        cnt = np.sum(np.isfinite(stack), axis=0)
        sums = np.nansum(stack, axis=0)
        neigh = np.divide(sums, cnt, out=np.full(sums.shape, np.nan), where=cnt > 0)
        fill = np.isnan(filled) & np.isfinite(neigh)
        if not fill.any():
            break
        filled[fill] = neigh[fill]
    filled[np.isnan(filled)] = 0.0
    volume = float(np.sum(filled) * du * dv)
    return HeightMapResult(
        height=filled, u_edges=u_edges, v_edges=v_edges, u_axis=u_axis,
        v_axis=v_axis, surface_height=hmap, observed=observed,
        coverage=coverage, volume_mm3=volume, base=base,
        footprint=(u0, u1, v0, v1),
    )


def evaluate_point_cloud(points_world, cell_mm: float = 0.5,
                         rng: np.random.Generator | None = None,
                         footprint: tuple[float, float, float, float] | None = None
                         ) -> tuple[BasePlane, HeightMapResult]:
    base = estimate_base_plane(points_world, rng)
    hm = height_map_volume(points_world, base, cell_mm=cell_mm, footprint=footprint)
    return base, hm


# ==========================================================================
# 解析对照
# ==========================================================================
def true_height_at(object_name: str, x, y):
    """给定世界 X/Y 的真实顶面高度（高度图方法的理想真值）。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    h = np.full(np.broadcast(x, y).shape, 50.0, dtype=np.float64)
    if object_name.lower().startswith("b"):
        in_slot = (x >= 40) & (x <= 60) & (y >= 30) & (y <= 50)
        h = np.where(in_slot, 40.0, 50.0)
    return h


def analytic_height_map_volume(object_name: str, cell_mm: float = 0.1) -> float:
    """在已知足迹上直接积分真实高度场，单位 cm^3；用于无噪声解析对照。"""
    nx = max(1, int(round(100.0 / cell_mm)))
    ny = max(1, int(round(80.0 / cell_mm)))
    dx = 100.0 / nx
    dy = 80.0 / ny
    xc = (np.arange(nx) + 0.5) * dx
    yc = (np.arange(ny) + 0.5) * dy
    xg, yg = np.meshgrid(xc, yc)
    h = true_height_at(object_name, xg, yg)
    return float(h.sum() * dx * dy / 1000.0)


def true_volume_cm3(object_name: str) -> float:
    return make_object(object_name).true_volume_cm3


# ==========================================================================
# 保存
# ==========================================================================
def save_png(path, image) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", np.asarray(image, dtype=np.uint8))
    if not ok:
        raise RuntimeError(f"PNG 编码失败: {path}")
    buf.tofile(str(path))


def save_ply(path, points_world) -> None:
    pts = np.asarray(points_world, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def write_csv(path, rows) -> None:
    path = Path(path)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_json(path, value) -> None:
    ll.write_json(path, ll.clean_json(value))


def base_error(base: BasePlane) -> dict:
    """基准面估计误差（仅评价用）；真实基准是 z=0。"""
    n_true = Z_AXIS
    cos = abs(float(base.normal @ n_true))
    angle = float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))
    # 真实基准面过原点；估计平面到原点的距离 = |d|
    return {"normal_angle_error_deg": angle, "offset_error_mm": abs(float(base.d))}
