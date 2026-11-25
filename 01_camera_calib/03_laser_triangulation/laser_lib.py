"""线激光三角测量核心库（纯 NumPy；只有处理真实图像/镜头畸变时才用 OpenCV）。

坐标与单位约定（和 01_camera_calib 工程保持一致）：

* 相机坐标系：X 向右，Y 向下，Z 沿光轴向前；所有长度单位统一为 **mm**。
* 像素坐标：u 向右，v 向下；整数坐标表示像素中心。
* 相机模型：``[u,v,1]^T ~ K [X,Y,Z]^T``；有畸变时用 OpenCV 的
  ``undistortPointsIter`` 把原始像素换成归一化射线。
* 激光平面：``n · X + d = 0``，n、d 都定义在相机坐标系中。
* 射线：本库使用 ``[x, y, 1]`` 形式（x=(u-cx)/fx, y=(v-cy)/fy），
  于是交点参数 t 恰好等于相机坐标 Z。

这个文件同时包含四部分：

1. 相机模型与激光平面（真实数据与仿真共用）；
2. 像素 -> 三维反算；
3. 台阶高度的稳健估计（自动分层 + 中位数 + MAD 剔除）；
4. 激光线提取、合成激光图像渲染（给真实 webcam 路径预留同一接口）。

``StepScene`` 是仿真用的已知真值场景，算法本身不会读它。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_RAY_EPS = 1e-10
_PLANE_EPS = 1e-12
_MAD_SCALE = 1.4826          # 高斯分布下 MAD -> sigma
_MEDIAN_SE = 1.2533          # 高斯分布下 median 的标准误系数


# ==========================================================================
# 小工具
# ==========================================================================
def _import_cv2():
    """延迟导入 OpenCV，保证纯仿真路径只依赖 NumPy。"""
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise ImportError(
            "这条路径需要 OpenCV：请在全局 Python 中执行 pip install opencv-python"
        ) from exc
    return cv2


def _as_points(points, name: str = "points") -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} 必须是 N×3 数组，当前 shape={arr.shape}")
    return arr


def _median_mad(values: np.ndarray) -> tuple[float, float]:
    """返回 (中位数, 1.4826*MAD)，MAD 是对离群点不敏感的 sigma 估计。"""
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med, _MAD_SCALE * mad


def display_width(text: str) -> int:
    """终端显示宽度：中文/全角字符算 2 格。"""
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def _pad(text: str, width: int, align: str = "<") -> str:
    text = str(text)
    gap = max(0, width - display_width(text))
    return text + " " * gap if align == "<" else " " * gap + text


def format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    aligns: Sequence[str] | None = None,
) -> str:
    """按显示宽度排版的简单文本表格（中文列名不会错位）。"""
    body = [[str(c) for c in row] for row in rows]
    widths = []
    for i, head in enumerate(headers):
        widths.append(max([display_width(head)] + [display_width(row[i]) for row in body]))
    aligns = list(aligns) if aligns is not None else ["<"] + [">"] * (len(headers) - 1)
    if len(aligns) != len(headers):
        raise ValueError("aligns 数量必须和 headers 一致")

    sep = "  "
    lines = [sep.join(_pad(h, widths[i], aligns[i]) for i, h in enumerate(headers))]
    lines.append(sep.join("-" * widths[i] for i in range(len(headers))))
    for row in body:
        lines.append(sep.join(_pad(row[i], widths[i], aligns[i]) for i in range(len(headers))))
    return "\n".join(lines)


def write_json(path, value) -> None:
    """写 JSON；把 NumPy 类型转成原生类型，遇到 NaN 直接报错而不是写出非法 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"不能序列化的类型: {type(obj)!r}")

    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=default)
    path.write_text(text, encoding="utf-8")


def clean_json(value):
    """递归把 NaN/Inf 换成 None，方便写严格的 JSON 报告。"""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


# ==========================================================================
# 相机模型
# ==========================================================================
@dataclass
class CameraModel:
    """针孔相机 + 可选 OpenCV 畸变模型。

    Parameters
    ----------
    K : (3,3) array
        内参矩阵，单位像素。
    dist : array
        OpenCV 畸变系数 (k1,k2,p1,p2,k3,...)；无畸变时给空数组或全 0。
    image_size : (width, height), optional
        图像分辨率。真实图像路径会用它检查分辨率是否与标定一致。
    source : str, optional
        参数来源说明，写进报告/JSON。
    """

    K: np.ndarray
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))
    image_size: tuple[int, int] | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        K = np.asarray(self.K, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"K 必须是 3×3，当前 shape={K.shape}")
        if not np.isfinite(K).all():
            raise ValueError("K 含有 NaN/Inf")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("K 的 fx/fy 必须为正")
        if abs(K[2, 2] - 1.0) > 1e-9:
            raise ValueError("K[2,2] 应为 1（齐次坐标约定）")
        self.K = K.copy()

        dist = np.asarray(self.dist, dtype=np.float64).reshape(-1)
        if dist.size == 0:
            dist = np.zeros(5, dtype=np.float64)
        if not np.isfinite(dist).all():
            raise ValueError("dist 含有 NaN/Inf")
        self.dist = dist.copy()

        if self.image_size is not None:
            w, h = self.image_size
            if int(w) <= 0 or int(h) <= 0:
                raise ValueError("image_size 必须为正")
            self.image_size = (int(w), int(h))

    # ---- 构造/序列化 -----------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict, source: str | None = None) -> "CameraModel":
        if "camera_matrix" in data:
            K = data["camera_matrix"]
        elif "K" in data:
            K = data["K"]
        else:
            raise KeyError("相机参数里找不到 camera_matrix（或 K）")
        dist = data.get("dist_coeffs", data.get("dist", np.zeros(5)))
        size = data.get("image_size")
        return cls(K=K, dist=dist, image_size=tuple(size) if size else None,
                   source=source or data.get("source"))

    @classmethod
    def from_calib_json(cls, path) -> "CameraModel":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"找不到相机标定文件: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, source=str(path))

    @classmethod
    def from_calib_lib_dict(cls, data: dict) -> "CameraModel":
        """兼容 calib_lib.load_calib() 返回的 dict。"""
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return {
            "unit": "mm",
            "frame": "camera: X right, Y down, Z forward",
            "image_size": list(self.image_size) if self.image_size else None,
            "camera_matrix": self.K.tolist(),
            "dist_coeffs": self.dist.ravel().tolist(),
            "source": self.source,
            "created": datetime.now().isoformat(timespec="seconds"),
        }

    # ---- 基本属性 ---------------------------------------------------------
    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def has_distortion(self) -> bool:
        return bool(np.any(np.abs(self.dist) > 1e-12))

    # ---- 投影 / 反投影 ----------------------------------------------------
    def project(self, points_mm, distort: bool = True) -> np.ndarray:
        """三维点 (N,3) -> 像素 (N,2)。相机后方的点返回 NaN。"""
        pts = _as_points(points_mm)
        if distort and self.has_distortion:
            cv2 = _import_cv2()
            uv = cv2.projectPoints(
                pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.dist
            )[0].reshape(-1, 2)
            return uv.astype(np.float64)

        hom = pts @ self.K.T
        z = hom[:, 2]
        uv = np.full((len(pts), 2), np.nan, dtype=np.float64)
        ok = z > 0
        uv[ok] = hom[ok, :2] / z[ok, None]
        return uv

    def normalized_rays(self, uv) -> np.ndarray:
        """像素 (N,2) -> 相机坐标射线 (N,3)，形式 [x, y, 1]。

        有畸变时先做 ``undistortPointsIter``，与 02_measurement_3d/measure3d.py
        的 intersect_pixels() 完全一致。要求 uv 全部有限。
        """
        uv = np.asarray(uv, dtype=np.float64)
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError(f"uv 必须是 N×2，当前 shape={uv.shape}")
        if len(uv) == 0:
            return np.empty((0, 3), dtype=np.float64)
        if not np.isfinite(uv).all():
            raise ValueError("uv 含有 NaN/Inf；请先过滤无效像素")

        if self.has_distortion:
            cv2 = _import_cv2()
            criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
            xy = cv2.undistortPointsIter(
                uv.reshape(-1, 1, 2), self.K, self.dist, None, None, criteria
            ).reshape(-1, 2)
        else:
            hom = np.column_stack([uv, np.ones(len(uv), dtype=np.float64)])
            xy = np.linalg.solve(self.K, hom.T).T[:, :2]

        return np.column_stack([xy, np.ones(len(xy), dtype=np.float64)])

    def pixel_ray(self, u: float, v: float) -> np.ndarray:
        """单个像素的射线，返回长度 3 的数组。"""
        return self.normalized_rays(np.array([[u, v]], dtype=np.float64))[0]


# ==========================================================================
# 激光平面
# ==========================================================================
@dataclass
class LaserPlane:
    """相机坐标系下的平面 ``n·X + d = 0``。

    n 不要求是单位向量；内部求交时统一归一化，因此 [0,1,0.2], -120
    和 [0,0.98058,0.19612], -117.67 表示同一个平面。
    """

    normal: np.ndarray
    d: float
    source: str | None = None

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, dtype=np.float64).reshape(-1)
        if n.size != 3:
            raise ValueError("激光平面法向必须是 3 个数")
        if not np.isfinite(n).all():
            raise ValueError("激光平面法向含有 NaN/Inf")
        if np.linalg.norm(n) < _PLANE_EPS:
            raise ValueError("激光平面法向不能为零向量")
        self.normal = n.copy()
        self.d = float(self.d)
        if not np.isfinite(self.d):
            raise ValueError("激光平面 d 必须是有限数")

    # ---- 构造 -------------------------------------------------------------
    @classmethod
    def from_abcd(cls, abcd, source: str | None = None) -> "LaserPlane":
        """从 [a,b,c,d] 构造，aX+bY+cZ+d=0。"""
        arr = np.asarray(abcd, dtype=np.float64).reshape(-1)
        if arr.size != 4:
            raise ValueError("平面必须是 4 个数 [a,b,c,d]")
        return cls(arr[:3], float(arr[3]), source=source)

    @classmethod
    def from_point_normal(cls, point, normal, source: str | None = None) -> "LaserPlane":
        p = np.asarray(point, dtype=np.float64).reshape(-1)
        n = np.asarray(normal, dtype=np.float64).reshape(-1)
        if p.size != 3 or n.size != 3:
            raise ValueError("point / normal 都必须是 3 个数")
        if not np.isfinite(p).all() or not np.isfinite(n).all():
            raise ValueError("point / normal 含有 NaN/Inf")
        return cls(n, float(-n @ p), source=source)

    @classmethod
    def from_dict(cls, data: dict, source: str | None = None) -> "LaserPlane":
        if "normal" in data and "d" in data:
            return cls(data["normal"], float(data["d"]), source=source or data.get("source"))
        if "unit_normal" in data and "normalized_d" in data:
            return cls(data["unit_normal"], float(data["normalized_d"]),
                       source=source or data.get("source"))
        for key in ("plane_abcd", "plane"):
            if key in data:
                return cls.from_abcd(data[key], source=source or data.get("source"))
        raise KeyError("平面文件里找不到 normal/d、unit_normal/normalized_d 或 plane_abcd")

    @classmethod
    def from_json(cls, path) -> "LaserPlane":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"找不到激光平面文件: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, source=str(path))

    # ---- 属性 -------------------------------------------------------------
    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.normal))

    @property
    def unit_normal(self) -> np.ndarray:
        return self.normal / self.norm

    @property
    def unit_d(self) -> float:
        return self.d / self.norm

    @property
    def abcd(self) -> np.ndarray:
        """归一化后的 [a,b,c,d]，可直接喂给 02_measurement_3d.intersect_pixels。"""
        return np.r_[self.unit_normal, self.unit_d]

    def to_dict(self) -> dict:
        n = self.unit_normal
        return {
            "unit": "mm",
            "frame": "camera: X right, Y down, Z forward",
            "normal": self.normal.tolist(),          # 保留输入形式
            "d": float(self.d),
            "unit_normal": n.tolist(),
            "normalized_d": float(self.unit_d),
            "plane_abcd": [float(n[0]), float(n[1]), float(n[2]), float(self.unit_d)],
            "equation": "n . X + d = 0",
            "source": self.source,
            "created": datetime.now().isoformat(timespec="seconds"),
        }

    # ---- 几何操作 ---------------------------------------------------------
    def distance(self, points_mm) -> np.ndarray:
        """点到平面的有符号距离（mm），正值在法向一侧。"""
        pts = _as_points(points_mm)
        return pts @ self.unit_normal + self.unit_d

    def y_at_xz(self, x, z):
        """已知 X、Z，求平面上的 Y（用于生成仿真台阶）。"""
        ny = self.normal[1]
        if abs(ny) < _PLANE_EPS:
            raise ValueError("该激光平面没有 Y 分量，不能由 X/Z 反求 Y")
        x = np.asarray(x, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        return -(self.normal[0] * x + self.normal[2] * z + self.d) / ny

    def translated(self, offset_mm: float) -> "LaserPlane":
        """沿单位法向平移 offset_mm（正值朝 n 方向）。"""
        if not np.isfinite(offset_mm):
            raise ValueError("offset_mm 必须是有限数")
        return LaserPlane(self.normal, self.d - float(offset_mm) * self.norm,
                          source=f"{self.source or 'plane'} + offset {offset_mm:+.3f} mm")

    def angle_to(self, other: "LaserPlane") -> float:
        """两个平面的夹角（度）。"""
        c = abs(float(self.unit_normal @ other.unit_normal))
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

    # ---- 射线求交 ---------------------------------------------------------
    def intersect(self, rays, eps: float = _RAY_EPS):
        """相机中心出发的射线与平面求交。

        Parameters
        ----------
        rays : (N,3)
            射线方向，不要求单位化。本库用 [x,y,1]，所以返回的 t 就是 Z。

        Returns
        -------
        points : (N,3) 交点；无效行为 NaN。
        valid : (N,) bool
        t : (N,) 射线参数；有效时 >0。
        """
        r = np.asarray(rays, dtype=np.float64)
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError(f"rays 必须是 N×3，当前 shape={r.shape}")
        n = self.unit_normal
        dd = self.unit_d
        denom = r @ n
        valid = np.isfinite(r).all(axis=1) & (np.abs(denom) > eps)
        t = np.full(len(r), np.nan, dtype=np.float64)
        np.divide(-dd, denom, out=t, where=valid)
        valid &= np.isfinite(t) & (t > 0)
        points = r * t[:, None]
        points[~valid] = np.nan
        return points, valid, t


# ==========================================================================
# 像素 -> 三维
# ==========================================================================
@dataclass
class Reconstruction:
    """一次像素反算的结果；数组顺序与输入像素一致。"""

    points_mm: np.ndarray
    valid: np.ndarray
    depth_mm: np.ndarray
    rays: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.points_mm)

    def finite_points(self) -> np.ndarray:
        return self.points_mm[self.valid]

    def valid_uv(self, uv) -> np.ndarray:
        return np.asarray(uv, dtype=np.float64)[self.valid]


def reconstruct(uv, camera: CameraModel, plane: LaserPlane,
                keep_rays: bool = False) -> Reconstruction:
    """像素 -> 三维点。

    允许 uv 中含有 NaN 行（测试 C 的“缺失点”直接置 NaN 即可），
    这些行在结果里 valid=False、坐标为 NaN。
    """
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"uv 必须是 N×2，当前 shape={uv.shape}")
    n = len(uv)
    if n == 0:
        return Reconstruction(np.empty((0, 3)), np.empty(0, dtype=bool),
                              np.empty(0), np.empty((0, 3)) if keep_rays else None)

    finite_rows = np.isfinite(uv).all(axis=1)
    rays = np.full((n, 3), np.nan, dtype=np.float64)
    if finite_rows.any():
        rays[finite_rows] = camera.normalized_rays(uv[finite_rows])

    points, valid, t = plane.intersect(rays)
    valid &= finite_rows
    points[~valid] = np.nan
    t[~valid] = np.nan
    return Reconstruction(points, valid, t, rays if keep_rays else None)


def intersect_pixels(pixels, K, dist, plane, eps: float = _RAY_EPS):
    """兼容 02_measurement_3d 的接口：原始畸变像素 + 相机坐标系平面 -> 点、有效掩码。"""
    camera = CameraModel(K, dist)
    pl = LaserPlane.from_abcd(plane)
    rec = reconstruct(pixels, camera, pl)
    return rec.points_mm, rec.valid


def distance_to_polyline(points, polyline) -> np.ndarray:
    """每个二维点到折线的最短距离（点到线段，避免只比较顶点带来的采样误差）。"""
    p = np.asarray(points, dtype=np.float64)
    q = np.asarray(polyline, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("points 必须是 N×2")
    if q.ndim != 2 or q.shape[1] != 2 or len(q) < 2:
        raise ValueError("polyline 必须是 M×2 且 M>=2")
    best = np.full(len(p), np.inf, dtype=np.float64)
    for i in range(len(q) - 1):
        a = q[i]
        b = q[i + 1]
        d = b - a
        length2 = float(d @ d)
        if length2 < 1e-12:
            dist = np.linalg.norm(p - a, axis=1)
        else:
            t = np.clip(((p - a) @ d) / length2, 0.0, 1.0)
            closest = a + t[:, None] * d
            dist = np.linalg.norm(p - closest, axis=1)
        best = np.minimum(best, dist)
    return best


def rmse_3d(points_a, points_b) -> float:
    """两组 N×3 点的三维 RMSE；自动忽略任一侧为 NaN 的行。"""
    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape 不一致: {a.shape} vs {b.shape}")
    ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if not ok.any():
        return float("nan")
    d = a[ok] - b[ok]
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


# ==========================================================================
# 仿真场景：31 mm 台阶（真值只用于最后算误差）
# ==========================================================================
@dataclass
class StepScene:
    """相机坐标系中的一个两阶台阶，表面分别位于 Z=z_far / Z=z_near。

    X<step_x 是低层（远，Z=z_far），X>=step_x 是上层（近，Z=z_near）。
    台阶高度 = z_far - z_near。台阶侧面（X=step_x）可选，用来让渲染出的
    激光图像更像真实拍摄（真实图像在台阶边缘会有一小段竖直亮线）。
    """

    z_far: float = 500.0
    z_near: float = 469.0
    x_min: float = -120.0
    x_max: float = 120.0
    step_x: float = 0.0
    n_surface: int = 481
    include_wall: bool = True
    wall_samples: int = 41

    def __post_init__(self) -> None:
        if not (self.z_far > self.z_near > 0):
            raise ValueError("要求 z_far > z_near > 0")
        if not (self.x_min < self.step_x < self.x_max):
            raise ValueError("要求 x_min < step_x < x_max")
        if self.n_surface < 4:
            raise ValueError("n_surface 至少为 4")
        if self.wall_samples < 2:
            raise ValueError("wall_samples 至少为 2")

    @property
    def height_mm(self) -> float:
        return float(self.z_far - self.z_near)

    def surface_points(self, plane: LaserPlane):
        """两个水平面上的激光交点，返回 (N,3) 和角色标签（0=远面, 1=近面）。"""
        n_low = max(2, int(self.n_surface) // 2)
        n_high = max(2, int(self.n_surface) - n_low)
        x_low = np.linspace(self.x_min, self.step_x, n_low, endpoint=False)
        x_high = np.linspace(self.step_x, self.x_max, n_high, endpoint=True)
        z_low = np.full_like(x_low, self.z_far, dtype=np.float64)
        z_high = np.full_like(x_high, self.z_near, dtype=np.float64)
        y_low = np.asarray(plane.y_at_xz(x_low, z_low), dtype=np.float64)
        y_high = np.asarray(plane.y_at_xz(x_high, z_high), dtype=np.float64)

        pts = np.vstack([
            np.column_stack([x_low, y_low, z_low]),
            np.column_stack([x_high, y_high, z_high]),
        ])
        roles = np.r_[np.zeros(n_low, dtype=int), np.ones(n_high, dtype=int)]
        return pts, roles

    def wall_points(self, plane: LaserPlane) -> np.ndarray:
        """台阶侧面上的激光交点，从远面走到近面。"""
        z = np.linspace(self.z_far, self.z_near, int(self.wall_samples))
        x = np.full_like(z, self.step_x, dtype=np.float64)
        y = np.asarray(plane.y_at_xz(x, z), dtype=np.float64)
        return np.column_stack([x, y, z])

    def curve_points(self, plane: LaserPlane):
        """按激光线顺序返回整条空间折线：远面 -> 侧面 -> 近面。

        返回 (points, roles)，roles: 0=远面, 1=近面, 2=台阶侧面。
        侧面点只用于渲染/可视化；高度估计不依赖 roles。
        """
        pts_low, roles_low = self.surface_points(plane)
        low_mask = roles_low == 0
        high_mask = roles_low == 1
        pieces = [pts_low[low_mask]]
        roles = [np.zeros(int(low_mask.sum()), dtype=int)]
        if self.include_wall:
            wall = self.wall_points(plane)
            pieces.append(wall)
            roles.append(np.full(len(wall), 2, dtype=int))
        pieces.append(pts_low[high_mask])
        roles.append(np.ones(int(high_mask.sum()), dtype=int))
        return np.vstack(pieces), np.concatenate(roles)

    def to_dict(self) -> dict:
        return {
            "unit": "mm",
            "frame": "camera: X right, Y down, Z forward",
            "z_far_mm": float(self.z_far),
            "z_near_mm": float(self.z_near),
            "height_mm": float(self.height_mm),
            "x_min_mm": float(self.x_min),
            "x_max_mm": float(self.x_max),
            "step_x_mm": float(self.step_x),
            "n_surface": int(self.n_surface),
            "include_wall": bool(self.include_wall),
        }


# ==========================================================================
# 平面拟合（给真实激光平面标定预留）
# ==========================================================================
def plane_from_three_points(p1, p2, p3) -> LaserPlane:
    """三点定平面；三点共线时报错。"""
    a = np.asarray(p1, dtype=np.float64).reshape(3)
    b = np.asarray(p2, dtype=np.float64).reshape(3)
    c = np.asarray(p3, dtype=np.float64).reshape(3)
    n = np.cross(b - a, c - a)
    if np.linalg.norm(n) < 1e-10:
        raise ValueError("三个点共线，无法确定平面")
    return LaserPlane(n, float(-n @ a), source="THREE_POINTS")


def fit_plane_svd(points_mm) -> LaserPlane:
    """用 SVD 拟合平面；返回的平面经过点云质心，法向为最小奇异向量。"""
    pts = _as_points(points_mm)
    if len(pts) < 3:
        raise ValueError("至少需要 3 个点才能拟合平面")
    if not np.isfinite(pts).all():
        raise ValueError("点云含有 NaN/Inf")
    center = pts.mean(axis=0)
    centered = pts - center
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if singular.size < 2 or singular[1] < 1e-12 * max(1.0, singular[0]):
        raise ValueError("点云近似共线，无法稳定拟合平面")
    normal = vt[-1]
    d = float(-normal @ center)
    # 法向符号本身不影响平面，但为了和示例/手写平面一致，尽量让 d<=0
    # （相机原点在平面负侧），报告里读起来更直观。
    if d > 0:
        normal = -normal
        d = -d
    return LaserPlane(normal, d, source="SVD_FIT")


# ==========================================================================
# 台阶高度：自动分层 + 中位数 + MAD 剔除
# ==========================================================================
def two_level_split(values) -> tuple[np.ndarray, float, float]:
    """一维两水平聚类（用中位数迭代，抗离群点）。

    返回 (labels, level_lo, level_hi)，labels 用 0 表示 level_lo、1 表示 level_hi。
    只对有限值计算；调用方应保证输入已经过滤 NaN。
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size == 0:
        return np.empty(0, dtype=int), float("nan"), float("nan")
    if not np.isfinite(x).all():
        raise ValueError("two_level_split 只接受有限值")

    lo, hi = np.percentile(x, [10.0, 90.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros(x.size, dtype=int), float(lo), float(lo)

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        below = x[x <= mid]
        above = x[x > mid]
        if below.size == 0 or above.size == 0:
            break
        new_lo = float(np.median(below))
        new_hi = float(np.median(above))
        if abs(new_lo - lo) < 1e-12 and abs(new_hi - hi) < 1e-12:
            lo, hi = new_lo, new_hi
            break
        lo, hi = new_lo, new_hi

    labels = (x > 0.5 * (lo + hi)).astype(int)
    return labels, float(lo), float(hi)


@dataclass
class StepEstimate:
    """一次台阶高度估计的结果。"""

    height_mm: float
    level_hi_mm: float          # 坐标较大的那一层
    level_lo_mm: float          # 坐标较小的那一层
    separation_mm: float
    has_step: bool
    labels: np.ndarray          # 0=lo, 1=hi, -1=非有限点
    inlier_mask: np.ndarray
    n_inliers: int
    n_outliers: int
    sigma_lo_mm: float
    sigma_hi_mm: float
    height_se_mm: float

    def summary(self) -> dict:
        return {
            "height_mm": self.height_mm,
            "level_hi_mm": self.level_hi_mm,
            "level_lo_mm": self.level_lo_mm,
            "separation_mm": self.separation_mm,
            "has_step": self.has_step,
            "n_inliers": self.n_inliers,
            "n_outliers": self.n_outliers,
            "sigma_lo_mm": self.sigma_lo_mm,
            "sigma_hi_mm": self.sigma_hi_mm,
            "height_se_mm": self.height_se_mm,
        }


def _empty_step_estimate(n: int, has_step: bool = False) -> StepEstimate:
    return StepEstimate(
        height_mm=float("nan"), level_hi_mm=float("nan"), level_lo_mm=float("nan"),
        separation_mm=float("nan"), has_step=has_step,
        labels=np.full(n, -1, dtype=int), inlier_mask=np.zeros(n, dtype=bool),
        n_inliers=0, n_outliers=n, sigma_lo_mm=float("nan"),
        sigma_hi_mm=float("nan"), height_se_mm=float("nan"),
    )


def robust_step_height(
    points_mm,
    axis: int = 2,
    min_separation_mm: float = 1.0,
    core_fraction: float = 0.30,
    reject_k: float = 4.0,
) -> StepEstimate:
    """从三维点中稳健估计台阶高度。

    步骤：
      1. 沿 ``axis`` 做一维两水平聚类；
      2. 用台阶间距和两簇 MAD 判断是否真的存在台阶（避免平面被误判成台阶）；
      3. 只保留离各自层中心 ``core_fraction*sep`` 以内的点（去掉台阶侧面/过渡点）；
      4. 每层再做 ``reject_k*MAD`` 剔除；
      5. 用两层的**中位数之差**作为高度，并给出中位数的标准误。

    这个函数不需要真值标签，真实图像路径可以直接用。
    """
    pts = _as_points(points_mm, "points_mm")
    if axis not in (0, 1, 2):
        raise ValueError("axis 必须是 0/1/2")
    n_total = len(pts)
    finite = np.isfinite(pts).all(axis=1)
    idx = np.flatnonzero(finite)
    if len(idx) < 4:
        return _empty_step_estimate(n_total)

    x = pts[idx, axis]
    labels, lo, hi = two_level_split(x)
    sep = float(hi - lo)

    # 用两簇的稳健 sigma 判断“这是台阶还是同一平面上的噪声”
    sigmas = []
    for k in (0, 1):
        sel = labels == k
        if sel.sum() >= 2:
            _, sigma = _median_mad(x[sel])
        else:
            sigma = 0.0
        sigmas.append(0.0 if not np.isfinite(sigma) else sigma)
    sigma_pooled = 0.5 * (sigmas[0] + sigmas[1])
    threshold = max(float(min_separation_mm), 6.0 * sigma_pooled)
    if not np.isfinite(sep) or sep < threshold:
        labels_full = np.full(n_total, -1, dtype=int)
        labels_full[idx] = labels.astype(int)
        return StepEstimate(
            height_mm=0.0, level_hi_mm=float(hi), level_lo_mm=float(lo),
            separation_mm=float(sep), has_step=False,
            labels=labels_full,
            inlier_mask=finite.copy(),
            n_inliers=int(finite.sum()), n_outliers=int(n_total - finite.sum()),
            sigma_lo_mm=float(sigmas[0]), sigma_hi_mm=float(sigmas[1]),
            height_se_mm=float("nan"),
        )

    centers = np.array([lo, hi], dtype=np.float64)
    dist_to_center = np.abs(x - centers[labels])
    core = dist_to_center <= float(core_fraction) * sep

    # 每层在 core 点上学 median / MAD，再按 MAD 剔除
    keep = core.copy()
    med_arr = np.empty(2, dtype=np.float64)
    sigma_arr = np.empty(2, dtype=np.float64)
    for k in (0, 1):
        sel_cluster = labels == k
        sel_core = core & sel_cluster
        sel = sel_core if int(sel_core.sum()) >= 3 else sel_cluster
        if int(sel.sum()) == 0:
            med_arr[k] = centers[k]
            sigma_arr[k] = 0.0
            continue
        med, sigma = _median_mad(x[sel])
        med_arr[k] = med
        sigma_arr[k] = 0.0 if not np.isfinite(sigma) else sigma
        if int(sel.sum()) >= 3:
            keep &= ~(sel_cluster & (np.abs(x - med) > float(reject_k) * max(sigma, 1e-12)))

    if int(keep.sum()) < 2:
        keep = core if int(core.sum()) >= 2 else np.ones(len(x), dtype=bool)

    lo_vals = x[keep & (labels == 0)]
    hi_vals = x[keep & (labels == 1)]
    if len(lo_vals) == 0 or len(hi_vals) == 0:
        return _empty_step_estimate(n_total)

    med_lo, sig_lo = _median_mad(lo_vals)
    med_hi, sig_hi = _median_mad(hi_vals)
    height = abs(float(med_hi - med_lo))
    se = _MEDIAN_SE * np.sqrt(
        (sig_lo ** 2) / max(1, len(lo_vals)) + (sig_hi ** 2) / max(1, len(hi_vals))
    )

    labels_full = np.full(n_total, -1, dtype=int)
    labels_full[idx] = labels
    inlier_full = np.zeros(n_total, dtype=bool)
    inlier_full[idx[keep]] = True

    return StepEstimate(
        height_mm=height,
        level_hi_mm=float(med_hi),
        level_lo_mm=float(med_lo),
        separation_mm=sep,
        has_step=True,
        labels=labels_full,
        inlier_mask=inlier_full,
        n_inliers=int(inlier_full.sum()),
        n_outliers=int(n_total - inlier_full.sum()),
        sigma_lo_mm=float(sig_lo),
        sigma_hi_mm=float(sig_hi),
        height_se_mm=float(se) if np.isfinite(se) else float("nan"),
    )


# ==========================================================================
# 激光线提取
# ==========================================================================
@dataclass
class LaserExtraction:
    """逐列/逐行提取激光条纹中心的结果。"""

    uv: np.ndarray               # 有效中心点 (N,2)
    scan: np.ndarray             # 扫描坐标（column 模式是 u，row 模式是 v）
    center: np.ndarray           # 中心坐标（column 模式是 v，row 模式是 u）；无效为 NaN
    valid: np.ndarray
    peak: np.ndarray
    snr: np.ndarray
    width: np.ndarray
    axis: str
    roi: tuple[int, int, int, int]
    image_shape: tuple[int, int]

    def __len__(self) -> int:
        return len(self.uv)

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.mean()) if self.valid.size else 0.0

    def summary(self) -> dict:
        valid = self.valid
        return {
            "axis": self.axis,
            "roi_xy": list(self.roi),
            "scan_count": int(len(self.scan)),
            "valid_count": int(valid.sum()),
            "valid_fraction": self.valid_fraction,
            "snr_median": float(np.nanmedian(self.snr[valid])) if valid.any() else None,
            "width_median_px": float(np.nanmedian(self.width[valid])) if valid.any() else None,
            "center_min": float(np.nanmin(self.center)) if valid.any() else None,
            "center_max": float(np.nanmax(self.center)) if valid.any() else None,
        }

    def to_csv(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["u", "v", "axis", "peak", "snr", "width_px"])
            for (u, v), peak, snr, width in zip(self.uv, self.peak[self.valid],
                                                 self.snr[self.valid],
                                                 self.width[self.valid]):
                writer.writerow([f"{u:.6f}", f"{v:.6f}", self.axis,
                                 f"{peak:.3f}", f"{snr:.3f}", f"{width:.4f}"])


def _gaussian_kernel(sigma: float) -> np.ndarray | None:
    if sigma is None or sigma <= 0:
        return None
    radius = max(1, int(np.ceil(4.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _smooth_along_axis(arr: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    kernel = _gaussian_kernel(sigma)
    if kernel is None:
        return arr
    out = np.empty_like(arr, dtype=np.float64)
    if axis == 0:
        for j in range(arr.shape[1]):
            out[:, j] = np.convolve(arr[:, j], kernel, mode="same")
    elif axis == 1:
        for i in range(arr.shape[0]):
            out[i, :] = np.convolve(arr[i, :], kernel, mode="same")
    else:
        raise ValueError("axis 必须是 0 或 1")
    return out


def _auto_axis(sub: np.ndarray) -> str:
    """根据亮像素的主方向决定逐列还是逐行扫描。"""
    p99 = float(np.percentile(sub, 99.0))
    p50 = float(np.percentile(sub, 50.0))
    if p99 - p50 < 5.0:
        return "column"
    ys, xs = np.nonzero(sub >= p99)
    if xs.size < 10:
        return "column"
    x = xs.astype(np.float64) - xs.mean()
    y = ys.astype(np.float64) - ys.mean()
    cov = np.cov(np.vstack([x, y]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, int(np.argmax(eigvals))]
    return "column" if abs(v[0]) >= abs(v[1]) else "row"


def extract_laser_centers(
    gray,
    axis: str = "auto",
    roi: tuple[int, int, int, int] | None = None,
    smooth_sigma: float = 1.0,
    threshold_sigma: float = 4.0,
    min_peak: float = 8.0,
    min_peak_fraction: float = 0.10,
    centroid_radius: int = 4,
    max_width: float = 9.0,
    background_percentile: float = 20.0,
) -> LaserExtraction:
    """从灰度图中提取激光条纹的亚像素中心。

    对每一列（column）或每一行（row）：

    1. 用该列/行背景灰度的低分位数做背景扣除；
    2. 沿扫描方向做一维高斯平滑（不展宽条纹截面）；
    3. 找峰值，峰不够高 / 贴边 / 太宽则标记无效；
    4. 在峰值附近 ±centroid_radius 内做强度加权质心，得到亚像素中心；
    5. 再用全图条纹峰值的高分位数做一次相对阈值过滤，去掉背景噪声
       造成的孤立假峰（真实条纹即使局部变暗，通常也不会低于峰值的 10%）。

    该函数既可用于合成图，也可直接用于 webcam 采到的真实灰度图；
    真实图通常需要调 ``min_peak``、``threshold_sigma``、``max_width``。
    """
    img = np.asarray(gray, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"gray 必须是二维灰度图，当前 shape={img.shape}")
    if not np.isfinite(img).all():
        raise ValueError("gray 含有 NaN/Inf")
    h, w = img.shape

    if roi is None:
        x0, x1, y0, y1 = 0, w, 0, h
    else:
        if len(roi) != 4:
            raise ValueError("roi 必须是 (x0, x1, y0, y1)")
        x0, x1, y0, y1 = (int(v) for v in roi)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(f"ROI 超出图像范围: {(x0, x1, y0, y1)}，图像 {w}x{h}")
    if centroid_radius < 1:
        raise ValueError("centroid_radius 至少为 1")

    sub = img[y0:y1, x0:x1]
    if axis == "auto":
        axis = _auto_axis(sub)
    if axis not in ("column", "row"):
        raise ValueError("axis 必须是 auto/column/row")

    if axis == "column":
        bg = np.percentile(sub, background_percentile, axis=0)
        prof = sub - bg[None, :]
        prof = _smooth_along_axis(prof, smooth_sigma, axis=1)
        n_scan = prof.shape[1]
        n_center = prof.shape[0]
        scan = np.arange(x0, x1, dtype=np.float64)
    else:
        bg = np.percentile(sub, background_percentile, axis=1)
        prof = sub - bg[:, None]
        prof = _smooth_along_axis(prof, smooth_sigma, axis=0)
        n_scan = prof.shape[0]
        n_center = prof.shape[1]
        scan = np.arange(y0, y1, dtype=np.float64)

    center = np.full(n_scan, np.nan, dtype=np.float64)
    peak = np.full(n_scan, np.nan, dtype=np.float64)
    snr = np.full(n_scan, np.nan, dtype=np.float64)
    width = np.full(n_scan, np.nan, dtype=np.float64)

    for j in range(n_scan):
        p = prof[:, j] if axis == "column" else prof[j, :]
        med = float(np.median(p))
        noise = _MAD_SCALE * float(np.median(np.abs(p - med)))
        thr = max(float(min_peak), float(threshold_sigma) * noise)
        k = int(np.argmax(p))
        pk = float(p[k])
        peak[j] = pk
        snr[j] = pk / max(noise, 1e-12)
        if pk < thr or k <= 0 or k >= n_center - 1:
            continue

        lo = max(0, k - centroid_radius)
        hi = min(n_center, k + centroid_radius + 1)
        weights = p[lo:hi] - 0.10 * pk
        weights = np.clip(weights, 0.0, None)
        total = float(weights.sum())
        if total <= 1e-12:
            continue
        coords = np.arange(lo, hi, dtype=np.float64)
        c = float((weights * coords).sum() / total)
        wd = float(np.sqrt((weights * (coords - c) ** 2).sum() / total))
        if wd > float(max_width):
            continue
        center[j] = c
        width[j] = wd

    valid = np.isfinite(center)
    if valid.any() and min_peak_fraction > 0:
        reference = float(np.nanpercentile(peak[valid], 95.0))
        valid &= peak >= max(float(min_peak), float(min_peak_fraction) * reference)
    center[~valid] = np.nan
    # 扫描坐标已经是全图坐标；中心坐标是在 ROI 子图里算的，要加回 ROI 原点。
    if axis == "column":
        center[valid] += y0
        uv = np.column_stack([scan[valid], center[valid]])
    else:
        center[valid] += x0
        uv = np.column_stack([center[valid], scan[valid]])

    return LaserExtraction(
        uv=uv, scan=scan, center=center, valid=valid, peak=peak, snr=snr,
        width=width, axis=axis, roi=(x0, x1, y0, y1), image_shape=(h, w),
    )


# ==========================================================================
# 合成激光图像渲染（纯 NumPy，用于验证“图像 -> 提取 -> 三维”整条链路）
# ==========================================================================
def _mask_any(layer: np.ndarray, x: float, y: float, radius: float) -> bool:
    h, w = layer.shape
    x0 = max(0, int(np.floor(x - radius)))
    x1 = min(w, int(np.ceil(x + radius + 1)))
    y0 = max(0, int(np.floor(y - radius)))
    y1 = min(h, int(np.ceil(y + radius + 1)))
    if x0 >= x1 or y0 >= y1:
        return False
    return bool(np.any(layer[y0:y1, x0:x1] > 1.0))


def render_laser_image(
    polyline_uv,
    image_size: tuple[int, int],
    sigma_px: float = 1.3,
    peak: float = 200.0,
    amplitude=None,
    background: float = 8.0,
    gradient: tuple[float, float] = (5.0, 3.0),
    read_sigma: float = 2.0,
    shot_gain: float = 0.3,
    dropout_spans: Iterable[tuple[float, float]] = (),
    hotspot_count: int = 0,
    hotspot_amp: tuple[float, float] = (120.0, 240.0),
    hotspot_sigma: tuple[float, float] = (1.5, 3.5),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """把一条像素折线渲染成带激光条纹、背景和噪声的灰度图（float, 0~255）。

    Parameters
    ----------
    polyline_uv : (N,2)
        有序的激光线像素坐标（由真值三维点投影得到）。
    image_size : (width, height)
        输出图像尺寸。
    sigma_px : float
        激光条纹的高斯截面宽度（像素）。
    peak : float
        条纹峰值亮度（灰度）。
    amplitude : (N,), optional
        沿线亮度包络（相对值），用于模拟激光功率不均匀/端部衰减。
    dropout_spans : Iterable[(a,b)]
        在扫描方向上的缺失区间（像素坐标）；column 模式是 u 区间。
    hotspot_count : int
        随机亮点（反光/杂散光）数量；会尽量避开激光线本身。
    """
    w, h = int(image_size[0]), int(image_size[1])
    if w <= 0 or h <= 0:
        raise ValueError("image_size 必须为正")
    pts = np.asarray(polyline_uv, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError("polyline_uv 必须是 N×2 且 N>=2")
    if not np.isfinite(pts).all():
        raise ValueError("polyline_uv 含有 NaN/Inf")
    if sigma_px <= 0 or peak <= 0:
        raise ValueError("sigma_px / peak 必须为正")

    rng = np.random.default_rng() if rng is None else rng
    amp = np.ones(len(pts), dtype=np.float64) if amplitude is None else np.asarray(amplitude, dtype=np.float64).ravel()
    if amp.size != len(pts):
        raise ValueError("amplitude 长度必须和 polyline_uv 一致")
    if not np.isfinite(amp).all() or amp.min() < 0:
        raise ValueError("amplitude 必须是非负有限值")
    if amp.max() <= 0:
        raise ValueError("amplitude 不能全为 0")
    amp = peak * amp / amp.max()

    layer = np.zeros((h, w), dtype=np.float64)
    radius = max(1, int(np.ceil(4.0 * sigma_px)))
    for i in range(len(pts) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        d = p1 - p0
        length2 = float(d @ d)
        if length2 < 1e-12:
            continue
        xmin = max(0, int(np.floor(min(p0[0], p1[0]) - radius)))
        xmax = min(w - 1, int(np.ceil(max(p0[0], p1[0]) + radius)))
        ymin = max(0, int(np.floor(min(p0[1], p1[1]) - radius)))
        ymax = min(h - 1, int(np.ceil(max(p0[1], p1[1]) + radius)))
        if xmin > xmax or ymin > ymax:
            continue
        ys, xs = np.mgrid[ymin:ymax + 1, xmin:xmax + 1]
        px = xs - p0[0]
        py = ys - p0[1]
        t = np.clip((px * d[0] + py * d[1]) / length2, 0.0, 1.0)
        qx = px - t * d[0]
        qy = py - t * d[1]
        dist2 = qx * qx + qy * qy
        a = 0.5 * (amp[i] + amp[i + 1])
        val = a * np.exp(-dist2 / (2.0 * sigma_px * sigma_px))
        view = layer[ymin:ymax + 1, xmin:xmax + 1]
        np.maximum(view, val, out=view)

    scan_is_x = float(np.ptp(pts[:, 0])) >= float(np.ptp(pts[:, 1]))
    for span in dropout_spans:
        if len(span) != 2:
            raise ValueError("dropout_spans 的每一项必须是 (a,b)")
        a, b = sorted(float(v) for v in span)
        if scan_is_x:
            i0 = max(0, int(np.floor(a)))
            i1 = min(w, int(np.ceil(b)))
            layer[:, i0:i1] = 0.0
        else:
            i0 = max(0, int(np.floor(a)))
            i1 = min(h, int(np.ceil(b)))
            layer[i0:i1, :] = 0.0

    yy, xx = np.mgrid[0:h, 0:w]
    bg = (float(background)
          + float(gradient[0]) * (xx / max(w - 1, 1))
          + float(gradient[1]) * (yy / max(h - 1, 1)))
    img = bg + layer

    for _ in range(int(hotspot_count)):
        x = y = None
        for _attempt in range(50):
            cx = float(rng.uniform(radius, max(radius + 1, w - radius)))
            cy = float(rng.uniform(radius, max(radius + 1, h - radius)))
            if not _mask_any(layer, cx, cy, radius + 2.0):
                x, y = cx, cy
                break
        if x is None:
            continue
        s = float(rng.uniform(*hotspot_sigma))
        a = float(rng.uniform(*hotspot_amp))
        x0 = max(0, int(np.floor(x - 4 * s)))
        x1 = min(w, int(np.ceil(x + 4 * s + 1)))
        y0 = max(0, int(np.floor(y - 4 * s)))
        y1 = min(h, int(np.ceil(y + 4 * s + 1)))
        ys, xs = np.mgrid[y0:y1, x0:x1]
        img[y0:y1, x0:x1] += a * np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * s * s))

    if read_sigma > 0:
        img = img + rng.normal(0.0, float(read_sigma), img.shape)
    if shot_gain > 0:
        img = img + rng.normal(0.0, np.sqrt(np.clip(img, 0, None)) * float(shot_gain), img.shape)
    return np.clip(img, 0.0, 255.0)


def to_uint8(img) -> np.ndarray:
    """float 灰度图 -> uint8（四舍五入后截断）。"""
    return np.clip(np.rint(np.asarray(img, dtype=np.float64)), 0, 255).astype(np.uint8)


def load_gray_image(path) -> np.ndarray:
    """读灰度图；优先 OpenCV，失败时退回 Pillow。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到图像: {path}")
    try:
        cv2 = _import_cv2()
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("读图需要 opencv-python 或 Pillow") from exc
    return np.asarray(Image.open(path).convert("L"))
