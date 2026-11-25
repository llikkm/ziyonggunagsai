"""共用函数：棋盘格检测、相机标定、平面（标定板所在平面）测量。

所有长度单位统一为毫米 (mm)，长度单位换算是本实验最容易出错的地方，
因此这里只用一个单位，避免 m / mm 混用。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# 亚像素迭代：最多 100 次或位移小于 1e-6 px
SUB_PIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

# 默认标定板：9×6 个方格，即 8×5 个内部角点，格边长 20 mm
DEFAULT_COLS = 9
DEFAULT_ROWS = 6
DEFAULT_SQUARE_MM = 20.0


# --------------------------------------------------------------------------
# 基本工具
# --------------------------------------------------------------------------
def inner_corners(cols: int = DEFAULT_COLS, rows: int = DEFAULT_ROWS) -> tuple[int, int]:
    """方格数 -> 内部角点数。OpenCV 只认内部角点，9×6 格对应 8×5 角点。"""
    return cols - 1, rows - 1


def object_points(pattern: tuple[int, int], square_mm: float) -> np.ndarray:
    """角点在标定板坐标系下的三维坐标 (N,3)，单位 mm，z=0。

    顺序与 findChessboardCorners 的输出一致：逐行、从左到右。
    """
    cols, rows = pattern
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square_mm)
    return pts


def list_images(folder) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def load_calib(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    d["camera_matrix"] = np.array(d["camera_matrix"], dtype=np.float64)
    d["dist_coeffs"] = np.array(d["dist_coeffs"], dtype=np.float64).reshape(1, -1)
    return d


def save_calib(path, data: dict) -> None:
    out = dict(data)
    out["camera_matrix"] = np.asarray(data["camera_matrix"]).tolist()
    out["dist_coeffs"] = np.asarray(data["dist_coeffs"]).ravel().tolist()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------
# 角点检测
# --------------------------------------------------------------------------
def find_corners(
    gray: np.ndarray,
    pattern: tuple[int, int],
    use_sb: bool = True,
    subpix: bool = True,
) -> tuple[bool, np.ndarray | None]:
    """在灰度图上找棋盘格内部角点。

    优先用 findChessboardCornersSB（OpenCV 4.6+ 自带，抗噪、自带亚像素），
    失败再退回经典的 findChessboardCorners + cornerSubPix。
    """
    found, corners = False, None

    if use_sb and hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)

    if not found:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
        if found and subpix:
            # 亚像素级精定位：在角点周围窗口内做迭代最优化
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), SUB_PIX_CRITERIA
            )

    return found, (corners if found else None)


# --------------------------------------------------------------------------
# 标定
# --------------------------------------------------------------------------
def calibrate_from_images(
    paths,
    pattern: tuple[int, int],
    square_mm: float,
    model: str = "standard",
    exclude: set[str] | None = None,
    verbose: bool = False,
) -> dict:
    """对一组标定图做内参标定。

    返回 dict：camera_matrix / dist_coeffs / rms / per_view / image_size ...
    """
    exclude = exclude or set()
    objp = object_points(pattern, square_mm)

    obj_all, img_all, used, skipped = [], [], [], []
    size = None

    for p in paths:
        p = Path(p)
        if p.name in exclude:
            skipped.append((p.name, "按 --exclude 排除"))
            continue
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            skipped.append((p.name, "读不了（文件损坏或不是图片）"))
            continue
        if size is None:
            size = (gray.shape[1], gray.shape[0])
        elif (gray.shape[1], gray.shape[0]) != size:
            skipped.append((p.name, f"分辨率 {gray.shape[1]}x{gray.shape[0]} 与首张 {size[0]}x{size[1]} 不一致"))
            continue

        ok, corners = find_corners(gray, pattern)
        if not ok:
            skipped.append((p.name, "未检出角点（模糊/过曝/棋盘不全/格数不对）"))
            continue

        obj_all.append(objp)
        img_all.append(corners)
        used.append(p)
        if verbose:
            print(f"  ✓ {p.name}")

    if len(used) < 6:
        raise RuntimeError(
            f"只有 {len(used)} 张图成功检出角点，不足以标定（至少 6 张，建议 15~25 张）。"
        )

    flags = 0
    if model == "rational":
        flags |= cv2.CALIB_RATIONAL_MODEL  # k4,k5,k6，广角/大畸变更稳

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_all, img_all, size, None, None, flags=flags
    )

    # 每张图的重投影误差（剔除坏图的主要依据）
    per_view = []
    for p, objp_i, imgp_i, rvec, tvec in zip(used, obj_all, img_all, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp_i, rvec, tvec, K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - imgp_i.reshape(-1, 2), axis=1)
        per_view.append(
            {
                "image": p.name,
                "rms_px": float(np.sqrt(np.mean(err ** 2))),
                "mean_px": float(np.mean(err)),
                "max_px": float(np.max(err)),
            }
        )

    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "image_size": [int(size[0]), int(size[1])],
        "pattern": {
            "inner_cols": pattern[0],
            "inner_rows": pattern[1],
            "squares_cols": pattern[0] + 1,
            "squares_rows": pattern[1] + 1,
            "square_mm": float(square_mm),
            "object_points_order": "逐行、从左到右",
        },
        "model": model,
        "rms_px": float(rms),
        "camera_matrix": K,
        "dist_coeffs": dist,
        "n_views": len(used),
        "per_view": sorted(per_view, key=lambda d: d["rms_px"], reverse=True),
        "skipped": [{"image": n, "reason": r} for n, r in skipped],
    }


# --------------------------------------------------------------------------
# 位姿与平面测量
# --------------------------------------------------------------------------
def solve_pose(corners: np.ndarray, pattern, square_mm: float, K, dist):
    """由一张图的角点求标定板位姿（板坐标系 -> 相机坐标系）。"""
    objp = object_points(pattern, square_mm)
    ok, rvec, tvec = cv2.solvePnP(
        objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None, None, None
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
    return rvec, tvec, float(np.sqrt(np.mean(err ** 2)))


def plane_homography(corners: np.ndarray, pattern, square_mm: float) -> np.ndarray:
    """板平面(mm) -> 图像像素 的单应矩阵 H，满足 [u,v,1] ~ H·[X,Y,1]。"""
    objp = object_points(pattern, square_mm)
    H, _ = cv2.findHomography(objp[:, :2].astype(np.float64), corners.reshape(-1, 2).astype(np.float64), 0)
    if H is None:
        H, _ = cv2.findHomography(
            objp[:, :2].astype(np.float64), corners.reshape(-1, 2).astype(np.float64),
            cv2.RANSAC, 2.0,
        )
    return H


def pixel_to_plane(H: np.ndarray, points) -> np.ndarray:
    """像素坐标 -> 板上毫米坐标（H 必须来自去畸变后的图像）。"""
    pts = np.array(points, dtype=np.float64).reshape(-1, 1, 2)
    Hi = np.linalg.inv(H)
    return cv2.perspectiveTransform(pts, Hi).reshape(-1, 2)


def plane_to_pixel(H: np.ndarray, points_mm) -> np.ndarray:
    pts = np.array(points_mm, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def distance_on_plane(H: np.ndarray, p1, p2) -> float:
    """两个像素点之间的距离，换算到标定板平面上，单位 mm。"""
    a, b = pixel_to_plane(H, [p1, p2])
    return float(np.linalg.norm(a - b))


def undistort_image(img: np.ndarray, K, dist, alpha: float = 0.0):
    """去畸变，并裁掉黑边。

    alpha=0 -> 裁到最大无黑边矩形（推荐，测量时画面干净）
    alpha=1 -> 保留全部像素（含黑边）

    返回 (去畸变图, K_adj)：K_adj 是对"裁剪后图像"的内参——
    裁剪使主点平移，必须同步修正 cx/cy，否则位姿/测距会带系统性偏差。
    """
    h, w = img.shape[:2]
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))
    und = cv2.undistort(img, K, dist, None, K_new)

    x, y, ww, hh = roi
    K_adj = np.array(K_new, dtype=np.float64, copy=True)
    if alpha == 0.0 and ww > 0 and hh > 0 and (ww < w or hh < h):
        und = und[y : y + hh, x : x + ww]
        K_adj[0, 2] -= x
        K_adj[1, 2] -= y
    return und, K_adj


def board_pose_summary(rvec, tvec):
    """板中心到相机的距离，以及板面法线与光轴的夹角（倾斜角）。"""
    R, _ = cv2.Rodrigues(rvec)
    normal = R[:, 2]
    tilt = float(np.degrees(np.arccos(np.clip(abs(normal[2]), -1.0, 1.0))))
    return {"distance_mm": float(np.linalg.norm(tvec)), "tilt_deg": tilt}


# --------------------------------------------------------------------------
# 文字报告
# --------------------------------------------------------------------------
def disp_width(s: str) -> int:
    """终端/文本里的显示宽度，中文算 2 格。"""
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def ljust(s: str, width: int) -> str:
    return s + " " * max(0, width - disp_width(s))


def rjust(s: str, width: int) -> str:
    return " " * max(0, width - disp_width(s)) + s


def format_calib_report(data: dict) -> str:
    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["dist_coeffs"], dtype=np.float64).ravel()
    w, h = data["image_size"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    hfov = 2 * np.degrees(np.arctan(w / (2 * fx)))
    vfov = 2 * np.degrees(np.arctan(h / (2 * fy)))

    lines = []
    lines.append("=" * 66)
    lines.append("相机内参标定结果")
    lines.append("=" * 66)
    lines.append(f"标定时间       : {data.get('created', '-')}")
    lines.append(f"图像分辨率     : {w} x {h}")
    pat = data["pattern"]
    sq_cols = pat.get("squares_cols", pat["inner_cols"] + 1)
    sq_rows = pat.get("squares_rows", pat["inner_rows"] + 1)
    lines.append(
        f"标定板         : {sq_cols}x{sq_rows} 格 -> "
        f"{pat['inner_cols']}x{pat['inner_rows']} 内部角点, 格边长 {pat['square_mm']} mm"
    )
    lines.append(f"参与标定图数   : {data['n_views']}   (畸变模型: {data.get('model')})")
    lines.append("")
    lines.append("内参矩阵 K (像素):")
    lines.append(f"  [ {fx:10.3f}   0        {cx:9.3f} ]")
    lines.append(f"  [ 0        {fy:10.3f}   {cy:9.3f} ]")
    lines.append(f"  [ 0         0          1        ]")
    lines.append("")
    names = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
    lines.append("畸变系数: " + ", ".join(f"{n}={v:+.6f}" for n, v in zip(names, dist)))
    lines.append("")
    lines.append(f"垂直/水平视场角: {hfov:.2f}° x {vfov:.2f}°  (由 fx,fy 反推)")
    lines.append(f"重投影 RMS 误差: {data['rms_px']:.4f} px")
    lines.append("")
    lines.append("质量判据:")
    rms = data["rms_px"]
    if rms < 0.5:
        lines.append("  ✓ RMS < 0.5 px，标定质量良好")
    elif rms < 1.0:
        lines.append("  ~ RMS 在 0.5~1.0 px，可用；若测量精度要求高，建议重拍更清晰的图")
    elif rms < 3.0:
        lines.append("  ✗ RMS > 1.0 px，偏大：检查是否模糊、标定板不平、某几张姿态异常")
    else:
        lines.append("  ✗✗ RMS > 3 px，数据本身有问题：角点是不是贴错格、"
                     "板子是否翘曲、有没有混进另一块板的图？")
    d_pp = abs(cx - w / 2) / w
    if d_pp < 0.05:
        lines.append(f"  ✓ 主点 cx={cx:.1f} 接近图像中心 {w/2:.1f}（偏差 {d_pp*100:.1f}%）")
    else:
        lines.append(f"  ~ 主点 cx={cx:.1f} 偏离图像中心 {w/2:.1f} 达 {d_pp*100:.1f}%，留意是否有系统误差")
    if abs(fx - fy) / fy < 0.01:
        lines.append(f"  ✓ fx 与 fy 相差 {abs(fx-fy)/fy*100:.2f}%，接近方形像素")
    else:
        lines.append(f"  ~ fx 与 fy 相差 {abs(fx-fy)/fy*100:.2f}%，偏大（可能是标定板不共面或图太少）")

    pv = data.get("per_view") or []
    if pv:
        lines.append("")
        lines.append("每张图的重投影误差（越大越可疑，>1 px 的图建议排除后重标）:")
        for row in pv:
            flag = "  ← 偏大" if row["rms_px"] > 1.0 else ""
            lines.append(f"  {row['image']:<28s} rms={row['rms_px']:6.3f} px  max={row['max_px']:6.3f} px{flag}")
    if data.get("skipped"):
        lines.append("")
        lines.append("被跳过的图:")
        for s in data["skipped"]:
            lines.append(f"  {s['image']:<28s} {s['reason']}")
    lines.append("=" * 66)
    return "\n".join(lines)
