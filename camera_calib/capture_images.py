"""摄像头采集标定图：实时检测棋盘格，满足"姿态够不同"才自动存盘。

用法：
    python capture_images.py                       # 默认 0 号相机，存到 data/calib
    python capture_images.py --camera 1 --target 20 --cols 9 --rows 6
    python capture_images.py --width 1920 --height 1080

快捷键：
    空格 / s : 立刻保存当前帧（手动）
    d        : 删除上一张
    q / Esc  : 退出

设计要点（为什么这样写）：
  1. 存 PNG 而不是 JPG：JPG 压缩会在黑白边缘产生振铃，直接拉低亚像素精度。
  2. 自动存图要求"和已有图的姿态差异足够大"，否则 20 张图全是一个角度，
     标定出来 fx/fy 会病态，RMS 好看但实际不能用。
  3. 保存前检查清晰度（Laplacian 方差）和帧间位移，避免存到运动模糊的图。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import calib_lib as cl


def board_feature(corners: np.ndarray, shape, pattern=(8, 5)) -> np.ndarray:
    """把一个姿态压缩成 4 维特征，用于判断两张图是否"太像"。

    特征 = [归一化中心x, 归一化中心y, sqrt(板面积占比), 水平/垂直间距比]
    面积和间距比会随倾斜、距离变化，能反映"远近 + 倾斜"的差异。
    """
    h, w = shape[:2]
    pts = corners.reshape(-1, 2)
    cx, cy = pts.mean(axis=0)
    hull = cv2.convexHull(pts.astype(np.float32))
    area = abs(cv2.contourArea(hull)) / (w * h)

    # 角点按 pattern[1] 行 x pattern[0] 列排列
    grid = pts.reshape(pattern[1], pattern[0], 2)
    dx = np.linalg.norm(np.diff(grid, axis=1), axis=2).mean()
    dy = np.linalg.norm(np.diff(grid, axis=0), axis=2).mean()
    aspect = dx / max(dy, 1e-6)
    return np.array([cx / w, cy / h, np.sqrt(max(area, 0)), aspect], dtype=np.float64)


def too_similar(f: np.ndarray, saved: list[np.ndarray]) -> bool:
    """与任意已存图都太像就返回 True。"""
    for s in saved:
        if (
            abs(f[0] - s[0]) < 0.07
            and abs(f[1] - s[1]) < 0.07
            and abs(f[2] - s[2]) / max(s[2], 1e-6) < 0.12
            and abs(f[3] - s[3]) / max(s[3], 1e-6) < 0.10
        ):
            return True
    return False


def sharpness(gray: np.ndarray, corners: np.ndarray) -> float:
    """棋盘区域内的 Laplacian 方差，越小越糊。"""
    pts = corners.reshape(-1, 2)
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def draw_overlay(frame, corners, pattern, n_saved, target, grid_counts, status, color):
    out = frame.copy()
    if corners is not None:
        cv2.drawChessboardCorners(out, pattern, corners, True)

    h, w = out.shape[:2]
    # 3x3 覆盖网格
    for i in range(1, 3):
        cv2.line(out, (w * i // 3, 0), (w * i // 3, h), (60, 60, 60), 1)
        cv2.line(out, (0, h * i // 3), (w, h * i // 3), (60, 60, 60), 1)
    for r in range(3):
        for c in range(3):
            cnt = grid_counts[r][c]
            col = (0, 180, 0) if cnt >= 2 else (0, 140, 255) if cnt else (90, 90, 90)
            cv2.putText(out, str(cnt), (c * w // 3 + 10, r * h // 3 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    bar = np.zeros((70, w, 3), np.uint8)
    out[0:70, :] = cv2.addWeighted(out[0:70, :], 0.25, bar, 0.75, 0)
    cv2.putText(out, f"{n_saved}/{target}", (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(out, status, (110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(out, "SPACE=save  d=undo  q=quit", (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return out


def main():
    ap = argparse.ArgumentParser(description="采集棋盘格标定图")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default="data/calib")
    ap.add_argument("--cols", type=int, default=9, help="方格列数")
    ap.add_argument("--rows", type=int, default=6, help="方格行数")
    ap.add_argument("--target", type=int, default=20, help="目标张数")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--min-sharp", type=float, default=40.0, help="清晰度阈值")
    ap.add_argument("--manual", action="store_true", help="关闭自动保存，完全手动按空格")
    args = ap.parse_args()

    pattern = cl.inner_corners(args.cols, args.rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = cl.list_images(out_dir)
    if existing:
        print(f"注意：{out_dir} 里已有 {len(existing)} 张图；本次从 calib_01.png 开始覆盖同名文件，"
              f"旧的多余文件不会被删除（标定时会一起用上）。")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"打不开相机 {args.camera}。检查是否被其它程序占用（微信/腾讯会议/浏览器）。")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 自动对焦先开着，拍的时候等它合焦
    time.sleep(0.8)

    saved: list[Path] = []
    feats: list[np.ndarray] = []
    grid = [[0] * 3 for _ in range(3)]
    prev_center = None
    last_save_t = 0.0
    win = "camera calibration capture"

    print("实时预览已打开。移动/旋转/倾斜标定板，让板子覆盖画面各个区域。")
    print("提示：板子要整块在画面内（四周至少留半格白边），并且不要反光/过曝。")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("读帧失败，退出。")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cl.find_corners(gray, pattern)

        status, color = "searching for board...", (0, 200, 255)
        can_save = False

        if found:
            f = board_feature(corners, gray.shape, pattern)
            sharp = sharpness(gray, corners)
            center = corners.reshape(-1, 2).mean(axis=0)
            moved = (
                prev_center is None
                or np.linalg.norm(center - prev_center) < 3.0
            )
            prev_center = center

            if sharp < args.min_sharp:
                status, color = f"blurry ({sharp:.0f}) - keep still", (0, 0, 255)
            elif too_similar(f, feats):
                status, color = "pose too similar to saved - move/tilt board", (0, 200, 255)
            elif not moved:
                status, color = "hold still...", (0, 200, 255)
            else:
                status, color = f"ok (sharp {sharp:.0f}) - auto saving", (0, 255, 0)
                can_save = True

        if (
            found
            and can_save
            and not args.manual
            and len(saved) < args.target
            and time.time() - last_save_t > 0.8
        ):
            p = out_dir / f"calib_{len(saved) + 1:02d}.png"
            cv2.imwrite(str(p), frame)  # PNG 无损
            saved.append(p)
            feats.append(f)
            x, y = f[0], f[1]
            grid[min(int(y * 3), 2)][min(int(x * 3), 2)] += 1
            last_save_t = time.time()
            print(f"  保存 {p.name}  ({len(saved)}/{args.target})")

        vis = draw_overlay(frame, corners if found else None, pattern, len(saved),
                           args.target, grid, status, color)
        cv2.imshow(win, vis)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            break
        elif key in (32, ord("s")) and found:
            p = out_dir / f"calib_{len(saved) + 1:02d}.png"
            cv2.imwrite(str(p), frame)
            saved.append(p)
            feats.append(board_feature(corners, gray.shape, pattern))
            f = feats[-1]
            grid[min(int(f[1] * 3), 2)][min(int(f[0] * 3), 2)] += 1
            print(f"  手动保存 {p.name}  ({len(saved)}/{args.target})")
        elif key == ord("d") and saved:
            p = saved.pop()
            feats.pop()
            p.unlink(missing_ok=True)
            grid = [[0] * 3 for _ in range(3)]
            for ff in feats:
                grid[min(int(ff[1] * 3), 2)][min(int(ff[0] * 3), 2)] += 1
            print(f"  已删除 {p.name}  (剩 {len(saved)} 张)")

    cap.release()
    cv2.destroyAllWindows()

    print()
    print(f"共 {len(saved)} 张，保存在 {out_dir.resolve()}")
    flat = [c for row in grid for c in row]
    if min(flat) == 0:
        print("注意：画面 3x3 网格里有格子是 0 张，边缘区域覆盖不足，标定外参外推会变差。")
    if len(saved) < 15:
        print("注意：张数少于 15，建议补拍。")
    print()
    print("下一步：")
    print(f"  python calibrate.py --dir {out_dir} --cols {args.cols} --rows {args.rows} --square 20.0")


if __name__ == "__main__":
    main()
