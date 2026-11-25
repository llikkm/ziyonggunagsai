"""端到端自检：没有相机和标定板时，用"已知真值"的合成图像验证整条链路。

做的事：
  1. 用已知内参 K_true 和畸变 dist_true 渲染 20 张不同位姿的棋盘格图；
     另渲染 5 张"没参与标定"的图作为验证集；
  2. 跑真实检测 + calibrateCamera，比较恢复出的 K、dist 与真值；
  3. 在验证图上测已知距离（模拟点击两个已知点），统计误差；
  4. 对比"没去畸变就测"的误差，说明第 2 步为什么不能省。

用法：
    python selftest_synthetic.py                # 默认 20 张标定 + 5 张验证
    python selftest_synthetic.py --views 15 --val 4

产物：out/selftest/ 下的合成图、report.txt、montage.png、measure_demo.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import calib_lib as cl

COLS, ROWS = 9, 6            # 方格数
SQ = 20.0                    # 格边长 mm
MARGIN_MM = 12.0             # 板上白边
PPM = 12.0                   # 纹理分辨率 px/mm


# --------------------------------------------------------------------------
# 合成数据
# --------------------------------------------------------------------------
def board_texture() -> np.ndarray:
    """按真实几何画一块 9x6、20mm 的板（含白边），单位 px = PPM/mm。"""
    w_mm = COLS * SQ + 2 * MARGIN_MM
    h_mm = ROWS * SQ + 2 * MARGIN_MM
    W, H = int(round(w_mm * PPM)), int(round(h_mm * PPM))
    tex = np.full((H, W), 255, np.uint8)
    for j in range(ROWS):
        for i in range(COLS):
            if (i + j) % 2 == 1:
                x0 = int(round((MARGIN_MM + i * SQ) * PPM))
                y0 = int(round((MARGIN_MM + j * SQ) * PPM))
                x1 = int(round((MARGIN_MM + (i + 1) * SQ) * PPM))
                y1 = int(round((MARGIN_MM + (j + 1) * SQ) * PPM))
                tex[y0:y1, x0:x1] = 0
    return tex


def random_pose(rng, K, size, sq=SQ, cols=COLS, rows=ROWS):
    """随机位姿：板面 0..160 x 0..100 mm，tilt 可达 50°，整板（含白边）在画面内。"""
    w, h = size
    fx, fy = K[0, 0], K[1, 1]
    board_w, board_h = (cols - 1) * sq, (rows - 1) * sq

    for _ in range(2000):
        alpha = np.radians(rng.uniform(-38, 38))
        beta = np.radians(rng.uniform(-38, 38))
        theta = np.radians(rng.uniform(-180, 180))
        Rx = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
        Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
        Rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        R = Rz @ Rx @ Ry
        rvec, _ = cv2.Rodrigues(R)

        # 先定板中心的像素位置和距离，再反推平移
        u0 = w / 2 + rng.uniform(-0.10, 0.10) * w
        v0 = h / 2 + rng.uniform(-0.10, 0.10) * h
        Z = rng.uniform(210, 430)
        center = np.array([board_w / 2, board_h / 2, 0.0])
        cam_pt = Z * np.linalg.inv(K) @ np.array([u0, v0, 1.0])
        t = cam_pt - R @ center
        tvec = t.reshape(3, 1).astype(np.float64)

        # 整块板（含白边）必须落在画面内，且大小合适
        xs = [-MARGIN_MM, board_w + MARGIN_MM]
        ys = [-MARGIN_MM, board_h + MARGIN_MM]
        border = np.array([[x, y, 0.0] for x in xs for y in ys], dtype=np.float64)
        proj, _ = cv2.projectPoints(border, rvec, tvec, K, np.zeros(5))
        p = proj.reshape(-1, 2)
        if p[:, 0].min() < 4 or p[:, 0].max() > w - 4 or p[:, 1].min() < 4 or p[:, 1].max() > h - 4:
            continue
        hull = cv2.convexHull(p.astype(np.float32))
        ratio = abs(cv2.contourArea(hull)) / (w * h)
        if not (0.18 < ratio < 0.62):
            continue
        return rvec, tvec
    raise RuntimeError("采样不出合适位姿，放宽约束")


def render_view(K, dist, rvec, tvec, tex, size, rng):
    """把板纹理投到画面上，再叠加畸变（逐像素反算，等价于真实成像）。"""
    w, h = size
    gy, gx = np.mgrid[0:h, 0:w]
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)

    # 畸变像素 -> 理想归一化坐标
    ideal = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)
    xn, yn = ideal[:, 0], ideal[:, 1]

    R, _ = cv2.Rodrigues(rvec)
    M = np.column_stack([R[:, 0], R[:, 1], tvec.ravel()])  # [r1 r2 t]
    Minv = np.linalg.inv(M)
    v = np.stack([xn, yn, np.ones_like(xn)], axis=1)
    q = v @ Minv.T

    with np.errstate(divide="ignore", invalid="ignore"):
        X = q[:, 0] / q[:, 2]
        Y = q[:, 1] / q[:, 2]

    mapx = ((X + MARGIN_MM) * PPM - 0.5).reshape(h, w).astype(np.float32)
    mapy = ((Y + MARGIN_MM) * PPM - 0.5).reshape(h, w).astype(np.float32)
    mapx[~np.isfinite(mapx)] = -1e6
    mapy[~np.isfinite(mapy)] = -1e6

    img = cv2.remap(tex, mapx, mapy, cv2.INTER_AREA,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # 轻量"真实感"：光照渐变 + 噪声 + 轻微模糊
    img = img.astype(np.float32)
    ramp = np.linspace(rng.uniform(0.86, 0.94), 1.0, w, dtype=np.float32)[None, :]
    img *= ramp
    img += rng.normal(0, 2.5, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    return img


# --------------------------------------------------------------------------
def model_gap(K_true, dist_true, K, dist, size):
    """真值畸变模型与估计模型在整个画面上的投影差异（像素）。

    单独比较 k2、k3 没意义——它们强相关、会互相补偿；
    看"同一个物理方向投到画面上的差异"才是真实影响。
    """
    w, h = size
    xs = np.linspace(-0.45, 0.45, 41)
    ys = np.linspace(-0.30, 0.30, 31)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], axis=1)

    a = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K_true, dist_true)[0].reshape(-1, 2)
    b = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
    d = np.linalg.norm(a - b, axis=1)
    return float(d.max()), float(d.mean())


def main():
    ap = argparse.ArgumentParser(description="合成数据端到端自检")
    ap.add_argument("--views", type=int, default=20)
    ap.add_argument("--val", type=int, default=5)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="out/selftest")
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    size = (w, h)
    out = Path(args.out)
    (out / "calib").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)

    K_true = np.array([[980.0, 0, 641.3], [0, 974.0, 356.8], [0, 0, 1.0]])
    dist_true = np.array([-0.28, 0.11, 0.0008, -0.0005, -0.018])

    rng = np.random.default_rng(args.seed)
    tex = board_texture()

    print(f"渲染 {args.views} 张标定图 + {args.val} 张验证图（用真值 K 和畸变）...")
    for i in range(args.views):
        rvec, tvec = random_pose(rng, K_true, size)
        img = render_view(K_true, dist_true, rvec, tvec, tex, size, rng)
        cv2.imwrite(str(out / "calib" / f"calib_{i+1:02d}.png"), img)
    val_poses = []
    for i in range(args.val):
        rvec, tvec = random_pose(rng, K_true, size)
        img = render_view(K_true, dist_true, rvec, tvec, tex, size, rng)
        cv2.imwrite(str(out / "val" / f"val_{i+1:02d}.png"), img)
        val_poses.append((rvec, tvec))

    # 缩略图拼版
    files = sorted((out / "calib").glob("*.png"))[:8]
    thumbs = [cv2.resize(cv2.imread(str(f)), (320, 180)) for f in files]
    if thumbs:
        cv2.imwrite(str(out / "montage.png"), np.vstack([np.hstack(thumbs[i:i + 4]) for i in range(0, 8, 4)]))

    # ---------------- 标定 ----------------
    print("检测角点并标定 ...")
    data = cl.calibrate_from_images(sorted((out / "calib").glob("*.png")),
                                    cl.inner_corners(COLS, ROWS), SQ)
    cl.save_calib(out / "calib.json", data)
    K = np.asarray(data["camera_matrix"])
    dist = np.asarray(data["dist_coeffs"]).ravel()

    fx_err = (K[0, 0] - K_true[0, 0]) / K_true[0, 0] * 100
    fy_err = (K[1, 1] - K_true[1, 1]) / K_true[1, 1] * 100
    cx_err = K[0, 2] - K_true[0, 2]
    cy_err = K[1, 2] - K_true[1, 2]

    lines = ["=" * 64, "合成数据自检报告", "=" * 64, ""]
    lines.append(f"真值 K: fx={K_true[0,0]:.2f} fy={K_true[1,1]:.2f} cx={K_true[0,2]:.2f} cy={K_true[1,2]:.2f}")
    lines.append(f"估计 K: fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    lines.append(f"误差  : fx {fx_err:+.3f}%  fy {fy_err:+.3f}%  cx {cx_err:+.3f} px  cy {cy_err:+.3f} px")
    lines.append(f"真值 dist: {np.round(dist_true, 5).tolist()}")
    lines.append(f"估计 dist: {np.round(dist, 5).tolist()}")
    lines.append(f"重投影 RMS: {data['rms_px']:.4f} px   (图数 {data['n_views']})")
    gap_max, gap_mean = model_gap(K_true, dist_true, K, dist, size)
    lines.append(f"全画面畸变模型差异: 最大 {gap_max:.3f} px，平均 {gap_mean:.3f} px")
    lines.append("  （k2/k3 等系数强相关会互相补偿，所以别单看系数，看这个差异）")
    ok_calib = abs(fx_err) < 1.0 and abs(fy_err) < 1.0 and data["rms_px"] < 0.5 and gap_max < 2.0
    lines.append("")

    # ---------------- 平面测量验证 ----------------
    # 板上两个"被测点"：故意不落在角点网格上，模拟被测物体端点
    P1 = np.array([23.4, 71.2])
    P2 = np.array([118.7, 33.9])
    true_len = float(np.linalg.norm(P1 - P2))

    errs_und, errs_raw, rows = [], [], []
    demo = None
    for p in sorted((out / "val").glob("*.png")):
        raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        und, K_adj = cl.undistort_image(raw, K, dist, alpha=0.0)

        found_u, corners_u = cl.find_corners(und, cl.inner_corners(COLS, ROWS))
        found_r, corners_r = cl.find_corners(raw, cl.inner_corners(COLS, ROWS))
        if not (found_u and found_r):
            lines.append(f"{p.name}: 未检出，跳过")
            continue

        H_und = cl.plane_homography(corners_u, cl.inner_corners(COLS, ROWS), SQ)   # 正确流程
        H_raw = cl.plane_homography(corners_r, cl.inner_corners(COLS, ROWS), SQ)   # 忘记去畸变

        # 真值位姿，把两个"被测点"投到两种图像上
        rvec, tvec = val_poses[int(p.stem.split("_")[-1]) - 1]
        P = np.array([[P1[0], P1[1], 0.0], [P2[0], P2[1], 0.0]])
        a_und, b_und = cv2.projectPoints(P, rvec, tvec, K_adj, np.zeros(5))[0].reshape(2, 2)
        a_raw, b_raw = cv2.projectPoints(P, rvec, tvec, K_true, dist_true)[0].reshape(2, 2)

        d_und = cl.distance_on_plane(H_und, a_und, b_und)
        d_raw = cl.distance_on_plane(H_raw, a_raw, b_raw)

        rows.append((p.name, d_und, d_und - true_len, d_raw, d_raw - true_len))
        errs_und.append(abs(d_und - true_len))
        errs_raw.append(abs(d_raw - true_len))

        # 标注一张图用于展示
        if demo is None:
            vis = cv2.cvtColor(und, cv2.COLOR_GRAY2BGR)
            cv2.line(vis, tuple(np.round(a_und).astype(int)), tuple(np.round(b_und).astype(int)), (0, 255, 0), 2)
            for pt in (a_und, b_und):
                cv2.circle(vis, tuple(np.round(pt).astype(int)), 5, (0, 0, 255), -1)
            cv2.putText(vis, f"measured {d_und:.2f} mm / true {true_len:.2f} mm",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            demo = vis

    lines.append(f"平面测量：被测两点真值距离 {true_len:.3f} mm（点坐标故意不在角点网格上）")
    lines.append(cl.ljust("图像", 12) + cl.rjust("去畸变后测量", 14) + cl.rjust("误差", 9)
                 + cl.rjust("忘了去畸变", 14) + cl.rjust("误差", 9))
    for name, d, e, d2, e2 in rows:
        lines.append(cl.ljust(name, 12) + f"{d:14.3f}{e:+9.3f}{d2:14.3f}{e2:+9.3f}")
    if rows:
        lines.append(f"\n去畸变后：平均绝对误差 {np.mean(errs_und):.4f} mm，最大 {np.max(errs_und):.4f} mm")
        lines.append(f"不去畸变：平均绝对误差 {np.mean(errs_raw):.4f} mm，最大 {np.max(errs_raw):.4f} mm")
        ok_measure = np.max(errs_und) < 0.5
    else:
        ok_measure = False

    lines.append("")
    lines.append(f"结论：内参 {'通过' if ok_calib else '未通过'}，平面测量 {'通过' if ok_measure else '未通过'}")
    lines.append("=" * 64)
    report = "\n".join(lines)
    print()
    print(report)
    (out / "report.txt").write_text(report, encoding="utf-8")
    if demo is not None:
        cv2.imwrite(str(out / "measure_demo.png"), demo)

    print(f"\n产物目录 {out.resolve()}")
    if not (ok_calib and ok_measure):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
