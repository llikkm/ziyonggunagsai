"""激光平面标定仿真主实验。

运行：
    python calibrate_plane_sim.py
    python calibrate_plane_sim.py --show
    python calibrate_plane_sim.py --synthetic-camera
    python calibrate_plane_sim.py --square 19      # 实物打印后实测 19mm 时用

默认相机内参读 01_camera_calib/out/calib.json（已有标定），棋盘格用 20mm 标称方格。
如果实物打印缩放后是 19mm，真实标定应传 --square 19；本仿真默认 20mm。

数据隔离：
* build_dataset() 知道真实平面（渲染需要）；
* calibrate_plane() 只接收 measured corners / measured laser UV / K,dist / board；
* 真值只在 evaluate_plane()、evaluate_validation() 和画图时使用。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import plane_calib_lib as pc
import laser_lib as ll


# ==========================================================================
# 基础工具
# ==========================================================================
def _get_plt(show: bool):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "figure.dpi": 110,
    })
    return plt


def _save(fig, out: Path, name: str, plt, show: bool) -> Path:
    path = out / "figs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def _fmt(value, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "-"
    v = float(value)
    if not np.isfinite(v):
        return "-"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


# ==========================================================================
# 相机 / 数据
# ==========================================================================
def build_camera(args) -> ll.CameraModel:
    if args.synthetic_camera:
        K = np.array([[806.0, 0.0, 640.0], [0.0, 806.0, 360.0], [0.0, 0.0, 1.0]])
        return ll.CameraModel(K, np.zeros(5), (1280, 720), source="SYNTHETIC_806")
    return ll.CameraModel.from_calib_json(args.calib)


def save_dataset(dataset: pc.Dataset, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    pc.write_json(folder / "measurements.json", dataset.measurements)
    pc.write_json(folder / "truth.json", dataset.truth)
    pc.write_json(folder / "orientation.json", {
        "rows": dataset.orientation_rows,
        "aggregate": pc.aggregate_orientation(dataset.orientation_rows),
        "note": "diagnostic only; calibration uses laser_uv_row",
    })


def save_validation(cases, folder: Path) -> None:
    pc.write_json(folder / "measurements.json", [c.measurement for c in cases])
    pc.write_json(folder / "truth.json", [c.truth for c in cases])


def build_coverage_datasets(camera, true_plane, board, out, sigma_corner,
                            sigma_laser, n_boards, seed):
    specs = {
        "fronto": (pc.PoseRanges(z=(420, 580), x=(-55, 55), y=(-30, 30),
                                 rx=(0, 0), ry=(0, 0), rz=(0, 0)), seed + 101),
        "one_axis": (pc.PoseRanges(z=(420, 580), x=(-55, 55), y=(-30, 30),
                                   rx=(-22, 22), ry=(0, 0), rz=(0, 0)), seed + 102),
        "same_plane": (pc.PoseRanges(z=(500, 500), x=(-55, 55), y=(-30, 30),
                                     rx=(0, 0), ry=(0, 0), rz=(0, 0)), seed + 103),
        "close": (pc.PoseRanges(z=(400, 430), x=(-55, 55), y=(-30, 30),
                                rx=(-22, 22), ry=(-22, 22), rz=(-15, 15)), seed + 104),
    }
    result = {}
    notes = {
        "fronto": "全部正对、无倾角，只变距离和画面位置",
        "one_axis": "只绕 X 轴倾斜，缺另一方向倾角",
        "same_plane": "全部在同一 Z=500 正对，板面重合，标定点退化成一条线",
        "close": "只在 400~430mm 近距离，验证会外推到更远位置",
    }
    for name, (ranges, rng_seed) in specs.items():
        rng = np.random.default_rng(rng_seed)
        poses = pc.sample_poses(n_boards, camera, true_plane, ranges, board, rng)
        folder = out / "data" / "coverage" / name
        dataset = pc.build_dataset(
            poses, camera, true_plane, board, folder, rng,
            sigma_corner=sigma_corner, sigma_laser=sigma_laser, save_images=True,
        )
        save_dataset(dataset, folder)
        pc.write_json(folder / "coverage_note.json", {"note": notes[name]})
        result[name] = pc.prepare_all(dataset.measurements, camera, board)
    return result, notes


# ==========================================================================
# 绘图
# ==========================================================================
def plot_geometry(plt, out, show, dataset, calibration, board, plane_eval):
    pts = np.vstack([p.points for p in calibration.prepared if len(p.points)])
    pose_idx = np.concatenate([
        np.full(len(p.points), p.index, dtype=int)
        for p in calibration.prepared if len(p.points)
    ])

    n = calibration.fit.normal
    d = calibration.fit.d
    center = pts.mean(axis=0)
    center = center - (float(n @ center) + d) * n
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    fig = plt.figure(figsize=(12.0, 5.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pose_idx,
                    cmap="tab20", s=2.5, alpha=0.75)
    proj = pts - center
    eu = max(float(np.max(np.abs(proj @ u))), 20.0) + 10.0
    ev = max(float(np.max(np.abs(proj @ v))), 20.0) + 10.0
    gu, gv = np.meshgrid(np.linspace(-eu, eu, 12), np.linspace(-ev, ev, 12))
    grid = center + gu[..., None] * u + gv[..., None] * v
    ax.plot_surface(grid[..., 0], grid[..., 1], grid[..., 2],
                    color="tab:blue", alpha=0.16, linewidth=0)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("calibration points in camera frame")
    ax.view_init(elev=20, azim=-62)
    ax.set_box_aspect((1.4, 0.8, 1.0))

    ax = fig.add_subplot(1, 2, 2)
    pc2 = ax.scatter(proj @ u, proj @ v, c=pose_idx, cmap="tab20", s=3, alpha=0.8)
    ax.set_xlabel("coordinate along plane u (mm)")
    ax.set_ylabel("coordinate along plane v (mm)")
    ax.set_title("coverage of the fitted laser plane (one color per board pose)")
    ax.set_aspect("equal", adjustable="datalim")
    cb = fig.colorbar(pc2, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("board index")
    fig.suptitle(f"{len(dataset.poses)} board poses -> one laser plane (fit angle error "
                 f"{plane_eval['normal_angle_error_deg']:.4f} deg)", y=1.02)
    fig.tight_layout()
    return _save(fig, out, "fig01_geometry.png", plt, show)


def plot_residuals(plt, out, show, calibration, per_pose):
    pts = np.vstack([p.points for p in calibration.prepared if len(p.points)])
    res = pts @ calibration.fit.normal + calibration.fit.d
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))

    ax = axes[0]
    ax.hist(res, bins=40, color="tab:blue", alpha=0.75)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("point-to-plane residual (mm)")
    ax.set_ylabel("count")
    ax.set_title(f"residuals: RMS={calibration.fit.residual_rms_mm:.4f} mm")

    idx = [r["index"] for r in per_pose]
    pnp = [r["pnp_rms_px"] for r in per_pose]
    ax = axes[1]
    ax.bar(idx, pnp, color="tab:green", alpha=0.75)
    ax.set_xlabel("board index")
    ax.set_ylabel("PnP RMS (px)")
    ax.set_title("corner / pose residual")

    angle = [r.get("board_plane_angle_error_deg", np.nan) for r in per_pose]
    ax = axes[2]
    ax.bar(idx, angle, color="tab:red", alpha=0.75)
    ax.set_xlabel("board index")
    ax.set_ylabel("board plane angle error (deg)")
    ax.set_title("PnP board plane vs truth")
    fig.tight_layout()
    return _save(fig, out, "fig02_residuals.png", plt, show)


def plot_board_count(plt, out, show, board_count):
    n = [r["n_boards"] for r in board_count]
    angle = [r["angle_mean_deg"] for r in board_count]
    angle_std = [r["angle_std_deg"] for r in board_count]
    val = [r["validation_mean_abs_mean_mm"] for r in board_count]
    val_std = [r["validation_mean_abs_std_mm"] for r in board_count]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    ax.errorbar(n, angle, yerr=angle_std, fmt="o-", capsize=3, color="tab:red")
    ax.set_xlabel("number of calibration boards")
    ax.set_ylabel("normal angle error (deg)")
    ax.set_title("plane normal vs board count")
    ax = axes[1]
    ax.errorbar(n, val, yerr=val_std, fmt="o-", capsize=3, color="tab:blue")
    ax.set_xlabel("number of calibration boards")
    ax.set_ylabel("validation mean |error| (mm)")
    ax.set_title("independent step validation vs board count")
    fig.tight_layout()
    return _save(fig, out, "fig03_board_count.png", plt, show)


def plot_coverage(plt, out, show, coverage):
    names = [r["coverage"] for r in coverage]
    angle = np.array([r["angle_deg"] for r in coverage])
    val = np.array([r["validation_mean_abs_mm"] for r in coverage])
    cond = np.array([r["condition"] for r in coverage])
    flags = [r["degenerate"] for r in coverage]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    x = np.arange(len(names))
    for ax, values, title, ylabel in [
        (axes[0], angle, "normal angle error", "angle error (deg)"),
        (axes[1], val, "independent validation", "validation mean |error| (mm)"),
    ]:
        ax.bar(x, values, color=["tab:green" if not f else "tab:red" for f in flags],
               alpha=0.8)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        # 给顶部的标注留空间，避免和标题重叠。
        ax.set_ylim(top=max(float(np.nanmax(values)) * 3.0, 1e-3))
        for xi, v, c, bad in zip(x, values, cond, flags):
            if not np.isfinite(v):
                ax.annotate(f"REJECTED\ncond={c:.0f}", (xi, 0),
                            textcoords="offset points", xytext=(0, 5),
                            ha="center", fontsize=7, color="tab:red")
                continue
            ax.annotate(f"{v:.3g}\ncond={c:.0f}" + ("\nDEGEN" if bad else ""),
                        (xi, v), textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=7)
    fig.tight_layout()
    return _save(fig, out, "fig04_coverage.png", plt, show)


def plot_noise(plt, out, show, noise):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for kind, color in (("corner", "tab:red"), ("laser", "tab:blue")):
        rows = [r for r in noise if r["noise_type"] == kind]
        s = [r["sigma_px"] for r in rows]
        a = [r["angle_mean_deg"] for r in rows]
        asd = [r["angle_std_deg"] for r in rows]
        v = [r["validation_mean_abs_mean_mm"] for r in rows]
        vsd = [r["validation_mean_abs_std_mm"] for r in rows]
        axes[0].errorbar(s, a, yerr=asd, fmt="o-", capsize=3, color=color,
                         label=f"{kind} noise")
        axes[1].errorbar(s, v, yerr=vsd, fmt="o-", capsize=3, color=color,
                         label=f"{kind} noise")
    axes[0].set_xlabel("additional pixel noise sigma (px)")
    axes[0].set_ylabel("normal angle error (deg)")
    axes[0].set_title("plane normal vs measurement noise")
    axes[0].legend()
    axes[1].set_xlabel("additional pixel noise sigma (px)")
    axes[1].set_ylabel("validation mean |error| (mm)")
    axes[1].set_title("independent step validation vs noise")
    axes[1].legend()
    fig.tight_layout()
    return _save(fig, out, "fig05_noise.png", plt, show)


def plot_validation(plt, out, show, val_rows):
    true_h = np.array([r["true_height_mm"] for r in val_rows])
    meas = np.array([r["measured_height_mm"] for r in val_rows])
    err = np.array([r["error_mm"] for r in val_rows])
    names = [r["case"] for r in val_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    ax.plot([0, 55], [0, 55], "k--", lw=1.0, label="1:1")
    sc = ax.scatter(true_h, meas, c=true_h, cmap="viridis", s=45, zorder=3)
    ax.set_xlabel("true step height (mm)")
    ax.set_ylabel("measured height (mm)")
    ax.set_title("independent validation (reconstruction uses fitted plane)")
    ax.legend()
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03, label="true height (mm)")

    ax = axes[1]
    ax.bar(np.arange(len(err)), err, color=["tab:red" if e < 0 else "tab:blue" for e in err])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(err)))
    ax.set_xticklabels(names, rotation=60, fontsize=7)
    ax.set_ylabel("height error (mm)")
    ax.set_title("signed validation error")
    fig.tight_layout()
    return _save(fig, out, "fig06_validation.png", plt, show)


def plot_orientation(plt, out, show, dataset, orientation_agg):
    m0 = dataset.measurements[0]
    img_path = out / "data" / m0["laser_image"]
    gray = ll.load_gray_image(img_path)
    uv_row = np.asarray(m0["laser_uv_row"])
    uv_col = np.asarray(m0["laser_uv_column"])
    line = np.asarray(dataset.truth["poses"][0]["line_uv"])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    ax = axes[0]
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255, aspect="auto")
    ax.plot(line[:, 0], line[:, 1], color="cyan", lw=1.0, label="true stripe")
    if len(uv_row):
        ax.scatter(uv_row[:, 0], uv_row[:, 1], s=2, color="red", label="row scan")
    if len(uv_col):
        ax.scatter(uv_col[:, 0], uv_col[:, 1], s=10, color="orange",
                   edgecolors="k", linewidths=0.3, label="column scan")
    lu = line
    ax.set_xlim(max(0, lu[:, 0].min() - 30), min(gray.shape[1], lu[:, 0].max() + 30))
    ax.set_ylim(min(gray.shape[0], lu[:, 1].max() + 30), max(0, lu[:, 1].min() - 30))
    ax.set_xlabel("u (pixel)")
    ax.set_ylabel("v (pixel)")
    ax.set_title("near-vertical stripe: row vs column extraction")
    ax.legend(loc="upper right", fontsize=7, markerscale=2)

    ax = axes[1]
    names = ["row", "column", "auto\nfull image", "auto\ntight ROI"]
    counts = [orientation_agg["row_count_mean"], orientation_agg["column_count_mean"],
              orientation_agg["auto_count_mean"], orientation_agg["auto_roi_count_mean"]]
    errs = [orientation_agg["row_error_mean_px"], orientation_agg["column_error_mean_px"],
            orientation_agg["auto_error_mean_px"], orientation_agg["auto_roi_error_mean_px"]]
    ax.bar(np.arange(4), counts, color=["tab:green", "tab:red", "tab:orange", "tab:blue"],
           alpha=0.8)
    for i, (c, e) in enumerate(zip(counts, errs)):
        ax.annotate(f"{c:.0f}\nerr {e:.3f}px", (i, c), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=7)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(names)
    ax.set_ylabel("valid centers per pose (mean)")
    ax.set_title("valid centers: row scan is the right mode")
    ax.set_ylim(0, max(counts) * 1.18)
    fig.tight_layout()
    return _save(fig, out, "fig07_orientation.png", plt, show)


# ==========================================================================
# 报告
# ==========================================================================
def build_report(args, camera, board, true_plane, dataset, calibration,
                 plane_eval, per_pose, ideal, val_rows, val_summary,
                 board_count, coverage, coverage_notes, noise, orientation_rows,
                 orientation_agg, checks, out):
    lines = []
    lines.append("=" * 74)
    lines.append(f"激光平面标定仿真（{board.square_mm:g} mm 方格）")
    lines.append("=" * 74)
    lines.append(f"相机           : fx={camera.fx:.3f}, fy={camera.fy:.3f}, "
                 f"cx={camera.cx:.3f}, cy={camera.cy:.3f}, "
                 f"{camera.image_size[0]}x{camera.image_size[1]}, "
                 f"{'有畸变' if camera.has_distortion else '无畸变'}")
    lines.append(f"相机来源       : {camera.source}")
    lines.append(f"棋盘格         : {board.cols}x{board.rows} 格, "
                 f"{board.inner[0]}x{board.inner[1]} 内部角点, "
                 f"方格 {board.square_mm:.3f} mm, 物理 {board.physical_width_mm:.1f}x"
                 f"{board.physical_height_mm:.1f} mm")
    lines.append(f"说明           : 本次方格 {board.square_mm:g} mm；实物打印缩放后若实测 19mm，"
                 "真实标定请传 --square 19（或实测值）。")
    lines.append(f"真实激光平面   : n={np.round(true_plane.unit_normal, 6).tolist()}, "
                 f"d={true_plane.unit_d:.4f}（仅评价/渲染用）")
    lines.append(f"标定板数量     : {len(dataset.poses)}")
    lines.append(f"测量噪声       : 角点 {args.sigma_corner:.3f} px, 激光中心 "
                 f"{args.sigma_laser:.3f} px")
    lines.append("数据隔离       : 标定函数只接收 measured corners / laser UV / K / board；"
                 "真实平面未传入。")
    lines.append("")

    lines.append("[1] 标定输入与棋盘格位姿")
    pnp = [r["pnp_rms_px"] for r in per_pose]
    line_len = [p.features["line_length_mm"] for p in dataset.poses]
    lines.append(f"PnP 重投影 RMS: 平均 {np.mean(pnp):.3f} px, 最大 {np.max(pnp):.3f} px")
    lines.append(f"每块板上的激光线长度: 平均 {np.mean(line_len):.1f} mm, "
                 f"范围 {np.min(line_len):.1f}~{np.max(line_len):.1f} mm")
    rows = []
    for r in per_pose:
        rows.append([
            str(r["index"] + 1),
            _fmt(r["pnp_rms_px"], 3),
            _fmt(r.get("board_plane_angle_error_deg"), 4),
            _fmt(r.get("board_plane_d_error_mm"), 4, signed=True),
            str(r["n_points"]),
            _fmt(r.get("point_residual_rms_mm"), 4),
        ])
    lines.append(ll.format_table(
        ["板号", "PnP RMS(px)", "板面角误差(°)", "板面d误差(mm)", "激光点", "点残差RMS(mm)"],
        rows, ["<", ">", ">", ">", ">", ">"],
    ))
    lines.append("")

    lines.append("[2] 拟合结果（标定程序输出）")
    fit = calibration.fit
    lines.append(f"归一化平面     : n={np.round(fit.normal, 8).tolist()}, d={fit.d:.6f}")
    lines.append(f"平面方程       : {fit.normal[0]:+.8f} X {fit.normal[1]:+.8f} Y "
                 f"{fit.normal[2]:+.8f} Z {fit.d:+.8f} = 0")
    lines.append(f"点到平面残差   : RMS {fit.residual_rms_mm:.6f} mm, "
                 f"max {fit.residual_max_mm:.6f} mm, 点数 {fit.n_points}")
    lines.append(f"点云条件数 s0/s1: {fit.condition:.3f}  "
                 f"({'退化/近共线' if fit.degenerate else '正常'})")
    lines.append(f"法向量角度误差 : {plane_eval['normal_angle_error_deg']:.6f}°")
    lines.append(f"平面 d 误差    : {plane_eval['d_error_mm']:+.6f} mm")
    lines.append(f"理想投影对照   : 角度 {ideal['evaluation']['normal_angle_error_deg']:.3e}°, "
                 f"残差 {ideal['fit']['residual_rms_mm']:.3e} mm, "
                 f"验证均值 {ideal['validation']['mean_abs_error_mm']:.3e} mm")
    lines.append("  （理想投影的拟合本身是精确的；验证均值非零，因为验证数据仍含")
    lines.append("    渲染/提取/额外 0.2px 噪声，它代表验证数据的采集噪声下限。）")
    lines.append("")

    lines.append("[3] 独立验证：10 / 31 / 50 mm 台阶（未参与标定）")
    rows = []
    for r in val_rows:
        rows.append([
            r["case"],
            _fmt(r["true_height_mm"], 1),
            _fmt(r["measured_height_mm"], 4),
            _fmt(r["error_mm"], 4, signed=True),
            _fmt(r["abs_error_mm"], 4),
            _fmt(r["height_se_mm"], 4),
        ])
    lines.append(ll.format_table(
        ["场景", "真值(mm)", "拟合平面测量(mm)", "误差(mm)", "|误差|(mm)", "标准误(mm)"],
        rows, ["<", ">", ">", ">", ">", ">"],
    ))
    lines.append(f"验证汇总       : 平均 |误差| {val_summary['mean_abs_error_mm']:.4f} mm, "
                 f"最大 {val_summary['max_abs_error_mm']:.4f} mm, "
                 f"{val_summary['n_valid']}/{val_summary['n_cases']} 有效")
    lines.append("验证反算只用拟合平面；真值只用于最后算误差。")
    lines.append("")

    lines.append("[4] 标定板数量对误差的影响")
    rows = []
    for r in board_count:
        rows.append([
            str(r["n_boards"]),
            str(r["n_subsets"]),
            _fmt(r["angle_mean_deg"], 4) + " ± " + _fmt(r["angle_std_deg"], 4),
            _fmt(r["rms_mean_mm"], 4),
            _fmt(r["validation_mean_abs_mean_mm"], 4) + " ± " +
            _fmt(r["validation_mean_abs_std_mm"], 4),
            _fmt(r["condition_median"], 1),
        ])
    lines.append(ll.format_table(
        ["板数", "子集数", "法向角误差(°)", "拟合RMS(mm)", "验证均值±std(mm)", "条件数中位数"],
        rows, ["<", ">", ">", ">", ">", ">"],
    ))
    lines.append("“板数”一行是随机抽子集的统计量；15 块时只有一个子集，std=0。")
    lines.append("本场景 12 块以上比较稳定（约 0.01° / 0.10 mm）；8~10 块已可用，")
    lines.append("但随机子集的方差明显更大；5 块最散。没有时间压力时建议 10~12 块，")
    lines.append("并保证姿态覆盖，而不是重复同一姿态。")
    lines.append("")

    lines.append("[5] 不同姿态覆盖")
    rows = []
    for r in coverage:
        rows.append([
            r["coverage"],
            str(r["n_boards"]),
            _fmt(r["angle_deg"], 4),
            _fmt(r["residual_rms_mm"], 4),
            _fmt(r["condition"], 1),
            "是" if r["degenerate"] else "否",
            _fmt(r["validation_mean_abs_mm"], 3),
            _fmt(r["validation_max_abs_mm"], 3),
        ])
    lines.append(ll.format_table(
        ["覆盖方式", "板数", "角误差(°)", "RMS(mm)", "条件数", "退化",
         "验证均值(mm)", "验证最大(mm)"],
        rows, ["<", ">", ">", ">", ">", ">", ">", ">"],
    ))
    for name, note in coverage_notes.items():
        lines.append(f"  {name}: {note}")
    lines.append("结论：全正对/无倾角会明显变差；同一平面重复摆板会让点云退化成一条线，")
    lines.append("条件数急剧变大，验证完全失效。标定板必须变位置、变距离、变两个方向的倾角。")
    lines.append("")

    lines.append("[6] 角点噪声 / 激光中心噪声")
    rows = []
    for r in noise:
        rows.append([
            "角点" if r["noise_type"] == "corner" else "激光",
            _fmt(r["sigma_px"], 2),
            _fmt(r["angle_mean_deg"], 4),
            _fmt(r["rms_mean_mm"], 4),
            _fmt(r["validation_mean_abs_mean_mm"], 4),
        ])
    lines.append(ll.format_table(
        ["噪声类型", "额外σ(px)", "法向角误差(°)", "拟合RMS(mm)", "验证均值(mm)"],
        rows, ["<", ">", ">", ">", ">"],
    ))
    lines.append("σ=0 是“只经过渲染/检测/提取，不再额外加噪”的基线，不是理想投影；")
    lines.append("理想投影对照见 [2]。本几何下角点噪声对法向和高度的影响比激光中心噪声更明显：")
    lines.append("角点噪声会先扰动棋盘格平面，再传到激光点；激光中心噪声主要抬高拟合残差。")
    lines.append("")

    lines.append("[7] 近竖直条纹：按列 vs 按行提取")
    rows = [
        ["row", _fmt(orientation_agg["row_count_mean"], 1),
         _fmt(orientation_agg["row_error_mean_px"], 4), "-"],
        ["column", _fmt(orientation_agg["column_count_mean"], 1),
         _fmt(orientation_agg["column_error_mean_px"], 4), "-"],
        ["auto 全图", _fmt(orientation_agg["auto_count_mean"], 1),
         _fmt(orientation_agg["auto_error_mean_px"], 4),
         f"{orientation_agg['auto_axis_row_fraction'] * 100:.0f}%"],
        ["auto 紧 ROI", _fmt(orientation_agg["auto_roi_count_mean"], 1),
         _fmt(orientation_agg["auto_roi_error_mean_px"], 4),
         f"{orientation_agg['auto_roi_axis_row_fraction'] * 100:.0f}%"],
    ]
    lines.append(ll.format_table(
        ["模式", "每姿态有效中心数", "中心误差均值(px)", "自动选 row 的比例"],
        rows, ["<", ">", ">", ">"],
    ))
    lines.append("结论：当前条纹近似竖直，必须用 axis='row'。按列提取只能得到十几个中心，")
    lines.append("且中心误差明显更大；全图 auto 也会误选 column，因为细条纹占全图像素不足 1%，")
    lines.append("现有 auto 的 99 分位主方向启发式被背景点带偏。把 ROI 收紧到条纹附近后，")
    lines.append("auto 才正确选 row。本次标定全部显式使用 row，不依赖 auto。")
    lines.append("这属于使用建议，基础 laser_lib 未做修改。")
    lines.append("")

    lines.append("[8] 数据隔离与检查")
    lines.append("  * data/measurements.json 只含角点、激光中心、图像名；无真实平面/真实位姿")
    lines.append("  * data/truth.json 单独保存真实平面和真实位姿，只用于评价")
    lines.append("  * validation/measurements.json 只含验证条纹中心；validation/truth.json 单独存真值")
    lines.append("  * calibrate_plane() 的签名里没有 true_plane")
    passed = sum(1 for c in checks if c["passed"])
    for c in checks:
        lines.append(f"  {'✓' if c['passed'] else '✗'} {c['name']}: {c['detail']}")
    lines.append(f"检查: {passed}/{len(checks)} 通过")
    lines.append("")

    lines.append("[产物]")
    for name in [
        "data/measurements.json", "data/truth.json", "data/orientation.json",
        "data/ideal_measurements.json", "data/calibration_points.npz",
        "validation/measurements.json", "validation/truth.json",
        "results/calibration.json", "results/laser_plane.json",
        "results/summary.json", "results/validation.csv",
        "results/board_count.csv", "results/coverage.csv", "results/noise.csv",
        "results/orientation.csv", "results/report.txt",
        "figs/fig01_geometry.png", "figs/fig02_residuals.png",
        "figs/fig03_board_count.png", "figs/fig04_coverage.png",
        "figs/fig05_noise.png", "figs/fig06_validation.png",
        "figs/fig07_orientation.png",
    ]:
        if (out / name).exists():
            lines.append(f"  {out.name}/{name}")
    lines.append("=" * 74)
    return "\n".join(lines)


# ==========================================================================
# 主流程
# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="激光平面标定仿真（默认 20mm 方格，可 --square 19）")
    ap.add_argument("--calib", type=Path, default=Path("../out/calib.json"),
                    help="已有相机内参 calib.json")
    ap.add_argument("--synthetic-camera", action="store_true",
                    help="用 fx=fy=806、无畸变的合成相机（不读 calib.json）")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boards", type=int, default=15)
    ap.add_argument("--square", type=float, default=20.0,
                    help="标称方格 mm；实物打印后若 19mm，真实标定传 19")
    ap.add_argument("--sigma-corner", type=float, default=0.2)
    ap.add_argument("--sigma-laser", type=float, default=0.2)
    ap.add_argument("--subsets", type=int, default=20)
    ap.add_argument("--noise-repeats", type=int, default=10)
    ap.add_argument("--coverage-boards", type=int, default=8)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="检查失败时返回非零退出码")
    args = ap.parse_args()

    if args.n_boards < 3:
        raise ValueError("至少需要 3 块标定板")
    out = Path(args.out)
    for sub in ("data", "validation", "results", "figs"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    camera = build_camera(args)
    true_plane = ll.LaserPlane.from_abcd(pc.DEFAULT_TRUE_PLANE_ABCD,
                                         source="SYNTHETIC_TRUE")
    board = pc.BoardSpec(square_mm=float(args.square))

    rng = np.random.default_rng(args.seed)
    ranges = pc.PoseRanges(
        z=(420, 580), x=(-55, 55), y=(-30, 30),
        rx=(-22, 22), ry=(-22, 22), rz=(-15, 15),
    )
    poses = pc.sample_poses(args.n_boards, camera, true_plane, ranges, board, rng)
    dataset = pc.build_dataset(
        poses, camera, true_plane, board, out / "data", rng,
        sigma_corner=args.sigma_corner, sigma_laser=args.sigma_laser,
        save_images=True,
    )
    save_dataset(dataset, out / "data")

    # 只把 measurements 传给标定；true_plane 不进入。
    calibration = pc.calibrate_plane(dataset.measurements, camera, board)
    plane_eval = pc.evaluate_plane(calibration.fit, true_plane)
    board_eval = pc.evaluate_board_planes(calibration, dataset.poses)
    board_by_idx = {r["index"]: r for r in board_eval}
    per_pose = []
    for r in calibration.per_pose:
        merged = dict(r)
        merged.update({k: v for k, v in board_by_idx.get(r["index"], {}).items()
                       if k != "index"})
        per_pose.append(merged)

    pts = np.vstack([p.points for p in calibration.prepared if len(p.points)])
    pose_idx = np.concatenate([
        np.full(len(p.points), p.index, dtype=int)
        for p in calibration.prepared if len(p.points)
    ])
    with (out / "data" / "calibration_points.npz").open("wb") as f:
        np.savez(f, points_mm=pts, pose_index=pose_idx,
                 fit_plane_abcd=calibration.fit.to_plane().abcd,
                 source="LASER_PLANE_CALIB_SIM")

    # 独立验证数据：先用真实平面渲染/提取，之后只把拟合平面交给反算。
    val_rng = np.random.default_rng(args.seed + 900)
    cases = pc.build_validation_cases(
        camera, true_plane, out / "validation", val_rng,
        sigma_laser=args.sigma_laser, save_images=True,
    )
    save_validation(cases, out / "validation")
    val_measurements = [c.measurement for c in cases]
    val_measured = pc.measure_validation(calibration.fit, val_measurements, camera)
    val_rows, val_summary = pc.evaluate_validation(val_measured, cases)

    # 理想投影对照（几何上限；仍只把测量值交给标定函数）
    ideal_meas = pc.build_ideal_measurements(poses, camera, board)
    pc.write_json(out / "data" / "ideal_measurements.json", ideal_meas)
    ideal_calib = pc.calibrate_plane(ideal_meas, camera, board)
    ideal_eval = pc.evaluate_plane(ideal_calib.fit, true_plane)
    ideal_val_meas = pc.measure_validation(ideal_calib.fit, val_measurements, camera)
    _, ideal_val_summary = pc.evaluate_validation(ideal_val_meas, cases)
    ideal = {
        "fit": ideal_calib.fit.to_dict(),
        "evaluation": ideal_eval,
        "validation": ideal_val_summary,
        "per_pose": ideal_calib.per_pose,
    }

    # 板数
    prepared = calibration.prepared
    board_count = pc.run_board_count(
        prepared, true_plane, cases, camera,
        counts=sorted(set(c for c in (5, 8, 10, 12, args.n_boards)
                          if c <= args.n_boards)),
        repeats=args.subsets, rng=np.random.default_rng(args.seed + 11), audit_dir=out,
    )
    # 覆盖
    coverage_sets, coverage_notes = build_coverage_datasets(
        camera, true_plane, board, out, args.sigma_corner, args.sigma_laser,
        args.coverage_boards, args.seed,
    )
    coverage_sets = {"good": prepared[:args.coverage_boards], **coverage_sets}
    coverage = pc.run_coverage(coverage_sets, true_plane, cases, camera, audit_dir=out)
    # 噪声
    noise = pc.run_noise(
        dataset.measurements, camera, board, true_plane, cases,
        corner_sigmas=(0.0, 0.1, 0.2, 0.3, 0.5, 1.0),
        laser_sigmas=(0.0, 0.1, 0.2, 0.3, 0.5, 1.0),
        repeats=args.noise_repeats, rng=np.random.default_rng(args.seed + 21), audit_dir=out,
    )
    orientation_agg = pc.aggregate_orientation(dataset.orientation_rows)

    # ---------------- 保存结果 ----------------
    calib_result = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": "LASER_PLANE_CALIB_SIM",
        "unit": "mm",
        "camera": {"camera_matrix": camera.K.tolist(),
                   "dist_coeffs": camera.dist.tolist(),
                   "image_size": list(camera.image_size) if camera.image_size else None,
                   "source": camera.source},
        "board": board.to_dict(),
        "n_boards": len(dataset.poses),
        "noise": {"corner_px": args.sigma_corner, "laser_px": args.sigma_laser},
        "fit": calibration.fit.to_dict(),
        "evaluation": plane_eval,
        "per_pose": per_pose,
        "ideal": ideal,
    }
    pc.write_json(out / "results" / "calibration.json", calib_result)
    fit_plane = calibration.fit.to_plane()
    fit_plane.source = "CALIBRATED_SIM"
    pc.write_json(out / "results" / "laser_plane.json", fit_plane.to_dict())
    pc.write_json(out / "results" / "validation.json", {
        "summary": val_summary,
        "rows": val_rows,
        "fit_plane": calibration.fit.to_dict(),
        "note": "reconstruction uses fitted plane; true height only for error",
    })
    pc.write_csv(out / "results" / "validation.csv", val_rows)
    pc.write_csv(out / "results" / "board_count.csv", board_count)
    pc.write_csv(out / "results" / "coverage.csv", coverage)
    pc.write_csv(out / "results" / "noise.csv", noise)
    pc.write_csv(out / "results" / "orientation.csv", dataset.orientation_rows)
    pc.write_json(out / "data" / "orientation.json", {
        "rows": dataset.orientation_rows,
        "aggregate": orientation_agg,
        "note": "diagnostic only; calibration uses laser_uv_row",
    })

    plt = _get_plt(args.show)
    figures = [
        plot_geometry(plt, out, args.show, dataset, calibration, board, plane_eval),
        plot_residuals(plt, out, args.show, calibration, per_pose),
        plot_board_count(plt, out, args.show, board_count),
        plot_coverage(plt, out, args.show, coverage),
        plot_noise(plt, out, args.show, noise),
        plot_validation(plt, out, args.show, val_rows),
        plot_orientation(plt, out, args.show, dataset, orientation_agg),
    ]

    # ---------------- 检查 ----------------
    checks = []
    checks.append(_check(
        "理想投影可精确恢复",
        ideal_eval["normal_angle_error_deg"] < 1e-3 and
        ideal_calib.fit.residual_rms_mm < 1e-3,
        f"angle={ideal_eval['normal_angle_error_deg']:.3e} deg, "
        f"rms={ideal_calib.fit.residual_rms_mm:.3e} mm",
    ))
    checks.append(_check(
        "主标定法向误差 < 0.5°",
        plane_eval["normal_angle_error_deg"] < 0.5,
        f"{plane_eval['normal_angle_error_deg']:.4f}°",
    ))
    checks.append(_check(
        "独立台阶验证均值 < 1 mm",
        val_summary["all_valid"] and val_summary["mean_abs_error_mm"] < 1.0,
        f"{val_summary['mean_abs_error_mm']:.4f} mm "
        f"(max {val_summary['max_abs_error_mm']:.4f} mm)",
    ))
    checks.append(_check(
        "近竖直条纹用 row 模式",
        orientation_agg["row_count_mean"] > 5 * max(1.0, orientation_agg["column_count_mean"])
        and orientation_agg["row_error_mean_px"] < orientation_agg["column_error_mean_px"],
        f"row {orientation_agg['row_count_mean']:.0f} centers / "
        f"{orientation_agg['row_error_mean_px']:.3f} px; "
        f"column {orientation_agg['column_count_mean']:.0f} / "
        f"{orientation_agg['column_error_mean_px']:.3f} px",
    ))
    checks.append(_check(
        "紧 ROI 后 auto 选 row",
        orientation_agg["auto_roi_axis_row_fraction"] > 0.9,
        f"row {orientation_agg['auto_roi_axis_row_fraction'] * 100:.0f}%",
    ))
    same_plane = [r for r in coverage if r["coverage"] == "same_plane"][0]
    checks.append(_check(
        "坏覆盖被条件数识别",
        same_plane["condition"] > 50 and same_plane["degenerate"],
        f"condition={same_plane['condition']:.1f}, degenerate={same_plane['degenerate']}",
    ))
    good = [r for r in coverage if r["coverage"] == "good"][0]
    fronto = [r for r in coverage if r["coverage"] == "fronto"][0]
    checks.append(_check(
        "好覆盖优于全正对",
        good["validation_mean_abs_mm"] < fronto["validation_mean_abs_mm"],
        f"good {good['validation_mean_abs_mm']:.3f} mm vs "
        f"fronto {fronto['validation_mean_abs_mm']:.3f} mm",
    ))
    corner05 = [r for r in noise if r["noise_type"] == "corner" and
                abs(r["sigma_px"] - 0.5) < 1e-9][0]
    laser05 = [r for r in noise if r["noise_type"] == "laser" and
               abs(r["sigma_px"] - 0.5) < 1e-9][0]
    checks.append(_check(
        "0.5 px 角点噪声验证仍 < 1 mm",
        corner05["validation_mean_abs_mean_mm"] < 1.0,
        f"{corner05['validation_mean_abs_mean_mm']:.4f} mm",
    ))
    checks.append(_check(
        "0.5 px 激光噪声验证仍 < 1 mm",
        laser05["validation_mean_abs_mean_mm"] < 1.0,
        f"{laser05['validation_mean_abs_mean_mm']:.4f} mm",
    ))
    leak = any(("true_plane" in m) or ("rvec" in m) or ("tvec" in m)
               for m in dataset.measurements)
    checks.append(_check(
        "标定输入不含真实平面/真实位姿",
        not leak,
        "measurements keys: " + ", ".join(sorted(dataset.measurements[0].keys())),
    ))
    checks.append(_check(
        "7 张图已生成",
        len([p for p in figures if Path(p).exists()]) == 7,
        f"{len(figures)} 张",
    ))

    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "unit": "mm",
        "n_boards": len(dataset.poses),
        "board_square_mm": board.square_mm,
        "fit": calibration.fit.to_dict(),
        "evaluation": plane_eval,
        "validation": val_summary,
        "ideal_validation": ideal_val_summary,
        "orientation": orientation_agg,
        "checks": checks,
    }
    pc.write_json(out / "results" / "summary.json", summary)

    report = build_report(
        args, camera, board, true_plane, dataset, calibration, plane_eval,
        per_pose, ideal, val_rows, val_summary, board_count, coverage,
        coverage_notes, noise, dataset.orientation_rows, orientation_agg,
        checks, out,
    )
    (out / "results" / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\n产物目录: {out.resolve()}")

    if not val_summary["all_valid"] or (args.strict and not all(c["passed"] for c in checks)):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(2)
