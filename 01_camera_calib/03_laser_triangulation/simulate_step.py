"""激光三角测量仿真：31 mm 台阶。

运行：
    python simulate_step.py                # 保存图/报告到 out/，不弹窗
    python simulate_step.py --show         # 同时显示图窗
    python simulate_step.py --no-image     # 跳过“渲染->提取”图像管线

这个脚本只把真值三维坐标用于最后算误差。反算函数 reconstruct() 只接收：
像素坐标、相机内参 K、激光平面 n/d。和真实硬件路径完全一致。
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import laser_lib as ll

K_TRUE = np.array([
    [806.0, 0.0, 640.0],
    [0.0, 806.0, 360.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
IMAGE_SIZE = (1280, 720)
PLANE_ABCD = (0.0, 1.0, 0.2, -120.0)


# ==========================================================================
# 基础工具
# ==========================================================================
def build_camera(fx: float = 806.0, fy: float | None = None) -> ll.CameraModel:
    K = K_TRUE.copy()
    K[0, 0] = float(fx)
    K[1, 1] = float(fx if fy is None else fy)
    return ll.CameraModel(K, np.zeros(5), IMAGE_SIZE, source="SYNTHETIC_NOMINAL")


def _fmt(value, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "-"
    v = float(value)
    if not np.isfinite(v):
        return "-"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _write_csv(path, rows) -> None:
    path = Path(path)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ==========================================================================
# 1. 像素噪声扫描
# ==========================================================================
def run_noise_sweep(camera, plane, scene, truth, uv_true, noise_levels, seed,
                    trials: int = 30):
    """每个噪声档做多次独立抽样；表中给均值/标准差，图里画其中一次。"""
    rows = []
    examples = {}
    for i, sigma in enumerate(noise_levels):
        n_trials = 1 if sigma <= 0 else max(2, int(trials))
        heights = []
        errors = []
        rmses = []
        ses = []
        example = None
        for t in range(n_trials):
            rng = np.random.default_rng(seed + 1000 * i + t)
            if sigma > 0:
                uv_meas = uv_true + rng.normal(0.0, sigma, uv_true.shape)
            else:
                uv_meas = uv_true.copy()

            rec = ll.reconstruct(uv_meas, camera, plane)
            est = ll.robust_step_height(rec.points_mm, axis=2)
            height = est.height_mm if est.has_step else float("nan")
            heights.append(height)
            errors.append(height - scene.height_mm)
            rmses.append(ll.rmse_3d(rec.points_mm, truth))
            ses.append(est.height_se_mm)
            if t == 0:
                example = {"uv": uv_meas, "rec": rec, "est": est}

        errors_arr = np.asarray(errors, dtype=np.float64)
        rows.append({
            "noise_px": float(sigma),
            "n_trials": n_trials,
            "height_mm": float(np.nanmean(heights)),
            "error_mm": float(np.nanmean(errors_arr)),
            "error_std_mm": float(np.nanstd(errors_arr, ddof=1)) if n_trials > 1 else 0.0,
            "max_abs_error_mm": float(np.nanmax(np.abs(errors_arr))),
            "height_se_mm": float(np.nanmean(ses)),
            "rmse_mm": float(np.nanmean(rmses)),
            "example_height_mm": float(heights[0]),
            "example_error_mm": float(errors[0]),
            "example_rmse_mm": float(rmses[0]),
            "n_valid": int(example["rec"].valid.sum()),
            "n_inliers": int(example["est"].n_inliers),
            "n_rejected": int(example["est"].n_outliers),
            "level_far_mm": float(example["est"].level_hi_mm),
            "level_near_mm": float(example["est"].level_lo_mm),
        })
        examples[float(sigma)] = example
    return rows, examples


# ==========================================================================
# 2. 测试 A：相机焦距错误
# ==========================================================================
def run_focal_test(camera, plane, scene, truth, uv_true, seed):
    rows = []
    profiles = []
    fx_values = [780.0, 800.0, 806.0, 820.0, 840.0]

    for fx in fx_values:
        # 真实相机的“焦距标定错”通常 fx/fy 一起偏；这个场景的激光平面是 Y-Z 平面，
        # 高度主要对 fy 敏感，所以两者都设成同一个错误值。
        cam_est = build_camera(fx, fx)
        uv_model = cam_est.project(truth)
        shift = np.linalg.norm(uv_model - uv_true, axis=1)

        rec = ll.reconstruct(uv_true, cam_est, plane)
        est = ll.robust_step_height(rec.points_mm, axis=2)
        height = est.height_mm if est.has_step else float("nan")
        error = height - scene.height_mm
        rmse = ll.rmse_3d(rec.points_mm, truth)

        rows.append({
            "f_est_px": float(fx),
            "fx_est_px": float(fx),
            "fy_est_px": float(fx),
            "height_mm": float(height),
            "error_mm": float(error),
            "rmse_mm": float(rmse),
            "model_shift_mean_px": float(np.mean(shift)),
            "model_shift_max_px": float(np.max(shift)),
            "n_inliers": int(est.n_inliers),
            "n_rejected": int(est.n_outliers),
        })

        if abs(fx - 806.0) > 0.5:  # 只给错误焦距画沿 X 的误差剖面
            profiles.append({
                "f_est_px": float(fx),
                "x_mm": truth[:, 0].tolist(),
                "z_error_mm": (rec.points_mm[:, 2] - truth[:, 2]).tolist(),
            })
    return {"rows": rows, "profiles": profiles}


# ==========================================================================
# 3. 测试 B：激光平面标定误差
# ==========================================================================
def run_plane_test(camera, plane, scene, truth, uv_true):
    y_far = float(plane.y_at_xz(0.0, scene.z_far))
    y_near = float(plane.y_at_xz(0.0, scene.z_near))
    pivot = np.array([0.0, 0.5 * (y_far + y_near), 0.5 * (scene.z_far + scene.z_near)])

    normal_rows = []
    profiles = []
    nz_values = [0.190, 0.195, 0.198, 0.200, 0.202, 0.205, 0.210]
    for nz in nz_values:
        normal = np.array([plane.normal[0], plane.normal[1], float(nz)], dtype=np.float64)
        plane_est = ll.LaserPlane.from_point_normal(pivot, normal,
                                                    source=f"PERTURBED n_z={nz:.4f}")
        rec = ll.reconstruct(uv_true, camera, plane_est)
        est = ll.robust_step_height(rec.points_mm, axis=2)
        height = est.height_mm if est.has_step else float("nan")
        error = height - scene.height_mm
        rmse = ll.rmse_3d(rec.points_mm, truth)
        normal_rows.append({
            "n_z": float(nz),
            "angle_deg": float(plane.angle_to(plane_est)),
            "d": float(plane_est.d),
            "height_mm": float(height),
            "error_mm": float(error),
            "rmse_mm": float(rmse),
            "n_inliers": int(est.n_inliers),
            "n_rejected": int(est.n_outliers),
        })
        if abs(nz - 0.2) > 1e-9:
            profiles.append({
                "n_z": float(nz),
                "x_mm": truth[:, 0].tolist(),
                "z_error_mm": (rec.points_mm[:, 2] - truth[:, 2]).tolist(),
            })

    offset_rows = []
    for offset in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        plane_est = plane.translated(offset)
        rec = ll.reconstruct(uv_true, camera, plane_est)
        est = ll.robust_step_height(rec.points_mm, axis=2)
        height = est.height_mm if est.has_step else float("nan")
        error = height - scene.height_mm
        offset_rows.append({
            "offset_mm": float(offset),
            "height_mm": float(height),
            "error_mm": float(error),
            "rmse_mm": float(ll.rmse_3d(rec.points_mm, truth)),
            "n_inliers": int(est.n_inliers),
            "n_rejected": int(est.n_outliers),
        })

    return {"pivot_mm": pivot.tolist(), "normal_rows": normal_rows,
            "offset_rows": offset_rows, "profiles": profiles}


# ==========================================================================
# 4. 测试 C：缺失点 / 异常点 / 非均匀噪声
# ==========================================================================
def _make_uv_case(kind, rng, uv_true, truth, scene):
    n = len(uv_true)
    sigma = np.full(n, 0.15, dtype=np.float64)
    if "nonuniform" in kind or kind == "combined":
        x_scale = max(abs(scene.x_min), abs(scene.x_max))
        sigma = 0.08 + 0.62 * (np.abs(truth[:, 0]) / x_scale) ** 2

    uv = np.array(uv_true, dtype=np.float64, copy=True)
    uv = uv + rng.normal(0.0, 1.0, uv.shape) * sigma[:, None]
    keep = np.ones(n, dtype=bool)
    if "missing" in kind or kind == "combined":
        drop = 0.25 if kind == "missing" else 0.20
        keep &= rng.random(n) > drop
    if "outliers" in kind or kind == "combined":
        candidates = np.flatnonzero(keep)
        n_out = max(1, int(round(0.04 * n)))
        idx = rng.choice(candidates, size=min(n_out, len(candidates)), replace=False)
        # 模拟反光/二次反射：向同一方向（v 正方向）偏 5~10 px，u 方向小幅抖动。
        uv[idx, 1] += rng.uniform(5.0, 10.0, len(idx))
        uv[idx, 0] += rng.uniform(-3.0, 3.0, len(idx))
    uv[~keep] = np.nan
    return uv


def run_robustness_test(camera, plane, scene, truth, roles, uv_true, seed):
    scenarios = [
        ("clean", "干净：0.15 px 均匀噪声"),
        ("missing", "缺失 25% 点"),
        ("outliers", "4% 异常点（同向偏 5~10 px）"),
        ("nonuniform", "非均匀噪声（0.08~0.70 px）"),
        ("combined", "缺失 20% + 异常 4% + 非均匀"),
    ]
    estimators = [
        "均值（已知分层）",
        "中位数（已知分层）",
        "自动分层+中位数",
        "自动分层+MAD剔除",
    ]
    detail_rows = []
    error_matrix = []
    summary_rows = []

    for i, (kind, desc) in enumerate(scenarios):
        rng = np.random.default_rng(seed + 301 + i)
        uv = _make_uv_case(kind, rng, uv_true, truth, scene)
        rec = ll.reconstruct(uv, camera, plane)
        valid = rec.valid
        pts = rec.points_mm[valid]
        truth_valid = truth[valid]
        roles_valid = roles[valid]

        far = roles_valid == 0
        near = roles_valid == 1
        if far.sum() < 2 or near.sum() < 2:
            raise RuntimeError(f"场景 {kind} 的有效点太少，无法做已知分层估计")

        h_mean = float(np.mean(pts[far, 2]) - np.mean(pts[near, 2]))
        h_median = float(np.median(pts[far, 2]) - np.median(pts[near, 2]))

        labels, _, _ = ll.two_level_split(pts[:, 2])
        h_auto_median = float(abs(np.median(pts[labels == 1, 2]) - np.median(pts[labels == 0, 2])))

        est = ll.robust_step_height(pts, axis=2)
        h_robust = float(est.height_mm) if est.has_step else float("nan")

        rmse_all = ll.rmse_3d(pts, truth_valid)
        if est.has_step and est.inlier_mask.any():
            rmse_robust = ll.rmse_3d(pts[est.inlier_mask], truth_valid[est.inlier_mask])
        else:
            rmse_robust = float("nan")

        values = [
            (estimators[0], h_mean, rmse_all, int(len(pts)), 0),
            (estimators[1], h_median, rmse_all, int(len(pts)), 0),
            (estimators[2], h_auto_median, rmse_all, int(len(pts)), 0),
            (estimators[3], h_robust, rmse_robust, int(est.n_inliers), int(est.n_outliers)),
        ]
        error_row = {"scenario": desc}
        for name, height, rmse, n_in, n_out in values:
            error_value = height - scene.height_mm if np.isfinite(height) else float("nan")
            detail_rows.append({
                "scenario": kind,
                "scenario_desc": desc,
                "estimator": name,
                "height_mm": float(height),
                "error_mm": float(error_value),
                "abs_error_mm": float(abs(error_value)) if np.isfinite(error_value) else float("nan"),
                "n_points": int(len(pts)),
                "n_inliers": int(n_in),
                "n_outliers": int(n_out),
                "rmse_mm": float(rmse),
            })
            error_row[name] = float(abs(error_value))
        error_matrix.append(error_row)

        summary_rows.append({
            "scenario": kind,
            "scenario_desc": desc,
            "n_points": int(len(pts)),
            "robust_height_mm": h_robust,
            "robust_abs_error_mm": abs(h_robust - scene.height_mm) if np.isfinite(h_robust) else float("nan"),
            "n_inliers": int(est.n_inliers),
            "n_outliers": int(est.n_outliers),
            "robust_rmse_mm": float(rmse_robust),
        })

    return {"scenarios": scenarios, "estimators": estimators,
            "detail_rows": detail_rows, "error_matrix": error_matrix,
            "summary_rows": summary_rows}


# ==========================================================================
# 5. 图像管线：渲染 -> 提取 -> 反算
# ==========================================================================
def run_image_pipeline(camera, plane, scene, truth, roles, uv_true, seed, out, plt, show):
    u_min, u_max = float(uv_true[:, 0].min()), float(uv_true[:, 0].max())
    v_min, v_max = float(uv_true[:, 1].min()), float(uv_true[:, 1].max())
    roi = (
        max(0, int(np.ceil(u_min))),
        min(IMAGE_SIZE[0], int(np.floor(u_max)) + 1),
        max(0, int(np.floor(v_min - 12))),
        min(IMAGE_SIZE[1], int(np.ceil(v_max + 12))),
    )

    envelope = 1.0 - 0.35 * np.exp(-((uv_true[:, 0] - 770.0) / 35.0) ** 2) \
        - 0.25 * np.exp(-((uv_true[:, 0] - 520.0) / 20.0) ** 2)
    envelope = np.clip(envelope, 0.25, 1.0)

    img_clean = ll.render_laser_image(
        uv_true, IMAGE_SIZE, sigma_px=1.3, peak=190.0, amplitude=None,
        background=9.0, gradient=(4.0, 2.0), read_sigma=2.0, shot_gain=0.25,
        rng=np.random.default_rng(seed + 501),
    )
    img_hard = ll.render_laser_image(
        uv_true, IMAGE_SIZE, sigma_px=1.3, peak=190.0, amplitude=envelope,
        background=10.0, gradient=(6.0, 3.0), read_sigma=3.0, shot_gain=0.40,
        dropout_spans=[(560.0, 590.0), (700.0, 720.0), (810.0, 830.0)],
        hotspot_count=8, hotspot_amp=(120.0, 240.0), hotspot_sigma=(1.5, 3.5),
        rng=np.random.default_rng(seed + 502),
    )

    cases = [("clean", "干净图", img_clean), ("hard", "缺失+反光+亮度不均", img_hard)]
    rows = []
    details = {}
    point_rows = []

    for name, desc, image in cases:
        img_path = out / f"laser_{name}.png"
        plt.imsave(img_path, ll.to_uint8(image), cmap="gray", vmin=0, vmax=255)

        ext = ll.extract_laser_centers(
            image, axis="auto", roi=roi, smooth_sigma=1.0, threshold_sigma=4.0,
            min_peak=12.0, centroid_radius=4, max_width=9.0,
        )
        rec = ll.reconstruct(ext.uv, camera, plane)
        est = ll.robust_step_height(rec.points_mm[rec.valid], axis=2)
        height = est.height_mm if est.has_step else float("nan")
        height_error = height - scene.height_mm

        center_err = ll.distance_to_polyline(ext.uv, uv_true) if len(ext.uv) else np.array([])
        if len(center_err):
            center_mean = float(np.mean(center_err))
            center_p95 = float(np.percentile(center_err, 95))
            center_max = float(np.max(center_err))
        else:
            center_mean = center_p95 = center_max = float("nan")

        # 最近真值点的 Z 作为该像素对应的近似真值深度（只用于误差统计）
        if len(ext.uv):
            d2 = ((ext.uv[:, None, :] - uv_true[None, :, :]) ** 2).sum(axis=-1)
            nn = np.argmin(d2, axis=1)
            truth_z = truth[nn, 2]
            z_err = rec.points_mm[rec.valid, 2] - truth_z
            z_rmse = float(np.sqrt(np.mean(z_err ** 2)))
        else:
            z_rmse = float("nan")

        row = {
            "case": name,
            "description": desc,
            "image": img_path.name,
            "extracted_points": int(len(ext.uv)),
            "valid_fraction": float(ext.valid_fraction),
            "center_error_mean_px": center_mean,
            "center_error_p95_px": center_p95,
            "center_error_max_px": center_max,
            "height_mm": float(height),
            "height_error_mm": float(height_error),
            "height_se_mm": float(est.height_se_mm),
            "z_rmse_mm": z_rmse,
            "n_inliers": int(est.n_inliers),
            "n_outliers": int(est.n_outliers),
            "roi_xy": list(roi),
        }
        rows.append(row)
        details[name] = {"image": image, "extraction": ext, "reconstruction": rec,
                         "estimate": est, "row": row}

        for j, (u, v) in enumerate(ext.uv):
            point_rows.append({
                "case": name,
                "u": float(u),
                "v": float(v),
                "center_error_px": float(center_err[j]) if len(center_err) else float("nan"),
                "X_mm": float(rec.points_mm[j, 0]),
                "Y_mm": float(rec.points_mm[j, 1]),
                "Z_mm": float(rec.points_mm[j, 2]),
                "truth_Z_mm": float(truth[np.argmin(((ext.uv[j] - uv_true) ** 2).sum(axis=1)), 2]),
            })

    _plot_image_pipeline(plt, out, show, IMAGE_SIZE, uv_true, details)
    return {"rows": rows, "roi_xy": list(roi), "point_rows": point_rows}


# ==========================================================================
# 绘图
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


def _save(fig, out, name, plt, show):
    path = out / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def _plot_pixels(plt, out, show, noise_rows, examples, uv_true, roles, scene):
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(len(noise_rows), 2, figsize=(11.5, 3.3 * len(noise_rows)))
    if len(noise_rows) == 1:
        axes = np.atleast_2d(axes)

    u_min, u_max = float(uv_true[:, 0].min()), float(uv_true[:, 0].max())
    v_min, v_max = float(uv_true[:, 1].min()), float(uv_true[:, 1].max())
    wall = uv_true[roles == 2]
    step_u = float(np.mean(wall[:, 0])) if len(wall) else float(uv_true[len(uv_true) // 2, 0])
    zoom = (step_u - 55.0, step_u + 55.0, v_min - 10.0, v_max + 10.0)

    for i, row in enumerate(noise_rows):
        sigma = row["noise_px"]
        uv = examples[sigma]["uv"]
        ax_full, ax_zoom = axes[i, 0], axes[i, 1]

        ax_full.scatter(uv[:, 0], uv[:, 1], s=2, alpha=0.55, color="tab:blue", label="measured")
        ax_full.plot(uv_true[:, 0], uv_true[:, 1], "k-", lw=0.9, label="truth")
        ax_full.set_xlim(u_min - 30, u_max + 30)
        ax_full.set_ylim(v_max + 18, v_min - 18)
        ax_full.set_xlabel("u (pixel)")
        ax_full.set_ylabel("v (pixel)")
        ax_full.set_title(f"full line, noise sigma = {sigma:.2f} px")
        if i == 0:
            ax_full.legend(loc="upper right", markerscale=2)

        ax_full.add_patch(Rectangle((zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2],
                                    fill=False, ec="tab:red", lw=1.0))

        ax_zoom.scatter(uv[:, 0], uv[:, 1], s=8, alpha=0.7, color="tab:blue")
        ax_zoom.plot(uv_true[:, 0], uv_true[:, 1], "k-", lw=1.0)
        ax_zoom.set_xlim(zoom[0], zoom[1])
        ax_zoom.set_ylim(zoom[3], zoom[2])
        ax_zoom.set_xlabel("u (pixel)")
        ax_zoom.set_ylabel("v (pixel)")
        ax_zoom.set_title(f"step edge zoom (true H = {scene.height_mm:.1f} mm)")

    fig.suptitle("Simulated laser-line pixels on the 31 mm step", y=1.00, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig01_laser_pixels.png", plt, show)


def _plot_profiles(plt, out, show, noise_rows, examples, truth, scene):
    fig, axes = plt.subplots(1, len(noise_rows), figsize=(4.4 * len(noise_rows), 4.0))
    axes = np.atleast_1d(axes)
    for ax, row in zip(axes, noise_rows):
        sigma = row["noise_px"]
        est = examples[sigma]["est"]
        pts = examples[sigma]["rec"].points_mm
        ax.plot(truth[:, 0], truth[:, 2], "k--", lw=1.1, label="ground truth")
        ax.plot(pts[:, 0], pts[:, 2], ".", ms=2.2, color="tab:red", label="reconstructed")
        ax.invert_yaxis()
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Z depth (mm)")
        ax.set_title(f"noise {sigma:.2f} px (one draw): H = {est.height_mm:.3f} mm")
        ax.annotate("", xy=(scene.x_max - 18, scene.z_near),
                    xytext=(scene.x_max - 18, scene.z_far),
                    arrowprops=dict(arrowstyle="<->", color="tab:blue", lw=1.0))
        ax.text(scene.x_max - 42, 0.5 * (scene.z_near + scene.z_far) + 4,
                f"{est.height_mm:.3f} mm", color="tab:blue", va="center", fontsize=8)
        ax.legend(loc="center left", fontsize=7)
    fig.suptitle("Reconstructed step profile (camera-frame Z)", y=1.02, fontsize=11)
    fig.tight_layout()
    return _save(fig, out, "fig02_step_profile.png", plt, show)


def _plot_3d(plt, out, show, noise_rows, examples, truth, scene):
    pick = noise_rows[min(1, len(noise_rows) - 1)]["noise_px"]
    pts = examples[pick]["rec"].points_mm
    est = examples[pick]["est"]

    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(truth[:, 0], truth[:, 1], truth[:, 2], color="0.55", ls="--", lw=1.0, label="ground truth")
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap="viridis", s=6, label="reconstructed")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"3D laser line in camera frame (noise {pick:.2f} px, H={est.height_mm:.3f} mm)")
    ax.view_init(elev=22, azim=-62)
    ax.set_box_aspect((1.7, 0.55, 0.9))
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.08)
    cb.set_label("Z (mm)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return _save(fig, out, "fig03_3d_view.png", plt, show)


def _plot_focal(plt, out, show, focal, scene):
    rows = focal["rows"]
    fx = np.array([r["f_est_px"] for r in rows])
    err = np.array([r["error_mm"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(806.0, color="tab:blue", ls="--", lw=1.0, label="true f = 806 px")
    ax.plot(fx, err, "o-", color="tab:red", label="height error")
    for x, y in zip(fx, err):
        ax.annotate(f"{y:+.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7)
    ax.set_xlabel("f used for reconstruction (fx=fy, px)")
    ax.set_ylabel("height error (mm)")
    ax.set_title("Test A: wrong camera focal length")
    ax.legend()

    ax = axes[1]
    ax.axhline(0.0, color="k", lw=0.8)
    for prof in focal["profiles"]:
        ax.plot(prof["x_mm"], prof["z_error_mm"], ".", ms=3,
                label=f"f={prof['f_est_px']:.0f} px")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z error (mm)")
    ax.set_title("Systematic Z error along the laser line")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7,
              borderaxespad=0.0)
    fig.suptitle(f"Known step: {scene.height_mm:.1f} mm; images generated with fx=fy=806 px", y=1.02)
    fig.tight_layout()
    return _save(fig, out, "fig04_focal_error.png", plt, show)


def _plot_plane(plt, out, show, plane_test, scene):
    rows = plane_test["normal_rows"]
    nz = np.array([r["n_z"] for r in rows])
    err = np.array([r["error_mm"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(0.2, color="tab:blue", ls="--", lw=1.0, label="nominal n_z = 0.2")
    ax.plot(nz, err, "o-", color="tab:red", label="height error")
    for x, y in zip(nz, err):
        ax.annotate(f"{y:+.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7)
    ax.set_xlabel("n_z used for reconstruction")
    ax.set_ylabel("height error (mm)")
    ax.set_title("Test B: laser-plane normal error")
    ax.legend()

    ax = axes[1]
    ax.axhline(0.0, color="k", lw=0.8)
    for prof in plane_test["profiles"]:
        ax.plot(prof["x_mm"], prof["z_error_mm"], ".", ms=3,
                label=f"n_z={prof['n_z']:.3f}")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z error (mm)")
    ax.set_title("Systematic Z error along the laser line")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7,
              borderaxespad=0.0)
    fig.suptitle(f"Known step: {scene.height_mm:.1f} mm; nominal plane n=[0,1,0.2], d=-120", y=1.02)
    fig.tight_layout()
    return _save(fig, out, "fig05_plane_error.png", plt, show)


def _plot_robustness(plt, out, show, robust, scene):
    scenarios = robust["scenarios"]
    estimators = robust["estimators"]
    rows = robust["error_matrix"]
    short = ["clean", "missing", "outliers", "nonuniform", "combined"]

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    x = np.arange(len(scenarios))
    width = 0.2
    legend_labels = ["mean (known split)", "median (known split)",
                     "auto split + median", "auto split + MAD"]
    for k, name in enumerate(estimators):
        vals = [r[name] for r in rows]
        ax.bar(x + (k - 1.5) * width, vals, width, label=legend_labels[k])
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.axhline(0.5, color="k", ls=":", lw=1.0, label="0.5 mm")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel("| height error | (mm)")
    ax.set_title(f"Test C: robustness to missing points, outliers and non-uniform noise (true H={scene.height_mm:.1f} mm)")
    ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    return _save(fig, out, "fig06_robustness.png", plt, show)


def _plot_image_pipeline(plt, out, show, image_size, uv_true, details):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0))
    for i, name in enumerate(["clean", "hard"]):
        if name not in details:
            continue
        d = details[name]
        row = d["row"]
        ext = d["extraction"]
        ax_img = axes[i, 0]
        ax_img.imshow(d["image"], cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax_img.plot(uv_true[:, 0], uv_true[:, 1], color="cyan", lw=0.9, label="truth")
        ax_img.scatter(ext.uv[:, 0], ext.uv[:, 1], s=2, color="red", label="extracted")
        x0, x1, y0, y1 = row["roi_xy"]
        ax_img.set_xlim(x0 - 20, x1 + 20)
        ax_img.set_ylim(y1 + 15, y0 - 15)
        ax_img.set_xlabel("u (pixel)")
        ax_img.set_ylabel("v (pixel)")
        ax_img.set_title(f"{name}: extracted {row['extracted_points']} centers, "
                         f"H={row['height_mm']:.3f} mm")
        ax_img.legend(loc="upper right", fontsize=7, markerscale=2)

        ax_prof = axes[i, 1]
        ax_prof.plot(ext.scan, ext.center, ".", ms=3, color="red", label="extracted center")
        ax_prof.plot(uv_true[:, 0], uv_true[:, 1], color="tab:cyan", lw=1.0, label="truth")
        ax_prof.invert_yaxis()
        ax_prof.set_xlabel("u (pixel)")
        ax_prof.set_ylabel("v (pixel)")
        ax_prof.set_title(f"{name}: v(u) profile; center error mean "
                          f"{row['center_error_mean_px']:.3f} px")
        ax_prof.legend(fontsize=7)

    fig.suptitle("Image pipeline: render -> extract -> reconstruct (same code path as webcam)", y=1.01)
    fig.tight_layout()
    return _save(fig, out, "fig07_image_pipeline.png", plt, show)


# ==========================================================================
# 报告
# ==========================================================================
def build_report(scene, camera, plane, noise_rows, focal, plane_test, robust,
                 image_pipeline, checks, out):
    lines = []
    lines.append("=" * 72)
    lines.append("线激光三角测量仿真：31 mm 台阶")
    lines.append("=" * 72)
    lines.append(f"真值高度       : {scene.height_mm:.3f} mm")
    lines.append(f"相机           : fx={camera.fx:.3f}, fy={camera.fy:.3f}, "
                 f"cx={camera.cx:.3f}, cy={camera.cy:.3f}, "
                 f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}, "
                 f"{'有畸变' if camera.has_distortion else '无畸变'}")
    lines.append(f"激光平面       : n={np.round(plane.normal, 6).tolist()}, d={plane.d:.3f}"
                 f"  (n·X+d=0, 相机坐标系, mm)")
    lines.append(f"台阶           : 远面 Z={scene.z_far:.3f}, 近面 Z={scene.z_near:.3f}, "
                 f"X∈[{scene.x_min:.1f},{scene.x_max:.1f}], 台阶边 X={scene.step_x:.1f}")
    lines.append("说明           : 反算只用像素、K、激光平面；真值只在最后算误差。")
    lines.append("")

    lines.append("[1] 像素噪声的影响（每档独立抽样 30 次，表中为均值/标准差）")
    rows = []
    for r in noise_rows:
        rows.append([
            f"{r['noise_px']:.2f}",
            _fmt(r["height_mm"]),
            _fmt(r["error_mm"], signed=True),
            _fmt(r["error_std_mm"], 4),
            _fmt(r["max_abs_error_mm"], 4),
            _fmt(r["height_se_mm"], 4),
            _fmt(r["rmse_mm"], 6),
            str(r["n_trials"]),
        ])
    lines.append(ll.format_table(
        ["噪声(px)", "平均高度(mm)", "平均误差(mm)", "误差标准差(mm)",
         "最大|误差|(mm)", "内部标准误(mm)", "平均3D RMSE(mm)", "抽样次数"],
        rows, ["<", ">", ">", ">", ">", ">", ">", ">"],
    ))
    lines.append("0 px 一档应在 1e-9 mm 量级内；0.15/0.5 px 一行是多次独立抽样的统计量，")
    lines.append("“误差标准差”反映随机误差大小，“内部标准误”是单次估计自身给出的不确定度。")
    lines.append("图窗只画其中一次抽样，因此图上的示例高度和表中的平均值会略有差别。")
    lines.append("")

    lines.append("[2] 测试 A：相机焦距错误（用 806 px 生成，故意用错误 fx=fy 反算）")
    rows = []
    for r in focal["rows"]:
        rows.append([
            f"{r['f_est_px']:.0f}",
            _fmt(r["height_mm"]),
            _fmt(r["error_mm"], signed=True),
            _fmt(r["rmse_mm"], 4),
            _fmt(r["model_shift_mean_px"], 4),
            _fmt(r["model_shift_max_px"], 4),
        ])
    lines.append(ll.format_table(
        ["f_est(px)", "高度(mm)", "高度误差(mm)", "三维RMSE(mm)",
         "模型位移均值(px)", "模型位移最大(px)"],
        rows, ["<", ">", ">", ">", ">", ">"],
    ))
    lines.append("806 是控制组；800/820 的系统性偏差说明焦距标定误差会直接进入高度。")
    lines.append("本场景的激光平面是 Y-Z 平面，高度主要对 fy 敏感，因此测试中让 fx=fy 一起偏。")
    lines.append("")

    lines.append("[3] 测试 B：激光平面标定误差")
    lines.append("  (a) 法向 n_z 扰动（绕测量体积中心旋转，保持枢轴点不变）")
    rows = []
    for r in plane_test["normal_rows"]:
        rows.append([
            f"{r['n_z']:.3f}",
            _fmt(r["angle_deg"], 4),
            _fmt(r["height_mm"]),
            _fmt(r["error_mm"], signed=True),
            _fmt(r["rmse_mm"], 4),
        ])
    lines.append(ll.format_table(
        ["n_z", "法向夹角(°)", "高度(mm)", "高度误差(mm)", "三维RMSE(mm)"],
        rows, ["<", ">", ">", ">", ">"],
    ))
    lines.append("  (b) 平面整体平移（沿单位法向）")
    rows = []
    for r in plane_test["offset_rows"]:
        rows.append([
            f"{r['offset_mm']:+.1f}",
            _fmt(r["height_mm"]),
            _fmt(r["error_mm"], signed=True),
            _fmt(r["rmse_mm"], 4),
        ])
    lines.append(ll.format_table(
        ["平移(mm)", "高度(mm)", "高度误差(mm)", "三维RMSE(mm)"],
        rows, ["<", ">", ">", ">"],
    ))
    lines.append("0.198/0.205 已经能让 31 mm 高度产生明显的系统性偏差；")
    lines.append("真实标定要把激光平面法向稳定到 1e-3 量级以内，并尽量把平面误差控制在 0.5 mm 以内。")
    lines.append("")

    lines.append("[4] 测试 C：缺失点、异常点、非均匀噪声（|高度误差|，mm）")
    rows = []
    for r in robust["error_matrix"]:
        rows.append([r["scenario"]] + [_fmt(r[name], 3) for name in robust["estimators"]])
    lines.append(ll.format_table(
        ["场景"] + robust["estimators"], rows, ["<", ">", ">", ">", ">"],
    ))
    lines.append("")
    rows = []
    for r in robust["summary_rows"]:
        rows.append([
            r["scenario"],
            str(r["n_points"]),
            _fmt(r["robust_height_mm"]),
            _fmt(r["robust_abs_error_mm"]),
            str(r["n_inliers"]),
            str(r["n_outliers"]),
            _fmt(r["robust_rmse_mm"], 4),
        ])
    lines.append(ll.format_table(
        ["场景", "有效点", "自动MAD高度(mm)", "|误差|(mm)", "内点", "剔除", "3D RMSE(mm)"],
        rows, ["<", ">", ">", ">", ">", ">", ">"],
    ))
    lines.append("“已知分层”两列只是用仿真标签做对照；真实算法是最后一列“自动分层+MAD剔除”。")
    lines.append("")

    if image_pipeline is not None:
        lines.append("[5] 图像管线：渲染 -> 提取 -> 反算（与真实相机路径相同）")
        rows = []
        for r in image_pipeline["rows"]:
            rows.append([
                r["case"],
                str(r["extracted_points"]),
                f"{r['valid_fraction'] * 100:.1f}%",
                _fmt(r["center_error_mean_px"], 3),
                _fmt(r["center_error_p95_px"], 3),
                _fmt(r["height_mm"]),
                _fmt(r["height_error_mm"], signed=True),
                _fmt(r["z_rmse_mm"], 3),
            ])
        lines.append(ll.format_table(
            ["图像", "有效列", "有效比例", "中心误差均值(px)", "中心误差P95(px)",
             "高度(mm)", "高度误差(mm)", "Z RMSE(mm)"],
            rows, ["<", ">", ">", ">", ">", ">", ">", ">"],
        ))
        lines.append("clean = 正常采集；hard = 缺失 + 反光热点 + 亮度不均。")
        lines.append("")
        lines.append("可直接用同一组配置复现真实图像路径：")
        lines.append("  python laser_measure.py image --image out/laser_clean.png "
                     "--calib out/synthetic_camera.json --plane out/synthetic_plane.json")
        lines.append("")

    lines.append("[6] 完成标准检查")
    for c in checks:
        mark = "✓" if c["passed"] else "✗"
        lines.append(f"  {mark} {c['name']}: {c['detail']}")
    passed = sum(1 for c in checks if c["passed"])
    lines.append(f"检查: {passed}/{len(checks)} 通过")
    lines.append("")

    lines.append("[产物]")
    for name in [
        "summary.json", "report.txt", "noise_sweep.csv", "test_a_focal.csv",
        "test_b_plane_normal.csv", "test_b_plane_offset.csv",
        "test_c_robustness.csv", "fig01_laser_pixels.png", "fig02_step_profile.png",
        "fig03_3d_view.png", "fig04_focal_error.png", "fig05_plane_error.png",
        "fig06_robustness.png", "fig07_image_pipeline.png",
        "synthetic_camera.json", "synthetic_plane.json", "laser_clean.png",
        "laser_hard.png", "image_pipeline.csv", "image_pipeline_points.csv",
    ]:
        if (out / name).exists():
            lines.append(f"  {out.name}/{name}")
    lines.append("=" * 72)
    return "\n".join(lines)


# ==========================================================================
# 主流程
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="31 mm 台阶的激光三角测量仿真")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--height", type=float, default=31.0, help="台阶真值高度 mm")
    ap.add_argument("--z-far", type=float, default=500.0, help="远面到相机距离 mm")
    ap.add_argument("--x-half", type=float, default=120.0, help="台阶半宽 mm")
    ap.add_argument("--show", action="store_true", help="显示 matplotlib 图窗（默认只存 PNG）")
    ap.add_argument("--no-image", action="store_true", help="跳过合成图像管线")
    ap.add_argument("--no-strict", action="store_true", help="检查失败也返回 0")
    args = ap.parse_args()

    if not (0 < args.height < args.z_far):
        raise ValueError("要求 0 < height < z_far")
    if args.x_half <= 0:
        raise ValueError("x-half 必须为正")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scene = ll.StepScene(
        z_far=float(args.z_far),
        z_near=float(args.z_far - args.height),
        x_min=-float(args.x_half),
        x_max=float(args.x_half),
    )
    camera = build_camera(806.0)
    plane = ll.LaserPlane.from_abcd(PLANE_ABCD, source="SYNTHETIC_NOMINAL")
    truth, roles = scene.curve_points(plane)
    uv_true = camera.project(truth)

    if not np.isfinite(uv_true).all():
        raise RuntimeError("场景投影出现 NaN，检查相机/平面参数")
    if uv_true[:, 0].min() < 0 or uv_true[:, 0].max() >= IMAGE_SIZE[0] \
            or uv_true[:, 1].min() < 0 or uv_true[:, 1].max() >= IMAGE_SIZE[1]:
        raise ValueError("台阶超出画面，减小 --x-half 或调整 --z-far")

    ll.write_json(out / "synthetic_camera.json", camera.to_dict())
    ll.write_json(out / "synthetic_plane.json", plane.to_dict())
    ll.write_json(out / "scene.json", scene.to_dict())

    noise_levels = [0.0, 0.15, 0.5]
    noise_rows, examples = run_noise_sweep(
        camera, plane, scene, truth, uv_true, noise_levels, args.seed)
    focal = run_focal_test(camera, plane, scene, truth, uv_true, args.seed)
    plane_test = run_plane_test(camera, plane, scene, truth, uv_true)

    robust = run_robustness_test(
        camera, plane, scene, truth, roles, uv_true, args.seed)

    plt = _get_plt(args.show)
    image_pipeline = None
    if not args.no_image:
        image_pipeline = run_image_pipeline(
            camera, plane, scene, truth, roles, uv_true, args.seed, out, plt, args.show)

    _plot_pixels(plt, out, args.show, noise_rows, examples, uv_true, roles, scene)
    _plot_profiles(plt, out, args.show, noise_rows, examples, truth, scene)
    _plot_3d(plt, out, args.show, noise_rows, examples, truth, scene)
    _plot_focal(plt, out, args.show, focal, scene)
    _plot_plane(plt, out, args.show, plane_test, scene)
    _plot_robustness(plt, out, args.show, robust, scene)
    if image_pipeline is not None:
        # 图像管线图已在 run_image_pipeline 内保存
        pass

    # ---------------- 检查 ----------------
    checks = []
    r0 = noise_rows[0]
    checks.append(_check(
        "无噪声精确恢复",
        abs(r0["height_mm"] - scene.height_mm) < 1e-6 and r0["rmse_mm"] < 1e-6,
        f"H={r0['height_mm']:.9f} mm, RMSE={r0['rmse_mm']:.3e} mm",
    ))
    noise_ok = all(np.isfinite(r["height_mm"])
                   and abs(r["error_mm"]) < 0.25
                   and r["error_std_mm"] < 0.35
                   for r in noise_rows[1:])
    checks.append(_check(
        "0.15/0.5 px 噪声可量化",
        noise_ok,
        "; ".join(f"{r['noise_px']:.2f} px -> {r['error_mm']:+.3f} ± {r['error_std_mm']:.3f} mm"
                  for r in noise_rows[1:]),
    ))

    focal_by_fx = {round(r["f_est_px"], 1): r for r in focal["rows"]}
    e800 = abs(focal_by_fx[800.0]["error_mm"])
    e820 = abs(focal_by_fx[820.0]["error_mm"])
    checks.append(_check(
        "焦距错误产生系统性偏差",
        e800 > 0.05 and e820 > 0.05,
        f"f=800 -> {e800:.3f} mm, f=820 -> {e820:.3f} mm (fx=fy)",
    ))

    plane_by_nz = {round(r["n_z"], 3): r for r in plane_test["normal_rows"]}
    e198 = abs(plane_by_nz[0.198]["error_mm"])
    e205 = abs(plane_by_nz[0.205]["error_mm"])
    checks.append(_check(
        "激光平面法向误差产生系统性偏差",
        e198 > 0.05 and e205 > 0.05,
        f"n_z=0.198 -> {e198:.3f} mm, n_z=0.205 -> {e205:.3f} mm",
    ))

    if image_pipeline is not None:
        rows_by_case = {r["case"]: r for r in image_pipeline["rows"]}
        clean_err = abs(rows_by_case["clean"]["height_error_mm"])
        hard_err = abs(rows_by_case["hard"]["height_error_mm"])
        checks.append(_check(
            "图像管线恢复台阶",
            clean_err < 0.3 and hard_err < 1.0,
            f"clean {clean_err:.3f} mm, hard {hard_err:.3f} mm",
        ))

    combined = [r for r in robust["summary_rows"] if r["scenario"] == "combined"][0]
    checks.append(_check(
        "缺失+异常+非均匀下仍稳健",
        combined["robust_abs_error_mm"] < 0.5,
        f"|error|={combined['robust_abs_error_mm']:.3f} mm, "
        f"内点 {combined['n_inliers']}, 剔除 {combined['n_outliers']}",
    ))

    figures = sorted(out.glob("fig*.png"))
    checks.append(_check(
        "图表已生成",
        len(figures) >= 6,
        f"{len(figures)} 张 PNG",
    ))

    # ---------------- 保存数据 ----------------
    # 逐点数据放 CSV（image_pipeline_points.csv），summary.json 只留汇总，避免文件过大。
    image_summary = None
    if image_pipeline is not None:
        image_summary = {"rows": image_pipeline["rows"], "roi_xy": image_pipeline["roi_xy"]}

    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "unit": "mm",
        "frame": "camera: X right, Y down, Z forward",
        "truth_height_mm": scene.height_mm,
        "config": {
            "camera": camera.to_dict(),
            "laser_plane": plane.to_dict(),
            "scene": scene.to_dict(),
            "seed": int(args.seed),
            "noise_levels_px": noise_levels,
        },
        "noise_sweep": noise_rows,
        "test_a_focal": focal,
        "test_b_plane": plane_test,
        "test_c_robustness": robust,
        "image_pipeline": image_summary,
        "checks": checks,
    }
    ll.write_json(out / "summary.json", ll.clean_json(summary))
    _write_csv(out / "noise_sweep.csv", noise_rows)
    _write_csv(out / "test_a_focal.csv", focal["rows"])
    _write_csv(out / "test_b_plane_normal.csv", plane_test["normal_rows"])
    _write_csv(out / "test_b_plane_offset.csv", plane_test["offset_rows"])
    _write_csv(out / "test_c_robustness.csv", robust["detail_rows"])
    if image_pipeline is not None:
        _write_csv(out / "image_pipeline.csv", image_pipeline["rows"])
        _write_csv(out / "image_pipeline_points.csv", image_pipeline["point_rows"])

    report = build_report(scene, camera, plane, noise_rows, focal, plane_test,
                          robust, image_pipeline, checks, out)
    (out / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\n产物目录: {out.resolve()}")

    if not all(c["passed"] for c in checks) and not args.no_strict:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(2)
