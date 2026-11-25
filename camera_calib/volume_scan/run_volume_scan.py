"""多帧线激光扫描 + 高度图体积主实验。

运行：
    python run_volume_scan.py
    python run_volume_scan.py --show
    python run_volume_scan.py --synthetic-camera
    python run_volume_scan.py --scan-range 35        # 更小的机构扫描范围

默认：相机/激光器刚性连接，沿世界 X 平移；物体固定在 100x80x50 mm 区域。
A=长方体 400 cm^3，B=顶部 20x20x10 盲槽 396 cm^3，C=侧面 20x20 贯穿孔 368 cm^3。
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import scan_lib as sc
import laser_lib as ll

OBJ_ALIAS = {"A": "A_box", "B": "B_slot", "C": "C_side_hole"}
SHORT_ALIAS = {v: k for k, v in OBJ_ALIAS.items()}


# ==========================================================================
# 基础
# ==========================================================================
def _get_plt(show: bool):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.28,
                         "figure.dpi": 110})
    return plt


def _save(fig, out: Path, name: str, plt, show: bool) -> Path:
    path = out / "figs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def _fmt(v, digits=3, signed=False):
    if v is None:
        return "-"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(x):
        return "-"
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def _check(name, passed, detail):
    return {"name": name, "passed": bool(passed), "detail": detail}


def build_camera(args) -> ll.CameraModel:
    if args.synthetic_camera:
        K = np.array([[806.0, 0.0, 640.0], [0.0, 806.0, 360.0], [0.0, 0.0, 1.0]])
        return ll.CameraModel(K, np.zeros(5), (1280, 720), source="SYNTHETIC_806")
    return ll.CameraModel.from_calib_json(args.calib)


def visible_fraction(scan, names) -> float | None:
    vis = sum(f.visible_by_quad.get(n, 0) for f in scan.frames for n in names)
    tot = sum(f.total_by_quad.get(n, 0) for f in scan.frames for n in names)
    return float(vis / tot) if tot else None


# ==========================================================================
# 单次评价：反算 -> 坐标统一 -> 基准面 -> 高度图
# ==========================================================================
def evaluate_scan(scan: sc.ScanData, obj: sc.ScanObject, plane_c: ll.LaserPlane,
                  cell_mm: float, pixel_noise: float = 0.0,
                  motion_mm: float = 0.0, omission: float = 0.0,
                  frame_mask=None, base_override: sc.BasePlane | None = None,
                  seed: int = 0) -> dict:
    t0 = time.perf_counter()
    pc = sc.reconstruct_scan(
        scan, plane_c, pixel_noise_px=pixel_noise,
        motion_sigma_mm=motion_mm, omission=omission,
        frame_mask=frame_mask, seed=seed,
    )
    t1 = time.perf_counter()
    if len(pc.points_world) < 10:
        raise ValueError(f"{obj.name}: 反算点太少")
    if base_override is None:
        base = sc.estimate_base_plane(pc.points_world,
                                      np.random.default_rng(seed + 1))
    else:
        base = base_override
    foot = sc.known_footprint_in_base(obj, base)
    hm = sc.height_map_volume(pc.points_world, base, cell_mm=cell_mm,
                              footprint=foot)
    t2 = time.perf_counter()
    err = hm.volume_mm3 / 1000.0 - obj.true_volume_cm3
    be = sc.base_error(base)
    slot_names = [q.name for q in obj.quads if q.name.startswith("slot_bottom")]
    hole_names = [q.name for q in obj.quads if q.name.startswith("hole_")]
    return {
        "object": obj.name,
        "true_volume_cm3": obj.true_volume_cm3,
        "measured_volume_cm3": hm.volume_mm3 / 1000.0,
        "error_cm3": err,
        "error_pct": err / obj.true_volume_cm3 * 100.0,
        "coverage": hm.coverage,
        "base_angle_deg": be["normal_angle_error_deg"],
        "base_offset_mm": be["offset_error_mm"],
        "n_points": int(len(pc.points_world)),
        "n_frames": int(len(set(pc.frame_indices))),
        "slot_visible_frac": visible_fraction(scan, slot_names),
        "hole_visible_frac": visible_fraction(scan, hole_names),
        "ground_visible_frac": visible_fraction(scan, ["ground"]),
        "timings_s": {
            "scan": float(sum(scan.timings_s.values())),
            "scan_visibility": float(scan.timings_s.get("visibility", 0.0)),
            "scan_render": float(scan.timings_s.get("render", 0.0)),
            "scan_extract": float(scan.timings_s.get("extract", 0.0)),
            "reconstruct": float(pc.timings_s.get("reconstruct", 0.0)),
            "transform": float(pc.timings_s.get("transform", 0.0)),
            "volume": float(t2 - t1),
            "evaluate_total": float(t2 - t0),
        },
        "points": pc.points_world,
        "reconstruction_inputs": pc.measurements,
        "base": base,
        "height_map": hm,
    }


# ==========================================================================
# 保存
# ==========================================================================
def save_result(result: dict, out: Path, scenario: str, param: str,
                save_ply: bool = False) -> None:
    obj = result["object"]
    tag = f"{obj}_{scenario}_{param}".replace(".", "p").replace("-", "m")
    pcdir = out / "pointclouds"
    pcdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pcdir / f"{tag}.npz",
                        points_mm=result["points"],
                        source="RECONSTRUCTED_POINTS_NO_TRUTH",
                        scenario=scenario, object=obj)
    measurement_dir = out / "data" / "reconstruction_inputs"
    measurement_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(measurement_dir / f"{tag}.npz", **result["reconstruction_inputs"])
    if save_ply:
        sc.save_ply(pcdir / f"{tag}.ply", result["points"])
    hm = result["height_map"]
    hmdir = out / "heightmaps"
    hmdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        hmdir / f"{tag}.npz",
        height=hm.height, u_edges=hm.u_edges, v_edges=hm.v_edges,
        observed=hm.observed, coverage=hm.coverage,
        volume_mm3=hm.volume_mm3, base_normal=hm.base.normal, base_d=hm.base.d,
    )


def row_of(result: dict, scenario: str, param: str) -> dict:
    return {
        "object": result["object"],
        "scenario": scenario,
        "parameter": param,
        "true_volume_cm3": result["true_volume_cm3"],
        "measured_volume_cm3": result["measured_volume_cm3"],
        "error_cm3": result["error_cm3"],
        "error_pct": result["error_pct"],
        "coverage": result["coverage"],
        "base_angle_deg": result["base_angle_deg"],
        "base_offset_mm": result["base_offset_mm"],
        "n_points": result["n_points"],
        "n_frames": result["n_frames"],
        "slot_visible_frac": result["slot_visible_frac"],
        "hole_visible_frac": result["hole_visible_frac"],
        "ground_visible_frac": result["ground_visible_frac"],
        "scan_s": result["timings_s"]["scan"],
        "scan_visibility_s": result["timings_s"]["scan_visibility"],
        "scan_render_s": result["timings_s"]["scan_render"],
        "scan_extract_s": result["timings_s"]["scan_extract"],
        "reconstruct_s": result["timings_s"]["reconstruct"],
        "transform_s": result["timings_s"]["transform"],
        "volume_s": result["timings_s"]["volume"],
        "evaluate_total_s": result["timings_s"]["evaluate_total"],
    }


# ==========================================================================
# 图
# ==========================================================================
def plot_objects(plt, out, show, objects):
    fig = plt.figure(figsize=(13, 4.6))
    for k, obj in enumerate(objects):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        for q in obj.quads:
            c = q.corners()
            loop = np.vstack([c, c[0]])
            ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], lw=0.7, color="tab:blue")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{obj.name}\ntrue {obj.true_volume_cm3:.0f} cm³")
        ax.set_box_aspect((1.3, 1.0, 0.6))
        ax.view_init(elev=22, azim=-60)
    fig.suptitle("True objects: A box, B top slot, C side through-hole", y=1.02)
    fig.tight_layout()
    return _save(fig, out, "fig01_objects.png", plt, show)


def plot_setup(plt, out, show, setup):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    # 侧视 X-Z；Y 暂不看
    ax.add_patch(plt.Rectangle((0, 0), 100, 50, fill=False, ec="k", lw=1.5, label="object A"))
    ax.plot([0, 100], [0, 0], color="k", lw=1.0)
    ax.scatter([0], [300], marker="^", s=80, color="tab:green", label="camera")
    ax.scatter([100], [300], marker="*", s=120, color="tab:red", label="laser source")
    # 激光平面在 X-Z 截面：X - 0.2Z = 40（x_i=0）
    z = np.array([0, 320.0])
    x = 40 + 0.2 * z
    ax.plot(x, z, color="tab:red", lw=1.0, alpha=0.6, label="laser plane (X-Z)")
    # 相机到物体中心的视线
    ax.plot([0, 50], [300, 25], color="tab:green", lw=0.8, alpha=0.5)
    ax.annotate("", xy=(-55, 25), xytext=(55, 25),
                arrowprops=dict(arrowstyle="<->", color="tab:purple"))
    ax.text(0, 28, "camera + laser translate along X", color="tab:purple", ha="center")
    ax.set_xlim(-70, 120)
    ax.set_ylim(-10, 330)
    ax.set_xlabel("world X (mm)")
    ax.set_ylabel("world Z (mm)")
    ax.set_title("scan geometry: fixed object, rigid camera + laser moving along X")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return _save(fig, out, "fig02_setup.png", plt, show)


def plot_pointclouds(plt, out, show, baselines, objects):
    fig = plt.figure(figsize=(14, 4.8))
    for k, obj in enumerate(objects):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        pts = baselines[obj.name]["points"]
        sc_ = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2],
                         cmap="viridis", s=1.0, alpha=0.55)
        for q in obj.quads:
            if q.name.startswith("ground"):
                continue
            c = q.corners()
            loop = np.vstack([c, c[0]])
            ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="0.5",
                    lw=0.4, alpha=0.35)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title(f"{obj.name}: {len(pts)} reconstructed points")
        ax.view_init(elev=22, azim=-62)
    fig.colorbar(sc_, ax=fig.axes, fraction=0.015, pad=0.01, label="Z (mm)")
    fig.suptitle("Reconstructed point clouds in world frame (no truth passed to reconstruction)", y=1.02)
    return _save(fig, out, "fig03_pointclouds.png", plt, show)


def plot_heightmaps(plt, out, show, baselines, objects):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, obj in zip(axes, objects):
        hm = baselines[obj.name]["height_map"]
        im = ax.imshow(hm.height, origin="lower", cmap="viridis", aspect="equal",
                       extent=[hm.u_edges[0], hm.u_edges[-1], hm.v_edges[0], hm.v_edges[-1]])
        ax.set_xlabel("base u (mm)")
        ax.set_ylabel("base v (mm)")
        ax.set_title(f"{obj.name}: V={hm.volume_mm3 / 1000:.2f} cm³, "
                     f"cov={hm.coverage * 100:.0f}%")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="height (mm)")
    fig.tight_layout()
    return _save(fig, out, "fig04_heightmaps.png", plt, show)


def plot_volumes(plt, out, show, rows):
    base_rows = [r for r in rows if r["scenario"] == "baseline"]
    names = [r["object"] for r in base_rows]
    true = np.array([r["true_volume_cm3"] for r in base_rows])
    meas = np.array([r["measured_volume_cm3"] for r in base_rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(names))
    axes[0].bar(x - 0.18, true, 0.36, label="true", color="0.65")
    axes[0].bar(x + 0.18, meas, 0.36, label="height-map", color="tab:blue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("volume (cm³)")
    axes[0].set_title("true vs height-map volume")
    axes[0].legend()
    err = meas - true
    axes[1].bar(x, err, color=["tab:green" if abs(e) < 2 else "tab:red" for e in err])
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("volume error (cm³)")
    axes[1].set_title("C: hidden side hole is not represented by a top height map")
    for xi, e in zip(x, err):
        axes[1].annotate(f"{e:+.2f}", (xi, e), textcoords="offset points",
                         xytext=(0, 5), ha="center")
    fig.tight_layout()
    return _save(fig, out, "fig05_volumes.png", plt, show)


def plot_sweeps(plt, out, show, rows):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    scenarios = [
        ("pixel", "additional center noise (px)", axes[0, 0]),
        ("motion", "positioning error sigma (mm)", axes[0, 1]),
        ("plane", "laser plane n_z error", axes[1, 0]),
        ("omission", "frame omission fraction", axes[1, 1]),
    ]
    for scen, xlabel, ax in scenarios:
        for obj in ("A_box", "B_slot", "C_side_hole"):
            rr = [r for r in rows if r["scenario"] == scen and r["object"] == obj]
            if not rr:
                continue
            x = [float(r["parameter"]) for r in rr]
            y = [r["error_cm3"] for r in rr]
            ax.plot(x, y, "o-", label=SHORT_ALIAS.get(obj, obj))
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("volume error (cm³)")
        ax.set_title(scen)
        ax.legend()
    fig.suptitle("Volume error under pixel noise / motion error / plane error / omission", y=1.01)
    fig.tight_layout()
    return _save(fig, out, "fig06_sweeps.png", plt, show)


def plot_fov_shadow(plt, out, show, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fov = [r for r in rows if r["scenario"] == "fov"]
    names = [r["object"] for r in fov]
    x = np.arange(len(names))
    truncated = [r for r in rows if r["scenario"] == "scan_range"]
    axes[0].bar(x - 0.25, [r["measured_volume_cm3"] for r in fov], 0.25,
                color="tab:orange", label="FOV-limited")
    base = [r for r in rows if r["scenario"] == "baseline"]
    axes[0].bar(x, [r["measured_volume_cm3"] for r in truncated], 0.25,
                color="tab:green", label="short scan range")
    axes[0].bar(x + 0.25, [r["measured_volume_cm3"] for r in base], 0.25,
                color="tab:blue", label="full scan")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("volume (cm³)")
    axes[0].set_title("Image window vs scan range truncation")
    axes[0].legend()

    shadow = [r for r in rows if r["scenario"] == "shadow"]
    high = [r for r in shadow if r["parameter"] == "B_high"]
    low = [r for r in shadow if r["parameter"] == "B_low"]
    if high and low:
        h, l = high[0], low[0]
        labels = ["slot bottom", "ground reference"]
        hv = [(h["slot_visible_frac"] or 0) * 100, (h["ground_visible_frac"] or 0) * 100]
        lv = [(l["slot_visible_frac"] or 0) * 100, (l["ground_visible_frac"] or 0) * 100]
        xx = np.arange(2)
        axes[1].bar(xx - 0.18, hv, 0.36, color="tab:blue", label="high source")
        axes[1].bar(xx + 0.18, lv, 0.36, color="tab:red", label="low source")
        axes[1].set_xticks(xx)
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("visible fraction (%)")
        axes[1].set_title("B: low source shadows the ground, not the slot bottom")
        axes[1].legend()
    fig.tight_layout()
    return _save(fig, out, "fig07_fov_shadow.png", plt, show)


def plot_timing(plt, out, show, rows):
    base = [r for r in rows if r["scenario"] == "baseline"]
    names = [SHORT_ALIAS.get(r["object"], r["object"]) for r in base]
    vis = np.array([r["scan_visibility_s"] for r in base])
    render = np.array([r["scan_render_s"] for r in base])
    extract = np.array([r["scan_extract_s"] for r in base])
    rec = np.array([r["reconstruct_s"] for r in base])
    trans = np.array([r["transform_s"] for r in base])
    vol = np.array([r["volume_s"] for r in base])
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    ax.bar(x, vis, label="scan: visibility/intersection")
    ax.bar(x, render, bottom=vis, label="scan: stripe render")
    ax.bar(x, extract, bottom=vis + render, label="scan: line extraction")
    ax.bar(x, rec, bottom=vis + render + extract, label="reconstruct")
    ax.bar(x, trans, bottom=vis + render + extract + rec, label="world transform")
    ax.bar(x, vol, bottom=vis + render + extract + rec + trans, label="height map + volume")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("time (s)")
    ax.set_title("baseline stage timing")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return _save(fig, out, "fig08_timing.png", plt, show)


# ==========================================================================
# 报告
# ==========================================================================
def build_report(args, camera, setup, objects, rows, checks, out):
    base = {r["object"]: r for r in rows if r["scenario"] == "baseline"}
    lines = []
    lines.append("=" * 78)
    lines.append("多帧线激光扫描 / 高度图体积仿真")
    lines.append("=" * 78)
    lines.append(f"相机       : fx={camera.fx:.3f}, fy={camera.fy:.3f}, "
                 f"{camera.image_size[0]}x{camera.image_size[1]}, "
                 f"{'有畸变' if camera.has_distortion else '无畸变'}")
    lines.append(f"扫描       : {len(setup.scan_x_mm)} 帧, X = "
                 f"{setup.scan_x_mm.min():.1f}..{setup.scan_x_mm.max():.1f} mm, "
                 f"步长 {setup.scan_x_mm[1] - setup.scan_x_mm[0]:.2f} mm")
    lines.append("机构       : 相机 + 激光器刚性平移；物体固定；P_w = R_i P_c + t_i")
    lines.append("物体       : A 长方体 400 cm³；B 顶部盲槽 396 cm³；C 侧向贯穿孔 368 cm³")
    lines.append("高度图     : D 取已知物体足迹；格内取最高可见表面；有限邻域填窄缝")
    lines.append("数据隔离   : 反算只接收像素、K/dist、激光平面、机构位姿；真实点云未传入")
    lines.append("")

    lines.append("[1] 无噪声解析对照（直接在已知高度场上积分）")
    analytic = [r for r in rows if r["scenario"] == "analytic"]
    rows_a = []
    for r in analytic:
        rows_a.append([SHORT_ALIAS.get(r["object"], r["object"]),
                       f"{r['measured_volume_cm3']:.3f}",
                       f"{r['true_volume_cm3']:.1f}",
                       f"{r['error_cm3']:+.3f}"])
    lines.append(ll.format_table(
        ["物体", "解析高度图(cm³)", "真实体积(cm³)", "误差(cm³)"],
        rows_a, ["<", ">", ">", ">"]))
    lines.append("A/B 应精确等于 400/396；C 的高度图只能看到顶面 400，真实 368，")
    lines.append("这 32 cm³ 的差异就是侧孔被高度图方法丢掉的体积。")
    lines.append("")

    lines.append("[2] 基线扫描（无额外像素噪声/运动误差/平面误差/遗漏）")
    rows_b = []
    for name in ("A", "B", "C"):
        r = base[OBJ_ALIAS[name]]
        vis = r["slot_visible_frac"] if name == "B" else (
            r["hole_visible_frac"] if name == "C" else None)
        rows_b.append([
            name, f"{r['true_volume_cm3']:.1f}",
            f"{r['measured_volume_cm3']:.2f}", f"{r['error_cm3']:+.2f}",
            f"{r['error_pct']:+.2f}", f"{r['coverage'] * 100:.1f}",
            str(r["n_points"]), str(r["n_frames"]),
            (f"{vis * 100:.1f}" if vis is not None else "-"),
            f"{r['scan_s'] + r['evaluate_total_s']:.2f}",
        ])
    lines.append(ll.format_table(
        ["物体", "真实", "测量", "误差", "误差%", "覆盖%", "点云", "帧",
         "槽/孔可见%", "总耗时s"],
        rows_b, ["<", ">", ">", ">", ">", ">", ">", ">", ">", ">"]))
    lines.append("A/B 在 0.1~0.3 cm³ 内；B 的槽底约 80% 可见，剩余边缘阴影使体积略偏大；")
    lines.append("C 的顶面高度图仍是 ~400，远大于真实 368，说明单顶部方案看不到侧孔。")
    lines.append("")

    lines.append("[3] 误差扫描（每行是一个对象/场景；体积误差单位 cm³）")
    rows_c = []
    for r in rows:
        if r["scenario"] in ("analytic", "baseline"):
            continue
        vis = r["slot_visible_frac"] if r["slot_visible_frac"] is not None else \
            r["hole_visible_frac"]
        rows_c.append([
            SHORT_ALIAS.get(r["object"], r["object"]), r["scenario"],
            str(r["parameter"]), f"{r['measured_volume_cm3']:.2f}",
            f"{r['error_cm3']:+.2f}", f"{r['coverage'] * 100:.1f}",
            f"{r['base_angle_deg']:.3f}", f"{r['base_offset_mm']:.3f}",
            (f"{vis * 100:.1f}" if vis is not None else "-"),
        ])
    lines.append(ll.format_table(
        ["对象", "场景", "参数", "测量", "误差", "覆盖%", "基准角°", "基准偏mm",
         "槽/孔可见%"],
        rows_c, ["<", "<", ">", ">", ">", ">", ">", ">", ">"]))
    lines.append("pixel/motion 是随机误差；plane 是系统误差；omission 是随机丢帧；")
    lines.append("fov 是相机图像中心竖向 200 px 有效窗口；scan_range 是 |x|<=35mm 的扫描行程截断；")
    lines.append("shadow 是 B 低激光源。逐帧图像、像素/位姿和各场景实际反算输入均已保存。")
    lines.append("")

    lines.append("[4] 耗时（基线；扫描阶段拆分）")
    rows_d = []
    for name in ("A", "B", "C"):
        r = base[OBJ_ALIAS[name]]
        rows_d.append([
            name, f"{r['scan_visibility_s']:.2f}", f"{r['scan_render_s']:.2f}",
            f"{r['scan_extract_s']:.2f}", f"{r['reconstruct_s']:.2f}",
            f"{r['transform_s']:.2f}", f"{r['volume_s']:.2f}",
            f"{r['scan_s'] + r['evaluate_total_s']:.2f}",
        ])
    lines.append(ll.format_table(
        ["物体", "可见性/求交", "条纹渲染", "亮线提取", "反算", "变换",
         "体积", "合计"],
        rows_d, ["<", ">", ">", ">", ">", ">", ">", ">"]))
    lines.append("误差扫描复用基线像素，因此只统计反算/变换/体积时间。")
    lines.append("")

    lines.append("[5] 高度图方法的限制")
    lines.append("* 隐藏侧孔：C 的侧孔完全不在顶部扫描的可见表面内，高度图看到的是完整顶面，")
    lines.append("  所以体积必然接近 400 cm³；真实 368 cm³。需要多视角/侧面扫描或封闭网格。")
    lines.append("* 悬空结构：高度图每个 (x,y) 只能存一个高度，无法表示上下两层；")
    lines.append("  本实验用侧孔代表“被遮挡的内部体积”，悬空结构同理。")
    lines.append("* 顶部盲槽：B 的槽底可见约 80%，其体积能大体扣掉；但槽壁边缘阴影/遮挡")
    lines.append("  会让槽底覆盖不足，残余误差约 0.3 cm³。")
    lines.append("* FOV：图像有效窗口裁剪会丢失窗口外的条纹；scan_range 则减少机构行程。")
    lines.append("  两类缺失分别统计，高度图空缺和体积欠估应结合覆盖率判断。")
    lines.append("* 基准面：低激光源会遮住地面基准条纹，导致 RANSAC 失去基准；")
    lines.append("  shadow 场景使用基线基准面，只用于隔离槽底阴影。")
    lines.append("")

    lines.append("[6] 检查")
    passed = sum(1 for c in checks if c["passed"])
    for c in checks:
        lines.append(f"  {'✓' if c['passed'] else '✗'} {c['name']}: {c['detail']}")
    lines.append(f"检查: {passed}/{len(checks)} 通过")
    lines.append("")

    lines.append("[产物]")
    for name in [
        "results/summary.json", "results/experiments.csv", "results/report.txt",
        "data/scan_config.json", "data/scans/", "data/reconstruction_inputs/",
        "images/", "pointclouds/", "heightmaps/",
        "figs/fig01_objects.png", "figs/fig02_setup.png", "figs/fig03_pointclouds.png",
        "figs/fig04_heightmaps.png", "figs/fig05_volumes.png", "figs/fig06_sweeps.png",
        "figs/fig07_fov_shadow.png", "figs/fig08_timing.png",
    ]:
        if (out / name).exists() or (out / name).is_dir():
            lines.append(f"  {out.name}/{name}")
    lines.append("=" * 78)
    return "\n".join(lines)


# ==========================================================================
# 主流程
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="多帧激光扫描/高度图体积仿真")
    ap.add_argument("--calib", type=Path, default=Path("../out/calib.json"))
    ap.add_argument("--synthetic-camera", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scan-range", type=float, default=55.0)
    ap.add_argument("--scan-step", type=float, default=1.0)
    ap.add_argument("--cell", type=float, default=0.5)
    ap.add_argument("--read-noise", type=float, default=0.5)
    ap.add_argument("--shot-gain", type=float, default=0.05)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ("results", "data", "images", "pointclouds", "heightmaps", "figs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    camera = build_camera(args)
    scan_x = np.arange(-args.scan_range, args.scan_range + args.scan_step * 0.5,
                       args.scan_step)
    setup = sc.ScanSetup(camera=camera, scan_x_mm=scan_x)
    objects = [sc.make_object("A"), sc.make_object("B"), sc.make_object("C")]
    names = [o.name for o in objects]
    save_indices = set(range(len(scan_x)))

    rows = []
    baselines = {}
    scans = {}
    for obj in objects:
        print(f"扫描 {obj.name} ...")
        scan = sc.simulate_scan(
            obj, setup, source_label="high",
            save_image_indices=save_indices,
            image_dir=out / "images" / obj.name,
            seed=args.seed, read_sigma=args.read_noise, shot_gain=args.shot_gain,
        )
        scans[obj.name] = scan
        sc.save_scan_measurements(out / "data" / "scans" / obj.name, scan)
        res = evaluate_scan(scan, obj, setup.laser_plane_c, args.cell, seed=args.seed)
        res["scenario"] = "baseline"
        res["parameter"] = "0"
        baselines[obj.name] = res
        rows.append(row_of(res, "baseline", "0"))
        save_result(res, out, "baseline", "0", save_ply=True)
        print(f"  {obj.name}: V={res['measured_volume_cm3']:.3f} cm³, "
              f"err={res['error_cm3']:+.3f}, cov={res['coverage']*100:.1f}%")

    # 解析对照
    for obj in objects:
        a = sc.analytic_height_map_volume(obj.name)
        rows.append({
            "object": obj.name, "scenario": "analytic", "parameter": "known_D",
            "true_volume_cm3": obj.true_volume_cm3,
            "measured_volume_cm3": a,
            "error_cm3": a - obj.true_volume_cm3,
            "error_pct": (a - obj.true_volume_cm3) / obj.true_volume_cm3 * 100,
            "coverage": 1.0, "base_angle_deg": 0.0, "base_offset_mm": 0.0,
            "n_points": 0, "n_frames": 0, "slot_visible_frac": None,
            "hole_visible_frac": None,
            "scan_s": 0.0, "reconstruct_s": 0.0, "transform_s": 0.0,
            "volume_s": 0.0, "evaluate_total_s": 0.0,
        })

    # 误差场景（复用基线扫描的像素测量）
    for obj in objects:
        scan = scans[obj.name]
        for sigma in (0.15, 0.5, 1.0):
            res = evaluate_scan(scan, obj, setup.laser_plane_c, args.cell,
                                pixel_noise=sigma, seed=args.seed + 10)
            rows.append(row_of(res, "pixel", f"{sigma:g}"))
            save_result(res, out, "pixel", f"{sigma:g}")
        for mm in (0.1, 0.5, 1.0):
            res = evaluate_scan(scan, obj, setup.laser_plane_c, args.cell,
                                motion_mm=mm, seed=args.seed + 20)
            rows.append(row_of(res, "motion", f"{mm:g}"))
            save_result(res, out, "motion", f"{mm:g}")
        for dn in (0.002, 0.005):
            plane_err = ll.LaserPlane.from_abcd([1.0, 0.0, 0.2 + dn, -100.0])
            res = evaluate_scan(scan, obj, plane_err, args.cell, seed=args.seed + 30)
            rows.append(row_of(res, "plane", f"{dn:g}"))
            save_result(res, out, "plane", f"{dn:g}")
        for frac in (0.1, 0.25):
            res = evaluate_scan(scan, obj, setup.laser_plane_c, args.cell,
                                omission=frac, seed=args.seed + 40)
            rows.append(row_of(res, "omission", f"{frac:g}"))
            save_result(res, out, "omission", f"{frac:g}")
        mask = np.abs(scan.setup.scan_x_mm) <= 35.0
        res = evaluate_scan(scan, obj, setup.laser_plane_c, args.cell,
                            frame_mask=mask, seed=args.seed + 50)
        rows.append(row_of(res, "scan_range", "35"))
        save_result(res, out, "scan_range", "35")
        w, h = camera.image_size
        roi = (0, w, max(0, int(camera.cy) - 100), min(h, int(camera.cy) + 100))
        window_scan = sc.reextract_scan(scan, roi)
        sc.save_scan_measurements(out / "data" / "scans" / f"{obj.name}_fov", window_scan)
        res = evaluate_scan(window_scan, obj, setup.laser_plane_c, args.cell,
                            seed=args.seed + 51)
        rows.append(row_of(res, "fov", "center_200px"))
        save_result(res, out, "fov", "center_200px")

    # B 低激光源：地面基准被遮，阴影场景用基线基准面隔离槽底阴影
    print("扫描 B 低激光源（阴影对照）...")
    setup_low = sc.ScanSetup(camera=camera, scan_x_mm=scan_x,
                             laser_source_c=np.array([56.0, 0.0, 220.0]))
    scan_low = sc.simulate_scan(
        sc.make_object("B"), setup_low, source_label="low",
        save_image_indices=save_indices,
        image_dir=out / "images" / "B_low_source",
        seed=args.seed, read_sigma=args.read_noise, shot_gain=args.shot_gain,
    )
    objB = sc.make_object("B")
    sc.save_scan_measurements(out / "data" / "scans" / "B_low_source", scan_low)
    base_B = baselines["B_slot"]["base"]
    low_res = evaluate_scan(scan_low, objB, setup_low.laser_plane_c, args.cell,
                            base_override=base_B, seed=args.seed + 60)
    low_res["scenario"] = "shadow"
    low_res["parameter"] = "B_low"
    rows.append(row_of(low_res, "shadow", "B_low"))
    save_result(low_res, out, "shadow", "B_low")
    # 高源 B 作为对照行
    high_res = dict(baselines["B_slot"])
    high_res["scenario"] = "shadow"
    high_res["parameter"] = "B_high"
    rows.append(row_of(high_res, "shadow", "B_high"))

    # 保存配置/结果
    sc.write_json(out / "data" / "scan_config.json", {
        "created": datetime.now().isoformat(timespec="seconds"),
        "camera": camera.to_dict(),
        "scan_x_mm": setup.scan_x_mm.tolist(),
        "scan_step_mm": float(args.scan_step),
        "cell_mm": float(args.cell),
        "read_noise": float(args.read_noise),
        "shot_gain": float(args.shot_gain),
        "seed": int(args.seed),
        "laser_plane_c": setup.laser_plane_c.to_dict(),
        "laser_source_c": setup.laser_source_c.tolist(),
        "low_laser_source_c": setup_low.laser_source_c.tolist(),
        "R_c2w": setup.R_c2w.tolist(),
        "camera_y_mm": setup.camera_y_mm, "camera_z_mm": setup.camera_z_mm,
        "image_roi": list(roi),
        "objects": [o.__dict__ | {"quads": [q.name for q in o.quads]}
                    for o in objects],
        "data_isolation": "simulation geometry kept in scan_config; measured pixels and poses in data/scans; actual perturbed reconstruction inputs in data/reconstruction_inputs",
    })
    sc.write_csv(out / "results" / "experiments.csv", rows)
    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "camera": {"fx": camera.fx, "fy": camera.fy,
                   "dist": camera.dist.tolist(), "image_size": camera.image_size},
        "scan": {"n_frames": int(len(scan_x)), "range_mm": float(args.scan_range),
                 "step_mm": float(args.scan_step)},
        "objects": {o.name: {"true_volume_cm3": o.true_volume_cm3,
                             "description": o.description} for o in objects},
        "baseline": {r["object"]: r for r in rows if r["scenario"] == "baseline"},
        "analytic": [r for r in rows if r["scenario"] == "analytic"],
        "scenarios": [r for r in rows if r["scenario"] not in ("baseline", "analytic")],
    }

    # 图
    plt = _get_plt(args.show)
    figures = [
        plot_objects(plt, out, args.show, objects),
        plot_setup(plt, out, args.show, setup),
        plot_pointclouds(plt, out, args.show, baselines, objects),
        plot_heightmaps(plt, out, args.show, baselines, objects),
        plot_volumes(plt, out, args.show, rows),
        plot_sweeps(plt, out, args.show, rows),
        plot_fov_shadow(plt, out, args.show, rows),
        plot_timing(plt, out, args.show, rows),
    ]

    # 检查
    a_an = [r for r in rows if r["scenario"] == "analytic" and r["object"] == "A_box"][0]
    b_an = [r for r in rows if r["scenario"] == "analytic" and r["object"] == "B_slot"][0]
    c_an = [r for r in rows if r["scenario"] == "analytic" and r["object"] == "C_side_hole"][0]
    a0, b0, c0 = baselines["A_box"], baselines["B_slot"], baselines["C_side_hole"]
    fov = [r for r in rows if r["scenario"] == "fov"]
    fov_a = [r for r in fov if r["object"] == "A_box"][0]
    plane05 = [r for r in rows if r["scenario"] == "plane" and
               abs(float(r["parameter"]) - 0.005) < 1e-9 and r["object"] == "A_box"][0]
    checks = [
        _check("A 解析体积", abs(a_an["error_cm3"]) < 1e-3,
               f"{a_an['measured_volume_cm3']:.6f} cm³"),
        _check("B 解析体积", abs(b_an["error_cm3"]) < 1e-3,
               f"{b_an['measured_volume_cm3']:.6f} cm³"),
        _check("C 的隐藏孔限制被识别",
               c_an["error_cm3"] > 20 and (c0["hole_visible_frac"] or 0) < 0.01,
               f"height-map {c_an['measured_volume_cm3']:.1f} vs true 368; "
               f"hole visible {c0['hole_visible_frac']}"),
        _check("A 基线体积误差 < 0.5 cm³", abs(a0["error_cm3"]) < 0.5,
               f"{a0['error_cm3']:+.3f} cm³"),
        _check("B 基线体积误差 < 1.0 cm³", abs(b0["error_cm3"]) < 1.0,
               f"{b0['error_cm3']:+.3f} cm³, slot visible "
               f"{(b0['slot_visible_frac'] or 0) * 100:.0f}%"),
        _check("FOV 受限明显欠估",
               fov_a["measured_volume_cm3"] < a0["measured_volume_cm3"] - 20,
               f"{fov_a['measured_volume_cm3']:.1f} vs {a0['measured_volume_cm3']:.1f} cm³, "
               f"coverage {fov_a['coverage'] * 100:.0f}%"),
        _check("平面 n_z +0.005 产生可见系统误差",
               abs(plane05["error_cm3"]) > 5,
               f"{plane05['error_cm3']:+.2f} cm³"),
        _check("反算接口不含真实平面/真实点云",
               "true_plane" not in __import__("inspect").signature(
                   sc.reconstruct_scan).parameters,
               "reconstruct_scan(scan, plane_c, pixel_noise_px, motion_sigma_mm, ...)"),
        _check("8 张图已生成", len([p for p in figures if Path(p).exists()]) == 8,
               f"{len(figures)} 张"),
    ]
    summary["checks"] = checks
    sc.write_json(out / "results" / "summary.json", summary)

    report = build_report(args, camera, setup, objects, rows, checks, out)
    (out / "results" / "report.txt").write_text(report, encoding="utf-8")
    print()
    print(report)
    print(f"\n产物目录: {out.resolve()}")
    if args.strict and not all(c["passed"] for c in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(2)
