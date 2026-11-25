"""诊断为什么某张（或某目录的）图检不出棋盘格 / 能不能用于验证。

用法：
    python diagnose.py --dir data/val
    python diagnose.py --dir data/val --calib out/calib.json
    python diagnose.py --image data/val/obj01.png --calib out/calib.json

它逐张检查：能不能读、分辨率是否和标定一致、原始图能不能检出、
去畸变裁边后能不能检出、清晰度、板子像素尺寸、倾角、到画面边的距离，
然后给一句人话结论（"太糊"、"板子出画了"、"图被转了 90°"…）。
结论表写 out/diagnosis.txt，缩略图写 out/diagnosis.png。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import calib_lib as cl

# 经验阈值（1280 宽画面、20mm 格的典型设置下比较稳）
MIN_SQUARE_PX = 25.0    # 单格短边小于它，SB 检测器开始不稳
MIN_SHARP = 40.0        # 板区域的 Laplacian 方差
MAX_TILT = 55.0         # 板法线与光轴夹角
MIN_MARGIN_PX = 8.0     # 板到画面边的最小留白


def crop_info(K, dist, size):
    """去畸变(alpha=0)实际会裁掉多少边，返回新内参和 roi。"""
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist, size, 0.0, size)
    return K_new, roi


def metrics(gray, corners, pattern):
    h, w = gray.shape[:2]
    pts = corners.reshape(-1, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    grid = pts.reshape(pattern[1], pattern[0], 2)
    sq = np.concatenate(
        [np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
         np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel()]
    )
    roi = gray[int(max(y0, 0)):int(y1), int(max(x0, 0)):int(x1)]
    rng = float(np.percentile(roi, 98) - np.percentile(roi, 2)) if roi.size else 0.0
    return {
        "bbox": (float(x0), float(y0), float(x1), float(y1)),
        "board_px": float(x1 - x0),
        "square_px": float(sq.min()),
        "sharp": float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0,
        "margin": float(min(x0, y0, w - x1, h - y1)),
        "area_frac": float((x1 - x0) * (y1 - y0) / (w * h)),
        "range": rng,
    }


def global_stats(gray):
    return {
        "std": float(gray.std()),
        "dark": float((gray < 40).mean()),
        "bright": float((gray > 215).mean()),
        "sharp": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def best_block_sharp(gray, block: int = 64) -> float:
    """画面里"最锐利的那一块"的 Laplacian 方差。

    整幅均值会被大片白底拉低（板子只占一角时尤其明显），
    所以判断"是不是糊了"要看最锐的地方糊不糊。
    """
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    h, w = lap.shape
    best = 0.0
    for y in range(0, max(h - block, 0) + 1, block):
        for x in range(0, max(w - block, 0) + 1, block):
            patch = lap[y:y + block, x:x + block]
            best = max(best, float(patch.var()))
    return best


def dark_bbox_hint(gray, thr: int = 60):
    """暗像素的整体分布是否贴着画面边 —— 贴边且密度像棋盘，多半是板子出画。

    直接用暗像素的外接框（而不是最大连通域）：棋盘的黑格彼此不连通，
    按连通域找只会找到一个小方块。
    """
    mask = gray < thr
    frac = float(mask.mean())
    if frac < 0.002:
        return None
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    H, W = gray.shape
    touches = x0 <= 2 or y0 <= 2 or x1 >= W - 3 or y1 >= H - 3
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    density = float(mask[y0:y1 + 1, x0:x1 + 1].mean())
    return {"touches": bool(touches), "frac": frac, "w": bw, "h": bh,
            "density": density, "W": W, "H": H}


def analyze(path: Path, pattern, calib):
    r = {"name": path.name, "level": "bad", "why": "", "fix": "",
         "found_raw": False, "found_und": None, "size": "-"}
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        r["why"] = "读不了"
        r["fix"] = "文件损坏 / HEIC 等专有格式 / 不是图片"
        return r

    h, w = gray.shape[:2]
    r["size"] = f"{w}x{h}"
    g = global_stats(gray)

    # 1) 原始图检出
    found, corners = cl.find_corners(gray, pattern)
    r["found_raw"] = found

    # 2) 按 (rows, cols) 反着找一次，能识破"图被转了 90°"
    swapped = None
    if not found and pattern[0] != pattern[1]:
        sw = (pattern[1], pattern[0])
        ok_sw, c_sw = cl.find_corners(gray, sw)
        if ok_sw:
            swapped = (sw, c_sw)

    # 3) 去畸变(裁边)后再检出，模拟 measure.py 的实际流程
    found_und, corners_und, K_adj = False, None, None
    lost_by_crop = False
    if calib is not None:
        K, dist = calib["camera_matrix"], calib["dist_coeffs"]
        K_new, roi = crop_info(K, dist, (w, h))
        x, y, ww, hh = roi
        und, K_adj = cl.undistort_image(gray, K, dist, alpha=0.0)
        found_und, corners_und = cl.find_corners(und, pattern)
        r["found_und"] = found_und
        if found and not found_und:
            pts = cv2.undistortPoints(corners, K, dist, P=K_new).reshape(-1, 2)
            pts -= np.array([x, y], dtype=np.float64)
            inside = ((pts[:, 0] >= 0) & (pts[:, 0] < ww) &
                      (pts[:, 1] >= 0) & (pts[:, 1] < hh)).mean()
            lost_by_crop = inside < 0.999

    # 4) 结论（按优先级）
    if calib is not None:
        cw, ch = calib["image_size"]
        if (w, h) != (cw, ch):
            r["level"] = "warn"
            r["why"] = f"分辨率 {w}x{h} 与标定的 {cw}x{ch} 不同"
            r["fix"] = "同一次实验别改分辨率/裁剪；否则内参不适用，需重标定"
            if found:
                r.update(metrics(gray, corners, pattern))
            return r

    if found_und:
        m = metrics(gray, corners, pattern)
        r.update(m)
        r["level"] = "ok"
        r["why"] = "OK"
        if calib is not None and K_adj is not None:
            square = calib["pattern"]["square_mm"]
            rvec, tvec, _ = cl.solve_pose(corners_und, pattern, square, K_adj, np.zeros((1, 5)))
            if rvec is not None:
                r["tilt"] = cl.board_pose_summary(rvec, tvec)["tilt_deg"]

        if m["square_px"] < MIN_SQUARE_PX:
            r["level"] = "warn"
            r["why"] = f"单格只有 {m['square_px']:.0f} px"
            r["fix"] = "拍近一点或提高分辨率（建议单格 ≥ 40 px）"
        elif m["range"] < 90:
            r["level"] = "warn"
            r["why"] = f"对比度偏低（灰阶跨度 {m['range']:.0f}）"
            r["fix"] = "补光/降曝光/避开反光；对比度低会先丢亚像素精度，再丢检出"
        elif m["sharp"] < MIN_SHARP:
            r["level"] = "warn"
            r["why"] = f"偏糊（清晰度 {m['sharp']:.0f} < {MIN_SHARP:.0f}）"
            r["fix"] = "三脚架/定时，等自动对焦合焦；别在按快门瞬间动"
        elif r.get("tilt", 0) > MAX_TILT:
            r["level"] = "warn"
            r["why"] = f"太斜（{r['tilt']:.0f}° > {MAX_TILT:.0f}°）"
            r["fix"] = "倾角控制在 45° 以内，格子被压得太扁会检不稳"
        elif m["margin"] < MIN_MARGIN_PX:
            r["level"] = "warn"
            r["why"] = f"板离画面边仅 {m['margin']:.0f} px"
            r["fix"] = "退一点，四边至少留 1 格（20 mm）"
    elif found and calib is None:
        r.update(metrics(gray, corners, pattern))
        r["level"] = "ok"
        r["why"] = "OK（未提供标定，只查了原始图）"
    elif found:
        r.update(metrics(gray, corners, pattern))
        r["level"] = "bad"
        if lost_by_crop:
            r["why"] = "去畸变裁边把板子切掉了"
            r["fix"] = "板子太靠边；退一步重拍，四边留 ≥1 格"
        else:
            r["why"] = "原始图能检出，去畸变后检不出"
            r["fix"] = "检查畸变系数是否离谱；必要时用 --alpha 1 保留全部像素"
    elif swapped:
        r["level"] = "bad"
        r["why"] = f"按 {pattern} 检不出，按 {swapped[0]} 能检出"
        r["fix"] = "板和参数对不上：确认 --cols/--rows 填的是方格数（9x6 格填 9 和 6）"
    else:
        r["level"] = "bad"
        sharp_best = best_block_sharp(gray)
        blob = dark_bbox_hint(gray)
        board_like = (
            blob is not None
            and blob["touches"]
            and (blob["w"] > 0.15 * blob["W"] or blob["h"] > 0.15 * blob["H"])
            and 0.15 < blob["density"] < 0.8
        )
        if sharp_best < 30:
            r["why"] = f"整幅都糊（最锐块清晰度 {sharp_best:.0f}）"
            r["fix"] = "对焦/抖动；手持 1/30 s 以下基本必糊"
        elif board_like:
            r["why"] = "黑白区域贴着画面边 —— 板子出画了（被裁掉一部分）"
            r["fix"] = "退一步或调整构图，四边至少留 1 格（20 mm）白边"
        elif g["bright"] > 0.6 and g["std"] < 60:
            r["why"] = f"大面积过曝或强反光（{g['bright']*100:.0f}% 像素接近纯白）"
            r["fix"] = "降曝光/换角度避开反光；黑格边缘被冲掉就再也找不回来"
        elif g["std"] < 25:
            r["why"] = f"对比度不足（std {g['std']:.0f}）"
            r["fix"] = "补光或换角度，别逆光拍"
        elif g["dark"] > 0.6:
            r["why"] = f"整体欠曝（{g['dark']*100:.0f}% 像素很暗）"
            r["fix"] = "加光或提高曝光"
        else:
            r["why"] = "画面里找不到 8x5 角点"
            r["fix"] = ("板子整块在画面内吗？格数参数对吗？换了另一块板？"
                        "板子是不是太小（单格 < 25 px）？")
    return r


def report(rows, out_dir: Path, calib):
    lines = ["=" * 112, "棋盘格检测诊断", "=" * 112]
    if calib is not None:
        K, dist = calib["camera_matrix"], calib["dist_coeffs"]
        size = tuple(calib["image_size"])
        _, roi = crop_info(K, dist, size)
        x, y, ww, hh = roi
        lines.append(f"标定图 {size[0]}x{size[1]}；去畸变(alpha=0)会裁掉 "
                     f"左{x} 右{size[0]-ww-x} 上{y} 下{size[1]-hh-y} px，保留 {ww}x{hh}")
    lines.append("")
    lines.append(cl.ljust("文件", 20) + cl.rjust("分辨率", 11) + cl.rjust("原始", 6)
                 + cl.rjust("去畸变", 8) + cl.rjust("清晰度", 8) + cl.rjust("灰阶", 7)
                 + cl.rjust("板宽px", 9) + cl.rjust("最小格px", 10)
                 + cl.rjust("倾角°", 8) + cl.rjust("边距px", 8) + "  结论")
    lines.append("-" * 112)

    for r in rows:
        def g(key, width, fmt="{:.0f}"):
            v = r.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return cl.rjust("-", width)
            return cl.rjust(fmt.format(v), width)

        lines.append(
            cl.ljust(r["name"], 20) + cl.rjust(r["size"], 11)
            + cl.rjust("✓" if r["found_raw"] else "✗", 6)
            + cl.rjust("-" if r["found_und"] is None else ("✓" if r["found_und"] else "✗"), 8)
            + g("sharp", 8) + g("range", 7) + g("board_px", 9) + g("square_px", 10)
            + g("tilt", 8) + g("margin", 8)
            + "  " + r["why"]
        )
        if r["why"] and r["why"] != "OK" and r["fix"]:
            lines.append(cl.ljust("", 24) + "→ " + r["fix"])

    good = sum(1 for r in rows if r["level"] == "ok")
    warn = sum(1 for r in rows if r["level"] == "warn")
    bad = sum(1 for r in rows if r["level"] == "bad")
    lines += ["", f"可用 {good} 张 | 有隐患 {warn} 张 | 不可用 {bad} 张",
              "不可用的优先重拍（比修图快）；有隐患的先看它会不会让 RMS 变大。",
              "=" * 112]
    text = "\n".join(lines)
    print(text)
    (out_dir / "diagnosis.txt").write_text(text, encoding="utf-8")


TAGS = [
    ("读不了", "UNREADABLE"),
    ("分辨率", "RES MISMATCH"),
    ("单格只有", "SQUARE TOO SMALL"),
    ("对比度偏低", "LOW CONTRAST"),
    ("整幅都糊", "BLURRY"),
    ("偏糊", "BLURRY"),
    ("太斜", "TOO TILTED"),
    ("板离画面边", "NEAR EDGE"),
    ("裁边把板子切掉", "CROPPED"),
    ("去畸变后检不出", "UNDISTORT FAIL"),
    ("检不出，按", "PATTERN MISMATCH"),
    ("贴着画面边", "BOARD CROPPED"),
    ("过曝", "OVEREXPOSED"),
    ("对比度不足", "LOW CONTRAST"),
    ("欠曝", "UNDEREXPOSED"),
    ("找不到 8x5", "NO BOARD FOUND"),
    ("OK", "OK"),
]


def ascii_tag(why: str) -> str:
    """cv2.putText 画不了中文，缩略图上用英文短标签。"""
    for key, tag in TAGS:
        if key in why:
            return tag
    return "?"


def montage(rows, out_dir: Path, pattern):
    tiles = []
    for r in rows:
        img = cv2.imread(str(r["path"]))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = cl.find_corners(gray, pattern)
        if ok:
            cv2.drawChessboardCorners(img, pattern, corners, True)
        hh = 240
        img = cv2.resize(img, (int(img.shape[1] * hh / img.shape[0]), hh))
        color = {"ok": (0, 170, 0), "warn": (0, 165, 255), "bad": (0, 0, 255)}[r["level"]]
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, 3)
        cv2.rectangle(img, (0, 0), (img.shape[1], 22), color, -1)
        cv2.putText(img, f"{r['name']}  {ascii_tag(r['why'])}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(img)

    if not tiles:
        return None
    cols = min(3, len(tiles))
    tw = max(t.shape[1] for t in tiles)
    padded = []
    for t in tiles:
        canvas = np.full((240, tw, 3), 30, np.uint8)
        x = (tw - t.shape[1]) // 2
        canvas[:, x:x + t.shape[1]] = t
        padded.append(canvas)
    while len(padded) % cols:
        padded.append(np.full((240, tw, 3), 30, np.uint8))
    out = np.vstack([np.hstack(padded[i:i + cols]) for i in range(0, len(padded), cols)])
    p = out_dir / "diagnosis.png"
    cv2.imwrite(str(p), out)
    return p


def main():
    ap = argparse.ArgumentParser(description="棋盘格检测诊断")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--calib", default="out/calib.json")
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    pattern = cl.inner_corners(args.cols, args.rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = cl.load_calib(args.calib) if Path(args.calib).exists() else None
    if calib is None:
        print(f"[提示] 没找到 {args.calib}：只查原始图（不查去畸变裁边/分辨率一致性）")

    if args.image:
        paths = [Path(args.image)]
    elif args.dir:
        paths = [p for p in cl.list_images(args.dir) if not p.name.startswith("diagnosis")]
    else:
        raise SystemExit("给 --dir 或 --image")
    if not paths:
        raise SystemExit(f"{args.dir} 里没有图片")

    rows = []
    for p in paths:
        r = analyze(p, pattern, calib)
        r["path"] = p
        rows.append(r)

    report(rows, out_dir, calib)
    m = montage(rows, out_dir, pattern)
    if m:
        print(f"标注图 {m}")


if __name__ == "__main__":
    main()
