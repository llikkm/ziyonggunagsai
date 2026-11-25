"""相机内参标定：角点检测 -> 亚像素精定位 -> calibrateCamera -> 质量报告。

用法：
    python calibrate.py --dir data/calib
    python calibrate.py --dir data/calib --square 19.96        # 按打印实测格边长
    python calibrate.py --dir data/calib --exclude calib_07.png calib_13.png
    python calibrate.py --dir data/calib --show                # 弹窗看检测结果

产物（默认写到 out/）：
    calib.json            内参 + 畸变系数 + 每张图误差，后续测量直接读它
    calib_report.txt      文字报告（可以直接贴进实验记录）
    detections.png        每张图的角点检测缩略图，肉眼检查有没有检歪
    undistort_*.png       去畸变前后对比
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import calib_lib as cl


def detection_contact_sheet(paths, pattern, K, dist, per_view, out_path: Path, cols: int = 4):
    """把所有标定图缩略拼一张，画上检测到的角点，方便一眼看出哪张有问题。

    同时按每张图的测得的角点画一个极小的十字；误差大的图边框标红。
    """
    err_map = {row["image"]: row["rms_px"] for row in (per_view or [])}
    thumbs = []
    for p in paths[:24]:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        ok, corners = cl.find_corners(gray, pattern)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if ok:
            cv2.drawChessboardCorners(bgr, pattern, corners, True)
        rms = err_map.get(p.name)
        color = (0, 200, 0)
        if rms is not None:
            color = (0, 0, 255) if rms > 1.0 else (0, 200, 0)
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1] - 1, bgr.shape[0] - 1), color, 3)

        h = 260
        scale = h / bgr.shape[0]
        thumb = cv2.resize(bgr, (int(bgr.shape[1] * scale), h))

        # 标签画在缩略图上，否则缩完就看不清
        label = p.name if rms is None else f"{p.name}  rms {rms:.2f}px"
        cv2.rectangle(thumb, (0, 0), (10 + 11 * len(label), 22), color, -1)
        cv2.putText(thumb, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        thumbs.append(thumb)

    if not thumbs:
        return None
    tw = max(t.shape[1] for t in thumbs)
    rows = []
    for i in range(0, len(thumbs), cols):
        chunk = thumbs[i : i + cols]
        while len(chunk) < cols:
            chunk.append(np.full_like(thumbs[0], 40))
        rows.append(np.hstack(chunk))
    sheet = np.vstack(rows)
    cv2.imwrite(str(out_path), sheet)
    return out_path


def undistort_demo(paths, K, dist, out_dir: Path, n: int = 3):
    made = []
    for p in paths[:n]:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        und, _ = cl.undistort_image(img, K, dist, alpha=0.0)
        h = img.shape[0]
        und = cv2.resize(und, (int(und.shape[1] * h / und.shape[0]), h))
        sep = np.full((h, 8, 3), 255, np.uint8)
        combo = np.hstack([img, sep, und])
        cv2.putText(combo, "BEFORE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(combo, "AFTER (undistorted)", (img.shape[1] + 20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 0), 2)
        out = out_dir / f"undistort_{p.stem}.png"
        cv2.imwrite(str(out), combo)
        made.append(out)
    return made


def main():
    ap = argparse.ArgumentParser(description="相机内参标定")
    ap.add_argument("--dir", default="data/calib", help="标定图目录")
    ap.add_argument("--cols", type=int, default=9, help="方格列数（不是角点数）")
    ap.add_argument("--rows", type=int, default=6, help="方格行数")
    ap.add_argument("--square", type=float, default=20.0, help="格边长 mm（用打印后实测值）")
    ap.add_argument("--model", default="standard", choices=["standard", "rational"],
                    help="standard=k1,k2,p1,p2,k3；rational 多估 k4,k5,k6")
    ap.add_argument("--out", default="out")
    ap.add_argument("--exclude", nargs="*", default=[], help="排除的文件名")
    ap.add_argument("--show", action="store_true", help="弹窗逐张显示检测结果")
    args = ap.parse_args()

    pattern = cl.inner_corners(args.cols, args.rows)
    paths = cl.list_images(args.dir)
    if not paths:
        raise SystemExit(f"{args.dir} 里没有图片。先用 capture_images.py 拍，或把照片拷进去。")

    print(f"目录 {args.dir}: {len(paths)} 张图，棋盘格 {args.cols}x{args.rows} 格 -> 内部角点 {pattern[0]}x{pattern[1]}，格边长 {args.square} mm")
    print("正在检测角点并标定 ...")

    data = cl.calibrate_from_images(
        paths, pattern, args.square, model=args.model, exclude=set(args.exclude)
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cl.save_calib(out_dir / "calib.json", data)

    report = cl.format_calib_report(data)
    print()
    print(report)
    (out_dir / "calib_report.txt").write_text(report, encoding="utf-8")

    used_names = {row["image"] for row in data["per_view"]}
    used_paths = [p for p in paths if p.name in used_names]
    sheet = detection_contact_sheet(used_paths, pattern, data["camera_matrix"],
                                    data["dist_coeffs"], data["per_view"],
                                    out_dir / "detections.png")
    demos = undistort_demo(used_paths, data["camera_matrix"], data["dist_coeffs"], out_dir)

    print()
    print(f"已写出: {out_dir / 'calib.json'}")
    print(f"        {out_dir / 'calib_report.txt'}")
    if sheet:
        print(f"        {sheet}   <- 先看这张，确认角点都贴在格点上")
    for d in demos:
        print(f"        {d}   <- 直线应该变直")

    if args.show:
        for p in used_paths:
            gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            ok, corners = cl.find_corners(gray, pattern)
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if ok:
                cv2.drawChessboardCorners(vis, pattern, corners, True)
            cv2.imshow("detection (any key / q to quit)", vis)
            if (cv2.waitKey(0) & 0xFF) in (27, ord("q")):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
