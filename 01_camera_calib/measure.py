"""平面尺寸测量与验证。

两个子命令：

1) board  —— 用"没有参与标定的另一组图"测板上已知角点距离（一致性检查）
      python measure.py board --dir data/val --calib out/calib.json
   注意：这里的"已知距离"仍然由格边长推出，所以它检验的是
   "检测+畸变+位姿"这一整条链路，不能发现格边长本身打印错了。

2) object —— 在板面上放一个用卡尺量过的物体（硬币/螺丝/笔杆/小零件），
   点它的两个端点，比较视觉测量值与卡尺值（独立尺度检查）
      python measure.py object --image data/val/obj01.png --calib out/calib.json --known-mm 24.85
   GUI 里点两下 = 一次测量；右键撤销；q 结束并输出 CSV。

原理：
   标定板平面上的点与其像素坐标满足单应变换 [u,v,1] ~ H·[X,Y,1]。
   H 由该图检测到的角点和已知板坐标求出，反变换即可把像素点映回毫米。
   前提：被测物与板面共面（薄物体直接平放即可），且图像已去畸变。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import calib_lib as cl


# --------------------------------------------------------------------------
# 通用
# --------------------------------------------------------------------------
def prepare(image_path: Path, calib: dict, pattern, square: float):
    """读图 -> 去畸变 -> 找角点 -> 求 H（板 mm <-> 去畸变像素）。"""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    und, K_new = cl.undistort_image(img, calib["camera_matrix"], calib["dist_coeffs"], alpha=0.0)
    gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
    found, corners = cl.find_corners(gray, pattern)
    if not found:
        return {"image": und, "found": False, "corners": None, "H": None, "K": K_new}
    H = cl.plane_homography(corners, pattern, square)
    return {"image": und, "found": True, "corners": corners, "H": H, "K": K_new}


def corner_pairs(pattern):
    """board 模式要测的三条已知距离。"""
    cols, rows = pattern
    idx = lambda c, r: r * cols + c
    return [
        ("横向上边缘 8 格", (idx(0, 0), idx(cols - 1, 0))),
        ("纵向左边缘 6 格", (idx(0, 0), idx(0, rows - 1))),
        ("对角线", (idx(0, 0), idx(cols - 1, rows - 1))),
    ]


# --------------------------------------------------------------------------
# board 模式
# --------------------------------------------------------------------------
def cmd_board(args):
    calib = cl.load_calib(args.calib)
    pattern = cl.inner_corners(args.cols, args.rows)
    objp = cl.object_points(pattern, args.square)
    paths = cl.list_images(args.dir)
    if not paths:
        raise SystemExit(f"{args.dir} 里没有图片")

    pairs = corner_pairs(pattern)
    rows_out = []
    print(cl.ljust("图像", 18) + cl.rjust("板距mm", 9) + cl.rjust("倾角°", 8) + "  "
          + "".join(cl.rjust(name, 18) for name, _ in pairs))
    print(cl.ljust("", 18) + cl.rjust("", 9) + cl.rjust("", 8) + "  "
          + "".join(cl.rjust("实测/误差 mm", 18) for _ in pairs))

    for p in paths:
        r = prepare(p, calib, pattern, args.square)
        if r is None or not r["found"]:
            print(f"{p.name:<18}  未检出棋盘格，跳过")
            continue

        H = r["H"]
        corners = r["corners"].reshape(-1, 2).astype(np.float64)
        rvec, tvec, _ = cl.solve_pose(corners, pattern, args.square,
                                      r["K"], np.zeros((1, 5)))
        pose = cl.board_pose_summary(rvec, tvec) if rvec is not None else {"distance_mm": np.nan, "tilt_deg": np.nan}

        line = cl.ljust(p.name, 18) + f"{pose['distance_mm']:9.1f}{pose['tilt_deg']:8.1f}  "
        rec = {"image": p.name, "distance_mm": round(pose["distance_mm"], 2),
               "tilt_deg": round(pose["tilt_deg"], 2)}
        for name, (i, j) in pairs:
            expected = float(np.linalg.norm(objp[i, :2] - objp[j, :2]))
            measured = cl.distance_on_plane(H, corners[i], corners[j])
            err = measured - expected
            line += f"{measured:9.3f}/{err:+.3f}".rjust(18)
            rec[f"{name}_measured_mm"] = round(measured, 4)
            rec[f"{name}_error_mm"] = round(err, 4)
        print(line)
        rows_out.append(rec)

    if not rows_out:
        raise SystemExit("没有一张图可用于验证")

    print()
    all_err = []
    for name, (i, j) in pairs:
        expected = float(np.linalg.norm(objp[i, :2] - objp[j, :2]))
        errs = np.array([r[f"{name}_error_mm"] for r in rows_out], dtype=np.float64)
        all_err.append(errs)
        print(cl.ljust(name, 18) + f" 标称 {expected:7.2f} mm | 平均误差 {errs.mean():+.3f} mm "
              f"({errs.mean()/expected*100:+.3f}%) | 最大 {np.abs(errs).max():.3f} mm")
    flat = np.concatenate([np.abs(e) for e in all_err])
    print(f"\n所有距离的绝对误差：平均 {flat.mean():.3f} mm，最大 {flat.max():.3f} mm")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(f"明细已写入 {out}")
    print("\n提示：这些距离都由格边长推出，只能证明链路一致；"
          "要验证尺度请用 object 模式配卡尺。")


# --------------------------------------------------------------------------
# object 模式
# --------------------------------------------------------------------------
def draw_measure(img, a, b, dist_mm, known_mm=None):
    vis = img.copy()
    cv2.line(vis, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), (0, 255, 0), 2)
    for p in (a, b):
        cv2.circle(vis, tuple(np.round(p).astype(int)), 5, (0, 0, 255), -1)
    mid = (np.asarray(a) + np.asarray(b)) / 2
    if known_mm:
        err = dist_mm - known_mm
        txt = f"{dist_mm:.2f} mm  (caliper {known_mm:.2f}, err {err:+.2f})"
    else:
        txt = f"{dist_mm:.2f} mm"
    cv2.putText(vis, txt, tuple(np.round(mid + np.array([10, -12])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return vis


def cmd_object(args):
    calib = cl.load_calib(args.calib)
    pattern = cl.inner_corners(args.cols, args.rows)

    r = prepare(Path(args.image), calib, pattern, args.square)
    if r is None:
        raise SystemExit(f"读不了 {args.image}")
    if not r["found"]:
        raise SystemExit("这张图里没检出棋盘格，测不了（标定板必须整块清楚地在画面里）")

    img, H = r["image"], r["H"]
    corners = r["corners"].reshape(-1, 2).astype(np.float64)
    rvec, tvec, pose_rms = cl.solve_pose(corners, pattern, args.square, r["K"], np.zeros((1, 5)))
    pose = cl.board_pose_summary(rvec, tvec)

    print(f"图像 {args.image}  板距 {pose['distance_mm']:.1f} mm  倾角 {pose['tilt_deg']:.1f}°  "
          f"位姿拟合 {pose_rms:.3f} px")
    if pose["tilt_deg"] > 45:
        print("注意：板子倾角偏大，平面外误差会被放大，尽量让板面接近正对相机。")

    measurements = []

    # 非交互模式：直接给两个点
    if args.p1 and args.p2:
        a = np.array([float(v) for v in args.p1.split(",")])
        b = np.array([float(v) for v in args.p2.split(",")])
        d = cl.distance_on_plane(H, a, b)
        measurements.append({"image": Path(args.image).name, "p1": a.tolist(), "p2": b.tolist(),
                             "measured_mm": round(d, 3), "known_mm": args.known_mm})
        vis = draw_measure(img, a, b, d, args.known_mm)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), vis)
        print(f"测量结果 {d:.3f} mm" + (f"  卡尺 {args.known_mm:.2f} mm  误差 {d-args.known_mm:+.3f} mm" if args.known_mm else ""))
        print(f"标注图 {out}")
        _write_object_csv(measurements, args.csv)
        return

    state = {"pts": [], "vis": img.copy()}
    win = "measure: click 2 points on the object (q=quit, r=undo)"

    def redraw(msg=""):
        vis = state["vis"].copy()
        for i, p in enumerate(state["pts"]):
            cv2.circle(vis, tuple(np.round(p).astype(int)), 5, (0, 0, 255), -1)
            cv2.putText(vis, ["A", "B"][i], tuple(np.round(p + np.array([8, -8])).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis, msg or "left click = endpoint, q = done, r = undo",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(win, vis)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pts"].append(np.array([x, y], dtype=np.float64))
            if len(state["pts"]) == 2:
                a, b = state["pts"]
                d = cl.distance_on_plane(H, a, b)
                measurements.append({"image": Path(args.image).name, "p1": a.tolist(),
                                     "p2": b.tolist(), "measured_mm": round(d, 3),
                                     "known_mm": args.known_mm})
                print(f"[{len(measurements)}] {d:.3f} mm"
                      + (f"  卡尺 {args.known_mm:.2f}  误差 {d-args.known_mm:+.3f} mm" if args.known_mm else ""))
                state["vis"] = draw_measure(state["vis"], a, b, d, args.known_mm)
                state["pts"] = []
                redraw(f"{d:.2f} mm  (press q when done)")
                return
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if state["pts"]:
                state["pts"].pop()
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    redraw("click endpoint A")

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord("r") and state["pts"]:
            state["pts"].pop()
            redraw()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), state["vis"])
    cv2.destroyAllWindows()

    print()
    if measurements:
        ds = [m["measured_mm"] for m in measurements]
        print(f"本次共 {len(ds)} 次测量：{', '.join(f'{d:.2f}' for d in ds)} mm")
        if args.known_mm:
            errs = np.array(ds) - args.known_mm
            print(f"相对卡尺 {args.known_mm:.2f} mm：平均误差 {errs.mean():+.3f} mm "
                  f"({errs.mean()/args.known_mm*100:+.2f}%)，最大 {np.abs(errs).max():.3f} mm")
        _write_object_csv(measurements, args.csv)
    print(f"标注图 {out}")


def _write_object_csv(rows, path):
    if not rows:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"明细 {p}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="平面尺寸测量/验证")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("board", "object"):
        p = sub.add_parser(name)
        p.add_argument("--calib", default="out/calib.json")
        p.add_argument("--cols", type=int, default=9)
        p.add_argument("--rows", type=int, default=6)
        p.add_argument("--square", type=float, default=20.0)
        if name == "board":
            p.add_argument("--dir", default="data/val")
            p.add_argument("--out", default="out/measure_board.csv")
        else:
            p.add_argument("--image", required=True)
            p.add_argument("--known-mm", type=float, default=None, help="卡尺实测长度")
            p.add_argument("--p1", default=None, help="非交互：x,y 像素")
            p.add_argument("--p2", default=None, help="非交互：x,y 像素")
            p.add_argument("--out", default="out/measure_object.png")
            p.add_argument("--csv", default="out/measure_object.csv")

    args = ap.parse_args()
    if args.cmd == "board":
        cmd_board(args)
    else:
        cmd_object(args)


if __name__ == "__main__":
    main()
