"""生成可精确打印的棋盘格标定板（矢量 PDF，尺寸不依赖打印机 DPI）。

用法：
    python make_board.py                 # 默认 9x6 格、每格 20 mm、A4
    python make_board.py --cols 9 --rows 6 --square 20 --paper a4

输出：
    board/chessboard_9x6_20mm.pdf   打印用（按 100% / 实际大小打印）
    board/chessboard_9x6_20mm.png   屏幕预览用（不要用它去打印，容易被缩放）
"""

from __future__ import annotations

import argparse
from pathlib import Path

MM = 72.0 / 25.4  # 1 mm = 2.8346 pt

PAPERS = {  # 单位 pt (宽, 高)，纵向
    "a4": (595.276, 841.890),
    "letter": (612.0, 792.0),
    "a3": (841.890, 1190.551),
}


def build_pdf(content: bytes, page_w: float, page_h: float) -> bytes:
    """极简 PDF 1.4 生成器：单页、内置 Helvetica、内容流为原始绘图指令。"""
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w:.3f} {page_h:.3f}] "
            f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ).encode("ascii"),
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


def make_page(w_mm: float, h_mm: float, cols: int, rows: int, square_mm: float):
    """返回绘图指令（pt）与版面信息。"""
    page_w, page_h = w_mm * MM, h_mm * MM
    sq = square_mm * MM
    board_w, board_h = cols * sq, rows * sq

    mx = (page_w - board_w) / 2
    # 顶部留 45 pt 边距，板子尽量靠上，下方留给比例尺和说明
    top = 45.0
    y_top = page_h - top
    y0 = y_top - board_h

    cmds = ["1 g", f"0 0 {page_w:.3f} {page_h:.3f} re f", "0 g"]

    # 棋盘：左上角为白格，黑白相间
    for j in range(rows):
        for i in range(cols):
            if (i + j) % 2 == 1:
                x = mx + i * sq
                y = y0 + (rows - 1 - j) * sq
                cmds.append(f"{x:.3f} {y:.3f} {sq:.3f} {sq:.3f} re f")

    # 说明文字（PDF 内置字体不含中文，用英文避免乱码）
    tx = mx
    ty = y0 - 60
    lines = [
        (10, "0.08 0.08 0.08", f"Chessboard {cols} x {rows} squares, {square_mm:.2f} mm each, "
                             f"{cols-1} x {rows-1} inner corners"),
        (10, "0.75 0.15 0.15", "PRINT AT 100% / ACTUAL SIZE - do NOT use 'fit to page' / 'scale to fit'"),
        (8, "0.08 0.08 0.08",
         f"Verify with a caliper: the bar below is {cols*square_mm:.2f} mm. "
         f"If it reads D, use square size = {square_mm:.2f} x D / {cols*square_mm:.2f}"),
    ]
    cmds.append("BT")
    cmds.append(f"1 0 0 1 {tx:.2f} {ty:.2f} Tm")
    for size, color, text in lines:
        cmds.append(f"/F1 {size} Tf {color} rg")
        cmds.append(f"({text}) Tj")
        cmds.append(f"0 -{size * 1.9:.2f} Td")
    cmds.append("ET")

    # 比例尺：长度 = 板宽 = cols 格，每格一个刻度，用于卡尺校验打印比例
    bar_y = ty - 62
    bar_x0 = mx
    cmds.append(f"0.6 w 0 g {bar_x0:.3f} {bar_y:.3f} m {bar_x0 + board_w:.3f} {bar_y:.3f} l S")
    for k in range(cols + 1):
        x = bar_x0 + k * sq
        cmds.append(f"0.6 w 0 g {x:.3f} {bar_y:.3f} m {x:.3f} {bar_y + 6:.3f} l S")
        cmds.append(
            f"BT /F1 7 Tf 0 g 1 0 0 1 {x - 4:.2f} {bar_y - 11:.2f} Tm ({k*square_mm:.0f}) Tj ET"
        )
    cmds.append(
        f"BT /F1 7 Tf 0 g 1 0 0 1 {bar_x0:.2f} {bar_y - 22:.2f} Tm "
        f"(unit: mm, full length = {cols*square_mm:.0f} mm) Tj ET"
    )

    info = dict(page_w=page_w, page_h=page_h, board_w=board_w, board_h=board_h,
                margin_x=mx, board_bottom=y0)
    return ("\n".join(cmds) + "\n").encode("ascii"), info


def preview_png(path: Path, w_mm: float, h_mm: float, cols: int, rows: int,
                square_mm: float, dpi: int = 150):
    """生成屏幕预览图（仅供确认版式，不要用它打印）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(w_mm / 25.4, h_mm / 25.4), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_mm)
    ax.set_ylim(h_mm, 0)  # 顶部为 y=0，方便按"从上到下"思考
    ax.axis("off")

    board_w, board_h = cols * square_mm, rows * square_mm
    mx = (w_mm - board_w) / 2
    my = 45.0 / MM  # 与 PDF 的 top=45pt 对应
    for j in range(rows):
        for i in range(cols):
            if (i + j) % 2 == 1:
                ax.add_patch(
                    Rectangle((mx + i * square_mm, my + j * square_mm),
                              square_mm, square_mm, facecolor="black", edgecolor="none")
                )
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="生成棋盘格标定板 PDF")
    ap.add_argument("--cols", type=int, default=9, help="横向方格数（默认 9）")
    ap.add_argument("--rows", type=int, default=6, help="纵向方格数（默认 6）")
    ap.add_argument("--square", type=float, default=20.0, help="每格边长 mm（默认 20）")
    ap.add_argument("--paper", default="a4", choices=sorted(PAPERS), help="纸张")
    ap.add_argument("--out", default="board", help="输出目录")
    args = ap.parse_args()

    pw, ph = PAPERS[args.paper]
    w_mm, h_mm = pw / MM, ph / MM
    if args.cols * args.square > w_mm - 20 or args.rows * args.square > h_mm - 100:
        raise SystemExit("板子放不下这张纸，减小 --square 或换更大的 --paper / 更多列行")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"chessboard_{args.cols}x{args.rows}_{args.square:g}mm"
    content, info = make_page(w_mm, h_mm, args.cols, args.rows, args.square)
    pdf_path = out / f"{stem}.pdf"
    pdf_path.write_bytes(build_pdf(content, info["page_w"], info["page_h"]))

    png_path = out / f"{stem}_preview.png"
    try:
        preview_png(png_path, w_mm, h_mm, args.cols, args.rows, args.square)
    except ImportError:
        png_path = None
        print("[warn] 没装 matplotlib，跳过 PNG 预览（PDF 已正常生成）")

    print(f"[OK] PDF : {pdf_path.resolve()}")
    if png_path:
        print(f"[OK] 预览: {png_path.resolve()}  （仅屏幕预览，打印请用 PDF）")
    print()
    print(f"版面：{args.cols}x{args.rows} 格，每格 {args.square} mm，"
          f"板面 {args.cols*args.square:.1f} x {args.rows*args.square:.1f} mm")
    print(f"内部角点：{args.cols-1} x {args.rows-1}  （OpenCV 参数用 {args.cols-1}, {args.rows-1}）")
    print()
    print("打印要点：")
    print("  1. 打印对话框选“实际大小 / 100%”，关闭“适合页面/缩放”。")
    print("  2. 打印后用游标卡尺量 PDF 上那条标称 "
          f"{args.cols*args.square:.0f} mm 的横线，记下实测值 D。")
    print("  3. 贴到平整硬板上（亚克力/铝板/厚卡纸），务必压平，翘曲会让 RMS 变差。")
    print("  4. 实际格边长 = "
          f"{args.square} * D / {args.cols*args.square:.0f}，后续脚本用 --square 传入该值。")


if __name__ == "__main__":
    main()
