# 三维测量实验操作指南

> 项目总览、模块地图与全套测试：见 [PROJECT_README.md](../PROJECT_README.md)。

先用已有照片检查棋盘格位姿，再实拍一组垫高前后的棋盘格照片，测量高度差。射线与激光平面求交使用合成数据验证，最后用 Open3D 查看三维点。

直接使用电脑现有的全局 Python 和已装依赖。所有命令在 PowerShell 中执行，先进入工程目录：

```powershell
cd 'C:\Users\wangjianing\Desktop\光赛\01_camera_calib'
```

程序默认读取 `out/calib.json` 中的相机内参及畸变系数。当前参数对应 1280×720 图像，棋盘为 9×6 个方格，内部角点为 8×5，格边长为 20 mm。下面的拍照命令按这组参数设置。

### 当前手机实验的修正参数

本轮棋盘格实测边长为 19 mm。手机旧内参的标定优化在默认 30 次迭代时尚未收敛；新增 `refine_calibration.py` 使用 300 次迭代完成细化。已将新的手机参数另存为 `out/phone_refined/calib.json`，原文件保留。

使用新参数重算现有的 31 mm 垫高实验：

```powershell
python 02_measurement_3d/measure3d.py --calib out/phone_refined/calib.json height --reference 02_measurement_3d/data/phone_reference/frame_01.png --targets 02_measurement_3d/data/phone_height20 --known-mm 31 --out 02_measurement_3d/results/phone_height31_refined.json
```

后续手机采集时也在 `capture3d.py` 命令中指定 `--calib out/phone_refined/calib.json`。手机标定照片的板面倾角目前约 0°～14°；下一轮内参标定应补充约 15°～40° 的左右、上下倾斜姿态，并保持整块棋盘格可见。

## 摄像头选择与实时角度显示

新增 `capture3d.py` 可选择笔记本内置摄像头、USB 摄像头或 OBS 虚拟摄像头。手机通过 scrcpy 显示后置摄像头画面，再由 OBS 输出为虚拟摄像头。

先列出设备编号，再预览确认画面：

```powershell
python 02_measurement_3d/capture3d.py --list-cameras
python 02_measurement_3d/capture3d.py --camera 0 --preview
```

继续用内置摄像头时，可以使用以下新命令。第一条拍基准，第二条在垫高后拍 10 张并实时显示相对基准的板面夹角：

```powershell
python 02_measurement_3d/capture3d.py --source builtin --camera 0 --count 1 --out 02_measurement_3d/data/builtin_reference
python 02_measurement_3d/capture3d.py --source builtin --camera 0 --reference 02_measurement_3d/data/builtin_reference/frame_01.png --count 10 --out 02_measurement_3d/data/builtin_height20
python 02_measurement_3d/measure3d.py height --reference 02_measurement_3d/data/builtin_reference/frame_01.png --targets 02_measurement_3d/data/builtin_height20 --known-mm 20.00 --out 02_measurement_3d/results/builtin_height20.json
```

按空格保存，拍满指定张数后自动结束。窗口中的 `relative board angle` 是垫高前后两个板面的夹角，要求在 2° 以内。相机从斜上方观察时也可满足这个条件：相机固定，等厚垫块让棋盘格整体平行升高即可。

USB 摄像头及手机的完整接入、单独标定和采集命令见 [摄像头选择指南](CAMERAS.md)。以下章节中的原采集命令继续可用；原命令照片名为 `calib_01.png`，新入口照片名为 `frame_01.png`，请使用与采集命令对应的路径。

新入口加 `--calibration` 时显示九宫格及各区已保存照片数；采集相机内参时，建议每区至少 2 张，并变化棋盘格倾角。

## 一、先用已有照片运行一次位姿求解

目的：确认程序能找到棋盘格，并求出棋盘格相对相机的位置。

运行：

```powershell
python 02_measurement_3d/measure3d.py pose --image data/calib/calib_03.png --out 02_measurement_3d/results/pose_check.json --view
```

程序会在终端输出结果，并打开 Open3D 窗口显示棋盘格角点的三维位置。鼠标拖动可旋转视角，滚轮可缩放；关闭窗口后返回终端。

结果保存在 `02_measurement_3d/results/pose_check.json`。用记事本打开：

```powershell
notepad 02_measurement_3d/results/pose_check.json
```

主要看这两个字段：

| 字段 | 含义 |
| --- | --- |
| `tvec_mm` | 棋盘首个内部角点在相机坐标系中的 `[X,Y,Z]`，单位 mm |
| `reprojection_rms_px` | 棋盘格重投影误差，单位像素；越小表示位姿与图像角点拟合得越好 |

相机坐标系：X 向右，Y 向下，Z 向前。本工程已有 `calib_03.png` 的重投影 RMS 约为 0.65 px，可用来核对运行结果。

## 二、准备高度差实验

目的：给棋盘格垫上已知厚度的垫块，用图像测得的高度差与卡尺测量值比较。

准备相机、贴在平整硬板上的原棋盘格，以及游标卡尺。再准备两块或多块等厚垫块，让棋盘格能够平稳架起。

摆放方法：

1. 将棋盘格背板平放在桌面上，固定相机，使整块棋盘格都出现在画面中。相机可以从斜上方观察，让板面有适度倾斜。
2. 将相机分辨率设为 1280×720，保持与已有标定相同的焦距和对焦状态。拍摄期间固定相机位置。
3. 用卡尺测量垫块厚度。垫高时把等厚垫块放在背板下方的支撑位置，让板面整体平行抬高。

例如：同一块棋盘格背板从桌面抬到 20.00 mm 厚的垫块上，棋盘图案平面也随之升高 20.00 mm。如果卡尺实测为 19.86 mm，后续命令中的 `--known-mm` 填 `19.86`。

这轮需要拍的照片如下：

| 状态 | 拍多少张 | 存放目录 |
| --- | --- | --- |
| 背板直接放在桌面上，作为基准 | 1 张 | `02_measurement_3d/data/reference/` |
| 背板放到垫块上，保持同一高度 | 10 张 | `02_measurement_3d/data/height20/` |

每张照片都要有清晰、完整的棋盘格，并在四周留出边缘。照片按原始尺寸保存为 PNG。10 张目标照片分别拍摄，每次等板子静止后再保存。

每张照片都拍整块棋盘格。预览画面上的 3×3 九宫格用于原来的内参标定，帮助采集不同位置的标定照片。本次高度差实验采用固定构图：选好一个能看清整板的位置，拍完基准照后只垫高棋盘格，随后在同一高度连续拍 10 张。九宫格内的计数无需作为这次实验的采集目标。

## 三、拍基准照片

棋盘格背板直接放在桌面上，运行：

拍这 1 张之前，先完成构图和对焦：

- 使用后面要垫高的同一块棋盘格，背板平放、支撑稳定。
- 把整块棋盘格放在画面中央附近，所有 8×5 个内部角点都清楚可见，板四周留出空白。让棋盘格占据画面中较大区域，并预留垫高后仍能完整入镜的空间。
- 相机从斜上方观察板面，调到黑白边缘清晰、没有明显反光。出现完整彩色角点标记后，再保存基准照。
- 构图确定后固定相机。接下来垫高和拍 10 张期间，保持相机位置、拍摄方向及成像设置一致。

```powershell
python capture_images.py --camera 0 --width 1280 --height 720 --cols 9 --rows 6 --manual --target 1 --out 02_measurement_3d/data/reference
```

操作摄像头窗口：

1. 棋盘格角点出现彩色标记后，等画面稳定。
2. 按一次空格保存照片，终端会显示保存成功。
3. 按 `q` 退出。

得到的基准照片是：

```text
02_measurement_3d/data/reference/calib_01.png
```

如果相机编号为 1，将命令中的 `--camera 0` 改为 `--camera 1`。这里使用 `--manual`，由空格控制保存；达到 `--target` 指定张数后，按 `q` 结束采集。

## 四、垫高后拍 10 张照片

保持相机固定，将等厚垫块放在背板下方，使棋盘格平行抬高。放稳后运行：

```powershell
python capture_images.py --camera 0 --width 1280 --height 720 --cols 9 --rows 6 --manual --target 10 --out 02_measurement_3d/data/height20
```

看到角点标记后，每隔约 1 秒按一次空格，共保存 10 张，再按 `q`。拍摄这组照片时保持垫块和棋盘格的位置固定。

10 张照片使用同一构图、同一垫高量。每次按空格保存当前的新画面，整块棋盘格始终可见。保持板面朝向不变，连续拍摄；相机和棋盘格在这 10 次保存之间都保持静止。程序分别处理这 10 张图，得到 10 个高度值，用于计算平均值和标准差。

文件名依次为 `calib_01.png` 至 `calib_10.png`，都保存在 `02_measurement_3d/data/height20/`。

采集脚本每次启动会从 `calib_01.png` 开始编号。重做整组实验时，给 `--out` 换一个新目录名，例如 `02_measurement_3d/data/height20_run2`，保存不同轮次的照片。

## 五、计算高度差并查看结果

以下命令按卡尺参考值 20.00 mm 计算。将 `--known-mm` 后面的数值换成垫块的实测厚度：

```powershell
python 02_measurement_3d/measure3d.py height --reference 02_measurement_3d/data/reference/calib_01.png --targets 02_measurement_3d/data/height20 --known-mm 20.00 --out 02_measurement_3d/results/height20.json
```

程序会逐张求出棋盘格位姿，沿基准板面的法向计算两个平面之间的距离，再输出汇总统计。

打开结果：

```powershell
notepad 02_measurement_3d/results/height20.json
```

| 字段 | 怎么看 |
| --- | --- |
| `count` | 参与统计的目标照片数，这组应为 10 |
| `known_mm` | 输入的卡尺参考值 |
| `mean_mm` | 10 次视觉高度测量的平均值 |
| `absolute_error_mm` | 平均值与卡尺参考值之差的绝对值 |
| `std_mm` | 10 次测量的样本标准差，反映这一组照片的重复性 |

例如，卡尺参考值为 20.00 mm，若 `mean_mm` 为 20.15，则 `absolute_error_mm` 为 0.15 mm。`std_mm` 越小，表示这组重复测量越集中。

逐张结果在 `02_measurement_3d/results/height20.csv` 中，可以用 Excel 打开。每行包括高度测量值、该次绝对误差，以及前后板面的夹角。`mean_absolute_error_mm` 是逐张绝对误差的平均值，`std_mm` 使用样本标准差公式，分母为 `N-1`。

建议再选一个高度，例如 10 mm，重复第 4 步，将照片存到 `02_measurement_3d/data/height10/`，然后运行：

```powershell
python 02_measurement_3d/measure3d.py height --reference 02_measurement_3d/data/reference/calib_01.png --targets 02_measurement_3d/data/height10 --known-mm 10.00 --out 02_measurement_3d/results/height10.json
```

`--known-mm` 同样填该组垫块的实测厚度。记录各高度的均值、绝对误差和标准差，即可形成高度测量实验表。

## 六、验证射线与平面求交，再查看三维点

当前的射线求交实验使用程序生成的已知三维点。此步骤直接运行命令：

```powershell
python 02_measurement_3d/measure3d.py synthetic --view
```

程序在平面 `0.2X - 0.1Y + Z - 550 = 0` 上生成 315 个三维点，用已有相机参数投影成带畸变的像素，再通过射线与平面求交恢复三维坐标。Open3D 窗口显示恢复出的点。

结果保存到：

```text
02_measurement_3d/results/synthetic/synthetic_report.json
02_measurement_3d/results/synthetic/synthetic_points.npz
```

查看报告：

```powershell
notepad 02_measurement_3d/results/synthetic/synthetic_report.json
```

报告中 `max_3d_error_mm` 是恢复坐标与预设真值之间的最大距离误差。程序要求它小于 `1e-5 mm`；本机已验证的结果约为 `5.97e-13 mm`。同一命令还验证了合成棋盘角点的 PnP 高度计算，预设高度为 19.8/20.0/20.2 mm，对应均值 20 mm、样本标准差 0.2 mm。

再次查看已经生成的点云：

```powershell
python 02_measurement_3d/measure3d.py view --points 02_measurement_3d/results/synthetic/synthetic_points.npz
```

将点云导出为 PLY 文件：

```powershell
python 02_measurement_3d/measure3d.py view --points 02_measurement_3d/results/synthetic/synthetic_points.npz --export-ply 02_measurement_3d/results/synthetic/synthetic_points.ply --no-window
```

合成数据的文件名和来源字段带有 `synthetic` 标记。后续接入线激光器时，需要实测相机坐标系中的激光平面方程，并从照片提取激光条纹中心像素。

## 七、遇到报错怎么处理

| 报错 | 处理方法 |
| --- | --- |
| `image size must be [1280, 720]` | 将相机设为 1280×720，使用原始照片重新拍摄 |
| `chessboard not detected` | 让整块棋盘进入画面，调整清晰度和光照，等角点标记出现后再保存 |
| `Nonparallel board` | 检查垫块是否等厚、背板是否架稳，恢复前后平行后重拍 |
| `RMS ... > 3.0 px` | 根据报错中的文件名找到照片，检查模糊、反光和板面弯曲，重拍该组 |
| `At least two repeated target images` | 检查目标目录，补拍到至少 2 张，完整实验建议 10 张 |
| 相机打不开 | 关闭占用相机的程序，或改用 `--camera 1` |

## 附：程序接口与数学公式

`02_measurement_3d/measure3d.py` 复用原 `calib_lib.py` 的参数读取、棋盘格检测和 `solve_pose()`。位姿满足：

```text
X_camera = R @ X_board + t
```

高度使用基准板面的单位法向 `n0` 计算：

```text
h_i = abs(n0 · (t_i - t_reference))
mean_mm = mean(h_i)
absolute_error_mm = abs(mean_mm - known_mm)
std_mm = sqrt(sum((h_i - mean_mm)^2) / (N-1))
```

目标板面与基准板面的法向夹角默认要求在 2° 以内，每张图的重投影 RMS 要求在 3 px 以内。

射线求交函数接收原始图像上的 N×2 像素坐标，返回 N×3 相机坐标及有效性掩码：

```python
from measure3d import intersect_pixels
points_mm, valid = intersect_pixels(pixels_uv, K, dist, [a, b, c, d])
```

平面方程 `aX+bY+cZ+d=0` 定义在相机坐标系，XYZ 单位为 mm。函数先去畸变得到射线 `r`，再求 `lambda=-d/(n·r)`，交点为 `lambda*r`。平行射线或相机后方交点对应 `valid=False`，坐标为 NaN。

从 CSV 输入像素时，先创建带 `u,v` 表头的文件。下面使用两个示例像素和合成平面：

```powershell
@'
u,v
647.61625628,405.42811581
700,420
'@ | Set-Content -Encoding utf8 02_measurement_3d/pixels_synthetic.csv
python 02_measurement_3d/measure3d.py intersect --pixels 02_measurement_3d/pixels_synthetic.csv --plane 0.2 -0.1 1 -550 --source SYNTHETIC_EXAMPLE --out 02_measurement_3d/results/intersections_synthetic.npz
python 02_measurement_3d/measure3d.py view --points 02_measurement_3d/results/intersections_synthetic.npz
```

运行数学单元测试：

```powershell
python -m unittest discover -s 02_measurement_3d -p test_measure3d.py -v
```

测试通过时终端末尾显示 `OK`。
