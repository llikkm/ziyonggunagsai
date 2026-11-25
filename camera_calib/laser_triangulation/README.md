# 激光三角测量仿真：31 mm 台阶

> 项目总览、模块地图与全套测试：见 [PROJECT_README.md](../PROJECT_README.md)。

这套程序解决一个很具体的问题：**没有激光器、没有真实照片，先在计算机里把“像素 + 相机内参 + 激光平面 → 三维点 → 台阶高度”这条数学链路跑通，并定量知道每一类误差会带来多大偏差。**

等激光器到手，只需要替换两个输入：

1. 把仿真生成的激光像素，换成从真实图像里提取的激光条纹中心；
2. 把标称激光平面，换成实际标定得到的平面参数。

相机内参、射线求交、高度估计、异常点剔除这些代码可以直接继续用。

---

## 0. 先说清楚：验证了什么，没验证什么

**已经验证**

- 针孔相机投影与反投影；
- 相机射线与激光平面求交；
- 从像素坐标反算相机坐标系三维点；
- 31 mm 台阶高度的稳健估计（自动分层 + 中位数 + MAD 剔除）；
- 像素噪声、焦距误差、激光平面法向/平移误差对高度的影响；
- 缺失点、异常点、非均匀噪声下的稳健性；
- “渲染激光图像 → 亚像素中心提取 → 三维反算”整条图像链路，和真实 webcam 路径共用同一套提取/反算函数；
- 有镜头畸变时，用 OpenCV `undistortPointsIter` 去畸变后再求交。

**没有验证**

- 真实激光平面的标定流程本身（本程序只提供平面 JSON 格式和 `fit-plane` 拟合接口）；
- 真实激光线的散斑、光斑宽度、景深、过曝、二次反射；
- 遮挡/阴影（仿真只画了台阶侧面，没有模拟激光被台阶挡住的阴影）；
- 完整三维扫描（目前是一条线、一个截面）；
- 相机噪声的相关性、固定模式噪声、卷帘快门等。

这些要等真实硬件到位后逐项补。当前这一步的价值是：**先把不依赖硬件的几何和误差传播做扎实。**

---

## 1. 快速开始

在全局 Python 3.12 下运行（不要建虚拟环境）：

```powershell
cd 'C:\Users\wangjianing\Desktop\光赛\camera_calib\laser_triangulation'

python simulate_step.py                 # 跑完整仿真，图和报告写到 out/，不弹窗
python simulate_step.py --show          # 同时显示 matplotlib 图窗
python simulate_step.py --no-image      # 跳过“渲染→提取”图像管线
python -m unittest test_laser_lib -v    # 19 个单元测试
```

依赖：

```text
numpy>=1.22
matplotlib>=3.5
opencv-python>=4.6     # 只有真实图像/webcam/畸变路径需要
```

纯仿真只需要 NumPy + Matplotlib；OpenCV 是延迟导入的。本机已用全局 Python 3.12、NumPy 2.4、Matplotlib 3.10、OpenCV 4.13 跑通。

运行后重点看：

| 文件 | 内容 |
| --- | --- |
| `out/report.txt` | 终端同款中文报告，含全部表格和检查结果 |
| `out/summary.json` | 机器可读的完整结果（配置、各实验、检查项） |
| `out/fig01_laser_pixels.png` | 三种噪声下的激光线像素位置 + 台阶边缘放大 |
| `out/fig02_step_profile.png` | 重建的 X–Z 台阶截面 |
| `out/fig03_3d_view.png` | 相机坐标系中的三维激光线 |
| `out/fig04_focal_error.png` | 焦距误差 → 高度误差 |
| `out/fig05_plane_error.png` | 激光平面法向误差 → 高度误差 |
| `out/fig06_robustness.png` | 缺失/异常/非均匀噪声下的估计器对比 |
| `out/fig07_image_pipeline.png` | 渲染图、提取中心、v(u) 剖面 |
| `out/laser_clean.png` / `laser_hard.png` | 合成激光图像，可直接喂给 `laser_measure.py` |
| `out/synthetic_camera.json` | 仿真相机内参（fx=fy=806, 1280×720, 无畸变） |
| `out/synthetic_plane.json` | 仿真激光平面（n=[0,1,0.2], d=-120） |

`simulate_step.py` 默认把检查结果当退出码：7 项全过返回 0，有失败返回 1。调试时可加 `--no-strict`。

---

## 2. 本机实际运行结果

下面全部是 **seed=42** 的实际输出，不是示意值。噪声档每档做 30 次独立抽样，表中给均值和标准差；图窗只画其中一次，所以图上的示例高度会和表里的平均值略有差别。

### 2.1 像素噪声的影响

| 噪声 (px) | 平均高度 (mm) | 平均误差 (mm) | 误差标准差 (mm) | 最大 \|误差\| (mm) | 平均 3D RMSE (mm) |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 31.000 | -0.000 | 0.000 | 0.000 | 0.000000 |
| 0.15 | 30.993 | -0.007 | 0.040 | 0.068 | 0.387 |
| 0.50 | 30.928 | -0.072 | 0.191 | 0.553 | 1.290 |

无噪声一档的高度误差在 `1e-13 mm` 量级，三维点 RMSE 约 `4.9e-14 mm`，说明反算公式本身没有系统偏差。

0.15 px 时高度误差约 0.04 mm；0.5 px 时误差标准差约 0.19 mm，单次最大偏差约 0.55 mm。这个量级和“31 mm 的 0.6%”相当，符合预期：**像素噪声先传到 Z，再被台阶差分放大。** 本场景的灵敏度约 `0.41 px/mm`（Z 变化 1 mm 对应激光线在 v 方向移动约 0.41 px），所以 0.5 px 噪声对应约 1.2 mm 的单点 Z 噪声；但高度取的是两层各几百个点的中位数之差，标准误会降到 0.1~0.2 mm。

### 2.2 测试 A：相机焦距错误

用真实 `fx=fy=806 px` 生成图像，反算时故意用错误的焦距。这个场景的激光平面是 Y–Z 平面，高度主要对 `fy` 敏感，所以测试让 `fx=fy` 一起偏。

| 反算焦距 (px) | 高度 (mm) | 高度误差 (mm) | 三维 RMSE (mm) | 模型位移均值 (px) |
| --- | --- | --- | --- | --- |
| 780 | 31.626 | +0.626 | 3.62 | 3.39 |
| 800 | 31.143 | +0.143 | 0.82 | 0.78 |
| **806** | **31.000** | **-0.000** | **0.00** | **0.00** |
| 820 | 30.672 | -0.328 | 1.87 | 1.83 |
| 840 | 30.214 | -0.786 | 4.46 | 4.44 |

结论：

- 806 控制组精确恢复；
- 800/820 产生**系统性**偏差，不是随机波动；
- 大致 `1 px` 焦距误差 → `0.02~0.03 mm` 高度误差；14 px 误差（820 vs 806）→ 0.33 mm；
- 误差沿 X 是分段常数（见 `fig04` 右图），在台阶处跳变，因为两层对同一条射线的 Z 敏感度不同。

如果你们真实标定的焦距有 0.5% 误差（806 的 0.5% ≈ 4 px），高度误差大约 0.1 mm 量级；如果差 2%（16 px），就会到 0.4 mm 量级。**焦距标定不能只做到“能跑”，要做到 0.1% 量级才不拖后腿。**

### 2.3 测试 B：激光平面标定误差

#### (a) 法向 n_z 扰动

绕测量体积中心（两条激光线的中点）旋转平面，保持枢轴点不变。这样 0.2 → 0.198/0.205 主要是在改变平面姿态，而不是整体平移。

| n_z | 法向夹角 (°) | 高度 (mm) | 高度误差 (mm) | 三维 RMSE (mm) |
| --- | --- | --- | --- | --- |
| 0.190 | 0.552 | 32.304 | +1.304 | 0.643 |
| 0.195 | 0.276 | 31.639 | +0.639 | 0.315 |
| 0.198 | 0.110 | 31.252 | +0.252 | 0.124 |
| **0.200** | **0.000** | **31.000** | **-0.000** | **0.000** |
| 0.202 | 0.110 | 30.752 | -0.248 | 0.122 |
| 0.205 | 0.275 | 30.387 | -0.613 | 0.302 |
| 0.210 | 0.550 | 29.797 | -1.203 | 0.593 |

#### (b) 平面整体平移（沿单位法向）

| 平移 (mm) | 高度 (mm) | 高度误差 (mm) |
| --- | --- | --- |
| -1.0 | 30.737 | -0.263 |
| -0.5 | 30.868 | -0.132 |
| **0.0** | **31.000** | **-0.000** |
| +0.5 | 31.132 | +0.132 |
| +1.0 | 31.263 | +0.263 |

结论（这是本轮最值得记住的数字）：

- 平面法向只改 `0.002`（约 `0.11°`），31 mm 高度就差 `0.25 mm`；
- 改 `0.005`（约 `0.28°`），高度差 `0.61 mm`；
- 平面整体平移 `1 mm`，高度差约 `0.26 mm`。

所以如果目标高度误差是 `0.5 mm` 以内，激光平面大致需要：

- **法向稳定到 0.1°~0.2° 量级**（对应 n_z 的约 ±0.002~0.004）；
- **平面位置稳定到 1 mm 以内**（沿法向）；
- 并且要在整个测量视场内有约束，不能只在中心一个点对上。

这也解释了为什么真实系统里激光平面标定比相机内参更“娇气”：它不是简单的像素误差，而是直接把平面姿态误差放大到深度。等激光器到手，建议优先把平面标定的重复性做到 0.1° 以内，再谈更高精度。

### 2.4 测试 C：缺失点、异常点、非均匀噪声

| 场景 | 均值（已知分层） | 中位数（已知分层） | 自动分层+中位数 | **自动分层+MAD 剔除** |
| --- | --- | --- | --- | --- |
| 干净：0.15 px 均匀噪声 | 0.003 | 0.033 | 0.101 | **0.050** |
| 缺失 25% 点 | 0.008 | 0.019 | 0.074 | **0.020** |
| 4% 异常点（同向偏 5~10 px） | 0.193 | 0.001 | 0.050 | **0.011** |
| 非均匀噪声（0.08~0.70 px） | 0.097 | 0.046 | 0.046 | **0.012** |
| 缺失 20% + 异常 4% + 非均匀 | 0.016 | 0.016 | 0.063 | **0.003** |

表中是 `|高度误差|`，单位 mm。

怎么看：

- “均值（已知分层）”和“中位数（已知分层）”用了仿真标签来分层，只是**对照基线**，真实程序拿不到这些标签；
- 真实程序用的是最后一列“自动分层+MAD 剔除”；
- 同向异常点（模拟反光/二次反射）会把均值拉偏到 0.19 mm，中位数基本不受影响；
- 自动分层会把台阶侧面/过渡点分到某一层，所以“自动+中位数”会比“已知分层中位数”略差；加 MAD 剔除后回到 0.01 mm 量级；
- 组合场景（缺失+异常+非均匀）仍然恢复到 31.003 mm。

对应的稳健估计算法：

1. 沿 Z 做一维两水平聚类（用中位数迭代，抗离群点）；
2. 用台阶间距和两簇的 MAD 判断“这是台阶还是同一平面的噪声”；
3. 只保留离各自层中心 `0.30 × 台阶间距` 以内的点，去掉台阶侧面/边缘过渡点；
4. 每层再按 `4 × MAD` 剔除；
5. 高度 = 两层中位数之差，并用中位数的标准误给出不确定度。

### 2.5 图像管线：渲染 → 提取 → 反算

这一节和真实 webcam 路径走同一套 `extract_laser_centers()` 和 `reconstruct()`，只是图像由程序渲染。

| 图像 | 有效列 | 有效比例 | 中心误差均值 (px) | 中心误差 P95 (px) | 高度 (mm) | 高度误差 (mm) | Z RMSE (mm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 400 | 100.0% | 0.021 | 0.034 | 31.035 | +0.035 | 0.125 |
| hard | 336 | 84.0% | 0.030 | 0.046 | 31.032 | +0.032 | 0.146 |

- **clean**：高斯截面激光线 + 背景渐变 + 读出噪声 + 散粒噪声；
- **hard**：在 clean 基础上加了 3 段缺失（模拟遮挡/弱反射）、8 个随机亮点（模拟反光）、沿线亮度不均（局部只剩 25% 强度）；
- 两张图的提取中心误差都在 `0.02~0.05 px`，高度误差 `0.03 mm` 量级；
- hard 图有效列降到 84%，但稳健估计仍然恢复到 31.032 mm。

复现命令：

```powershell
python laser_measure.py image --image out/laser_clean.png `
    --calib out/synthetic_camera.json --plane out/synthetic_plane.json --plot `
    --out out/measure_clean

# hard 图建议加 ROI，避免全图背景热点被误提取
python laser_measure.py image --image out/laser_hard.png `
    --calib out/synthetic_camera.json --plane out/synthetic_plane.json `
    --roi 447 847 380 418 --plot --out out/measure_hard_roi
```

本机结果：clean `31.030 mm`，hard（带 ROI）`31.034 mm`，与仿真真值 31 mm 一致。

注意：合成图是用 `out/synthetic_camera.json`（fx=fy=806）生成的，必须配这个内参；如果误用真实标定 `../out/calib.json`（fx≈990、强畸变），会得到约 41 mm 的错误高度——这不是算法出错，而是内参和图像不匹配。真实图反过来也不能用 `synthetic_camera.json`。本机专门试过这个错配，结果确实偏到 41.3 mm，可作为“内参必须和图像对应”的反例。

### 2.6 完成标准检查

`simulate_step.py` 末尾会自动检查：

```text
✓ 无噪声精确恢复: H=31.000000000 mm, RMSE=4.895e-14 mm
✓ 0.15/0.5 px 噪声可量化: 0.15 px -> -0.007 ± 0.040 mm; 0.50 px -> -0.072 ± 0.190 mm
✓ 焦距错误产生系统性偏差: f=800 -> 0.143 mm, f=820 -> 0.328 mm (fx=fy)
✓ 激光平面法向误差产生系统性偏差: n_z=0.198 -> 0.252 mm, n_z=0.205 -> 0.613 mm
✓ 图像管线恢复台阶: clean 0.035 mm, hard 0.032 mm
✓ 缺失+异常+非均匀下仍稳健: |error|=0.003 mm, 内点 486, 剔除 36
✓ 图表已生成: 7 张 PNG
检查: 7/7 通过
```

这七项对应任务书里的四条完成标准，另加图像链路和稳健性两项。

---

## 3. 数学与代码

### 3.1 坐标系、单位

和 `camera_calib` 完全一致：

- 相机坐标系：X 向右，Y 向下，Z 沿光轴向前；
- 所有长度单位：**mm**；
- 像素坐标：u 向右，v 向下，整数坐标表示像素中心；
- 相机模型：`[u,v,1]^T ~ K [X,Y,Z]^T`；
- 激光平面：`n · X + d = 0`，n、d 都在相机坐标系下。

仿真场景：

- 远面（低层）`Z = 500 mm`，X 在 `[-120, 0)`；
- 近面（上层）`Z = 469 mm`，X 在 `[0, 120]`；
- 台阶边在 `X = 0`，高度 `31 mm`；
- 激光平面 `Y + 0.2 Z - 120 = 0`；
- 于是远面激光线在 `Y=20 mm`，近面在 `Y=26.2 mm`，图像上表现为一条近水平线，在台阶处上下跳约 13 px。

这里把示例里的 X/Y 角色对调了（示例是沿 Y 分界、平面法向主要在 X；本程序是沿 X 分界、平面法向主要在 Y）。这样做只是为了让激光线在图像里接近水平、更像真实 webcam 的构图；射线-平面求交的公式完全一样。相机的 X/Y/Z 约定、单位和求交代码都没有变。

### 3.2 正向投影

无畸变：

```text
[u, v, 1]^T ~ K [X, Y, Z]^T
```

有畸变时用 `cv2.projectPoints()`。本程序的 `CameraModel.project()` 会自动判断。

### 3.3 反向求交

像素转射线：

```text
r = K^{-1} [u, v, 1]^T = [x, y, 1]^T
```

有畸变时 `r` 由 `cv2.undistortPointsIter()` 得到（和 `measurement_3d/measure3d.py` 的 `intersect_pixels()` 一致）。

射线与平面求交：

```text
t = -d / (n · r)
P = t r
```

`t` 在 `[x,y,1]` 参数化下恰好就是相机坐标 Z。平行射线（`n·r≈0`）或交点在相机后方（`t<=0`）标记为无效，坐标为 NaN。

### 3.4 高度估计

`robust_step_height()` 不读真值，只接收三维点：

1. 沿 Z 做一维两水平聚类（中位数迭代）；
2. 用两簇的稳健 sigma 判断是否存在台阶：`sep > max(min_separation, 6σ)`；
3. 去掉离各自层中心超过 `0.30 × sep` 的点（台阶侧面、边缘过渡）；
4. 每层按 `4 × MAD` 剔除异常点；
5. 高度 = 两层中位数之差；
6. 中位数标准误 `1.2533 σ / sqrt(n)`，两层合成。

### 3.5 为什么算法不能读真值

`reconstruct()` 的入参只有：

- 像素坐标 `uv`；
- 相机内参 `K`（以及畸变 `dist`）；
- 激光平面 `n, d`。

它完全不知道 `StepScene` 的存在。真值三维点只在最后用于：

- `rmse_3d()` 计算三维误差；
- `points_est - points_true` 画剖面；
- 检查“无噪声是否精确恢复”。

这是这套仿真有意义的前提。

---

## 4. 三个测试怎么读

### 测试 A：相机参数错误

- 现象：高度误差随焦距误差近似线性变化，且沿 X 是分段常数；
- 原因：焦距改变射线方向，平面求交后 Z 系统性地偏大/偏小；
- 工程含义：相机标定后不能改分辨率、不能数字变焦、不能换镜头；重标定要检查 fx/fy 的重复性。

这个场景的高度主要对 `fy` 敏感（激光平面是 Y–Z 平面，Z 只由 v 方向射线决定）。如果真实系统的激光平面有 X 分量，`fx` 也会同样重要。测试里让 `fx=fy` 一起偏，代表“整体焦距标定错”。

### 测试 B：激光平面标定误差

- 法向误差：高度误差近似与 `Δn` 成正比，误差沿 X 分段常数；
- 平移误差：高度误差近似与 `Δd` 成正比，但比法向误差温和；
- 工程含义：平面法向比平面位置更敏感；标定时要保证多姿态、多位置的约束，不能只在一个点对准。

如果要把高度误差控制在 0.5 mm 内：

- 法向夹角 ≲ 0.2°；
- 平面平移 ≲ 1 mm；
- 并且要确认整个视场内残差均匀，而不是中心对、边缘偏。

### 测试 C：采集问题

- 缺失点：中位数/自动分层不受影响，只要每层还剩足够点；
- 异常点：均值会被拉偏，中位数和 MAD 剔除基本不受影响；
- 非均匀噪声：点多的区域权重大，中位数仍然稳；
- 组合场景：自动分层+MAD 剔除恢复到 31.003 mm。

真实图像里，“异常点”通常来自反光、二次反射、另一条亮线、过曝饱和；提取阶段用峰值/宽度/相对阈值过滤，三维阶段再用 MAD 剔除，两道防线都有必要。

---

## 5. 接入真实相机和激光器

### 5.1 相机内参

直接用现有 `camera_calib` 的标定结果：

```powershell
cd 'C:\Users\wangjianing\Desktop\光赛\camera_calib'
python calibrate.py --dir data/calib --square <实测格边长>
# 产物：out/calib.json
```

`laser_measure.py` 默认读 `../out/calib.json`。注意：

- 标定后**不要改分辨率、焦距、数字变焦、对焦**；
- 图像分辨率必须和 `calib.json` 里的 `image_size` 一致，程序会检查；
- 有畸变时会自动去畸变，不需要先手动 `undistort` 整张图；
- 如果裁剪了图像，`cx/cy` 必须同步修正，否则会有系统偏差。

### 5.2 激光平面标定

激光平面必须表示在**同一个相机坐标系**下，单位 mm：

```text
n · X + d = 0
```

推荐用棋盘格做约束（和现有 `measurement_3d` 的思路一致）：

1. 固定相机和激光器；
2. 把棋盘格放在多个不同位姿，每个位姿拍一张“棋盘 + 激光线”的图；
3. 对每个位姿，用 `solvePnP` 求棋盘平面在相机坐标系中的方程；
4. 提取该图激光线的像素中心，用 `intersect_pixels(pixels, K, dist, board_plane)` 求交，得到落在棋盘平面上的三维点；
5. 多个位姿的点合在一起，用 `fit_plane_svd()` / `laser_measure.py fit-plane` 拟合激光平面。

第 4 步的 `board_plane` 可以由棋盘位姿得到：板面法向是旋转矩阵第 3 列，`d = -n·t`。这和 `measurement_3d` 里的 `height_statistics()` 用的是同一套位姿约定。

也可以由机械设计值手写：

```powershell
python laser_measure.py make-plane `
    --point 0 20 500 --normal 0 1 0.2 `
    --out my_laser_plane.json
```

但机械值很难保证在相机坐标系里准确，建议只用它做初值，最终仍用棋盘格/已知平面拟合。

平面 JSON 格式（`fit-plane` 会自动写这个格式）：

```json
{
  "unit": "mm",
  "frame": "camera: X right, Y down, Z forward",
  "normal": [0.0, 1.0, 0.2],
  "d": -120.0,
  "source": "FIT_FROM_POINTS"
}
```

也接受 `plane_abcd: [a,b,c,d]` 或 `unit_normal` + `normalized_d`。`fit-plane` 会尽量让 `d<=0`（相机原点在平面负侧），只是为了让报告更好读，平面本身不变。

### 5.3 单张真实图像测量

```powershell
python laser_measure.py image `
    --image D:\data\laser_step.png `
    --calib ..\out\calib.json `
    --plane my_laser_plane.json `
    --roi 400 900 300 500 `
    --plot
```

输出：

- 终端：有效列数、中心误差（如果有真值）、高度、近/远面 Z、内点/剔除数；
- `out/measure/measure.json`：完整结果；
- `out/measure/points.csv` / `points.npz`：反算出的三维点；
- `out/measure/extraction.csv`：每个有效列的 u、v、峰值、SNR、宽度；
- `out/measure/measure.png`：图像叠加 + v(u) 剖面（`--plot`）。

可选：用现有 `measurement_3d` 的 Open3D 查看反算点（需要装 open3d）：

```powershell
python ..\measurement_3d\measure3d.py view --points out\measure_clean\points.npz
```

`points.npz` 里的 `points_mm` 是完整数组，无效交点为 NaN；`measure3d view` 会自动过滤。

`--roi` 很重要：真实图像里可能有背景反光、桌面边缘、另一条亮线。把 ROI 收到激光线附近，能显著减少误提取。不填 ROI 时程序扫全图，靠峰值阈值和 10% 相对阈值过滤背景，但反光点仍可能混进来。

### 5.4 webcam 实时预览

```powershell
python laser_measure.py live `
    --camera 0 `
    --calib ..\out\calib.json `
    --plane my_laser_plane.json `
    --width 1280 --height 720
```

窗口里：

- 红色点是提取到的激光中心；
- 绿色文字是当前高度 `H = xx.xx mm` 和有效点数；
- `q` 退出，`s` 保存当前帧 + JSON 结果到 `out/live/live_YYYYmmdd_HHMMSS/`。

如果打不开相机，先关掉微信/腾讯会议/浏览器里占用摄像头的页面，或换 `--camera 1`。背景有反光时，和 `image` 子命令一样加 `--roi X0 X1 Y0 Y1`。

### 5.5 真实图像参数怎么调

| 现象 | 调什么 |
| --- | --- |
| 提取到很多背景点 | 加大 `--min-peak`、`--threshold-sigma`，或加 `--roi` |
| 激光线局部太暗被丢掉 | 减小 `--min-peak-fraction`（默认 0.10），或降低 `--threshold-sigma` |
| 激光线很粗/散焦 | 加大 `--centroid-radius` 和 `--max-width` |
| 噪声大、中心抖动 | 加大 `--smooth`（沿扫描方向平滑，不展宽截面） |
| 激光线接近竖直 | `--axis row`（`auto` 通常也能判断） |
| 高度估不出来 | 检查 `--min-step`（默认 2 mm）是否大于噪声；检查平面是否在相机坐标系 |
| 图像分辨率与标定不一致 | 用标定时的分辨率重拍，或按新分辨率重标 |

真实采集时先看直方图：**激光峰值不要到 255（饱和会让中心偏移），背景尽量暗，条纹宽度 1~3 px 比较理想。**

---

## 6. 文件与接口

```text
laser_triangulation/
├─ laser_lib.py             核心库：相机模型、激光平面、求交、稳健高度、提取、渲染
├─ simulate_step.py         主仿真：噪声、测试 A/B/C、图像管线、图、报告
├─ laser_measure.py         真实图像 / webcam / 平面拟合 CLI
├─ test_laser_lib.py        19 个单元测试
├─ laser_plane.example.json 仿真用标称平面（真实测量必须替换）
├─ requirements.txt
└─ out/                     报告、CSV、JSON、PNG、合成图
```

核心 API：

```python
import numpy as np
import laser_lib as ll

# 1) 读现有标定和激光平面
cam = ll.CameraModel.from_calib_json("../out/calib.json")
plane = ll.LaserPlane.from_json("my_laser_plane.json")

# 2) 像素 -> 三维
rec = ll.reconstruct(uv, cam, plane)          # uv: N×2 原始像素
points = rec.points_mm[rec.valid]             # N×3, 相机坐标系, mm

# 3) 稳健台阶高度（不需要真值标签）
est = ll.robust_step_height(points, axis=2)
print(est.height_mm, est.level_hi_mm, est.level_lo_mm, est.height_se_mm)

# 4) 从灰度图提取激光中心（真实图像/合成图共用）
ext = ll.extract_laser_centers(gray, axis="auto", roi=(x0, x1, y0, y1))
rec = ll.reconstruct(ext.uv, cam, plane)

# 5) 拟合激光平面（真实标定用）
plane = ll.fit_plane_svd(points_on_laser_plane)
```

与现有 `measurement_3d` 的兼容接口：

```python
from laser_lib import intersect_pixels
points, valid = intersect_pixels(raw_pixels, K, dist, [a, b, c, d])
```

和 `measurement_3d/measure3d.py` 里的同名函数语义一致：原始畸变像素进，相机坐标系点出。

---

## 7. 单元测试

```powershell
python -m unittest test_laser_lib -v
```

本机结果：`Ran 19 tests ... OK`。覆盖：

- 无噪声投影/反投影往返（<1e-9 mm）；
- 焦距错误、平面法向错误产生系统性偏差；
- 平面 SVD 拟合、三点定平面、共线报错；
- 自动分层、异常点不改变高度、平面不被误判为台阶；
- 合成图提取（干净/缺失+热点/竖直线的 row 模式）；
- 有畸变时 `projectPoints` 与 `undistortPointsIter` 往返（<1e-6 mm）；
- 输入校验（零法向、坏 K、非二维图、越界 ROI）。

---

## 8. 限制与下一步

**当前限制**

- 只有一条激光线、一个截面，不是完整三维扫描；
- 没有模拟台阶对激光的遮挡/阴影；真实台阶边缘可能出现一段“无激光”的阴影区；
- 激光截面用高斯模型，没有散斑、衍射、景深、饱和；
- 相机噪声是简单的读出+散粒模型，没有固定模式噪声/卷帘快门；
- 激光平面标定流程只给了接口和拟合函数，没有一键采集脚本（需要和 `measurement_3d` 的棋盘位姿配合）；
- `robust_step_height()` 假设“两个水平层”，连续曲面/斜面不适用；
- 提取是逐列/逐行的一维方法，遇到强弯曲、多条线、严重过曝时需要更复杂的二维方法。

**等激光器到手后的最短路径**

1. 用现有 `camera_calib` 重标一次相机，确认 `image_size`、fx/fy 重复性；
2. 按 5.2 用棋盘格多姿态标定激光平面，把 `fit-plane` 的残差压到 0.1 mm 以内，并检查法向重复性；
3. 拍一张 31 mm 台阶（或量块/垫块）的激光图，用 `laser_measure.py image` 测量，和卡尺值对比；
4. 通过后，把 `laser_measure.py live` 接到 webcam，做实时预览；
5. 再考虑加一维扫描（平移台/传送带），把每帧的 `points.npz` 拼成点云。

**一个重要的心态**：这一步已经把“几何对不对”与“硬件好不好”分开了。等真实数据进来，如果高度不对，先分别检查：

- 相机内参（fx/fy/cx/cy、分辨率）；
- 激光平面（法向、d、坐标系、单位）；
- 激光线提取（中心是否偏、是否饱和、是否有反光）；
- 台阶区域分割（是否把侧面/阴影算进了某一层）。

这四类问题在仿真里都能单独复现，排查时不要混在一起。

---

## 附：最短命令序列

```powershell
cd 'C:\Users\wangjianing\Desktop\光赛\camera_calib\laser_triangulation'

# 1. 纯仿真 + 报告 + 图
python simulate_step.py

# 2. 单元测试
python -m unittest test_laser_lib -v

# 3. 用合成图验证真实图像路径
python laser_measure.py image --image out/laser_clean.png `
    --calib out/synthetic_camera.json --plane out/synthetic_plane.json `
    --plot --out out/measure_clean

# 4. 真实相机 + 实测激光平面（示例路径，按实际改）
python laser_measure.py image --image D:\data\laser_step.png `
    --calib ..\out\calib.json --plane my_laser_plane.json `
    --roi 400 900 300 500 --plot

# 5. webcam 实时
python laser_measure.py live --camera 0 `
    --calib ..\out\calib.json --plane my_laser_plane.json `
    --width 1280 --height 720
```

---

## 9. 下一阶段：激光平面标定仿真

真实系统里 `my_laser_plane.json` 不能靠猜，需要先用棋盘格标定。独立模块 [../laser_plane_calib/README.md](../laser_plane_calib/README.md) 做了这件事：

- 模拟 15 组不同位姿的 20 mm 标称棋盘格（实物打印后实测 19 mm 时传 `--square 19`）；
- 从 measured corners 做 PnP，得到棋盘格平面；相机射线与板面求交，得到板上的激光三维点；
- 用 SVD 拟合激光平面，输出归一化 `n, d`、残差、法向角度误差；
- 用未参与标定的 10 / 31 / 50 mm 台阶做独立验证；
- 分析标定板数量、姿态覆盖、角点/激光噪声，以及近竖直条纹应按 row 还是 column 提取。

它输出的 `laser_plane_calib/out/results/laser_plane.json` 可以直接用本模块的 `laser_measure.py --plane` 加载。本机实测：用 15 块板标定后，独立 31 mm 台阶在标定模块内测得 30.854 mm；把同一个平面 JSON 交给 `laser_measure.py` 处理同一张验证图，测得 31.047 mm。
