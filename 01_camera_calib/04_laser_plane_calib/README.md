# 激光平面标定仿真（20 mm 标称棋盘格）

> 项目总览、模块地图与全套测试：见 [PROJECT_README.md](../PROJECT_README.md)。

2026-11-17 修复说明：维持 20 mm 和原有基础几何算法。退化标定现在拒绝正常返回或导出；覆盖实验只保留退化诊断，不再对退化平面测量高度。验证反算仅接收测量字典，真值单独交给评价函数。漏测会计入失败，主验证漏测时程序返回非零状态。

`measure_validation(fit, [c.measurement for c in cases], camera)` 是新的验证调用方式。重复或未知场景会报错，缺少的场景按漏测处理。

完整仿真会把每次板数子集索引、噪声扰动后的测量记录分别存入 `data/experiments/`，拟合结果存入 `results/experiments/`，逐场景验证结果存入 `validation/experiments/`。使用根目录的 `results/calibration.json` 中的相机和棋盘格规格，即可重新拟合归档的测量记录。板数索引对应主数据的 `index`。JSON 中无效数值保存为 null。

下文数值是修复前的历史输出；正常主标定结果保持一致。`same_plane` 的高度验证现在被拦截，原表中的 165.660 mm 不再作为可用测量值输出。重新运行脚本即可更新输出。测试命令：`python -m unittest test_plane_calib test_regressions -v`。

这是 `03_laser_triangulation` 的独立扩展模块：**用 8~15 组不同位姿的模拟棋盘格，先恢复每块板上的激光三维点，再用 SVD 拟合激光平面，最后用未参与标定的 10 / 31 / 50 mm 台阶做独立验证。**

它不修改已通过的 `laser_lib.py` / `simulate_step.py`。核心算法只接收：

- 测量到的棋盘格角点；
- 测量到的激光条纹中心；
- 相机内参 `K`、畸变 `dist`；
- 棋盘格规格（方格数、方格边长）。

**真实激光平面不在标定函数的入参里。** 真值只用于三处：渲染模拟图像、生成真值条纹线、最后计算角度/高度误差。

---

## 0. 关于 20 mm 和 19 mm

这次仿真按 **20.0 mm 标称方格**生成。你们实物标定板打印时有缩放，实测是 19 mm；这是**打印误差**，不是棋盘格设计尺寸。

- 本仿真默认 `--square 20.0`，对应设计值；
- 如果要用真实打印板的数据重跑，传 `--square 19`（或卡尺实测值）：
  ```powershell
  python calibrate_plane_sim.py --square 19
  ```
- 真实相机标定也应按实测格边长传 `--square 19`，否则所有尺度会整体偏约 5%。

---

## 1. 快速开始

在全局 Python 3.12 下运行（不要建虚拟环境）：

```powershell
cd 'C:\Users\wangjianing\Desktop\光赛\01_camera_calib\04_laser_plane_calib'

python calibrate_plane_sim.py                  # 默认 15 块板，跑全部实验
python calibrate_plane_sim.py --show           # 同时显示图窗
python calibrate_plane_sim.py --synthetic-camera   # 不读 calib.json，用 fx=fy=806 合成相机
python calibrate_plane_sim.py --square 19      # 实物打印板实测 19mm
python -m unittest test_plane_calib test_regressions -v         # 20 个单元测试
```

依赖（与现有工程一致）：

```text
numpy>=1.22
matplotlib>=3.5
opencv-python>=4.6
```

默认相机内参读 `../out/calib.json`（现有 `01_camera_calib` 的标定结果，1280×720，有畸变）。如果 `calib.json` 不存在，可用 `--synthetic-camera` 跑通。

运行时间约 1~2 分钟；会生成约 100 张模拟图（15 块主数据 + 4 组覆盖实验 + 10 张验证图）。

---

## 2. 本机实际结果

下面全部是默认参数（seed=42、15 块板、20 mm 方格、角点/激光各 0.2 px 额外噪声）的实际输出。

### 2.1 相机与场景

| 项目 | 值 |
| --- | --- |
| 相机 | fx=989.768, fy=991.661, cx=647.616, cy=405.428, 1280×720, 有畸变 |
| 棋盘格 | 9×6 格，8×5 内部角点，20.000 mm，物理 204.0×144.0 mm |
| 真实激光平面 | `n=[0.980581, 0, 0.196116], d=-98.0581`（`X+0.2Z-100=0` 的归一化形式） |
| 标定板数 | 15 |
| 额外噪声 | 角点 0.200 px，激光中心 0.200 px |
| PnP 重投影 RMS | 平均 0.324 px，最大 0.405 px |
| 每块板激光线长度 | 144.0~150.8 mm，平均 146.1 mm |

### 2.2 拟合结果（标定程序输出）

```text
归一化平面: n = [0.98061111, 0.00006791, 0.19596387]
            d = -97.97481276
点到平面残差: RMS 0.118840 mm, max 0.398134 mm
点数: 4228
点云条件数 s0/s1: 1.189  (正常)
法向量角度误差: 0.009710°
平面 d 误差:   +0.083255 mm
```

理想投影对照（直接用真实位姿投影角点和条纹，不走渲染/检测）：

```text
角度误差 8.538e-07°
拟合残差 1.603e-08 mm
```

这说明几何链路本身是精确的；0.0097° 和 0.119 mm 主要来自模拟的角点检测/条纹提取噪声，而不是公式误差。

### 2.3 独立验证：10 / 31 / 50 mm 台阶

验证台阶**没有参与标定**；反算只用上面拟合出的平面，真值只用于最后算误差。

| 场景 | 真值 (mm) | 拟合平面测量 (mm) | 误差 (mm) |
| --- | --- | --- | --- |
| h10_center | 10 | 9.9359 | -0.0641 |
| h31_center | 31 | 30.8538 | -0.1462 |
| h50_center | 50 | 49.9579 | -0.0421 |
| h31_near_left | 31 | 30.9912 | -0.0088 |
| h31_far_right | 31 | 31.0621 | +0.0621 |
| h31_outside | 31 | 30.7989 | -0.2011 |
| h10_near | 10 | 10.1204 | +0.1204 |
| h10_far | 10 | 10.1576 | +0.1576 |
| h50_near | 50 | 50.0200 | +0.0200 |
| h50_far | 50 | 50.1668 | +0.1668 |

**汇总：平均 |误差| 0.0989 mm，最大 0.2011 mm，10/10 有效。**

`h31_outside` 在 650 mm（标定体 420~580 mm 之外）仍然只有 -0.20 mm，说明这个标定结果有一定外推能力；但外推不能当作常规使用，真实系统仍应让标定覆盖工作距离。

### 2.4 标定板数量

每档随机抽 20 个子集（15 块档只有全集）：

| 板数 | 法向角误差 (°) | 拟合 RMS (mm) | 验证均值±std (mm) |
| --- | --- | --- | --- |
| 5 | 0.0423 ± 0.0422 | 0.1145 | 0.1403 ± 0.0651 |
| 8 | 0.0308 ± 0.0342 | 0.1155 | 0.1265 ± 0.0525 |
| 10 | 0.0271 ± 0.0321 | 0.1190 | 0.1204 ± 0.0421 |
| 12 | 0.0102 ± 0.0050 | 0.1175 | 0.0998 ± 0.0024 |
| 15 | 0.0097 | 0.1188 | 0.0989 |

结论：

- **12 块以上比较稳定**（约 0.01° / 0.10 mm）；
- 8~10 块已经可用，但随机子集方差明显更大（0.03~0.04°）；
- 5 块最散（0.042° / 0.14 mm）；
- 如果时间允许，建议 **10~12 块以上**，并且优先保证姿态覆盖，而不是简单重复同一姿态。

### 2.5 不同姿态覆盖

每组 8 块板；除 `good` 外，其余几组是不同采样范围生成的新数据。

| 覆盖方式 | 角误差 (°) | 条件数 | 退化 | 验证均值 (mm) | 验证最大 (mm) |
| --- | --- | --- | --- | --- | --- |
| good（多位置/多距离/双向倾角） | 0.0057 | 1.2 | 否 | 0.103 | 0.283 |
| fronto（全正对、无倾角） | 0.0565 | 1.6 | 否 | 0.344 | 0.653 |
| one_axis（只绕 X 倾斜） | 0.0123 | 1.3 | 否 | 0.100 | 0.216 |
| same_plane（同一 Z=500 正对，板面重合） | 11.0935 | 94.9 | **是** | **165.660** | 375.913 |
| close（只在 400~430 mm） | 0.0359 | 3.4 | 否 | 0.113 | 0.301 |

结论：

- **全正对、无倾角**明显变差（验证 0.344 mm vs good 0.103 mm）；倾斜姿态对约束平面法向很重要。
- **同一平面重复摆板**是最坏情况：所有激光点落在同一条三维直线上，点云条件数从 1.2 涨到 94.9，SVD 法向绕这条线任意旋转，验证直接失效（165 mm）。
- 只绕一个轴倾斜（one_axis）在这个几何下仍然可用，但不要把它当成通用结论；两个方向都倾斜更稳。
- 只在近距离标定（close）本次还可以，但外推到更远位置的风险比 good 大。

### 2.6 角点噪声 / 激光中心噪声

σ=0 表示只经过渲染/检测/提取、不再额外加噪的基线；不是理想投影。

| 噪声 | 额外 σ (px) | 角误差 (°) | 拟合 RMS (mm) | 验证均值 (mm) |
| --- | --- | --- | --- | --- |
| 角点 | 0 | 0.0241 | 0.0392 | 0.0988 |
| 角点 | 0.5 | 0.0430 | 0.1319 | 0.1415 |
| 角点 | 1.0 | 0.0684 | 0.2347 | 0.2773 |
| 激光 | 0 | 0.0241 | 0.0392 | 0.0988 |
| 激光 | 0.5 | 0.0240 | 0.2629 | 0.0985 |
| 激光 | 1.0 | 0.0286 | 0.5223 | 0.1053 |

本几何下的结论：

- **角点噪声更关键**：它会先扰动棋盘格平面，再通过“射线 × 板面”传到激光点；0.5 px 角点噪声让验证均值到 0.14 mm，1.0 px 到 0.28 mm。
- **激光中心噪声主要抬高拟合残差**：0.5 px 时 RMS 从 0.039 涨到 0.263 mm，但法向和台阶验证几乎不变（点很多，平面拟合对它不敏感）。
- 这不是“激光提取不重要”，而是说：**先把棋盘格角点和板面位姿做好**，激光中心的随机噪声在多点平均下影响较小；如果激光提取有系统偏差（比如按列提取竖直条纹），那就是另一回事，见下一节。

### 2.7 近竖直条纹：按列 vs 按行

本场景的激光平面法向主要在 X，条纹在图像里近似竖直。结果：

| 模式 | 每姿态有效中心数 | 中心误差均值 (px) | 自动选 row 的比例 |
| --- | --- | --- | --- |
| row | 281.9 | 0.0526 | - |
| column | 15.7 | 1.1264 | - |
| auto 全图 | 15.7 | 1.1264 | 0% |
| auto 紧 ROI | 281.9 | 0.0531 | 100% |

**结论：当前条纹必须用 `axis="row"`。**

- 按列扫描每条竖线只能拿到十几个中心（沿 u 方向只跨十几列），而且中心误差 1.13 px；
- 按行扫描每条竖线能拿到约 282 个中心，误差 0.053 px；
- **全图 `axis="auto"` 会误选 column**：细条纹只占全图不到 1% 像素，现有 auto 的 99 分位主方向启发式被背景点带偏；
- 把 ROI 收紧到条纹附近后，auto 才 100% 选 row。

本次标定**全部显式使用 row**，不依赖 auto。这属于使用建议，基础 `laser_lib` 没有修改。

### 2.8 与已有激光测量程序的衔接

把拟合出的 `out/results/laser_plane.json` 直接喂给 `03_laser_triangulation/laser_measure.py`，处理独立验证图 `h31_center.png`：

```powershell
python ..\03_laser_triangulation\laser_measure.py image `
    --image out\validation\images\h31_center.png `
    --calib ..\out\calib.json `
    --plane out\results\laser_plane.json `
    --axis row --roi 620 690 230 580 `
    --out out\integration_h31 --plot
```

本机结果：

```text
台阶高度: 31.0467 mm
远面 499.975, 近面 468.928, 标准误 0.0034
有效 254/350 行
```

与标定模块自己的验证结果（30.8538 mm）相比，两者相差约 0.19 mm，都在 0.2 mm 内；差异来自 ROI、提取和噪声实现不同。这说明**标定输出的平面 JSON 可以直接被测量程序使用**。

---

## 3. 算法与数据流

### 3.1 模拟数据

每个位姿生成两张图：

- `board_XX.png`：棋盘格图（按真实位姿渲染，含镜头畸变）；
- `laser_XX.png`：同一姿态、同一真实激光平面的条纹图（暗背景 + 高斯截面）。

棋盘格图用于 `cv2.findChessboardCornersSB` 检角点；激光图用于条纹中心提取。真实系统可以用窄带滤光片得到类似的“只有激光”的图像。

角点检测后叠加可配置高斯噪声 `sigma_corner`；条纹中心提取后叠加 `sigma_laser`。两者都是像素单位。

### 3.2 标定（不读真值）

对每组测量：

1. 用 measured corners + 19/20 mm 方格 + `K, dist` 做 `solvePnP`，恢复棋盘格位姿 `X_cam = R X_board + t`；
2. 棋盘格平面：`n_board = R[:,2]`，`d_board = -n_board · t`；
3. 对每个 measured laser pixel，做去畸变得到相机射线，和棋盘格平面求交，得到激光三维点；
4. 所有位姿的激光点拼在一起，用 SVD 拟合平面：最小奇异向量为法向，`d = -n·centroid`；
5. 输出归一化 `n, d`、点到平面残差 RMS/max、条件数。

`calibrate_plane()` 的签名：

```python
calibrate_plane(measurements, camera, board)
```

没有 `true_plane`，也没有真实位姿。

### 3.3 数据隔离

| 文件 | 内容 | 是否含真值 |
| --- | --- | --- |
| `data/measurements.json` | 角点、激光中心、图像名 | 否 |
| `data/truth.json` | 真实平面、真实位姿 | 是 |
| `validation/measurements.json` | 验证条纹中心 | 否 |
| `validation/truth.json` | 验证台阶真值 | 是 |
| `data/coverage/<set>/measurements.json` | 覆盖实验测量值 | 否 |
| `data/coverage/<set>/truth.json` | 覆盖实验真值 | 是 |
| `results/calibration.json` | 拟合结果 + 评价 | 评价部分含真实平面对比 |
| `results/laser_plane.json` | 仅拟合平面，可直接给测量程序 | 否 |

---

## 4. 目录与产物

```text
04_laser_plane_calib/
├─ plane_calib_lib.py       核心库：棋盘格渲染、条纹渲染、PnP、求交、SVD、验证、实验
├─ calibrate_plane_sim.py   主实验：15 块板、板数/覆盖/噪声/方向分析、图、报告
├─ test_plane_calib.py      11 个单元测试
├─ README.md
├─ requirements.txt
└─ out/
   ├─ data/                 15 组主数据：board/laser 图、measurements.json、truth.json
   │  ├─ calibration_points.npz
   │  ├─ ideal_measurements.json
   │  ├─ orientation.json
   │  └─ coverage/          4 组覆盖实验（fronto/one_axis/same_plane/close）
   ├─ validation/           10 个独立台阶：图、measurements.json、truth.json
   ├─ results/              calibration.json、laser_plane.json、各类 CSV、summary.json、report.txt
   ├─ figs/                 7 张图
   └─ integration_h31/      用拟合平面 + laser_measure.py 的示例输出
```

7 张图：

| 图 | 内容 |
| --- | --- |
| `fig01_geometry.png` | 三维标定点 + 拟合平面；右图是拟合平面内的覆盖轨迹（每种颜色一块板） |
| `fig02_residuals.png` | 点到平面残差直方图、每块板 PnP RMS、板面法向误差 |
| `fig03_board_count.png` | 板数 vs 法向误差、板数 vs 验证误差 |
| `fig04_coverage.png` | 5 种姿态覆盖的角误差和验证误差（标注条件数/退化） |
| `fig05_noise.png` | 角点/激光噪声 vs 法向误差、验证误差 |
| `fig06_validation.png` | 10/31/50 mm 验证：测量值 vs 真值、误差 |
| `fig07_orientation.png` | 近竖直条纹的 row/column 提取、有效中心数 |

---

## 5. 真实硬件怎么接

### 5.1 相机

沿用现有 `01_camera_calib`：

```powershell
cd ..
python calibrate.py --dir data/calib --square <实测格边长>
```

`calibrate_plane_sim.py` 默认读 `../out/calib.json`。真实采集时：

- 相机、分辨率、焦距、对焦状态固定；
- 棋盘格实测 19 mm 就传 `--square 19`；
- 有畸变时程序用 `cv2.undistortPointsIter` 去畸变，不需要先手动 `undistort` 整张图。

### 5.2 采集标定板

- 10~15 个位姿；至少覆盖：
  - 近/中/远三个距离；
  - 画面左/中/右、上/中/下位置；
  - 绕 X、绕 Y 两个方向的倾角（±15°~25°）；
  - 少量绕光轴的旋转。
- **不要**把所有板放在同一个 Z、同一个正对姿态——那样板面重合，激光点会退化成一条线。
- 每张图里激光线要完整落在棋盘格上，不要被裁掉。
- 棋盘格整块可见，四周留边；避免反光/过曝。

### 5.3 真实角点和条纹

- 角点：现有 `calib_lib.find_corners` / `cv2.findChessboardCornersSB`；
- 条纹：真实图像用 `laser_lib.extract_laser_centers`；
  - 条纹近似竖直：`axis="row"`；
  - 条纹近似水平：`axis="column"`；
  - 不确定时先给紧 ROI，再用 `axis="auto"`；**不要直接对全图用 auto**（本实验已复现该问题）。
- 组装 measurements：

```python
measurements = [{
    "index": i,
    "corners": detected_corners,       # (40,2)
    "laser_uv_row": laser_centers,     # (N,2)
}, ...]
```

- 调用：

```python
calib = pc.calibrate_plane(measurements, camera, pc.BoardSpec(square_mm=19.0))
calib.fit.to_plane().to_dict()   # 写入 laser_plane.json
```

### 5.4 真实验证

- 找 10 / 31 / 50 mm 的台阶或量块，**不要参与平面拟合**；
- 用 `laser_measure.py image` + 拟合平面测高度，和卡尺值比较；
- 验证位置要覆盖标定体，最好包含一个接近工作范围边缘的位置；
- 如果高度误差随位置线性变化，优先怀疑平面法向/相机内参，而不是提取算法。

### 5.5 本次实验给出的标定目标

- 平面法向误差：**0.01°~0.05°** 量级是本次模拟能达到的；真实系统受角点噪声、板平整度限制，先争取 **<0.2°**；
- 拟合残差：本次 0.12 mm；真实板不平、角点噪声更大，先争取 **<0.5 mm**；
- 独立台阶高度：本次平均 0.10 mm、最大 0.20 mm；真实系统先争取 **<0.5 mm**；
- 角点噪声比激光中心噪声更关键，优先把棋盘格拍清楚、板压平。

---

## 6. 单元测试

```powershell
python -m unittest test_plane_calib test_regressions -v
```

本机：`Ran 20 tests ... OK`。覆盖：

- 20 mm / 19 mm 棋盘格几何、物理边界；
- 已知平面 SVD 拟合、共线退化检测、平面 JSON 往返；
- 棋盘格渲染 + 角点检测 + PnP 位姿恢复；
- 理想投影标定（`<1e-3°` / `<1e-3 mm`）；
- 含噪声标定 + 独立台阶验证；
- 近竖直条纹 row/column 有效数与中心误差；
- 紧 ROI 下 auto 选 row；
- 标定输入不含真实平面/真实位姿，标定函数签名没有 `true_plane`。

---

## 7. 限制

- 棋盘格是理想平面，没有厚度、翘曲、打印墨迹扩散；真实板不平会直接进板面法向误差。
- 激光条纹用高斯截面，没有散斑、衍射、景深、饱和、二次反射。
- 没有模拟台阶阴影/遮挡；真实台阶边缘可能有一段无激光区。
- 验证台阶是合成的；真实工件的边缘、材质、反光会带来额外误差。
- 标定只拟合一个静态激光平面；没有考虑温度漂移、激光器功率变化、结构振动。
- 全图 `auto` 对细竖直条纹不可靠；这是现有启发式的使用边界，本模块没有修改基础库。
- 当前只支持一个激光平面、一个相机；多线/多相机需要扩展。

---

## 附：最短命令序列

```powershell
cd 'C:\Users\wangjianing\Desktop\光赛\01_camera_calib\04_laser_plane_calib'

# 1. 默认 20mm，跑全部实验 + 图 + 报告
python calibrate_plane_sim.py

# 2. 实物打印板实测 19mm
python calibrate_plane_sim.py --square 19

# 3. 单元测试
python -m unittest test_plane_calib test_regressions -v

# 4. 把拟合平面交给已有测量程序（示例）
python ..\03_laser_triangulation\laser_measure.py image `
    --image out\validation\images\h31_center.png `
    --calib ..\out\calib.json `
    --plane out\results\laser_plane.json `
    --axis row --roi 620 690 230 580 `
    --out out\integration_h31 --plot
```
