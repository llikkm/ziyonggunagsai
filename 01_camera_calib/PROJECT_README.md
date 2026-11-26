# 光赛 / 01_camera_calib 项目总览

最后更新：2026-11-26

这是整个工程的**项目级入口**。以后新增模块、改接口、重跑实验或交接给另一个对话/智能体时，先读本文件，再按模块链接进入细节；改完请回到这里更新状态、命令和测试矩阵。文末有维护规则。

> 当前状态：新版 A/B/C 的六组归档仿真已完成扫描范围核查和自动体积估计；两组竞赛 P 均为 0.01。旧扫描及历史实验保留。真实激光硬件尚未接入。

---

## 1. 项目目标与当前进度

目标是用单目相机 + 线激光做三维测量：

1. 用棋盘格标定相机内参和畸变；
2. 用多姿态棋盘格 + 激光线标定激光平面；
3. 从激光条纹像素反算相机系三维点；
4. 让相机/激光器沿扫描轴移动，把多帧点云统一到世界系；
5. 估计基准面、建高度图、积分体积，并定量知道哪些结构测不了。

| 阶段 | 状态 | 当前证据 |
| --- | --- | --- |
| 相机内参/畸变 | 已完成（真实相机） | `out/calib.json`，RMS 1.703 px，1280×720，有畸变 |
| 棋盘格位姿/高度 | 已完成（真实照片 + 合成） | `02_measurement_3d`，31 mm 垫高实验，Open3D 查看 |
| 单帧激光三角测量 | 已完成（仿真） | 31 mm 台阶 7/7 检查，无噪声精确，19 个测试 |
| 激光平面标定 | 已完成（仿真，20 mm 标称板） | 15 块板，法向误差 0.009710°，独立验证 0.0989 mm |
| 多帧扫描/体积 | 已完成（仿真） | A 400.005、B 396.178、C 顶面 400.005（真实 368），21 个测试 |
| 真实激光硬件 | 未开始 | 等激光器/支架/同步采集；接口已留好 |

---

## 2. 模块地图

[10_generalization](10_generalization/README.md) 负责参数化测试件、冻结校验及独立评测；保留集结果不反馈到模块 09。

固定点云候选算法：[09_volume_refinement](09_volume_refinement/README.md)，使用编号 07 的估计结果及原 PLY，保留原有基线入口。

独立评价：[08_volume_diagnostics](08_volume_diagnostics/README.md)，只读三组理想扫描，正式估计器不依赖此模块。

新版体积模块：[07_heightmap_volume](07_heightmap_volume/README.md)，正式测量端仅读取点云与采样参数。

新版赛题模块：[06_standard_scan](06_standard_scan/README.md)，含标准件、渲染端、独立重放端及几何测试。

| 目录/文件 | 作用 | 入口 | 主要产物 | 测试 |
| --- | --- | --- | --- | --- |
| `README.md`（本目录） | 相机标定实操包（原始工程） | `calibrate.py` 等 | `out/calib.json`、报告、测量 CSV | 脚本自检 |
| `calib_lib.py` | 共用检测/标定/单应/去畸变 | 被各脚本导入 | — | 间接覆盖 |
| `make_board.py` | 生成 9×6、20 mm 矢量棋盘 PDF | `python make_board.py` | `board/chessboard_9x6_20mm.pdf` | — |
| `capture_images.py` | 摄像头采集合格标定姿态 | `python capture_images.py` | `data/calib`、`data/val` | — |
| `calibrate.py` | 内参/畸变标定与质量报告 | `python calibrate.py --square 19` | `out/calib.json` | — |
| `measure.py` | 留出集一致性、平面物体尺度验证 | `python measure.py ...` | `out/measure_*.csv` | — |
| `selftest_synthetic.py` | 无相机时验证整条标定链路 | `python selftest_synthetic.py` | 终端/报告 | 自检脚本 |
| `diagnose.py` | 单张图检不出棋盘格的原因 | `python diagnose.py --image ...` | 诊断输出 | — |
| `02_measurement_3d/` | 棋盘位姿、31 mm 垫高、Open3D 点云、摄像头选择 | `measure3d.py`、`capture3d.py`、`refine_calibration.py` | `results/`、`out/phone_refined/` | 4+4+1 |
| `03_laser_triangulation/` | 31 mm 台阶激光三角测量仿真 + 真实图像/网页摄像头接口 | `simulate_step.py`、`laser_measure.py` | `out/measures/`、报告、7 张图 | 19 |
| `04_laser_plane_calib/` | 多姿态棋盘格标定激光平面 + 独立台阶验证 | `calibrate_plane_sim.py` | `out/results/`、`laser_plane.json` | 11+9 |
| `05_volume_scan/` | 多帧扫描、可见性/阴影/FOV、点云、基准面、高度图体积 | `run_volume_scan.py` | `out/pointclouds/`、`heightmaps/`、报告、8 张图 | 20 |
| `PROJECT_README.md` | 本文件：项目地图/状态/接口/测试/维护 | — | — | — |

各模块的详细说明：

- [相机标定实操包](README.md)
- [02_measurement_3d 操作指南](02_measurement_3d/README.md)、[摄像头选择](02_measurement_3d/CAMERAS.md)、[验证记录](02_measurement_3d/VERIFICATION.md)
- [激光三角测量仿真](03_laser_triangulation/README.md)
- [激光平面标定仿真](04_laser_plane_calib/README.md)
- [多帧扫描/体积仿真](05_volume_scan/README.md)

---

## 3. 坐标、单位与数据约定

- 单位：长度 mm，体积 cm³；角度 °；像素 px。
- 相机系：X 右、Y 下、Z 前。
- 世界系（本工程仿真）：X 向右（物体长 100），Y 向里（宽 80），Z 向上（高 50）。
- 激光平面：相机系 `n·X + d = 0`，`n` 单位法向；`LaserPlane` 的 `d` 与 `n` 同尺度。
- 机构：`P_w = R_i P_c + t_i`；`R_i` 相机→世界旋转，`t_i` 相机中心世界坐标。
- 像素→点：`ll.reconstruct(uv, camera, plane)`；内部用 `undistortPointsIter`（有畸变时）和射线-平面求交。
- 数据隔离：**反算/拟合函数不接收真实平面、真实点云或真值位姿**；真值只用于渲染、生成候选和最后评价。各模块 README 有具体签名。
- **19 mm / 20 mm**：实物打印板实测边长 **19 mm**（打印缩放），设计标称 **20 mm**。所有仿真按 20 mm；真实标定/平面标定/位姿必须传 `--square 19`（或卡尺实测值），否则米制尺度整体偏约 5%。相机内参 K 本身不随方格尺寸变，但棋盘位姿、激光平面位置和高度尺度会变。

---

## 4. 端到端数据流

```text
make_board.py
   └─ board/chessboard_9x6_20mm.pdf
        └─ capture_images.py / capture3d.py
             └─ data/calib, data/val
                  └─ calibrate.py --square 19
                       └─ out/calib.json  (K, dist, image_size, rms)

棋盘格 + 激光线（多姿态）
   └─ 04_laser_plane_calib / 真实平面标定
        └─ laser_plane.json  (n, d, rms, ...)

条纹像素 uv + K/dist + laser_plane + 机构位姿 R_i,t_i
   └─ laser_lib.reconstruct / 05_volume_scan.reconstruct_scan
        └─ 相机系点 → P_w=R_i P_c+t_i → 世界点云
             └─ 基准面估计 → 高度图 → 体积/覆盖率
```

`05_volume_scan` 的仿真链路：

```text
真实物体四边形 + 理想激光平面
  → 平面/四边形交线
  → 激光入射面 + 相机可见面 + 相机遮挡 + 激光阴影 + FOV
  → 分段高斯条纹渲染
  → extract_laser_centers(axis="row")
  → uv（只把像素交给反算）
  → reconstruct_scan
  → 世界点云
  → RANSAC 基准面
  → 高度图（格内 max，D 已知，窄缝有限填充）
  → V=Σh·du·dv，coverage
```

---

## 5. 已验证结果（截至 2026-11-26）

[冻结泛化实验](10_generalization/runs/frozen_20261126/REPORT.md) 的分组统计优先用于判断新目标表现。下文六组标准件数据保留为历史开发基线。

09 固定六份 PLY 的候选算法：未舍入 P 理想组 0.012540 → 0.001507，扰动组 0.008014 → 0.001523（此处显示六位）；完整精度与逐件覆盖率见 09 说明。

三件理想扫描 0.5 mm 网格逐格复现；边界和缺口联合 oracle 的 P 为 0.003881，原始值为 0.012540。[误差诊断及优先级](08_volume_diagnostics/README.md)。

新版自动体积：[六组结果及扫描行程](07_heightmap_volume/runs/baseline_20261125/report.md)。原始 P 理想组 0.012540、扰动组 0.008014；直接命中和局部估计分别统计。

新版标准件：三件各扫描两次，每次 63 帧；6 组点云从独立测量目录逐点重放一致。[运行摘要](06_standard_scan/runs/baseline_20261118/summary.json)；下文 5.4 保留旧模型的体积结果。

### 5.1 相机标定

- 文件：`out/calib.json`
- 图像：1280×720；棋盘 9×6 方格，8×5 内角点；标称 20 mm。
- 结果：`fx=989.7676, fy=991.6615, cx=647.6163, cy=405.4281`；畸变 `[0.26633, -2.00127, 0.01107, -0.02728, 2.44187]`；RMS **1.703 px**。
- 注意：实物板实测 19 mm；真实位姿/平面/高度用 `--square 19`。详见 [相机标定 README](README.md) 和 [02_measurement_3d](02_measurement_3d/README.md)。

### 5.2 单帧激光三角测量

- 31 mm 台阶：无噪声精确；0.5 px 噪声约 `-0.072 ± 0.190 mm`；CLI 干净数据 `31.0298 mm`，困难 ROI `31.0339 mm`。
- 7/7 检查，19 个单元测试。
- 报告/图：`03_laser_triangulation/out/`。详见 [03_laser_triangulation README](03_laser_triangulation/README.md)。

### 5.3 激光平面标定（仿真）

- 15 块板，20 mm 标称；拟合 `n=[0.98061111, 0.00006791, 0.19596387]`，`d=-97.97481276`。
- 法向角误差 **0.009710°**；d 误差 **+0.083255 mm**；点到平面 RMS **0.118840 mm**，max 0.398134 mm；条件数 1.189。
- 独立 10/31/50 mm 验证：平均 |误差| **0.0989 mm**，最大 **0.2011 mm**，10/10 有效；11/11 检查。
- 平面文件：`04_laser_plane_calib/out/results/laser_plane.json`；把它的 `n,d` 喂给 `laser_measure.py` 得到 `31.0467 mm`（与独立验证 30.8538 mm 的差在验证噪声量级内）。
- 20 个测试（`test_plane_calib` 11 + `test_regressions` 9）。详见 [04_laser_plane_calib README](04_laser_plane_calib/README.md)。

### 5.4 多帧扫描 / 高度图体积（仿真）

相机：`out/calib.json` 的真实内参/畸变；111 帧，X=-55..55 mm，步长 1 mm。

| 物体 | 真实 | 基线测量 | 误差 | 覆盖 | 关键现象 |
| --- | ---: | ---: | ---: | ---: | --- |
| A 长方体 | 400.0 | 400.005 | +0.005 | 49.3% | 平面顶面，高度图适用 |
| B 顶部盲槽 | 396.0 | 396.178 | +0.178 | 48.8% | 槽底可见 80%，边缘遮挡残余 |
| C 侧向贯穿孔 | 368.0 | 400.005 | +32.005 | 49.3% | 顶部看不到侧孔，高度图必错 |

- 解析对照：A/B 400/396 精确；C 400 vs 368（差 32，方法上限）。
- 误差敏感性（A）：pixel 1 px `+8.29 cm³`；motion 1 mm `+9.11 cm³`；plane n_z+0.005 `-10.74 cm³`；omission 25% `-0.81 cm³`；FOV |x|≤35 `-105.80 cm³`（覆盖 41%）。
- B 低源：槽底仍 80%，但地面基准可见率约 42%→4%；shadow 用基线基准面，体积 396.19。
- 9/9 检查，21 个测试。详见 [05_volume_scan README](05_volume_scan/README.md)，完整表格 `05_volume_scan/out/results/report.txt`。

### 5.5 02_measurement_3d / 手机实验

- 手机旧内参 300 次迭代细化：`out/phone_refined/calib.json`；19 mm 实测方格用于手机实验。
- 31 mm 垫高、Open3D 点云、棋盘位姿检查；4+4+1 个测试。
- 细节见 [02_measurement_3d README](02_measurement_3d/README.md) 和 [VERIFICATION.md](02_measurement_3d/VERIFICATION.md)。

---

## 6. 命令索引

泛化测试进入 `10_generalization`，依次使用 `benchmark.py prepare/development/holdout`，再用 `report.py` 生成报告，完整命令见模块 README。

固定 PLY 优化：进入 `09_volume_refinement`，运行 `python -B compare.py`，默认新建目录。

误差诊断：进入 `08_volume_diagnostics`，运行 `python diagnose.py`，输出新目录。

新版体积：进入 `07_heightmap_volume`，运行 `python run_volume.py --pitch 0.5`；每次输出到新目录。

新版扫描：进入 `06_standard_scan`，运行 `python simulate.py`；历史基线复验用 `python verify_run.py runs/baseline_20261118`。

所有命令在 PowerShell、全局 Python 3.12 下执行（不要新建虚拟环境）。

### 相机标定

```powershell
cd 'C:\Users\wangjianing\Desktop\光赛\01_camera_calib'
python make_board.py
python capture_images.py
python calibrate.py --square 19          # 实物板实测 19mm；仿真/设计值用 20
python measure.py --help
python selftest_synthetic.py
python diagnose.py --help
```

### 02_measurement_3d

```powershell
cd 'C:\Users\wangjianing\Desktop\光赛\01_camera_calib'
python 02_measurement_3d/measure3d.py --help
python 02_measurement_3d/capture3d.py --list-cameras
python 02_measurement_3d/refine_calibration.py --help
```

### 激光三角测量

```powershell
cd 'C:\Users\wangjianing\Desktop\光赛\01_camera_calib\03_laser_triangulation'
python simulate_step.py
python simulate_step.py --show
python laser_measure.py --help          # image / live / fit-plane / make-plane
python -m unittest test_laser_lib -v    # 19
```

### 激光平面标定

```powershell
cd 'C:\Users\wangjianing\Desktop\光赛\01_camera_calib\04_laser_plane_calib'
python calibrate_plane_sim.py                 # 15 块板，20mm 标称
python calibrate_plane_sim.py --square 19     # 实物板
python calibrate_plane_sim.py --synthetic-camera
python -m unittest test_plane_calib test_regressions -v   # 20
```

### 多帧扫描/体积

```powershell
cd 'C:\Users\wangjianing\Desktop\光赛\01_camera_calib\05_volume_scan'
python run_volume_scan.py
python run_volume_scan.py --show
python run_volume_scan.py --synthetic-camera
python run_volume_scan.py --scan-range 35
python -m unittest test_volume_scan test_scan_repairs -v   # 20
```

---

## 7. 测试矩阵（92 个，全部通过）

| 模块 | 测试文件 | 数量 | 命令（在对应目录） |
| --- | --- | ---: | --- |
| 新版模型与可见性 | `test_standard.py` | 5 | `python -m unittest test_standard -v` |
| 新版高度积分 | `test_heightmap.py` | 7 | `python -m unittest test_heightmap -v` |
| 扫描修复 | `test_scan_repairs.py` | 9 | 和下行一起运行 |
| 多帧扫描/体积 | `test_volume_scan.py` | 12 | `python -m unittest test_volume_scan test_scan_repairs -v` |
| 激光三角测量 | `test_laser_lib.py` | 19 | `python -m unittest test_laser_lib -v` |
| 激光平面标定 | `test_plane_calib.py` | 11 | `python -m unittest test_plane_calib -v` |
| 激光平面回归 | `test_regressions.py` | 9 | `python -m unittest test_regressions -v` |
| 02_measurement_3d | `test_measure3d.py` | 4 | `python -m unittest test_measure3d -v` |
| 摄像头采集 | `test_capture3d.py` | 4 | `python -m unittest test_capture3d -v` |
| 标定细化 | `test_refine_calibration.py` | 1 | `python -m unittest test_refine_calibration -v` |
| 体积误差诊断 | `test_diagnostics.py` | 5 | `python -m unittest test_diagnostics -v` |
| 固定 PLY 优化 | `test_refined.py` | 3 | `python -B -m unittest test_refined -v` |
| 冻结和独立模型 | `test_protocol.py` | 3 | `python -B -m unittest test_protocol -v` |
| **合计** | | **92** | |

2026-11-25 全量回归：高度积分 7、新版扫描 5、旧扫描 21、三角测量 19、平面标定 20、实测工具 9，全部通过。独立重放和 `selftest_synthetic.py` 不计入 unittest 数量。

一键式（逐目录）：

```powershell
cd 'C:\Users\wangjianing\Desktop\光赛\01_camera_calib'
Push-Location 05_volume_scan;        python -m unittest test_volume_scan test_scan_repairs -v;        Pop-Location
Push-Location 06_standard_scan;      python -m unittest test_standard -v; Pop-Location
Push-Location 07_heightmap_volume;   python -m unittest test_heightmap -v; Pop-Location
Push-Location 08_volume_diagnostics; python -m unittest test_diagnostics -v; Pop-Location
Push-Location 09_volume_refinement;  python -B -m unittest test_refined -v; Pop-Location
Push-Location 10_generalization;     python -B -m unittest test_protocol -v; Pop-Location
Push-Location 03_laser_triangulation; python -m unittest test_laser_lib -v;         Pop-Location
Push-Location 04_laser_plane_calib;  python -m unittest test_plane_calib test_regressions -v; Pop-Location
Push-Location 02_measurement_3d;     python -m unittest test_measure3d test_capture3d test_refine_calibration -v; Pop-Location
```

---

## 8. 文件格式与接口

### 8.1 `out/calib.json`

相机标定结果；`calib_lib.load_calib()` / `laser_lib.CameraModel.from_calib_json()` 读取。核心字段：`image_size`（宽,高）、`K`（3×3）、`dist`（畸变）、`rms`。不同脚本另有 `pattern`、`square_mm` 等元数据；以实际 JSON 为准。

### 8.2 `04_laser_plane_calib/out/results/laser_plane.json`

激光平面 `n,d`（相机系）及拟合质量。`laser_measure.py --plane ...` 和 `laser_lib.LaserPlane.from_dict()` 可直接读。真实平面临界：板方格 19 mm、相机 K 正确、平面文件与当前相机一致。

### 8.3 点云/高度图/实验记录

- `05_volume_scan/out/pointclouds/*.npz`：`points_mm`（N×3，世界系）；`source="RECONSTRUCTED_POINTS_NO_TRUTH"`；基线另有 `.ply`。
- `05_volume_scan/out/heightmaps/*.npz`：`height`（ny×nx）、`u_edges/v_edges`、`observed`、`coverage`、`volume_mm3`、`base_normal/base_d`。
- `05_volume_scan/out/results/report.txt`：完整中文报告；`experiments.csv`：全部场景行；`summary.json`：机读汇总；`data/scan_config.json`：相机/扫描/物体/数据隔离。
- `03_laser_triangulation/out/`：仿真图、报告、测量 JSON；`04_laser_plane_calib/out/results/`：标定/验证 CSV、JSON、图。

### 8.4 关键接口

```python
# 单帧
ll.reconstruct(uv, camera, plane) -> points_camera, valid

# 多帧
scan = sc.simulate_scan(obj, setup, ...)          # 只产生 uv/pose（仿真）
pc   = sc.reconstruct_scan(scan, plane_c, ...)    # 只接收 uv/K/dist/plane/pose
base = sc.estimate_base_plane(pc.points_world)
hm   = sc.height_map_volume(pc.points_world, base, footprint=known_D)
```

真实硬件替换点：`ScanSetup.camera`、`ScanSetup.laser_plane_c`、`ScanSetup.pose(i)`、`ScanFrame.uv`。

---

## 9. 真实硬件接入顺序

1. **打印/量板**：100% 实际大小；游标卡尺量 180 mm 标称线，算实际格边长（本轮 19 mm）；全程用这个值。
2. **相机标定**：`python calibrate.py --square 19`；确认 RMS、畸变、图像尺寸；保留 `out/calib.json`。
3. **相机+激光器刚性安装**：激光平面与相机夹角尽量大（避免近平行退化）；先拍 10~15 个不同姿态的棋盘格 + 激光线；用 `04_laser_plane_calib` 的拟合接口/真实流程得到 `laser_plane.json`；检查法向、d、残差、条件数。
4. **机构**：记录每帧 `R_i,t_i`（编码器/平台读数）；先空程回零、重复定位测试；`R_i` 按实际安装旋转填。
5. **采集**：固定曝光/增益，避免过曝；条纹尽量竖直；用 `extract_laser_centers(axis="row")`；对多表面/多回波要升级多峰提取。
6. **反算/体积**：用真实 `uv/K/dist/plane/pose` 走 `reconstruct_scan`；基准面用静止的平面标定/最低点 RANSAC；高度图 D 用已知 CAD 或分割；同时看体积和覆盖率。
7. **验证**：先用 31 mm 台阶/已知长方体；再上 B 盲槽/C 侧孔类对象；C 类隐藏结构必须多视角或封闭网格。
8. **安全**：激光 class、护目镜、反射面、相机曝光/散热；运动平台限位。

---

## 10. 已知限制与下一步

理想扫描证据优先支持联合检查轮廓与边缘缺口；扰动组的诊断收益待验证。

新版模块下一步优先验证轮廓外扩和 B 孔槽边缘的缺测；高度图的有效面积包含受限局部估计。下列隐藏空腔等案例属于旧 `05_volume_scan` 压力测试。

1. **C 类隐藏空腔/侧孔**：高度图必错（本实验 +32 cm³）；下一步多视角/侧面扫描、占据栅格/TSDF、封闭网格。
2. **悬空/多层**：高度图每 (x,y) 仅一个高度；需要多层表面/多回波。
3. **单峰行提取**：地面基准和物体顶面同排不同 u 时会二选一；基线覆盖率 57% 含此因素；下一步多峰/多回波提取。
4. **FOV/扫描范围**：|x|≤35 时两端缺失，体积 -106 cm³；必须同时报告覆盖率。
5. **基准面**：低源会遮地面基准；shadow 用基线基准面；真实设备应保留基准条纹或单独基准标定。
6. **误差耦合**：本仿真把 pixel/motion/plane 分开注入；真实硬件会耦合（图像、机构、平面、温度）。下一步联合仿真/实测。
7. **理想化**：高斯条纹、无散斑/景深/过曝/二次反射/卷帘；真实激光线宽、功率、表面反射率会改变提取偏差。
8. **已知 D**：本实验用已知物体足迹；未知物体需要分割/凸包，边界缺失直接影响体积。
9. **19 mm 板**：所有仿真 20 mm；真实标定 19 mm；不要混用。`02_measurement_3d` 手机实验已改用细化参数/19 mm。
10. **真实激光平面**：`05_volume_scan` 目前用理想平面；接真实数据时换成 `laser_plane.json`，并重新验证法向/d/残差。
11. **实时性**：仿真 111 帧约 50 s 主要花在渲染；真实采集的瓶颈会在曝光、传输、提取和运动同步，不在 NumPy 几何。

---

<a id="handoff"></a>

## 11. 维护规则（给后续对话/智能体）

修改已有代码后必须完成一次 Git 远端备份：验证、提交、推送，核对 origin/main 的提交号。目录迁移单独提交并先上传，算法变更另行提交。工作目录内的原始采集文件和 PLY 归档随工程保留。

1. **本文件是项目级唯一入口**；模块细节写在各模块 README，不要在本文件复制大段实现。
2. 每次新增/修改模块，更新：最后更新日期、第 2 节模块地图、第 5 节结果、第 6 节命令、第 7 节测试数量、第 10 节限制。
3. **不要修改已通过验证的核心算法**（`laser_lib.py`、`simulate_step.py`、`laser_measure.py`、`plane_calib_lib.py`、`calibrate_plane_sim.py`）而不跑对应测试；新功能优先新建模块并 import。
4. **保持数据隔离**：反算/拟合函数不接收真实平面、真实点云、真值位姿；真值只用于渲染/评价。新增接口先检查签名和调用链。
5. **单位/坐标系不要隐式改**：mm、cm³、相机系 X右/Y下/Z前、世界 Z 上；改坐标约定必须同时改所有模块和测试。
6. **19 mm/20 mm 写清楚**：仿真 20 mm，实物 19 mm；真实标定传 19。
7. 按第 7 节当前测试矩阵跑完全量回归，再更新“全过”；局部验证写明模块和实际数量。
8. 产物路径尽量稳定（`out/...`）；报告/CSV/JSON 同源生成，避免手改数字。
9. 提交/交接时优先给出：本文件 + 对应模块 README + `out/results/report.txt` + 失败/限制。
10. 新对话先核对本文件日期、模块 README 和最近一次运行的摘要，再动代码。记忆摘要仅用于定位；如与代码或产物冲突，在变更记录注明复核命令和结果。赛题变动记录发文日期、附件页码及受影响模块。
11. 原始照片、既有标定及已交付实验均保留。每轮新实验创建独立运行目录；随机输入记下种子和扰动定义。变更测量接口时用仅含测量输入的目录执行重放，并报告核对范围。
12. 每次交接留一条带日期记录：本轮改了什么、如何复验、结果目录、未完成项和下一步。实物结果与仿真分开登记；模块 README 记录实现细节，根 README 负责导航。

---

## 12. 变更记录

- **2026-11-26 独立泛化**：冻结 09 和上游测量源码，模块 10 生成并锁定开发、保留两个清单，固定三种姿态和两种机构模式。逐例结果及失败信息在 `10_generalization/runs/frozen_20261126`。

- **2026-11-26 编号和固定点云优化**：全工程初始备份 `710fd05`，01～08 编号迁移并通过 86 项回归的备份 `255fa14`。模块 09 的结果在其 `runs/fixed_ply_20261126`。候选算法保留 C 的高度偏差供独立实验验证。

- **2026-11-26 体积归因**：新增独立模块 `08_volume_diagnostics`，结果在 `runs/ideal_20261126`（相对模块目录）。原 81 项及新增 5 项测试通过。原算法、原点云和既有测试均未修改。

- **2026-11-25 高度积分**：独立核查 378 帧后沿用原归档，新增 `07_heightmap_volume`。在模块目录运行 `python run_volume.py` 可复现，已交付产物在 `07_heightmap_volume/runs/baseline_20261125`。保留原 74 项测试，增加 7 项。

- **2026-11-18 新版标准件**：新增 `06_standard_scan`，运行 `python verify_run.py runs/baseline_20261118` 可复核六组扫描。已检查孔底、槽底和自由曲面内部区域。待接入自动分割及体积评测。

- **2026-11-17 扫描修复**：修复基准面低分位截断偏差；区分图像 FOV 与扫描行程截断；保存全部图像、逐帧像素/位姿及误差场景实际反算输入；增加 9 个回归测试，原有 60 个测试保留。修复后结果以本次新报告为准，文中的早期数值为历史记录。


- **2026-11-17**：新增 `05_volume_scan`（多帧扫描、可见性/阴影/FOV、点云、基准面、高度图体积、误差扫描、8 张图、12 个测试）；新增本 `PROJECT_README.md`；补齐 `04_laser_plane_calib` 的 `test_regressions` 9 个测试，项目测试合计 60；确认 A 400.04、B 396.19、C 400.04（真实 368）；确认低源遮的是地面基准而非 B 槽底；确认实物板 19 mm 与仿真 20 mm 的尺度注意事项。
- 更早：相机标定（RMS 1.703）、02_measurement_3d/手机细化、激光三角测量 31 mm（7/7、19 测试）、激光平面标定 15 板（0.009710°、0.0989 mm、20 测试）。
