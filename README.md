# 光赛项目

**Git 备份规定：**每次修改已有代码后，先复验相关入口并记录结果，再提交并推送到 [项目仓库](https://github.com/llikkm/ziyonggunagsai)。用 `git ls-remote origin refs/heads/main` 核对远端提交号与 `git rev-parse HEAD` 一致。目录迁移前后分别备份，确认远端后再开始下一阶段。上传失败时保留本地提交并告知接手人。

编号按功能加入顺序：01 相机标定，02 棋盘位姿与高度，03 单帧激光三角测量，04 光平面标定，05 多帧扫描，06 新版标准件，07 高度积分，08 体积诊断。数据目录和归档实验名保留。

相机标定 + 线激光三角测量 + 激光平面标定 + 多帧扫描/高度图体积。

**项目主文档（模块地图、当前状态、命令、测试、接口、维护规则）：**

[01_camera_calib/PROJECT_README.md](01_camera_calib/PROJECT_README.md)

```text
01_camera_calib/
├─ README.md                  相机标定实操包
├─ PROJECT_README.md          项目总览（先读这个）
├─ 02_measurement_3d/            棋盘位姿、31mm 垫高、Open3D
├─ 03_laser_triangulation/       单帧激光三角测量仿真
├─ 04_laser_plane_calib/         激光平面标定仿真
├─ 05_volume_scan/               多帧扫描与高度图体积
├─ 06_standard_scan/             新版 A/B/C 建模与扫描
├─ 07_heightmap_volume/          归档点云的自动高度积分及评分
├─ 08_volume_diagnostics/        体积误差归因
└─ 09_volume_refinement/         固定 PLY 的轮廓与孔槽保边积分
```

新版赛题入口：[标准模型与移动线激光扫描](01_camera_calib/06_standard_scan/README.md) → [高度图体积估计](01_camera_calib/07_heightmap_volume/README.md)。2026-11-25 已完成六组归档点云的覆盖核查和体积评测；[六组结果](01_camera_calib/07_heightmap_volume/runs/baseline_20261125/report.md)。真实激光硬件尚未接入。

跨对话接手时先读[项目主文档第 11 节](01_camera_calib/PROJECT_README.md#handoff)，再看对应模块的运行记录。旧 `05_volume_scan` 中的 C 表示隐藏结构实验，新版赛题的 C 为平滑自由曲面。

[理想扫描体积误差诊断报告（2026-11-26）](01_camera_calib/08_volume_diagnostics/README.md)：固定基线、真值对照、网格敏感性及优化顺序。

[09 固定 PLY 优化对照](01_camera_calib/09_volume_refinement/README.md)：原六组点云保持不变，分别比较轮廓定位和保边估计后的体积。原始备份提交 `710fd05`，编号迁移备份 `255fa14`。
