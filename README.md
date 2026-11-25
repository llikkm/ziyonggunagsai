# 光赛项目

相机标定 + 线激光三角测量 + 激光平面标定 + 多帧扫描/高度图体积。

**项目主文档（模块地图、当前状态、命令、测试、接口、维护规则）：**

[camera_calib/PROJECT_README.md](camera_calib/PROJECT_README.md)

```text
camera_calib/
├─ README.md                  相机标定实操包
├─ PROJECT_README.md          项目总览（先读这个）
├─ measurement_3d/            棋盘位姿、31mm 垫高、Open3D
├─ laser_triangulation/       单帧激光三角测量仿真
├─ laser_plane_calib/         激光平面标定仿真
├─ standard_scan/             新版 A/B/C 建模与扫描
├─ heightmap_volume/          归档点云的自动高度积分及评分
└─ volume_scan/               多帧扫描与高度图体积
```

新版赛题入口：[标准模型与移动线激光扫描](camera_calib/standard_scan/README.md) → [高度图体积估计](camera_calib/heightmap_volume/README.md)。2026-11-25 已完成六组归档点云的覆盖核查和体积评测；[六组结果](camera_calib/heightmap_volume/runs/baseline_20261125/report.md)。真实激光硬件尚未接入。

跨对话接手时先读[项目主文档第 11 节](camera_calib/PROJECT_README.md#handoff)，再看对应模块的运行记录。旧 `volume_scan` 中的 C 表示隐藏结构实验，新版赛题的 C 为平滑自由曲面。

[理想扫描体积误差诊断报告（2026-11-26）](camera_calib/volume_diagnostics/README.md)：固定基线、真值对照、网格敏感性及优化顺序。
