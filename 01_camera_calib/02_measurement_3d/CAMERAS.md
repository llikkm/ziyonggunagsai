# 内置、USB 和手机摄像头的选择

在 PowerShell 中进入工程目录，使用现有全局 Python：

```powershell
cd 'C:\Users\wangjianing\Desktop\光赛\01_camera_calib'
```

## 1. 找到摄像头编号

```powershell
python 02_measurement_3d/capture3d.py --list-cameras
python 02_measurement_3d/capture3d.py --camera 0 --preview
python 02_measurement_3d/capture3d.py --camera 1 --preview
```

列表显示能读取画面的编号。逐个预览确认对应设备，按 `q` 关闭预览。下文以内置为 0、USB 为 1、OBS 为 2 举例，请按实际预览结果替换 `--camera` 后的数字。

`--camera` 负责选择设备；`--source builtin/usb/obs` 将设备用途记入采集记录。每次插拔设备后可以重新列出编号。

默认使用 Windows DirectShow 后端。如果某个设备打开失败，可用相同后端重新列出并预览：

```powershell
python 02_measurement_3d/capture3d.py --backend msmf --list-cameras
python 02_measurement_3d/capture3d.py --backend msmf --camera 0 --preview
```

后续采集也加上同一个 `--backend msmf`，保持编号含义一致。

## 2. 继续用笔记本内置摄像头

使用已有 `out/calib.json`。先将棋盘格背板平放在桌面，固定相机，再拍一张基准：

```powershell
python 02_measurement_3d/capture3d.py --source builtin --camera 0 --count 1 --out 02_measurement_3d/data/builtin_reference
```

出现绿色 `ready` 后按空格，照片保存为 `builtin_reference/frame_01.png`，窗口自动结束。

给同一块板加上等厚垫块，保持板面平行，运行：

```powershell
python 02_measurement_3d/capture3d.py --source builtin --camera 0 --reference 02_measurement_3d/data/builtin_reference/frame_01.png --count 10 --out 02_measurement_3d/data/builtin_height20
```

窗口显示重投影 RMS 和 `relative board angle`。保持相机及板子静止，每隔约 1 秒按一次空格，保存 10 张。超过 2° 时，检查板子是否架平、相机是否被碰动，调整到绿色 `ready` 后继续。

2° 指基准板面与当前板面的夹角。相机可以从斜上方观察板子，调整相机时以能清楚看到完整棋盘格为准。

计算高度，`--known-mm` 填卡尺测得的垫块厚度：

```powershell
python 02_measurement_3d/measure3d.py height --reference 02_measurement_3d/data/builtin_reference/frame_01.png --targets 02_measurement_3d/data/builtin_height20 --known-mm 20.00 --out 02_measurement_3d/results/builtin_height20.json
```

## 3. 后续接入 USB 摄像头

插入 USB 摄像头后，按第 1 节列出编号并预览。下面以编号 1 为例。

首次使用这只 USB 摄像头时，先拍 20 张内参标定图：

```powershell
python 02_measurement_3d/capture3d.py --source usb --camera 1 --calibration --count 20 --out 02_measurement_3d/data/usb_calib
```

这一组标定图要改变棋盘格的位置和倾角，覆盖画面不同区域；每换一个姿态，等清晰稳定后按空格。所有照片都包含完整棋盘格。相机输出保持为默认的 1280×720。

`--calibration` 模式显示九宫格。黄色框标出棋盘格中心当前所在区域，每个区域的 `N saved` 表示已经保存的照片数。建议每区拍至少 2 张，合计 18 张，再用另外 2 张补充倾斜姿态。计数按棋盘格中心归区，照片中仍要保留整块棋盘格。每次成功按空格保存后，区域计数加 1。

运行原工程的标定程序，结果独立保存到 `out/usb/`。`--square` 填棋盘格实测边长，以下按当前实测 19 mm 填写：

```powershell
python calibrate.py --dir 02_measurement_3d/data/usb_calib --cols 9 --rows 6 --square 19 --out out/usb
python 02_measurement_3d/refine_calibration.py --calib out/usb/calib.json --dir 02_measurement_3d/data/usb_calib --square 19 --out out/usb_refined/calib.json
```

随后固定 USB 摄像头，拍基准照和垫高照：

```powershell
python 02_measurement_3d/capture3d.py --source usb --camera 1 --calib out/usb_refined/calib.json --count 1 --out 02_measurement_3d/data/usb_reference
python 02_measurement_3d/capture3d.py --source usb --camera 1 --calib out/usb_refined/calib.json --reference 02_measurement_3d/data/usb_reference/frame_01.png --count 10 --out 02_measurement_3d/data/usb_height20
python 02_measurement_3d/measure3d.py --calib out/usb_refined/calib.json height --reference 02_measurement_3d/data/usb_reference/frame_01.png --targets 02_measurement_3d/data/usb_height20 --known-mm 20.00 --out 02_measurement_3d/results/usb_height20.json
```

每一组高度实验保持同一构图，拍摄动作与第 2 节相同。更换这只摄像头的镜头焦距或输出尺寸后，重新为该设置标定。

## 4. 手机通过 scrcpy 和 OBS 接入

电脑已有 scrcpy 3.3.4 和 OBS。Windows 下使用这条路径：

```text
手机后置摄像头 → scrcpy 窗口 → OBS 虚拟摄像头 → capture3d.py
```

### 打开手机摄像头画面

手机通过 USB 连接电脑，开启 USB 调试，在手机上同意电脑的调试授权。Android 12 及以上可以使用 scrcpy 的摄像头源。

在另一个 PowerShell 窗口运行：

```powershell
& 'C:\Users\wangjianing\Desktop\scrcpy-win64-v3.3.4\scrcpy.exe' --video-source=camera --camera-facing=back --camera-size=1280x720 --no-audio
```

如果手机提示不支持该尺寸，列出支持的尺寸：

```powershell
& 'C:\Users\wangjianing\Desktop\scrcpy-win64-v3.3.4\scrcpy.exe' --video-source=camera --camera-facing=back --list-camera-sizes
```

选择手机支持的尺寸重新运行，把输出画面通过 OBS 固定为下述 1280×720。后续用同一只手机镜头和这套固定画面设置完成标定及测量。

### 在 OBS 输出虚拟摄像头

1. 打开 OBS，新建一个场景，例如“手机测量”。
2. 在“设置 → 视频”中，将基础画布和输出分辨率都设为 1280×720。
3. 添加“窗口采集”源，选择 scrcpy 窗口，采集手机画面区域。将画面按原比例适配画布，排除桌面及窗口边框。
4. 点击“启动虚拟摄像头”。之后保持这个场景、源的缩放及裁剪设置固定。

回到工程目录，找到 OBS 的设备编号并预览：

```powershell
python 02_measurement_3d/capture3d.py --list-cameras
python 02_measurement_3d/capture3d.py --camera 2 --preview
```

下面的命令以 OBS 编号 2 为例。预览确认能看到手机摄像头画面后，再执行采集。

### 给手机这条视频链路单独标定

固定手机使用的后置镜头和对焦设置，采集 20 个不同位置、不同倾角的完整棋盘格画面：

采集窗口同样显示九宫格及分区保存计数，按每区至少 2 张分配这 20 张照片。

在不同区域主动改变板面倾角，补充约 15°～40° 的左右倾斜和上下倾斜姿态，保持所有角点可见。这样能增加标定照片对焦距的约束。

```powershell
python 02_measurement_3d/capture3d.py --source obs --camera 2 --calibration --count 20 --out 02_measurement_3d/data/phone_calib
python calibrate.py --dir 02_measurement_3d/data/phone_calib --cols 9 --rows 6 --square 19 --out out/phone
python 02_measurement_3d/refine_calibration.py --calib out/phone/calib.json --dir 02_measurement_3d/data/phone_calib --square 19 --out out/phone_refined/calib.json
```

第二条标定命令在单独文件中以 300 次迭代细化优化，并将实测格边长写入参数。这里按当前棋盘实测 19 mm 填写；更换棋盘时两条标定命令都填写新棋盘的实测值。`out/phone_refined/calib.json` 对应 OBS 实际输出的图像，包含该手机镜头及固定缩放后的内参。保留原来的内置相机参数 `out/calib.json`，可按设备随时切换。输出文件已经存在时，为新一轮另选文件名。

### 用手机拍基准和垫高照片

将手机固定在支架上，棋盘格背板平放在桌面，拍基准：

```powershell
python 02_measurement_3d/capture3d.py --source obs --camera 2 --calib out/phone_refined/calib.json --count 1 --out 02_measurement_3d/data/phone_reference
```

保持手机固定，给棋盘格加等厚垫块，拍 10 张：

```powershell
python 02_measurement_3d/capture3d.py --source obs --camera 2 --calib out/phone_refined/calib.json --reference 02_measurement_3d/data/phone_reference/frame_01.png --count 10 --out 02_measurement_3d/data/phone_height20
python 02_measurement_3d/measure3d.py --calib out/phone_refined/calib.json height --reference 02_measurement_3d/data/phone_reference/frame_01.png --targets 02_measurement_3d/data/phone_height20 --known-mm 31.00 --out 02_measurement_3d/results/phone_height31_refined.json
```

## 5. 新采集入口的文件规则

- 按空格或 `s` 保存，按 `q` 或 Esc 退出；保存满 `--count` 指定数量后自动结束。
- 保存的 PNG 使用原始视频帧，窗口中的角点标记和文字留在预览中。
- 每轮选择新的输出目录，照片编号为 `frame_01.png`、`frame_02.png` 等；程序拒绝覆盖已有采集目录。
- 同目录的 `capture.json` 保存来源标记、摄像头编号、实际视频后端及每张图的质量信息。

内参标定需要多姿态照片；高度实验在每个高度保持同一姿态，重复拍 10 张。原有 `capture_images.py` 和内置摄像头命令继续可用。
