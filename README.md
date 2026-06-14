# 相对坐标光谱重建系统

这是一个基于自制光谱仪图像的光谱重建项目。系统先从相机图像中提取 RGB 和灰度强度曲线，再通过相对坐标标定将像素位置映射到波长，最后使用一维神经网络重建 400-650 nm 范围内的光谱。

项目包含图像采集、相对坐标标定、训练数据构建、模型训练与评估，以及一个用于上传图片并查看预测结果的网页界面。

## 当前参数

| 项目 | 数值 |
| --- | --- |
| 波长范围 | 400-650 nm |
| 采样间隔 | 0.1 nm |
| 输出点数 | 2501 |
| 相对坐标锚点 | `y=286`、`y=537` |
| 标定锚点 | Hg 408.0 nm、HeNe 631.6 nm |
| 模型 | `SpectrumUNetTransformer1D` |
| 模型参数量 | 2,211,073 |
| 最佳验证损失 | 0.045239608498145144 |

当前使用的线性标定关系为：

```text
s = (y_local - 286.0) / (537.0 - 286.0)
wavelength_nm = 223.039214714 * s + 411.1831404
```

标定诊断中记录的最大绝对残差为 4.1923 nm。该误差需要结合光源、曝光、ROI 位置和装置稳定性判断，不能作为精密光谱仪标定结果使用。

## 项目结构

主要代码位于 [`relative_coordinate_spectrometer/`](relative_coordinate_spectrometer/)：

```text
relative_coordinate_spectrometer/
|-- systems/          四个功能系统
|   |-- capture/          相机采集和原始图像保存
|   |-- calibration/      相对坐标标定、峰值检查和曲线提取
|   |-- reconstruction/   数据构建、模型训练和评估
|   `-- web/              FastAPI 后端和 React 前端
|-- shared/           模型、损失函数和数据读取代码
|-- configs/          标定、数据生成和训练配置
|-- data/             原始测量数据及中间曲线
|-- results/          标定结果、训练记录和模型权重
`-- verification/     项目完整性与复现检查
```

`systems/capture/` 只负责采集图像和保存元数据；标定、数据构建和模型计算分别在其他系统目录中完成。

## 已包含的数据

- 12 张原始标定和光源图像
- 9 份参考光谱仪测量文本
- 12 份相对坐标四通道曲线
- 标定锚点、残差、诊断图和报告
- 最佳模型权重及完整训练记录

训练阶段生成的 `x.npy`、`y.npy` 和数据划分文件体积较大，没有放入仓库。对应生成脚本、随机种子和配置均已保留。

## 安装

模型权重使用 Git LFS 管理：

```powershell
git lfs install
git clone https://github.com/ruihaoma/spectrometer-ai.git
cd spectrometer-ai\relative_coordinate_spectrometer
git lfs pull

py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 快速检查

检查仓库全部受控文件是否存在且非空，并验证文本编码、结构化文件、图片、原始数据数量、标定关系、曲线重建误差和模型权重：

```powershell
python verification\verify_project.py
```

再运行一次小规模数据链路测试：

```powershell
python verification\verify_project.py --full-smoke
```

该测试临时生成 6 个实测样本、20 个合成样本和 26 个混合样本，不会把训练数据写入仓库。

## 主要流程

重新计算相对坐标标定：

```powershell
python systems\calibration\relative_spectral_coordinate_calibration_diagnostic.py
```

重新提取图像曲线：

```powershell
python systems\calibration\build_relative_calibration_profiles_v1.py
```

生成训练数据：

```powershell
python systems\reconstruction\build_relative_calib_paired_dataset_v1.py --overwrite
python systems\reconstruction\generate_relative_calib_synthetic_dataset_v1.py --sample-count 80000 --seed 42 --allow-large --overwrite
python systems\reconstruction\build_relative_calib_mixed_dataset_v1.py --overwrite
```

训练模型：

```powershell
python systems\reconstruction\train_spectrum_unet_transformer_1d.py --config configs\train\relative_calib_mixed_v1_80k_train.yaml
```

启动网页应用：

```powershell
.\systems\web\start_system.bat
```

浏览器访问 `http://127.0.0.1:5173/`。停止服务时运行：

```powershell
.\systems\web\stop_system.ps1
```

需要使用其他端口时：

```powershell
.\systems\web\start_system.bat --backend-port 8011 --frontend-port 5174
```

自定义端口启动后，可用 `.\systems\web\stop_system.ps1 -Ports 5174,8011` 停止服务。

更完整的参数、评估方法和复现说明见 [`relative_coordinate_spectrometer/README.md`](relative_coordinate_spectrometer/README.md)。

## 说明

- 当前标定属于实验装置的相对坐标标定，不等同于经过计量校准的绝对波长标定。
- 不同相机位置、曝光参数、裁剪区域和光路状态会影响输入曲线。
- 原训练环境没有完整记录 GPU、CUDA 和驱动版本，因此不同设备重新训练时不保证逐位一致。
- 最终模型文件的 SHA256 为 `dda1be29ae42f424d4ef000138c7e9de83d10cfbd4d968cef7304ea8f8b44ae0`。
