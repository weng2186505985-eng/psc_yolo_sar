# PSC-YOLOv8 SAR 舰船目标检测系统

本项目实现了一种在 SAR 遥感图像中检测舰船目标的系统，核心是将物理散射中心（PSC）模型嵌入到 YOLOv8 中。

## 功能特性
- **数据处理**: 提供 HRSID 数据集的加载支持，支持近岸/离岸场景筛选。
- **物理引导的深度学习**: 通过实现专用的 `PSCActivation`，在模型中嵌入散射中心响应的物理特征。
- **多种基线对比**:
  - `PSC-YOLOv8`
  - `Plain YOLOv8`
  - `CA-CFAR` (传统雷达目标检测基准)
- **复合评估指标**:
  - `mAP@0.5` 与 `mAP@0.5:0.95` (利用 pycocotools 确保标准化)
  - `Pd` (检测概率)，`Pfa` (虚警概率) 与 `ROC`。

## 安装与运行

### 1. 环境准备
```bash
conda create -n psc_yolo python=3.10
conda activate psc_yolo
pip install -r requirements.txt
```

### 2. 数据准备
下载 [HRSID Dataset] 并在 `HRSID_jpg` 中解压。
你可以通过设置环境变量 `HRSID_DATA_ROOT` 修改数据路径，否则将使用默认相对路径。

### 3. 开始训练
运行以下命令启动训练流程（可通过 base_config.py 调整参数）：
```bash
python scripts/train.py
```

### 4. 评估代码
训练完成后对多场景评估：
```bash
python scripts/evaluate.py
```

### 5. 基线对比
运行不同基线模型对比检测性能：
```bash
python scripts/run_baselines.py
```
