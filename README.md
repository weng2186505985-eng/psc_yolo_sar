# PSC-YOLOv8 SAR 舰船目标检测系统

本项目实现了一种在 SAR 遥感图像中检测舰船目标的系统，核心是将物理散射中心（PSC）模型嵌入到 YOLOv8 中。

## 功能特性
- **数据处理**: 提供 HRSID 数据集的自动兼容处理，通过 `convert_coco_to_yolo` 模块完全支持 Ultralytics 最新引擎，并且支持近岸/离岸场景的 DataLoader 高级切分过滤。
- **物理引导的深度学习**: 通过实现专用的 `PSCActivation`，深入 YOLOv8 的主干网络 `C2f` 与 `SPPF` 中，完美替换 SiLU，在模型中嵌入散射中心响应的物理特征。
- **多种基线对比**: 提供开箱即用的比较测试环境，涵盖：
  - `PSC-YOLOv8` (本文全量创新方案)
  - `Plain YOLOv8` (原版空白网络，公平对照)
  - `CA-CFAR` (采用 boolean mask 卷积加速的传统雷达恒虚警率检测器，极速版)
- **多维度复合评估指标**:
  - `mAP@0.5` 与 `mAP@0.5:0.95` (利用 pycocotools 绝对标准还原)
  - `Pd` (雷达系统目标级检测概率)与 `Pfa` (雷达系统虚警概率)。

---

## 1. 安装环境

建议使用 Python 自带的 `venv` 虚拟环境。

**Windows (PowerShell/CMD)**:
```powershell
# 1. 创建虚拟环境 (在项目根目录下)
python -m venv venv

# 2. 激活虚拟环境
.\venv\Scripts\activate

# 3. 安装依赖 (如遇下载慢可追加 -i https://pypi.tuna.tsinghua.edu.cn/simple)
pip install -r requirements.txt
```

---

## 2. 数据准备 & 环境变量

请提前准备好 [HRSID Dataset](https://github.com/chaozhong2010/HRSID) 并在本地解压（目录内需包含 `images/` 和 `annotations/` 子文件夹）。
项目代码高度依赖根路径，**运行以下所有脚本前，请必须先配置好数据集环境变量（`HRSID_DATA_ROOT`）**。

**Windows (PowerShell)**:
```powershell
$env:HRSID_DATA_ROOT="D:\你的路径\HRSID_jpg"
```

**Windows (CMD)**:
```cmd
set HRSID_DATA_ROOT=D:\你的路径\HRSID_jpg
```

**Linux / macOS**:
```bash
export HRSID_DATA_ROOT="/你的路径/HRSID_jpg"
```

> *注意：如果不设置环境变量，系统在 `configs/base_config.py` 中会默认尝试回退寻找相对路径 `../../HRSID_jpg`，非常容易出现报错。强烈推荐显式设置。*

---

## 3. 命令行执行清单

### 🚀 启动训练 (Train)
开始从零训练 PSC-YOLOv8。第一次运行该脚本时，程序会自动在 `HRSID_DATA_ROOT` 下建立供 Ultralytics 训练专用的 `labels/` 文件夹（转换 COCO 到 YOLO_txt）并生成 `hrsid.yaml` 数据配置文件，随后启动带有 EMA 的原生安全训练进程：

```bash
python scripts/train.py
```
**产出**：训练结束后，最优的权重文件会自动从 `logs/train/weights/best.pt` 转移备份到项目根目录下的 `checkpoints/best.pth`。

### 📊 综合评估 (Evaluate)
如果已经有训练好的模型权重位于 `checkpoints/best.pth`，可以通过该脚本一次性分别在**Overall (全局测试)**、**Inshore (近岸子集)**、**Offshore (离岸子集)** 三个场景切分下评估当前模型。

```bash
python scripts/evaluate.py
```
**产出**：计算检测指标（mAP）和雷达特征指标（目标级框覆盖 Pd/Pfa），最后在 `logs/` 下生成一份 `evaluation_results.txt` 总结报告文档。

### ⚖️ 基线消融实验对比 (Run Baselines)
自动串联运行三种不同工作流的模型，快速比较出所有基线（Baseline）的数据差值。涵盖最底层的 CA-CFAR 到深度学习基础版的 YOLOv8n，以及物理改进后的 PSC-YOLOv8n 模型。

```bash
python scripts/run_baselines.py
```
**产出**：将首先进行 CFAR 掩码推断计算 像素级别 Pd/Pfa，然后针对基础和物理创新版网络加载权重评测，最后自动在终端打印出一份 Markdown 格式对齐的**消融实验对比表**。
