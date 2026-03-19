import os
import torch

class BaseConfig:
    # 数据路径（优先读环境变量）
    DATA_ROOT = os.environ.get(
        "HRSID_DATA_ROOT",
        os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "HRSID_jpg")
    )
    
    # 训练参数
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE  = 8
    NUM_WORKERS = 4
    EPOCHS      = 20
    LR          = 1e-4
    WEIGHT_DECAY = 1e-5
    IMG_SIZE    = 640        # YOLOv8 标准输入尺寸
    
    # 模型参数
    YOLO_MODEL  = "yolov8n.pt"   # 预训练权重
    NUM_CLASSES = 1               # 只有舰船一类
    
    # 评估参数
    IOU_THRESHOLD   = 0.5
    CONF_THRESHOLD  = 0.25
    VAL_INTERVAL    = 1
    EARLY_STOP_PATIENCE = 10
    
    # 输出路径
    PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    LOG_DIR        = os.path.join(PROJECT_ROOT, "logs")

    @classmethod
    def setup_dirs(cls):
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
