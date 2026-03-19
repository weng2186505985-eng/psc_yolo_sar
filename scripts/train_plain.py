import os
import sys
import yaml
import shutil

# 确保能导入 configs/engine 等模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import BaseConfig
from baselines.plain_yolo import build_plain_yolo

def train_plain_yolo():
    config = BaseConfig()
    config.setup_dirs()
    
    # 1. 准备数据集配置文件 (同 psc_yolo)
    data_yaml = {
        'path': config.DATA_ROOT,
        'train': 'images',
        'val': 'images',
        'test': 'images',
        'names': {0: 'ship'}
    }
    
    yaml_path = os.path.join(config.DATA_ROOT, 'data_plain.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    # 2. 构建标准模型 (无 PSC)
    model = build_plain_yolo()
    
    # 3. 开始训练
    print("Starting Training for Plain YOLOv8 Baseline...")
    results = model.train(
        data=yaml_path,
        epochs=config.EPOCHS,
        imgsz=config.IMG_SIZE,
        batch=config.BATCH_SIZE,
        project=config.LOG_DIR,
        name='train_plain',
        device=0 if config.DEVICE == 'cuda' else 'cpu',
        workers=config.NUM_WORKERS,
        lr0=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        exist_ok=True
    )
    
    # 4. 复制最优权重到 checkpoints
    best_weights = os.path.join(config.LOG_DIR, 'train_plain', 'weights', 'best.pt')
    if os.path.exists(best_weights):
        shutil.copy(best_weights, os.path.join(config.CHECKPOINT_DIR, "best_plain.pt"))
        print(f"Copied best plain model weights to {config.CHECKPOINT_DIR}/best_plain.pt")
    else:
        print("Error: Could not find best.pt for plain training!")

if __name__ == "__main__":
    train_plain_yolo()
