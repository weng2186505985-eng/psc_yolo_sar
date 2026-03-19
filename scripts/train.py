import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import BaseConfig
from models.psc_yolo import build_psc_yolo
from engine.trainer import train_yolo

def main():
    config = BaseConfig()
    config.setup_dirs()
    
    print(f"Loading data from {config.DATA_ROOT}")
    
    print("Building PSC-YOLOv8...")
    model = build_psc_yolo()
    
    print("Starting Training...")
    # trainer.py implementation natively builds its own pipeline
    train_yolo(model, config, None)
    
    print("Training finished. Checkpoints saved to", config.CHECKPOINT_DIR)

if __name__ == "__main__":
    main()
