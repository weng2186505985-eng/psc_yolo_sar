import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import BaseConfig
from models.psc_yolo import build_psc_yolo
from data.dataloader import get_dataloaders
from engine.trainer import train_yolo

def main():
    config = BaseConfig()
    config.setup_dirs()
    
    print(f"Loading data from {config.DATA_ROOT}")
    dataloaders = get_dataloaders(config)
    
    print("Building PSC-YOLOv8...")
    model = build_psc_yolo()
    
    print("Starting Training...")
    # trainer.py implementation uses dataloaders directly since we implemented custom loop
    train_yolo(model, config, dataloaders)
    
    print("Training finished. Checkpoints saved to", config.CHECKPOINT_DIR)

if __name__ == "__main__":
    main()
