import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import BaseConfig
from models.psc_yolo import build_psc_yolo, replace_activations
from data.dataloader import get_dataloaders
from engine.evaluator import Evaluator
from utils.checkpoint import load_checkpoint
from ultralytics import YOLO

def main():
    config = BaseConfig()
    device = torch.device(config.DEVICE)
    
    _, test_loader, inshore_loader, offshore_loader = get_dataloaders(config)
    
    best_weights = os.path.join(config.CHECKPOINT_DIR, "best.pt")
    if os.path.exists(best_weights):
        print(f"Loading checkpoint from {best_weights}")
        model = YOLO(best_weights, task='detect')
        replace_activations(model.model)
    else:
        print("Warning: No best.pt found, using pretrained base weights.")
        model = build_psc_yolo()
        
    model.model.to(device)
    
    results = {}
    print("Starting Evaluation on Overall Test Set...")
    eval_overall = Evaluator(test_loader, device, iou_thresh=config.IOU_THRESHOLD, conf_thresh=config.CONF_THRESHOLD)
    results["Overall"] = eval_overall.evaluate(model, scene_name="Overall", save_dir=config.LOG_DIR)
    
    print("\nStarting Evaluation on Inshore Test Set...")
    eval_inshore = Evaluator(inshore_loader, device, iou_thresh=config.IOU_THRESHOLD, conf_thresh=config.CONF_THRESHOLD)
    results["Inshore"] = eval_inshore.evaluate(model, scene_name="Inshore", save_dir=config.LOG_DIR)
    
    print("\nStarting Evaluation on Offshore Test Set...")
    eval_offshore = Evaluator(offshore_loader, device, iou_thresh=config.IOU_THRESHOLD, conf_thresh=config.CONF_THRESHOLD)
    results["Offshore"] = eval_offshore.evaluate(model, scene_name="Offshore", save_dir=config.LOG_DIR)
    
    # Save results to txt file
    with open(os.path.join(config.LOG_DIR, "evaluation_results.txt"), 'w') as f:
        for scene, metrics in results.items():
            f.write(f"--- {scene} ---\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
            
    print("Evaluation completed. Summaries generated.")

if __name__ == "__main__":
    main()
