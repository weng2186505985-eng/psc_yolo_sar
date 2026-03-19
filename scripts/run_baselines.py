import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import BaseConfig
from data.dataloader import get_dataloaders
from models.psc_yolo import build_psc_yolo
from baselines.plain_yolo import build_plain_yolo
from baselines.cfar import evaluate_cfar
from engine.evaluator import Evaluator
from utils.checkpoint import load_checkpoint

def evaluate_model(model_name, get_model_func, dataloader, config, load_path=None):
    device = torch.device(config.DEVICE)
    model = get_model_func()
    model.model.to(device)
    
    if load_path and os.path.exists(load_path):
        load_checkpoint(model, None, load_path)
    
    evaluator = Evaluator(dataloader, device, iou_thresh=config.IOU_THRESHOLD, conf_thresh=config.CONF_THRESHOLD)
    return evaluator.evaluate(model, scene_name=f"{model_name}-Overall")

def main():
    config = BaseConfig()
    config.BATCH_SIZE = 1 # CA-CFAR requires batch iterating clearly, standard is to use evaluation dataloader
    
    # We just need the overall test loader for baseline comparison table
    _, test_loader, _, _ = get_dataloaders(config)
    
    results = {}
    
    # 1. CA-CFAR
    print("Running CA-CFAR Baseline...")
    cfar_res = evaluate_cfar(test_loader)
    results["CA-CFAR"] = {
        "mAP@0.5": "-",
        "Pd": cfar_res["Pd"],
        "Pfa": cfar_res["Pfa"]
    }
    
    # 2. Plain YOLOv8
    print("\nRunning Plain YOLOv8n Baseline...")
    # NOTE: In a real scenario, this should point to a newly trained YOLOv8n on this dataset,
    # If the user hasn't trained it natively yet via run_baselines, we evaluate its loaded form.
    plain_res = evaluate_model("YOLOv8n", build_plain_yolo, test_loader, config)
    results["YOLOv8n"] = {
        "mAP@0.5": plain_res["mAP@0.5"],
        "Pd": plain_res["Pd"],
        "Pfa": plain_res["Pfa"]
    }
    
    # 3. PSC-YOLOv8
    print("\nRunning PSC-YOLOv8n...")
    psc_weights = os.path.join(config.CHECKPOINT_DIR, "best.pth")
    psc_res = evaluate_model("PSC-YOLOv8n", build_psc_yolo, test_loader, config, load_path=psc_weights)
    results["PSC-YOLOv8n"] = {
        "mAP@0.5": psc_res["mAP@0.5"],
        "Pd": psc_res["Pd"],
        "Pfa": psc_res["Pfa"]
    }
    
    # Print Ablation Table
    print("\n" + "="*40)
    print("消融实验对比表:")
    print(f"| {'模型':<12} | {'mAP@0.5':<7} | {'Pd':<5} | {'Pfa':<5} |")
    print("|--------------|---------|-------|-------|")
    for b in ["CA-CFAR", "YOLOv8n", "PSC-YOLOv8n"]:
        res_b = results[b]
        map_str = f"{res_b['mAP@0.5']:.4f}" if isinstance(res_b['mAP@0.5'], float) else res_b['mAP@0.5']
        print(f"| {b:<12} | {map_str:<7} | {res_b['Pd']:.4f} | {res_b['Pfa']:.4f} |")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
