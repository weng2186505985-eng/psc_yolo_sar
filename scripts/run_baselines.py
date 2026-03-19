import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import BaseConfig
from data.dataloader import get_dataloaders
from models.psc_yolo import build_psc_yolo, replace_activations
from baselines.plain_yolo import build_plain_yolo
from baselines.cfar import evaluate_cfar
from engine.evaluator import Evaluator
from utils.checkpoint import load_checkpoint
from ultralytics import YOLO

def evaluate_model(model_name, get_model_func, dataloader, config, load_path=None):
    device = torch.device(config.DEVICE)
    model = get_model_func()
    # If it's a YOLOv8 standard model, use its high-level predict inside Evaluator now
    evaluator = Evaluator(dataloader, device, iou_thresh=config.IOU_THRESHOLD, conf_thresh=config.CONF_THRESHOLD)
    return evaluator.evaluate(model, scene_name=f"{model_name}-Overall")

def main():
    config = BaseConfig()
    config.BATCH_SIZE = 8 # 多批次提高评测速度
    
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
    plain_weights = os.path.join(config.CHECKPOINT_DIR, "best_plain.pt")
    if os.path.exists(plain_weights):
        model_plain = YOLO(plain_weights, task='detect')
        evaluator_plain = Evaluator(test_loader, torch.device(config.DEVICE), iou_thresh=config.IOU_THRESHOLD, conf_thresh=config.CONF_THRESHOLD)
        plain_res = evaluator_plain.evaluate(model_plain, scene_name="Plain-YOLOv8n-Overall")
    else:
        print("Warning: No best_plain.pt found, evaluating with pre-trained weights only.")
        plain_res = evaluate_model("YOLOv8n-Pretrained", build_plain_yolo, test_loader, config)
    
    results["YOLOv8n"] = {
        "mAP@0.5": plain_res["mAP@0.5"],
        "Pd": plain_res["Pd"],
        "Pfa": plain_res["Pfa"]
    }
    
    # 3. PSC-YOLOv8
    print("\nRunning PSC-YOLOv8n...")
    psc_weights = os.path.join(config.CHECKPOINT_DIR, "best.pt")
    if os.path.exists(psc_weights):
        model = YOLO(psc_weights, task='detect')
        # 注意：不再调用 replace_activations，权重已在 pt 中
        evaluator = Evaluator(test_loader, torch.device(config.DEVICE), iou_thresh=config.IOU_THRESHOLD, conf_thresh=config.CONF_THRESHOLD)
        psc_res = evaluator.evaluate(model, scene_name="PSC-YOLOv8n-Overall")
    else:
        print("Warning: No best.pt found, skipping PSC-YOLOv8n baseline.")
        psc_res = {"mAP@0.5": 0.0, "Pd": 0.0, "Pfa": 0.0}
        
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
