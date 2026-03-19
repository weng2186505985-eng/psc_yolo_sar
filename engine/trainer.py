import os
import torch
import torch.optim as optim
from ultralytics import YOLO
from tqdm import tqdm

from configs.base_config import BaseConfig
from utils.checkpoint import save_checkpoint
from utils.logger import Logger
from engine.evaluator import Evaluator

def train_yolo(model, config: BaseConfig, dataloaders):
    """
    使用 Ultralytics 原生的训练接口进行训练。
    由于 Ultralytics YOLOv8 内置了强大的训练循环、Loss计算（包括 DFL, Box, Cls）以及 EMA等，
    同时其 Dataset 格式非常特定，如果使用原生训练，我们需要生成符合 Ultralytics 格式的 YAML 和 txt 标签文件。
    但在本项目中我们被要求 "dataset.py 使用 pycocotools 读取 COCO 格式标注" 并放入 DataLoader。
    因此我们需要一个 "自定义训练循环" 来调用 model.model (底层的 nn.Module)，
    并且使用 ultralytics 的 v8DetectionLoss。
    
    或者为了严谨：我们将 COCO JSON 转换为 Ultralytics YAML 格式，然后使用它自带的 trainer。
    由于要求实现：trainer.py -> "使用 ultralytics 的 YOLOv8 训练接口，或自定义训练循环"。
    为满足使用自主 dataloader 的逻辑，我们实现自定义训练循环。
    """
    from ultralytics.utils.loss import v8DetectionLoss
    
    device = torch.device(config.DEVICE)
    train_loader, val_loader, _, _ = dataloaders
    
    # 获取 PyTorch 底层模型
    pt_model = model.model.to(device)
    
    # 构造自定义的优化器
    optimizer = optim.AdamW(
        pt_model.parameters(), 
        lr=config.LR, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    # YOLOv8 损失函数初始化需要 model
    loss_fn = v8DetectionLoss(pt_model)
    
    logger = Logger(config.LOG_DIR)
    evaluator = Evaluator(val_loader, device, iou_thresh=config.IOU_THRESHOLD, conf_thresh=config.CONF_THRESHOLD)
    
    best_map = 0.0
    patience_counter = 0
    
    for epoch in range(1, config.EPOCHS + 1):
        pt_model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Train]")
        for batch_idx, batch in enumerate(pbar):
            images = batch["images"].to(device)
            # 转为 3 通道 (YOLOv8 默认输入)
            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)
                
            # Ultralytics loss_fn 取要求非常严格的 dict 结构
            # 格式要求：
            # targets: (N, 6) tensor => [batch_idx, class_id, x, y, w, h] normalized!
            targets_list = []
            for b_i, target in enumerate(batch["targets"]):
                boxes = target["boxes"]
                labels = target["labels"]
                if len(boxes) > 0:
                    # 将 xyxy 转换为 cxcywh，并归一化
                    _, _, H, W = images.shape
                    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0 / W
                    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0 / H
                    w = (boxes[:, 2] - boxes[:, 0]) / W
                    h = (boxes[:, 3] - boxes[:, 1]) / H
                    
                    b_idx = torch.full((len(boxes), 1), b_i, dtype=torch.float32)
                    cls_id = torch.zeros((len(boxes), 1), dtype=torch.float32)  # 单类为0
                    
                    targets_list.append(torch.cat([b_idx, cls_id, cx.unsqueeze(1), cy.unsqueeze(1), w.unsqueeze(1), h.unsqueeze(1)], dim=1))
            
            if len(targets_list) > 0:
                batch_targets = torch.cat(targets_list, dim=0).to(device)
            else:
                batch_targets = torch.empty((0, 6)).to(device)
                
            # 为了适配 ultralytics 损失输入
            ultralytics_batch = {
                "img": images,
                "bboxes": batch_targets[:, 2:] if len(batch_targets) > 0 else torch.empty((0, 4), device=device),
                "cls": batch_targets[:, 1].unsqueeze(1) if len(batch_targets) > 0 else torch.empty((0, 1), device=device),
                "batch_idx": batch_targets[:, 0] if len(batch_targets) > 0 else torch.empty((0,), device=device)
            }
            
            optimizer.zero_grad()
            
            # Forward pass
            # YOLOv8 pt_model(img) 返回结果
            preds = pt_model(images)
            
            loss, loss_items = loss_fn(preds, ultralytics_batch)
            
            # backward
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_loss = train_loss / len(train_loader)
        logger.log_scalars({'Train/Loss': avg_loss}, epoch)
        
        # Validation
        if epoch % config.VAL_INTERVAL == 0:
            metrics = evaluator.evaluate(model, scene_name="Validation", save_dir=config.LOG_DIR)
            map50 = metrics["mAP@0.5"]
            
            logger.log_scalars({
                'Val/mAP@0.5': map50,
                'Val/mAP@0.5:0.95': metrics["mAP@0.5:0.95"],
                'Val/Pd': metrics["Pd"],
                'Val/Pfa': metrics["Pfa"]
            }, epoch)
            logger.log_metrics_csv(metrics, epoch)
            
            save_path = os.path.join(config.CHECKPOINT_DIR, "last.pth")
            save_checkpoint(pt_model, optimizer, epoch, metrics, save_path)
            
            if map50 > best_map:
                best_map = map50
                patience_counter = 0
                best_path = os.path.join(config.CHECKPOINT_DIR, "best.pth")
                save_checkpoint(pt_model, optimizer, epoch, metrics, best_path)
                print(f"New Best mAP@0.5: {best_map:.4f}! Checkpoint saved.")
            else:
                patience_counter += 1
                
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
    logger.close()
    return best_map
