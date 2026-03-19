import os
import torch
import numpy as np
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
try:
    from sklearn.metrics import auc
except ImportError:
    # Handle if sklearn is not installed fully to just use numpy integration
    pass

from utils.visualize import plot_roc_curve

class Evaluator:
    def __init__(self, dataloader, device, iou_thresh=0.5, conf_thresh=0.001):
        """
        初始化评估器
        """
        self.dataloader = dataloader
        self.coco_gt = dataloader.dataset.coco
        self.device = device
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh

    @torch.no_grad()
    def evaluate(self, model, scene_name="Overall", save_dir=None):
        """
        评估函数。只接受 ultralytics YOLO 对象。
        """
        assert isinstance(model, YOLO) or hasattr(model, 'task'), "model must be an ultralytics YOLO object, not nn.Module"
        
        # evaluation can use pt_model for fast forward? No, use the ultralytics interface completely or just eval()
        model.model.eval()
        
        coco_results = []
        
        # 目标级 Pd / Pfa 统计所需变量
        total_TP = 0
        total_FP = 0
        total_GT = 0
        # 为了 Pfa 计算，我们需要图级别虚警（即没有任何 GT，但却检出了框）
        # 或者像素级/目标级。根据用户：
        # Pd（检测概率）：图像级别，该图有检测框即为正例 -> 但是图像全是正图？
        # Wait, if all images have ships, Pd will be related to how many images we successfully detect at least one box.
        # Pfa（虚警概率）：图像级别 (Images without GT but we detected something? But HRSID has no pure negative samples...)
        # Let's define:
        # Pd: count(detected ships) / count(GT ships) Or image level? "Pd（检测概率）：图像级别，该图有检测框即为正例" - Wait, this means: True Positives Images / All Positive Images
        # Let's count image-level Pd. 
        # For object-level Pd/Pfa, Pd = TP / (TP+FN), Pfa = FP / (FP+TN).
        # We will collect maximum confidence per image for image-level ROC.
        
        image_max_confs = []
        image_has_gt = []
        
        for batch in tqdm(self.dataloader, desc=f"Evaluating {scene_name}"):
            images = batch["images"].to(self.device)  # [B, 1, H, W]
            
            # 由于 YOLOv8 输入要求是 3 通道，如果这里是单通道灰度图，可能需要扩展
            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)
                
            # Ultralytics model predict
            # 注意: pt_model 预测输出可能是 [B, num_classes+4, max_preds]
            # 为了方便，我们可以直接用 Ultralytics 的 predict 接口，或者直接 forward
            # 如果使用 raw YOLOv8 forward：
            preds = model(images)
            
            # 使用非极大值抑制（NMS）等后处理。如果直接调 ultralytics 模型：
            # 这里为了避免复杂 NMS，最好直接用 Ultralytics 自带的方法。
            # preds 会返回 Results 对象！但这取决于传入的 model 是 ultralytics.YOLO 实例还是 nn.Module
            
            for i, target in enumerate(batch["targets"]):
                image_id = int(target["image_id"])
                gt_boxes = target["boxes"]
                
                result = preds[i] 
                boxes = result.boxes
                
                # COCO submission format [x, y, width, height]
                img_conf_list = []
                pred_boxes = []
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                    conf = box.conf[0].item()
                    
                    if conf > self.conf_thresh:
                        w = x2 - x1
                        h = y2 - y1
                        coco_results.append({
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [x1, y1, w, h],
                            "score": conf
                        })
                        pred_boxes.append([x1, y1, x2, y2])
                        img_conf_list.append(conf)
                
                # 计算目标级别 TP 和 FP
                total_GT += len(gt_boxes)
                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    import torchvision.ops as ops
                    pb_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
                    gb_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
                    ious = ops.box_iou(pb_tensor, gb_tensor) # [num_preds, num_gts]
                    max_iou, gt_idx = ious.max(dim=1)
                    
                    matched_gt = set()
                    for j, iou in enumerate(max_iou):
                        if iou > self.iou_thresh:
                            g = gt_idx[j].item()
                            if g not in matched_gt:
                                matched_gt.add(g)
                                total_TP += 1
                            else:
                                total_FP += 1
                        else:
                            total_FP += 1
                elif len(pred_boxes) > 0 and len(gt_boxes) == 0:
                    total_FP += len(pred_boxes)
                
                max_conf = max(img_conf_list) if img_conf_list else 0.0
                image_max_confs.append(max_conf)
                image_has_gt.append(1 if len(gt_boxes) > 0 else 0)

        # ====== 1. 目标检测指标 ======
        mAP50 = 0.0
        mAP50_95 = 0.0
        
        if len(coco_results) > 0:
            coco_dt = self.coco_gt.loadRes(coco_results)
            coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            mAP50_95 = coco_eval.stats[0] # AP @[ IoU=0.50:0.95 ]
            mAP50 = coco_eval.stats[1]    # AP @[ IoU=0.50 ]

        # ====== 2. 雷达检测指标 (目标级别) ======
        # Pd = TP / (TP + FN)  -> FN = total_GT - TP
        # Pfa = FP / 总检测框数 -> 总检测框数 = TP + FP
        
        Pd = total_TP / total_GT if total_GT > 0 else 0.0
        total_preds = total_TP + total_FP
        Pfa = total_FP / total_preds if total_preds > 0 else 0.0
        
        roc_auc = 0.0
        # ROC 图基于图级别的最大置信度（仍然可用但只供参考）
        image_max_confs = np.array(image_max_confs)
        image_has_gt = np.array(image_has_gt)
        pos_mask = (image_has_gt == 1)
        neg_mask = (image_has_gt == 0)
        # 如果存在阴性样本才能画图级别 ROC
        if total_neg > 0 and save_dir:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(image_has_gt, image_max_confs)
            roc_auc = auc(fpr, tpr)
            plot_roc_curve(fpr, tpr, roc_auc, os.path.join(save_dir, f"roc_{scene_name}.png"), scene_name)
        elif save_dir:
            print(f"[{scene_name}] No negative samples (pure background) found at image level. ROC curve skipped.")

        metrics = {
            "mAP@0.5": mAP50,
            "mAP@0.5:0.95": mAP50_95,
            "Pd": Pd,
            "Pfa": Pfa
        }
        
        print(f"--- {scene_name} Results ---")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            
        return metrics
