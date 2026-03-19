import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class HRSIDDataset(Dataset):
    def __init__(self, data_root, split="train", scene_filter_file=None, transforms=None):
        self.data_root = data_root
        self.split = split
        self.transforms = transforms
        
        self.img_dir = os.path.join(data_root, "images")
        
        # Load COCO annotations
        ann_file = os.path.join(data_root, "annotations", f"{split}2017.json")
        try:
            self.coco = COCO(ann_file)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load annotation file: {ann_file}. Error: {e}")
            
        self.img_ids = list(self.coco.imgs.keys())
        
        # Filter by scene if provided
        if scene_filter_file is not None:
            scene_path = os.path.join(data_root, "annotations", scene_filter_file)
            try:
                with open(scene_path, 'r') as f:
                    scene_ids = json.load(f)
                # Keep only image IDs that are in the scene list
                self.img_ids = [img_id for img_id in self.img_ids if img_id in scene_ids]
            except Exception as e:
                raise FileNotFoundError(f"Failed to load scene filter: {scene_path}. Error: {e}")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        # Load image in grayscale format as requested
        img = Image.open(img_path).convert("L")
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        for ann in anns:
            # COCO bbox is [x, y, w, h]
            x, y, w, h = ann['bbox']
            # Convert to [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)  # Ship class
            
        boxes = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.empty((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id
        }
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return {
            "image": img,         # FloatTensor [1, H, W]
            "boxes": target["boxes"],   # FloatTensor [N, 4]
            "labels": target["labels"], # LongTensor [N]
            "image_id": target["image_id"]
        }
