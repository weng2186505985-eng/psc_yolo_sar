import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class HRSIDDataset(Dataset):
    def __init__(self, data_root, split="train", scene_filter_file=None, exclude_filter_file=None, transforms=None):
        """
        :param scene_filter_file: 包含图片的 ID 列表或 COCO 字典（白名单）
        :param exclude_filter_file: 要排除的图片 ID 列表或 COCO 字典（黑名单）
        """
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
            if os.path.exists(scene_path):
                try:
                    with open(scene_path, 'r') as f:
                        scene_data = json.load(f)
                    
                    # Determine if it is a list of IDs or a COCO dict
                    if isinstance(scene_data, dict) and "images" in scene_data:
                        scene_ids = [int(img["id"]) for img in scene_data["images"]]
                    elif isinstance(scene_data, list):
                        scene_ids = [int(i) for i in scene_data]
                    else:
                        scene_ids = []
                        print(f"Warning: Scene filter file {scene_filter_file} has an unknown format. No filtering applied.")
                        
                    # Filter IDs
                    self.img_ids = [img_id for img_id in self.img_ids if int(img_id) in scene_ids]
                    if len(self.img_ids) == 0:
                        print(f"Warning: Scene filter {scene_filter_file} resulted in an empty dataset.")
                except Exception as e:
                    raise FileNotFoundError(f"Failed to load or parse scene filter: {scene_path}. Error: {e}")
            else:
                raise FileNotFoundError(f"Scene filter file not found: {scene_path}")
        
        # Exclude by scene if provided (for creating disjoint sets like Pure Offshore)
        if exclude_filter_file is not None:
            excl_path = os.path.join(data_root, "annotations", exclude_filter_file)
            if os.path.exists(excl_path):
                try:
                    with open(excl_path, 'r') as f:
                        excl_data = json.load(f)
                    if isinstance(excl_data, dict) and "images" in excl_data:
                        excl_ids = [int(img["id"]) for img in excl_data["images"]]
                    elif isinstance(excl_data, list):
                        excl_ids = [int(i) for i in excl_data]
                    else:
                        excl_ids = []
                    
                    self.img_ids = [img_id for img_id in self.img_ids if int(img_id) not in excl_ids]
                except Exception as e:
                    print(f"Error loading exclude filter: {e}")

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
