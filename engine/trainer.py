import os
import shutil
import torch
from ultralytics import YOLO

from configs.base_config import BaseConfig

def convert_coco_to_yolo(data_root, split="train2017"):
    import pycocotools.coco as coco
    import tqdm
    ann_file = os.path.join(data_root, "annotations", f"{split}.json")
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
        return None
    
    coco_api = coco.COCO(ann_file)
    labels_dir = os.path.join(data_root, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    img_paths = []
    
    print(f"Converting COCO format to YOLO format for {split}...")
    for img_id in tqdm.tqdm(coco_api.imgs.keys(), desc=f"Converting {split}"):
        img_info = coco_api.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Write .txt file
        base_name = os.path.splitext(file_name)[0]
        txt_path = os.path.join(labels_dir, f"{base_name}.txt")
        
        ann_ids = coco_api.getAnnIds(imgIds=img_id)
        anns = coco_api.loadAnns(ann_ids)
        
        with open(txt_path, 'w') as f:
            for ann in anns:
                x, y, w, h = ann['bbox']
                # convert to center format and normalize
                cx = (x + w / 2.0) / img_width
                cy = (y + h / 2.0) / img_height
                nw = w / img_width
                nh = h / img_height
                f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                
        img_paths.append(os.path.join(data_root, "images", file_name))
        
    # Write list file
    list_file = os.path.join(data_root, f"{split}.txt")
    with open(list_file, 'w') as f:
        for p in img_paths:
            f.write(p + "\n")
            
    return list_file

def prepare_yolo_yaml(data_root):
    train_txt = convert_coco_to_yolo(data_root, "train2017")
    val_txt = convert_coco_to_yolo(data_root, "test2017")
    
    yaml_path = os.path.join(data_root, "hrsid.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"path: {data_root}\n")
        
        # Use absolute paths or paths relative to YAML
        if train_txt:
            f.write(f"train: {train_txt}\n")
        if val_txt:
            f.write(f"val: {val_txt}\n")
            
        f.write("names:\n")
        f.write("  0: ship\n")
        
    return yaml_path

def train_yolo(model, config: BaseConfig, dataloaders=None):
    """
    使用 Ultralytics 原生的 model.train() 接口，避开使用极脆弱的 v8DetectionLoss 内部接口。
    """
    print("Preparing YOLO dataset format from COCO annotations...")
    yaml_path = prepare_yolo_yaml(config.DATA_ROOT)
    
    # 为了避免 Ultralytics 新建线程中报错找不到 PSCActivation，手动显式设定
    # 不过 Ultralytics 的 train() 方法对底层模型修改支持较好
    
    print("Start Training using Ultralytics native trainer...")
    # model must be ultralytics.YOLO
    model.train(
        data=yaml_path,
        epochs=config.EPOCHS,
        batch=config.BATCH_SIZE,
        workers=config.NUM_WORKERS,
        device=config.DEVICE,
        lr0=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        imgsz=config.IMG_SIZE,
        project=config.PROJECT_ROOT,
        name="logs/train",
        exist_ok=True,
        patience=config.EARLY_STOP_PATIENCE,
        save=True
    )
    
    # Training finishes. The best weights are saved internally by Ultralytics.
    # We will copy them to our configured checkpoint folder for uniform access.
    best_weights = os.path.join(config.PROJECT_ROOT, "logs", "train", "weights", "best.pt")
    if os.path.exists(best_weights):
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        shutil.copy(best_weights, os.path.join(config.CHECKPOINT_DIR, "best.pt"))
        print(f"Copied best model weights to {os.path.join(config.CHECKPOINT_DIR, 'best.pt')}")
    else:
        print("Warning: best.pt not found in ultralytics run dir.")
