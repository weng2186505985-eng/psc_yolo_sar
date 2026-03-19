import torch
from torch.utils.data import DataLoader
from configs.base_config import BaseConfig
from data.dataset import HRSIDDataset
from data.transforms import get_train_transforms, get_val_transforms

def identity_collate(batch):
    """
    Collate function to ensure compatibility with num_workers > 0 by avoiding lambda functions.
    It returns a list of dictionaries.
    """
    images = []
    targets = []
    
    for item in batch:
        images.append(item["image"])
        # We need a unified format for Ultralytics if using standard training loop,
        # but the prompt requires YOLOv8 native loss + custom trainer, 
        # so keeping as list or packed tensor is up to the custom trainer.
        targets.append({
            "boxes": item["boxes"],
            "labels": item["labels"],
            "image_id": item["image_id"]
        })
        
    return {
        "images": torch.stack(images, dim=0),      # [B, 1, H, W]
        "targets": targets                         # list of dicts
    }

def get_dataloaders(config: BaseConfig):
    """
    Returns four DataLoaders: train / test / inshore / offshore
    """
    train_dataset = HRSIDDataset(
        data_root=config.DATA_ROOT,
        split="train",
        transforms=get_train_transforms()
    )
    
    test_dataset = HRSIDDataset(
        data_root=config.DATA_ROOT,
        split="test",
        transforms=get_val_transforms()
    )
    
    inshore_dataset = HRSIDDataset(
        data_root=config.DATA_ROOT,
        split="test",
        scene_filter_file="inshore.json",
        transforms=get_val_transforms()
    )
    
    offshore_dataset = HRSIDDataset(
        data_root=config.DATA_ROOT,
        split="test",
        scene_filter_file="offshore.json",
        transforms=get_val_transforms()
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=identity_collate,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=identity_collate,
        pin_memory=True
    )
    
    inshore_loader = DataLoader(
        inshore_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=identity_collate,
        pin_memory=True
    )
    
    offshore_loader = DataLoader(
        offshore_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=identity_collate,
        pin_memory=True
    )
    
    return train_loader, test_loader, inshore_loader, offshore_loader
