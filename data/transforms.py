import torch
import torchvision.transforms.functional as F
import random

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
            # PIL Image size is (W, H)
            w, h = image.size 
            boxes = target["boxes"]
            if len(boxes) > 0:
                # boxes: [x1, y1, x2, y2]
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target

class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.vflip(image)
            w, h = image.size
            boxes = target["boxes"]
            if len(boxes) > 0:
                # boxes: [x1, y1, x2, y2]
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                target["boxes"] = boxes
        return image, target

class RandomRotation90:
    def __call__(self, image, target):
        angle = random.choice([0, 90, 180, 270])
        if angle == 0:
            return image, target
            
        w, h = image.size
        # The rotation center is the center of the image.
        # But F.rotate handles it, so we need to map boxes.
        image = F.rotate(image, angle)
        
        boxes = target["boxes"].clone()
        if len(boxes) > 0:
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            
            # w and h might be swapped after 90 or 270 rotation for bounding boxes calculation
            if angle == 90:
                # Note: F.rotate rotates counter-clockwise. 
                # x' = y, y' = w - x
                boxes[:, 0] = y1
                boxes[:, 1] = w - x2
                boxes[:, 2] = y2
                boxes[:, 3] = w - x1
            elif angle == 180:
                boxes[:, 0] = w - x2
                boxes[:, 1] = h - y2
                boxes[:, 2] = w - x1
                boxes[:, 3] = h - y1
            elif angle == 270:
                # x' = h - y, y' = x
                boxes[:, 0] = h - y2
                boxes[:, 1] = x1
                boxes[:, 2] = h - y1
                boxes[:, 3] = x2

            # Ensure box format is (min_x, min_y, max_x, max_y)
            boxes[:, 0], boxes[:, 2] = torch.min(boxes[:, 0].clone(), boxes[:, 2].clone()), \
                                       torch.max(boxes[:, 0].clone(), boxes[:, 2].clone())
            boxes[:, 1], boxes[:, 3] = torch.min(boxes[:, 1].clone(), boxes[:, 3].clone()), \
                                       torch.max(boxes[:, 1].clone(), boxes[:, 3].clone())
            target["boxes"] = boxes
            
        return image, target

class ToTensorAndNormalize:
    def __call__(self, image, target):
        image = F.to_tensor(image) # Normalizes to [0, 1] automatically
        # [1, H, W] for grayscale is retained.
        return image, target

def get_train_transforms():
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation90(),
        ToTensorAndNormalize()
    ])

def get_val_transforms():
    return Compose([
        ToTensorAndNormalize()
    ])
