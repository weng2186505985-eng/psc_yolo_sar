import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics import YOLO

from configs.base_config import BaseConfig
from models.psc_activation import PSCActivation

def replace_activations(model):
    """
    Traverse the model and replace SiLU with PSCActivation in C2f and SPPF modules.
    We need to track output channels. Usually, we can figure out the channels by
    inspecting the weights/biases of the preceding Conv layer.
    """
    for module in model.modules():
        if isinstance(module, (C2f, SPPF)):
            for submodule in module.modules():
                if hasattr(submodule, 'act') and isinstance(submodule.act, nn.SiLU):
                    if hasattr(submodule, 'conv') and isinstance(submodule.conv, nn.Conv2d):
                        num_channels = submodule.conv.out_channels
                        submodule.act = PSCActivation(num_channels)

def build_psc_yolo(weights_path=BaseConfig.YOLO_MODEL, num_classes=BaseConfig.NUM_CLASSES):
    """
    Load base YOLOv8 and apply PSC activation injects.
    """
    model = YOLO(weights_path)
    
    # We modify the underlying PyTorch model
    pt_model = model.model
    replace_activations(pt_model)
    
    return model

if __name__ == "__main__":
    model = build_psc_yolo()
    print("Model built successfully with PSCActivations.")
