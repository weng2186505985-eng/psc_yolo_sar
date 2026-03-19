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
    for name, module in model.named_modules():
        # Look for specific modules in YOLO that contain activations we want to replace
        # Ultralytics uses Conv which has (conv, bn, act)
        if hasattr(module, "act") and isinstance(module.act, nn.SiLU):
            # C2f and SPPF use Conv blocks. 
            # We want to replace act in these specific higher-level modules if they are in backbone or so.
            # But the requirement says: "替换范围：backbone 的 C2f 模块内部和 SPPF 模块"
            # We can check if name contains 'model.' where the index <= 9 (typically backbone ends around index 9)
            # Actually, to be safe, we can just check if it's inside a C2f or SPPF in the backbone.
            pass

    # A more robust iteration:
    for name, m in model.named_modules():
        is_in_c2f_or_sppf = False
        parts = name.split('.')
        # Check if any parent module is C2f or SPPF
        parent_module = model
        for part in parts:
            if hasattr(parent_module, part):
                parent_module = getattr(parent_module, part)
                if isinstance(parent_module, (C2f, SPPF)):
                    is_in_c2f_or_sppf = True
                    break
        
        # In Ultralytics, individual convolution blocks are typically 'Conv' containing 'conv', 'bn', 'act'
        if is_in_c2f_or_sppf and hasattr(m, 'act') and isinstance(m.act, nn.SiLU):
            if hasattr(m, 'conv') and isinstance(m.conv, nn.Conv2d):
                num_channels = m.conv.out_channels
                m.act = PSCActivation(num_channels)

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
