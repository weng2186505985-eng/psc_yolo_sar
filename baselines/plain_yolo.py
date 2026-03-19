from ultralytics import YOLO
from configs.base_config import BaseConfig

def build_plain_yolo(weights_path=BaseConfig.YOLO_MODEL, num_classes=BaseConfig.NUM_CLASSES):
    """
    Standard YOLOv8n without PSC activation
    """
    model = YOLO(weights_path)
    # ultralytics model handles num_classes configuration during dataset passing,
    # but we will just return the standard pre-loaded model here since we will override its loss in trainer
    return model
