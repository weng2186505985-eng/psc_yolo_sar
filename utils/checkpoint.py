import os
import torch

def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """保存 checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict() if not hasattr(model, 'model') else model.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)
    
def load_checkpoint(model, optimizer, load_path):
    """加载 checkpoint 并且必须使用 weights_only=True"""
    if os.path.isfile(load_path):
        # 加上 weights_only=True 按要求
        checkpoint = torch.load(load_path, map_location='cpu', weights_only=True)
        if hasattr(model, 'model'):
            model.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})
    else:
        raise FileNotFoundError(f"No checkpoint found at {load_path}")
