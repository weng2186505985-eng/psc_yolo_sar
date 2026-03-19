import torch
import torch.nn.functional as F
import numpy as np

class CACFAR:
    def __init__(self, guard_win=2, train_win=5, pfa=1e-3):
        self.guard_win = guard_win
        self.train_win = train_win
        self.pfa = pfa
        
        # 构建 boolean mask 算子 (N, C, H, W) -> (1, 1, 11, 11) for convolution
        sz = 2 * train_win + 1
        self.kernel = torch.ones((1, 1, sz, sz), dtype=torch.float32)
        
        # inner guard window to 0
        gw_sz = 2 * guard_win + 1
        start = train_win - guard_win
        end = train_win + guard_win + 1
        self.kernel[:, :, start:end, start:end] = 0.0
        
        # Number of training cells
        self.N_c = self.kernel.sum().item()
        
        # 门限乘子 alpha
        self.alpha = self.N_c * (pfa ** (-1.0 / self.N_c) - 1.0)

    @torch.no_grad()
    def detect(self, img_tensor):
        """
        img_tensor: [1, 1, H, W]
        returns mask of detected targets
        """
        device = img_tensor.device
        kernel = self.kernel.to(device)
        
        # 使用 boolean mask conv 提取均值
        # padding 保证输出尺寸和输入相同
        padding = self.train_win
        
        # 计算背景之和
        bg_sum = F.conv2d(img_tensor, kernel, padding=padding)
        
        # 背景平均值
        P_n = bg_sum / self.N_c
        
        # 判断阈值 T = alpha * P_n
        threshold = self.alpha * P_n
        
        # 超过阈值为检测出目标
        detected = img_tensor > threshold
        return detected

def evaluate_cfar(dataloader):
    cfar = CACFAR()
    # evaluate Pd, Pfa globally at the pixel level
    total_pd_count = 0
    total_target = 0
    
    total_pfa_count = 0
    total_bk = 0
    
    for batch in dataloader:
        images = batch["images"]  # [B, 1, H, W]
        # target box existence mask
        gt_mask = torch.zeros_like(images, dtype=torch.bool)
        for i, target in enumerate(batch["targets"]):
            for box in target["boxes"]:
                x1, y1, x2, y2 = box.int().tolist()
                # 越界保护
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(images.shape[3], x2), min(images.shape[2], y2)
                gt_mask[i, :, y1:y2, x1:x2] = True
            
        detected = cfar.detect(images).cpu()
        
        pos_mask = gt_mask
        neg_mask = ~gt_mask
        
        total_pd_count += (detected & pos_mask).sum().item()
        total_target += pos_mask.sum().item()
        
        total_pfa_count += (detected & neg_mask).sum().item()
        total_bk += neg_mask.sum().item()
        
    Pd = total_pd_count / total_target if total_target > 0 else 0.0
    Pfa = total_pfa_count / total_bk if total_bk > 0 else 0.0
    
    return {"Pd": Pd, "Pfa": Pfa}
