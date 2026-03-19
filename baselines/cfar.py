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
    # evaluate Pd, Pfa globally
    total_pd_count = 0
    total_target = 0
    
    total_pfa_count = 0
    total_bk = 0
    
    for batch in dataloader:
        images = batch["images"]  # [B, 1, H, W]
        # target box existence
        has_gt = []
        for target in batch["targets"]:
            has_gt.append(1 if len(target["boxes"]) > 0 else 0)
            
        detected = cfar.detect(images)
        
        # Image level detection: if any pixel is detected, image is positive detection
        # Since CA-CFAR fires at pixel level, we use max per image instance
        img_detected = detected.view(images.shape[0], -1).any(dim=1).cpu().numpy()
        has_gt = np.array(has_gt)
        
        pos_mask = (has_gt == 1)
        neg_mask = (has_gt == 0)
        
        total_pd_count += np.sum(img_detected[pos_mask] == 1)
        total_target += np.sum(pos_mask)
        
        total_pfa_count += np.sum(img_detected[neg_mask] == 1)
        total_bk += np.sum(neg_mask)
        
    Pd = total_pd_count / total_target if total_target > 0 else 0.0
    Pfa = total_pfa_count / total_bk if total_bk > 0 else 0.0
    
    return {"Pd": Pd, "Pfa": Pfa}
