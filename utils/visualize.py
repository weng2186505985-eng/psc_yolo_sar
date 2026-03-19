import matplotlib
matplotlib.use('Agg') # 避免无头服务器阻塞

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def plot_boxes(image_tensor, boxes, save_path, labels=None, scores=None):
    """
    可视化并将结果保存到文件
    image_tensor: [1, H, W] 归一化到 [0, 1] 的灰度图
    boxes: [N, 4] xyxy 格式
    """
    img = image_tensor.squeeze(0).cpu().numpy()
    
    fig, ax = plt.subplots(1)
    ax.imshow(img, cmap='gray')
    
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            
            rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            label_str = ""
            if labels is not None:
                label_str += f"Cls {int(labels[i])} "
            if scores is not None:
                label_str += f"{float(scores[i]):.2f}"
                
            if label_str:
                plt.text(x1, y1, label_str, color='r', fontsize=8, backgroundcolor="white")
                
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)

def plot_roc_curve(fpr, tpr, roc_auc, save_path, scene="Overall"):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Pfa)')
    plt.ylabel('True Positive Rate (Pd)')
    plt.title(f'Receiver Operating Characteristic - {scene}')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
