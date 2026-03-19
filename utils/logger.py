import os
import torch
import csv
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.csv_path = os.path.join(log_dir, 'metrics.csv')
        
    def log_scalars(self, summary_dict, step):
        """记录多个标量到 TensorBoard"""
        for k, v in summary_dict.items():
            self.writer.add_scalar(k, v, step)
            
    def log_metrics_csv(self, metrics_dict, epoch):
        """记录指标到 CSV"""
        file_exists = os.path.isfile(self.csv_path)
        
        # Merge epoch into dictionary to write
        row = {'epoch': epoch}
        row.update(metrics_dict)
        
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            
    def close(self):
        self.writer.close()
