"""
工具函数
"""

import os
import random
import numpy as np
import torch
import json
from datetime import datetime


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, loss, filepath, metrics=None, model_config=None):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    if model_config is not None:
        checkpoint['model_config'] = model_config
    torch.save(checkpoint, filepath)
    print(f"检查点已保存到: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """加载检查点"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"检查点文件不存在: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)
    metrics = checkpoint.get('metrics', None)
    
    print(f"检查点已加载: epoch={epoch}, loss={loss}")
    
    return epoch, loss, metrics


def create_output_dir(base_dir, prefix='sinca_net'):
    """创建输出目录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_metrics(metrics, filepath):
    """保存评估指标到JSON文件"""
    # 将numpy数组转换为列表
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_serializable[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            metrics_serializable[key] = float(value)
        else:
            metrics_serializable[key] = value
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"指标已保存到: {filepath}")


def load_metrics(filepath):
    """从JSON文件加载评估指标"""
    with open(filepath, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    return metrics
