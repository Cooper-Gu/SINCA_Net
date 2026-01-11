"""
评估指标模块
包含PCC, SSIM, RMAE, COSSIM等评估指标
"""

import torch
import numpy as np
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cosine


def pearson_correlation_coefficient(y_true, y_pred):
    """
    计算皮尔逊相关系数 (PCC)
    
    Args:
        y_true: 真实值 [N, ...]
        y_pred: 预测值 [N, ...]
    
    Returns:
        pcc: 皮尔逊相关系数的绝对值（表示相关性强度）
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # 展平
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # 计算皮尔逊相关系数
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    
    corr, _ = pearsonr(y_true, y_pred)
    # 返回绝对值，表示相关性强度
    return float(abs(corr)) if not np.isnan(corr) else 0.0


def structural_similarity_index(y_true, y_pred, data_range=None):
    """
    计算结构相似性指数 (SSIM)
    
    Args:
        y_true: 真实值 [N, ...] 或 [H, W, ...]
        y_pred: 预测值 [N, ...] 或 [H, W, ...]
        data_range: 数据范围，如果为None则自动计算
    
    Returns:
        ssim_value: SSIM值
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # 如果是2D或3D，尝试计算SSIM
    # 对于基因表达数据，我们可以将其视为"图像"
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    
    if data_range is None:
        data_range = max(y_true.max() - y_true.min(), y_pred.max() - y_pred.min())
        if data_range == 0:
            data_range = 1.0
    
    # 如果是一维数据，reshape为2D
    if y_true.ndim == 1:
        # 尝试找到合适的2D形状
        n = len(y_true)
        h = int(np.sqrt(n))
        w = (n + h - 1) // h
        pad_size = h * w - n
        if pad_size > 0:
            y_true = np.pad(y_true, (0, pad_size), mode='constant')
            y_pred = np.pad(y_pred, (0, pad_size), mode='constant')
        y_true = y_true[:h*w].reshape(h, w)
        y_pred = y_pred[:h*w].reshape(h, w)
    
    # 如果是多通道，对每个通道计算SSIM后取平均
    if y_true.ndim > 2:
        ssim_values = []
        for i in range(y_true.shape[-1]):
            ssim_val = ssim(y_true[..., i], y_pred[..., i], data_range=data_range)
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    else:
        return ssim(y_true, y_pred, data_range=data_range)


def root_mean_absolute_error(y_true, y_pred):
    """
    计算均方根绝对误差 (RMAE) - 实际上应该是RMSE (Root Mean Squared Error)
    但根据项目需求，这里计算RMAE (Root Mean Absolute Error)
    
    Args:
        y_true: 真实值 [N, ...]
        y_pred: 预测值 [N, ...]
    
    Returns:
        rmae: 均方根绝对误差
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # 计算绝对误差的均方根
    mae = np.mean(np.abs(y_true - y_pred))
    rmae = np.sqrt(mae)
    
    return float(rmae)


def cosine_similarity(y_true, y_pred):
    """
    计算余弦相似性 (COSSIM)
    
    Args:
        y_true: 真实值 [N, ...]
        y_pred: 预测值 [N, ...]
    
    Returns:
        cos_sim: 余弦相似性的绝对值（表示相似度强度）
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # 展平
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # 计算余弦相似性 (1 - cosine distance)
    if np.linalg.norm(y_true) == 0 or np.linalg.norm(y_pred) == 0:
        return 0.0
    
    cos_sim = 1 - cosine(y_true, y_pred)
    # 返回绝对值，表示相似度强度
    return float(abs(cos_sim)) if not np.isnan(cos_sim) else 0.0


def compute_all_metrics(y_true, y_pred, prefix=''):
    """
    计算所有评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        prefix: 指标名称前缀
    
    Returns:
        metrics: 包含所有指标的字典
    """
    metrics = {}
    
    # PCC
    pcc = pearson_correlation_coefficient(y_true, y_pred)
    metrics[f'{prefix}PCC'] = pcc
    
    # SSIM
    try:
        ssim_value = structural_similarity_index(y_true, y_pred)
        metrics[f'{prefix}SSIM'] = ssim_value
    except Exception as e:
        print(f"Warning: SSIM calculation failed: {e}")
        metrics[f'{prefix}SSIM'] = 0.0
    
    # RMAE
    rmae = root_mean_absolute_error(y_true, y_pred)
    metrics[f'{prefix}RMAE'] = rmae
    
    # COSSIM
    cos_sim = cosine_similarity(y_true, y_pred)
    metrics[f'{prefix}COSSIM'] = cos_sim
    
    return metrics


def compute_metrics_per_gene(y_true, y_pred):
    """
    按基因计算评估指标
    
    Args:
        y_true: 真实值 [N, num_genes]
        y_pred: 预测值 [N, num_genes]
    
    Returns:
        metrics: 每个基因的指标字典
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    num_genes = y_true.shape[1]
    metrics = {
        'PCC': [],
        'SSIM': [],
        'RMAE': [],
        'COSSIM': []
    }
    
    for i in range(num_genes):
        gene_true = y_true[:, i]
        gene_pred = y_pred[:, i]
        
        metrics['PCC'].append(pearson_correlation_coefficient(gene_true, gene_pred))
        
        try:
            metrics['SSIM'].append(structural_similarity_index(gene_true, gene_pred))
        except:
            metrics['SSIM'].append(0.0)
        
        metrics['RMAE'].append(root_mean_absolute_error(gene_true, gene_pred))
        metrics['COSSIM'].append(cosine_similarity(gene_true, gene_pred))
    
    return metrics
