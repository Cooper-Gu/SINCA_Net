"""
评估指标：PCC, SSIM, RMAE, JS, ACC
"""

import numpy as np
from scipy import stats
from skimage.metrics import structural_similarity as ssim
import warnings

warnings.filterwarnings('ignore')


def calculate_pcc(pred, true):
    """
    计算皮尔逊相关系数 (Pearson Correlation Coefficient)
    
    Args:
        pred: [N, M] 预测值
        true: [N, M] 真实值
    
    Returns:
        pcc: 标量或数组，每个基因的PCC
    """
    pred = np.array(pred)
    true = np.array(true)
    
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
        true = true.reshape(-1, 1)
    
    # 计算每个基因的PCC
    pccs = []
    for i in range(pred.shape[1]):
        if np.std(pred[:, i]) == 0 or np.std(true[:, i]) == 0:
            pccs.append(0.0)
        else:
            pcc, _ = stats.pearsonr(pred[:, i], true[:, i])
            pccs.append(pcc)
    
    return np.array(pccs)


def calculate_ssim(pred, true, data_range=None):
    """
    计算结构相似性指数 (Structural Similarity Index)
    
    Args:
        pred: [N, M] 预测值
        true: [N, M] 真实值
        data_range: 数据范围，如果为None则自动计算
    
    Returns:
        ssim_value: 标量，整体SSIM
    """
    pred = np.array(pred)
    true = np.array(true)
    
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
        true = true.reshape(-1, 1)
    
    # 将数据reshape为2D图像格式 (spots作为空间维度)
    # 对于基因表达数据，我们可以将每个基因视为一个通道
    if data_range is None:
        data_range = max(pred.max() - pred.min(), true.max() - true.min())
        if data_range == 0:
            data_range = 1.0
    
    # 计算每个基因的SSIM，然后取平均
    ssims = []
    for i in range(pred.shape[1]):
        # 将基因表达值reshape为2D（假设spots可以排列成网格）
        n_spots = pred.shape[0]
        grid_size = int(np.sqrt(n_spots))
        if grid_size * grid_size < n_spots:
            grid_size += 1
        
        # 填充到grid_size x grid_size
        pred_gene = pred[:, i].copy()
        true_gene = true[:, i].copy()
        
        if len(pred_gene) < grid_size * grid_size:
            padding = grid_size * grid_size - len(pred_gene)
            pred_gene = np.pad(pred_gene, (0, padding), mode='constant', constant_values=0)
            true_gene = np.pad(true_gene, (0, padding), mode='constant', constant_values=0)
        
        pred_2d = pred_gene[:grid_size * grid_size].reshape(grid_size, grid_size)
        true_2d = true_gene[:grid_size * grid_size].reshape(grid_size, grid_size)
        
        try:
            ssim_val = ssim(true_2d, pred_2d, data_range=data_range)
            ssims.append(ssim_val)
        except:
            ssims.append(0.0)
    
    return np.mean(ssims) if ssims else 0.0


def calculate_rmae(pred, true):
    """
    计算均方根误差 (Root Mean Absolute Error)
    
    Args:
        pred: [N, M] 预测值
        true: [N, M] 真实值
    
    Returns:
        rmae: 标量，整体RMAE
    """
    pred = np.array(pred)
    true = np.array(true)
    
    mae = np.mean(np.abs(pred - true))
    rmae = np.sqrt(mae)
    
    return rmae


def calculate_js(pred, true, threshold=None):
    """
    计算Jaccard相似性 (Jaccard Similarity)
    将基因表达值二值化后计算Jaccard相似性
    
    Args:
        pred: [N, M] 预测值
        true: [N, M] 真实值
        threshold: 二值化阈值，如果为None则使用中位数
    
    Returns:
        js: 标量，整体JS
    """
    pred = np.array(pred)
    true = np.array(true)
    
    if threshold is None:
        # 使用中位数作为阈值
        threshold = np.median(true)
    
    # 二值化
    pred_binary = (pred > threshold).astype(int)
    true_binary = (true > threshold).astype(int)
    
    # 计算Jaccard相似性
    intersection = np.sum(pred_binary & true_binary)
    union = np.sum(pred_binary | true_binary)
    
    if union == 0:
        return 0.0
    
    js = intersection / union
    return js


def calculate_acc(pred, true):
    """
    计算准确度评分 (Accuracy Score)
    使用相对值排名方法来评估生成数据的质量
    
    Args:
        pred: [N, M] 预测值
        true: [N, M] 真实值
    
    Returns:
        acc: 标量，整体ACC
    """
    pred = np.array(pred)
    true = np.array(true)
    
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
        true = true.reshape(-1, 1)
    
    # 对每个基因计算排名准确度
    accs = []
    for i in range(pred.shape[1]):
        pred_ranks = stats.rankdata(pred[:, i], method='average')
        true_ranks = stats.rankdata(true[:, i], method='average')
        
        # 计算排名相关性
        if np.std(pred_ranks) == 0 or np.std(true_ranks) == 0:
            accs.append(0.0)
        else:
            acc, _ = stats.pearsonr(pred_ranks, true_ranks)
            accs.append(acc)
    
    return np.mean(accs) if accs else 0.0


def calculate_all_metrics(pred, true):
    """
    计算所有评估指标
    
    Args:
        pred: [N, M] 预测值
        true: [N, M] 真实值
    
    Returns:
        metrics: dict，包含所有指标
    """
    pred = np.array(pred)
    true = np.array(true)
    
    # 计算各个指标
    pccs = calculate_pcc(pred, true)
    pcc_mean = np.mean(pccs)
    
    ssim_value = calculate_ssim(pred, true)
    rmae_value = calculate_rmae(pred, true)
    js_value = calculate_js(pred, true)
    acc_value = calculate_acc(pred, true)
    
    metrics = {
        'PCC_mean': pcc_mean,
        'PCC_per_gene': pccs,
        'SSIM': ssim_value,
        'RMAE': rmae_value,
        'JS': js_value,
        'ACC': acc_value
    }
    
    return metrics
