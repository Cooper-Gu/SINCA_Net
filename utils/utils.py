"""
工具函数模块
包含训练和评估过程中的常用工具函数
"""

import os
import json
import torch
import numpy as np
import scanpy as sc
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_spatial_data(data_path: str):
    """
    加载空间转录组数据
    
    Args:
        data_path: h5ad文件路径
    
    Returns:
        adata: AnnData对象
        expr: 表达矩阵 [num_spots, num_genes]
        coords: 空间坐标 [num_spots, 2]
    """
    adata = sc.read_h5ad(data_path)
    
    # 获取表达矩阵
    if isinstance(adata.X, np.ndarray):
        expr = adata.X
    else:
        expr = adata.X.toarray()
    
    # 获取空间坐标
    if 'spatial' in adata.obsm:
        coords = adata.obsm['spatial']
    elif 'x' in adata.obs and 'y' in adata.obs:
        coords = np.stack([adata.obs['x'].values, adata.obs['y'].values], axis=1)
    else:
        raise ValueError("无法找到空间坐标信息，请确保数据包含'spatial' in obsm或'x', 'y' in obs")
    
    return adata, expr, coords


def load_sc_data(data_path: str):
    """
    加载单细胞RNA数据
    
    Args:
        data_path: h5ad文件路径
    
    Returns:
        adata: AnnData对象
        expr: 表达矩阵 [num_cells, num_genes]
        cluster: 聚类标签 [num_cells] (可选)
    """
    adata = sc.read_h5ad(data_path)
    
    # 获取表达矩阵
    if isinstance(adata.X, np.ndarray):
        expr = adata.X
    else:
        expr = adata.X.toarray()
    
    # 获取聚类标签
    cluster = None
    if 'cluster' in adata.obs:
        cluster = adata.obs['cluster'].values
    
    return adata, expr, cluster


def normalize_expression(expr, method='log1p'):
    """
    归一化表达矩阵
    
    Args:
        expr: 表达矩阵
        method: 归一化方法 ('log1p', 'minmax', 'zscore', None)
    
    Returns:
        normalized_expr: 归一化后的表达矩阵
    """
    if method == 'log1p':
        # log(1 + x) 归一化
        return np.log1p(expr)
    elif method == 'minmax':
        # Min-Max归一化到[0, 1]
        expr_min = expr.min(axis=0, keepdims=True)
        expr_max = expr.max(axis=0, keepdims=True)
        return (expr - expr_min) / (expr_max - expr_min + 1e-8)
    elif method == 'zscore':
        # Z-score归一化
        expr_mean = expr.mean(axis=0, keepdims=True)
        expr_std = expr.std(axis=0, keepdims=True)
        return (expr - expr_mean) / (expr_std + 1e-8)
    else:
        return expr


def create_mask(num_spots, num_genes, mask_ratio=0.1, random_seed=None):
    """
    创建随机掩码用于模拟缺失基因
    
    Args:
        num_spots: spot数量
        num_genes: 基因数量
        mask_ratio: 掩码比例
        random_seed: 随机种子
    
    Returns:
        mask: 掩码矩阵 [num_spots, num_genes], True表示缺失
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    mask = np.random.rand(num_spots, num_genes) < mask_ratio
    return mask


def save_checkpoint(model, optimizer, epoch, loss, filepath, **kwargs):
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        loss: 当前loss
        filepath: 保存路径
        **kwargs: 其他需要保存的信息
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    
    torch.save(checkpoint, filepath)
    print(f"检查点已保存到: {filepath}")


def load_checkpoint(model, filepath, optimizer=None, device='cpu'):
    """
    加载模型检查点
    
    Args:
        model: 模型
        filepath: 检查点路径
        optimizer: 优化器（可选）
        device: 设备
    
    Returns:
        checkpoint: 检查点字典
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def save_metrics(metrics: Dict, filepath: str):
    """
    保存评估指标到JSON文件
    
    Args:
        metrics: 指标字典
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 转换numpy类型为Python原生类型
    metrics_serializable = {}
    for k, v in metrics.items():
        if isinstance(v, (np.ndarray, np.generic)):
            metrics_serializable[k] = v.tolist() if isinstance(v, np.ndarray) else v.item()
        elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (np.ndarray, np.generic)):
            metrics_serializable[k] = [x.item() if isinstance(x, np.generic) else x.tolist() for x in v]
        else:
            metrics_serializable[k] = v
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"指标已保存到: {filepath}")


def load_metrics(filepath: str) -> Dict:
    """
    从JSON文件加载评估指标
    
    Args:
        filepath: JSON文件路径
    
    Returns:
        metrics: 指标字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    return metrics


def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def count_parameters(model):
    """统计模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model):
    """打印模型信息"""
    total_params = count_parameters(model)
    print(f"模型总参数数量: {total_params:,}")
    print(f"模型结构:")
    print(model)


def find_common_genes(spatial_genes, sc_genes):
    """
    找到空间转录组和单细胞数据的共同基因
    
    Args:
        spatial_genes: 空间转录组基因列表
        sc_genes: 单细胞基因列表
    
    Returns:
        common_genes: 共同基因列表（排序后）
        spatial_indices: 空间转录组中的索引
        sc_indices: 单细胞数据中的索引
    """
    spatial_genes_set = set(spatial_genes)
    sc_genes_set = set(sc_genes)
    common_genes_set = spatial_genes_set & sc_genes_set
    common_genes = sorted(list(common_genes_set))
    
    # 获取索引
    spatial_indices = [spatial_genes.index(g) for g in common_genes]
    sc_indices = [sc_genes.index(g) for g in common_genes]
    
    return common_genes, spatial_indices, sc_indices


def expand_predictions_to_full_genes(
    predictions, 
    common_genes, 
    original_spatial_genes,
    original_spatial_expr=None,
    fill_non_common_with_original=True
):
    """
    将共同基因的预测结果扩展到空间转录组的全部基因
    
    Args:
        predictions: 共同基因的预测结果 [num_spots, num_common_genes]
        common_genes: 共同基因列表
        original_spatial_genes: 原始空间转录组的所有基因列表
        original_spatial_expr: 原始空间转录组表达矩阵 [num_spots, num_original_genes] (可选)
        fill_non_common_with_original: 如果为True，非共同基因使用原始值；如果为False，填充0
    
    Returns:
        full_predictions: 扩展到全部基因的预测结果 [num_spots, num_original_genes]
        gene_mapping: 基因映射信息字典
    """
    import numpy as np
    
    num_spots = predictions.shape[0]
    num_common_genes = len(common_genes)
    num_original_genes = len(original_spatial_genes)
    
    # 创建基因到索引的映射
    original_gene_to_idx = {gene: idx for idx, gene in enumerate(original_spatial_genes)}
    common_gene_to_idx = {gene: idx for idx, gene in enumerate(common_genes)}
    
    # 初始化全部基因的预测矩阵
    full_predictions = np.zeros((num_spots, num_original_genes), dtype=predictions.dtype)
    
    # 将共同基因的预测结果映射到对应位置
    for common_gene in common_genes:
        if common_gene in original_gene_to_idx:
            common_idx = common_gene_to_idx[common_gene]
            original_idx = original_gene_to_idx[common_gene]
            full_predictions[:, original_idx] = predictions[:, common_idx]
    
    # 处理非共同基因
    if fill_non_common_with_original and original_spatial_expr is not None:
        # 使用原始值填充非共同基因
        non_common_mask = np.ones(num_original_genes, dtype=bool)
        for common_gene in common_genes:
            if common_gene in original_gene_to_idx:
                non_common_mask[original_gene_to_idx[common_gene]] = False
        
        # 填充非共同基因的原始值
        full_predictions[:, non_common_mask] = original_spatial_expr[:, non_common_mask]
    # 如果fill_non_common_with_original为False，非共同基因保持为0（已在初始化时设置）
    
    # 创建基因映射信息
    gene_mapping = {
        'common_genes': common_genes,
        'original_genes': original_spatial_genes,
        'num_common_genes': num_common_genes,
        'num_original_genes': num_original_genes,
        'non_common_genes': [g for g in original_spatial_genes if g not in common_genes]
    }
    
    return full_predictions, gene_mapping