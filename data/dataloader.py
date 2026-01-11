"""
数据加载器模块
用于加载和处理空间转录组及单细胞RNA数据
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict
import scanpy as sc
from utils.utils import (
    load_spatial_data, load_sc_data, 
    normalize_expression, create_mask, find_common_genes
)


class SpatialTranscriptomicsDataset(Dataset):
    """
    空间转录组数据集
    支持加载空间转录组数据和可选的单细胞参考数据
    """
    
    def __init__(
        self,
        spatial_data_path: str,
        sc_data_path: Optional[str] = None,
        mask_ratio: float = 0.1,
        normalize: str = 'log1p',
        use_common_genes: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Args:
            spatial_data_path: 空间转录组数据路径
            sc_data_path: 单细胞数据路径（可选）
            mask_ratio: 掩码比例，用于模拟缺失基因
            normalize: 归一化方法 ('log1p', 'minmax', 'zscore', None)
            use_common_genes: 是否只使用共同基因
            random_seed: 随机种子
        """
        self.spatial_data_path = spatial_data_path
        self.sc_data_path = sc_data_path
        self.mask_ratio = mask_ratio
        self.normalize = normalize
        self.use_common_genes = use_common_genes
        self.random_seed = random_seed  # 基础随机种子
        self.current_epoch = 0  # 当前epoch，用于生成不同的掩码
        
        # 加载数据
        self._load_data()
        
    def _load_data(self):
        """加载和预处理数据"""
        # 加载空间转录组数据
        self.spatial_adata, spatial_expr, self.spatial_coords = load_spatial_data(
            self.spatial_data_path
        )
        self.spatial_genes = list(self.spatial_adata.var_names)
        self.num_spots = spatial_expr.shape[0]
        
        # 保存原始空间转录组的所有基因信息（用于推理时扩展到全部基因）
        # 注意：在归一化之前保存，以便后续可以正确填充非共同基因
        self.original_spatial_genes = self.spatial_genes.copy()
        self.original_spatial_expr_raw = spatial_expr.copy()  # 原始未归一化的表达
        
        # 加载单细胞数据（如果提供）
        self.sc_expr = None
        self.sc_cluster = None
        self.sc_genes = None
        if self.sc_data_path is not None and os.path.exists(self.sc_data_path):
            self.sc_adata, sc_expr, self.sc_cluster = load_sc_data(self.sc_data_path)
            self.sc_genes = list(self.sc_adata.var_names)
            self.sc_expr = sc_expr
            self.num_cells = sc_expr.shape[0]
        else:
            self.num_cells = 0
        
        # 处理共同基因
        # 如果存在单细胞数据，必须使用共同基因，否则模型无法工作
        if self.sc_expr is not None:
            common_genes, spatial_indices, sc_indices = find_common_genes(
                self.spatial_genes, self.sc_genes
            )
            if len(common_genes) == 0:
                raise ValueError("空间转录组和单细胞数据没有共同基因！")
            self.spatial_expr = spatial_expr[:, spatial_indices]
            self.spatial_genes = common_genes
            self.sc_expr = self.sc_expr[:, sc_indices]
            self.sc_genes = common_genes
            # 保存共同基因的索引映射（用于推理时扩展）
            self.common_gene_indices = spatial_indices
            self.common_genes = common_genes
            print(f"使用共同基因: {len(common_genes)} 个")
            if not self.use_common_genes:
                print("警告: 检测到单细胞数据，自动使用共同基因对齐")
        else:
            self.spatial_expr = spatial_expr
            # 如果没有单细胞数据，所有基因都是"共同基因"
            self.common_gene_indices = list(range(len(self.spatial_genes)))
            self.common_genes = self.spatial_genes
        
        self.num_genes = self.spatial_expr.shape[1]
        
        # 将基因划分为训练集和验证集（9:1）
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        gene_indices = np.arange(self.num_genes)
        np.random.shuffle(gene_indices)
        train_gene_end = int(self.num_genes * 0.9)
        self.train_gene_indices = np.sort(gene_indices[:train_gene_end])
        self.val_gene_indices = np.sort(gene_indices[train_gene_end:])
        self.num_train_genes = len(self.train_gene_indices)
        self.num_val_genes = len(self.val_gene_indices)
        print(f"基因划分: 训练集 {self.num_train_genes} 个基因, 验证集 {self.num_val_genes} 个基因")
        
        # 归一化
        if self.normalize:
            self.spatial_expr = normalize_expression(self.spatial_expr, self.normalize)
            if self.sc_expr is not None:
                self.sc_expr = normalize_expression(self.sc_expr, self.normalize)
            # 对原始表达也进行归一化（用于推理时填充非共同基因）
            self.original_spatial_expr = normalize_expression(self.original_spatial_expr_raw, self.normalize)
        else:
            self.original_spatial_expr = self.original_spatial_expr_raw
        
        # 转换为float32
        self.spatial_expr = self.spatial_expr.astype(np.float32)
        self.spatial_coords = self.spatial_coords.astype(np.float32)
        if self.sc_expr is not None:
            self.sc_expr = self.sc_expr.astype(np.float32)
        
        print(f"空间转录组数据: {self.num_spots} spots, {self.num_genes} genes")
        if self.sc_expr is not None:
            print(f"单细胞数据: {self.num_cells} cells, {self.num_genes} genes")
    
    def set_epoch(self, epoch):
        """
        设置当前epoch，用于生成不同的掩码
        
        Args:
            epoch: 当前epoch编号
        """
        self.current_epoch = epoch
    
    def __len__(self):
        return 1  # 每个数据集返回一个样本（整个数据集）
    
    def __getitem__(self, idx):
        """默认返回训练模式的数据"""
        return self._get_item_by_mode(idx, mode='train')
    
    def _get_item_by_mode(self, idx, mode='train'):
        """
        返回一个数据样本
        
        Args:
            idx: 样本索引（未使用，因为每个数据集返回一个样本）
            mode: 'train' 或 'val'，用于决定使用哪些基因
        
        Returns:
            sample: 包含以下字段的字典:
                - spatial_expr: [num_spots, num_genes] 空间转录组表达（可能被掩码）
                - spatial_coords: [num_spots, 2] 空间坐标
                - mask: [num_spots, num_genes] 掩码（True表示缺失）
                - target: [num_spots, num_genes] 真实值
                - gene_mask: [num_genes] 基因掩码（True表示该基因用于当前模式）
                - sc_expr: [num_cells, num_genes] 单细胞表达（可选）
                - sc_cluster: [num_cells] 单细胞聚类（可选）
        """
        # 根据模式选择基因
        if mode == 'train':
            gene_indices = self.train_gene_indices
            gene_mask = np.zeros(self.num_genes, dtype=bool)
            gene_mask[gene_indices] = True
        else:  # mode == 'val'
            gene_indices = self.val_gene_indices
            gene_mask = np.zeros(self.num_genes, dtype=bool)
            gene_mask[gene_indices] = True
        
        # 创建掩码：使用 base_seed + epoch 来确保每个epoch使用不同的掩码
        # 如果random_seed为None，则使用epoch作为种子
        mask_seed = None
        if self.random_seed is not None:
            mask_seed = self.random_seed + self.current_epoch
        else:
            mask_seed = self.current_epoch
        
        # 只对当前模式的基因创建掩码
        mask = create_mask(
            self.num_spots, self.num_genes, 
            mask_ratio=self.mask_ratio,
            random_seed=mask_seed
        )
        # 只对当前模式的基因应用掩码，其他基因保持完整
        mask = mask & gene_mask[np.newaxis, :]
        
        # 应用掩码（将缺失位置设为0）
        masked_expr = self.spatial_expr.copy()
        masked_expr[mask] = 0.0
        
        sample = {
            'spatial_expr': torch.FloatTensor(masked_expr),
            'spatial_coords': torch.FloatTensor(self.spatial_coords),
            'mask': torch.BoolTensor(mask),
            'target': torch.FloatTensor(self.spatial_expr),  # 真实值
            'gene_mask': torch.BoolTensor(gene_mask),  # 基因掩码
        }
        
        # 添加单细胞数据（如果存在）
        if self.sc_expr is not None:
            sample['sc_expr'] = torch.FloatTensor(self.sc_expr)
            if self.sc_cluster is not None:
                # 将聚类标签转换为整数索引
                unique_clusters = sorted(np.unique(self.sc_cluster))
                cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
                cluster_indices = np.array([cluster_to_idx[c] for c in self.sc_cluster])
                sample['sc_cluster'] = torch.LongTensor(cluster_indices)
            else:
                sample['sc_cluster'] = None
        
        return sample
    
    def get_full_data(self, mode='train'):
        """
        获取完整数据（不应用掩码）
        
        Args:
            mode: 'train' 或 'val'，用于决定使用哪些基因
        """
        # 根据模式选择基因
        if mode == 'train':
            gene_indices = self.train_gene_indices
            gene_mask = np.zeros(self.num_genes, dtype=bool)
            gene_mask[gene_indices] = True
        else:  # mode == 'val'
            gene_indices = self.val_gene_indices
            gene_mask = np.zeros(self.num_genes, dtype=bool)
            gene_mask[gene_indices] = True
        
        sample = {
            'spatial_expr': torch.FloatTensor(self.spatial_expr),
            'spatial_coords': torch.FloatTensor(self.spatial_coords),
            'mask': torch.zeros(self.num_spots, self.num_genes, dtype=torch.bool),
            'target': torch.FloatTensor(self.spatial_expr),
            'gene_mask': torch.BoolTensor(gene_mask),
        }
        
        if self.sc_expr is not None:
            sample['sc_expr'] = torch.FloatTensor(self.sc_expr)
            if self.sc_cluster is not None:
                unique_clusters = sorted(np.unique(self.sc_cluster))
                cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
                cluster_indices = np.array([cluster_to_idx[c] for c in self.sc_cluster])
                sample['sc_cluster'] = torch.LongTensor(cluster_indices)
            else:
                sample['sc_cluster'] = None
        
        return sample


def collate_fn(batch):
    """
    自定义collate函数，将多个样本组合成一个batch
    
    由于每个数据集返回整个数据集，我们直接返回第一个样本
    """
    return batch[0]


def get_batch_by_mode(dataset, mode='train'):
    """
    根据模式从数据集中获取一个batch
    
    Args:
        dataset: SpatialTranscriptomicsDataset对象
        mode: 'train' 或 'val'
    
    Returns:
        batch: 数据字典
    """
    return dataset._get_item_by_mode(0, mode=mode)


def create_dataloader(
    spatial_data_path: str,
    sc_data_path: Optional[str] = None,
    mask_ratio: float = 0.1,
    normalize: str = 'log1p',
    use_common_genes: bool = True,
    batch_size: int = 1,
    shuffle: bool = False,
    random_seed: Optional[int] = None
):
    """
    创建数据加载器
    
    Args:
        spatial_data_path: 空间转录组数据路径
        sc_data_path: 单细胞数据路径（可选）
        mask_ratio: 掩码比例
        normalize: 归一化方法
        use_common_genes: 是否只使用共同基因
        batch_size: batch大小（通常为1，因为每个数据集是一个样本）
        shuffle: 是否打乱
        random_seed: 随机种子
    
    Returns:
        dataloader: DataLoader对象
        dataset: Dataset对象
    """
    dataset = SpatialTranscriptomicsDataset(
        spatial_data_path=spatial_data_path,
        sc_data_path=sc_data_path,
        mask_ratio=mask_ratio,
        normalize=normalize,
        use_common_genes=use_common_genes,
        random_seed=random_seed
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0  # Windows上设为0
    )
    
    return dataloader, dataset


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, random_seed=42):
    """
    划分数据集为训练集、验证集和测试集（基于spots）
    
    Args:
        dataset: SpatialTranscriptomicsDataset对象
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        random_seed: 随机种子
    
    Returns:
        train_indices, val_indices, test_indices: 索引列表
    """
    np.random.seed(random_seed)
    num_spots = dataset.num_spots
    indices = np.random.permutation(num_spots)
    
    train_end = int(num_spots * train_ratio)
    val_end = train_end + int(num_spots * val_ratio)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    return train_indices, val_indices, test_indices
