"""
数据加载器：加载空间转录组和单细胞RNA数据
"""

import os
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')


class SpatialTranscriptomicsDataset(Dataset):
    """空间转录组数据集"""
    
    def __init__(
        self,
        dataset_dir,
        known_gene_indices=None,
        unknown_gene_indices=None,
        normalize='log1p',
        use_ground_truth=True
    ):
        """
        Args:
            dataset_dir: 数据集目录路径
            known_gene_indices: 已知基因的索引列表
            unknown_gene_indices: 未知基因的索引列表
            normalize: 归一化方法 ('log1p', 'standardize', None)
            use_ground_truth: 是否使用ground truth（用于训练/验证）
        """
        self.dataset_dir = dataset_dir
        self.normalize = normalize
        self.use_ground_truth = use_ground_truth
        
        # 加载数据
        self._load_data()
        
        # 设置基因索引
        if known_gene_indices is None or unknown_gene_indices is None:
            # 如果没有指定，使用所有基因作为已知基因（用于推理）
            if known_gene_indices is None:
                known_gene_indices = list(range(self.num_total_genes))
            if unknown_gene_indices is None:
                unknown_gene_indices = []
        
        self.known_gene_indices = np.array(known_gene_indices)
        self.unknown_gene_indices = np.array(unknown_gene_indices)
        
        # 提取特征
        self._prepare_features()
    
    def _load_data(self):
        """加载空间转录组和单细胞数据"""
        # 加载空间转录组数据
        spatial_path = os.path.join(self.dataset_dir, 'Spatial_count.h5ad')
        if not os.path.exists(spatial_path):
            raise FileNotFoundError(f"空间转录组数据不存在: {spatial_path}")
        
        self.spatial_adata = sc.read_h5ad(spatial_path)
        
        # 获取空间坐标
        if 'spatial' in self.spatial_adata.obsm:
            self.coords = self.spatial_adata.obsm['spatial'].astype(np.float32)
        elif 'x' in self.spatial_adata.obs and 'y' in self.spatial_adata.obs:
            self.coords = np.stack([
                self.spatial_adata.obs['x'].values,
                self.spatial_adata.obs['y'].values
            ], axis=1).astype(np.float32)
        else:
            raise ValueError("无法找到空间坐标信息（obsm['spatial']或obs['x']/'y']）")
        
        # 获取基因表达数据
        if isinstance(self.spatial_adata.X, np.ndarray):
            self.spatial_expr = self.spatial_adata.X.astype(np.float32)
        else:
            # 如果是稀疏矩阵，转换为密集矩阵
            self.spatial_expr = self.spatial_adata.X.toarray().astype(np.float32)
        
        self.num_spots = self.spatial_expr.shape[0]
        self.num_total_genes = self.spatial_expr.shape[1]
        self.gene_names = self.spatial_adata.var_names.tolist()
        
        # 加载单细胞数据
        scrna_path = os.path.join(self.dataset_dir, 'scRNA_count_cluster.h5ad')
        if not os.path.exists(scrna_path):
            raise FileNotFoundError(f"单细胞数据不存在: {scrna_path}")
        
        self.sc_adata = sc.read_h5ad(scrna_path)
        
        if isinstance(self.sc_adata.X, np.ndarray):
            self.sc_expr = self.sc_adata.X.astype(np.float32)
        else:
            self.sc_expr = self.sc_adata.X.toarray().astype(np.float32)
        
        self.num_cells = self.sc_expr.shape[0]
        self.sc_gene_names = self.sc_adata.var_names.tolist()
        
        # 对齐基因（确保空间和单细胞数据使用相同的基因顺序）
        self._align_genes()
    
    def _align_genes(self):
        """对齐空间和单细胞数据的基因"""
        # 找到共同基因
        common_genes = list(set(self.gene_names) & set(self.sc_gene_names))
        common_genes = sorted(common_genes)  # 排序以确保一致性
        
        if len(common_genes) == 0:
            raise ValueError("空间和单细胞数据没有共同基因！")
        
        # 获取基因索引
        spatial_gene_indices = [self.gene_names.index(g) for g in common_genes]
        sc_gene_indices = [self.sc_gene_names.index(g) for g in common_genes]
        
        # 重新排列数据
        self.spatial_expr = self.spatial_expr[:, spatial_gene_indices]
        self.sc_expr = self.sc_expr[:, sc_gene_indices]
        
        # 更新基因信息
        self.gene_names = common_genes
        self.num_total_genes = len(common_genes)
        
        print(f"对齐后基因数量: {self.num_total_genes}")
        print(f"空间spots数量: {self.num_spots}")
        print(f"单细胞数量: {self.num_cells}")
    
    def _normalize(self, data):
        """归一化数据"""
        if self.normalize == 'log1p':
            return np.log1p(data)
        elif self.normalize == 'standardize':
            # 标准化
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True) + 1e-8
            return (data - mean) / std
        else:
            return data
    
    def _prepare_features(self):
        """准备特征"""
        # 归一化
        self.spatial_expr_norm = self._normalize(self.spatial_expr)
        self.sc_expr_norm = self._normalize(self.sc_expr)
        
        # 提取已知和未知基因
        self.known_genes = self.spatial_expr_norm[:, self.known_gene_indices]
        if self.use_ground_truth and len(self.unknown_gene_indices) > 0:
            self.unknown_genes = self.spatial_expr_norm[:, self.unknown_gene_indices]
        else:
            self.unknown_genes = None
    
    def __len__(self):
        return self.num_spots
    
    def __getitem__(self, idx):
        """
        返回一个样本
        
        Returns:
            known_genes: [num_known_genes] 已知基因表达
            coords: [2] 空间坐标
            unknown_genes: [num_unknown_genes] 未知基因表达（如果use_ground_truth=True）
            sc_data: [num_cells, num_total_genes] 单细胞数据（所有样本共享）
        """
        known_genes = torch.FloatTensor(self.known_genes[idx])
        coords = torch.FloatTensor(self.coords[idx])
        sc_data = torch.FloatTensor(self.sc_expr_norm)
        
        if self.use_ground_truth and self.unknown_genes is not None:
            unknown_genes = torch.FloatTensor(self.unknown_genes[idx])
            return known_genes, coords, unknown_genes, sc_data
        else:
            return known_genes, coords, sc_data
    
    def get_all_data(self):
        """获取所有数据（用于批量处理）"""
        known_genes = torch.FloatTensor(self.known_genes)
        coords = torch.FloatTensor(self.coords)
        sc_data = torch.FloatTensor(self.sc_expr_norm)
        
        if self.use_ground_truth and self.unknown_genes is not None:
            unknown_genes = torch.FloatTensor(self.unknown_genes)
            return known_genes, coords, unknown_genes, sc_data
        else:
            return known_genes, coords, sc_data


def create_dataloader(
    dataset_dir,
    known_gene_indices,
    unknown_gene_indices,
    batch_size=32,
    shuffle=True,
    normalize='log1p',
    use_ground_truth=True
):
    """创建数据加载器"""
    dataset = SpatialTranscriptomicsDataset(
        dataset_dir=dataset_dir,
        known_gene_indices=known_gene_indices,
        unknown_gene_indices=unknown_gene_indices,
        normalize=normalize,
        use_ground_truth=use_ground_truth
    )
    
    # 自定义collate函数，因为sc_data对所有样本都是相同的
    def collate_fn(batch):
        if use_ground_truth:
            known_genes, coords, unknown_genes, sc_data = zip(*batch)
            return (
                torch.stack(known_genes),
                torch.stack(coords),
                torch.stack(unknown_genes),
                sc_data[0]  # 所有样本共享相同的sc_data
            )
        else:
            known_genes, coords, sc_data = zip(*batch)
            return (
                torch.stack(known_genes),
                torch.stack(coords),
                sc_data[0]
            )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0  # Windows上可能需要设置为0
    )
    
    return dataloader, dataset


def create_kfold_splits(dataset_dir, n_splits=10, normalize='log1p'):
    """
    创建K折交叉验证的数据划分（在已知基因上划分）
    
    Returns:
        splits: list of (train_indices, val_indices) tuples
        dataset: 完整数据集（用于获取基因信息）
    """
    # 先加载数据集获取基因信息
    temp_dataset = SpatialTranscriptomicsDataset(
        dataset_dir=dataset_dir,
        known_gene_indices=list(range(100)),  # 临时，用于获取基因数量
        unknown_gene_indices=[],
        normalize=normalize,
        use_ground_truth=False
    )
    
    num_genes = temp_dataset.num_total_genes
    
    # 在基因上做K折划分
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    gene_indices = np.arange(num_genes)
    
    splits = []
    for train_gene_idx, val_gene_idx in kf.split(gene_indices):
        train_gene_indices = gene_indices[train_gene_idx].tolist()
        val_gene_indices = gene_indices[val_gene_idx].tolist()
        splits.append((train_gene_indices, val_gene_indices))
    
    return splits, temp_dataset
