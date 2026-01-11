"""
SINCA-Net: Spatial Inpainting Network with Cross-Modal Attention
用于空间转录组基因插补的深度学习网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt


class PositionalEncoding(nn.Module):
    """位置编码模块，将空间坐标转换为位置编码"""
    
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, coords):
        """
        Args:
            coords: [N, 2] 空间坐标
        Returns:
            pos_encoding: [N, d_model] 位置编码
        """
        # 将坐标归一化并映射到位置编码
        # 使用坐标的线性组合生成位置编码
        N = coords.shape[0]
        coords_normalized = coords / (coords.abs().max() + 1e-8)  # 归一化
        
        # 使用MLP将坐标映射到d_model维度
        pos_encoding = torch.zeros(N, self.d_model, device=coords.device)
        for i in range(0, self.d_model, 2):
            if i < self.d_model:
                pos_encoding[:, i] = torch.sin(coords_normalized[:, 0] * (i + 1) * np.pi)
            if i + 1 < self.d_model:
                pos_encoding[:, i + 1] = torch.cos(coords_normalized[:, 1] * (i + 1) * np.pi)
        
        return pos_encoding


class SpatialConvolutionModule(nn.Module):
    """空间卷积模块：将空间坐标转换为2D网格，应用卷积操作"""
    
    def __init__(self, d_model, kernel_size=3):
        super(SpatialConvolutionModule, self).__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        
        # 卷积层
        self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def coords_to_grid(self, coords, features, grid_size=None):
        """
        将空间坐标和特征转换为2D网格
        
        Args:
            coords: [N, 2] 空间坐标
            features: [N, d_model] 特征
            grid_size: (H, W) 网格大小，如果为None则自动计算
        
        Returns:
            grid_features: [1, d_model, H, W] 网格特征
        """
        N, d_model = features.shape
        device = features.device
        
        # 归一化坐标到[0, 1]
        coords_min = coords.min(dim=0)[0]
        coords_max = coords.max(dim=0)[0]
        coords_range = coords_max - coords_min + 1e-8
        coords_normalized = (coords - coords_min) / coords_range
        
        # 自动计算网格大小
        if grid_size is None:
            # 根据spots密度估算网格大小
            grid_size = (int(sqrt(N) * 1.5), int(sqrt(N) * 1.5))
        
        H, W = grid_size
        
        # 将坐标映射到网格索引
        grid_indices = (coords_normalized * torch.tensor([H-1, W-1], device=device, dtype=coords.dtype)).long()
        grid_indices[:, 0] = torch.clamp(grid_indices[:, 0], 0, H-1)
        grid_indices[:, 1] = torch.clamp(grid_indices[:, 1], 0, W-1)
        
        # 创建网格并填充特征
        grid_features = torch.zeros(1, d_model, H, W, device=device)
        grid_counts = torch.zeros(1, 1, H, W, device=device)
        
        for i in range(N):
            h_idx, w_idx = grid_indices[i]
            grid_features[0, :, h_idx, w_idx] += features[i]
            grid_counts[0, 0, h_idx, w_idx] += 1
        
        # 平均化（处理多个spots映射到同一网格点的情况）
        grid_counts = torch.clamp(grid_counts, min=1)
        grid_features = grid_features / grid_counts
        
        return grid_features, grid_size
    
    def grid_to_coords(self, grid_features, coords, grid_size):
        """
        将网格特征转换回原始坐标位置
        
        Args:
            grid_features: [1, d_model, H, W] 网格特征
            coords: [N, 2] 原始坐标
            grid_size: (H, W) 网格大小
        
        Returns:
            features: [N, d_model] 特征
        """
        N = coords.shape[0]
        device = coords.device
        H, W = grid_size
        
        # 归一化坐标
        coords_min = coords.min(dim=0)[0]
        coords_max = coords.max(dim=0)[0]
        coords_range = coords_max - coords_min + 1e-8
        coords_normalized = (coords - coords_min) / coords_range
        
        # 映射到网格索引
        grid_indices = (coords_normalized * torch.tensor([H-1, W-1], device=device, dtype=coords.dtype)).long()
        grid_indices[:, 0] = torch.clamp(grid_indices[:, 0], 0, H-1)
        grid_indices[:, 1] = torch.clamp(grid_indices[:, 1], 0, W-1)
        
        # 从网格中提取特征
        features = torch.zeros(N, grid_features.shape[1], device=device)
        for i in range(N):
            h_idx, w_idx = grid_indices[i]
            features[i] = grid_features[0, :, h_idx, w_idx]
        
        return features
    
    def forward(self, features, coords):
        """
        Args:
            features: [N, d_model] 输入特征
            coords: [N, 2] 空间坐标
        
        Returns:
            output: [N, d_model] 输出特征
        """
        # 转换为网格
        grid_features, grid_size = self.coords_to_grid(coords, features)
        
        # 应用卷积
        x = grid_features
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        
        # 转换回坐标空间
        output = self.grid_to_coords(x, coords, grid_size)
        
        # 残差连接和归一化
        output = self.norm1(output + features)
        
        return output


class CrossModalAttention(nn.Module):
    """跨模态注意力机制：融合空间转录组和单细胞RNA数据"""
    
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, spatial_features, sc_features):
        """
        Args:
            spatial_features: [N_spots, d_model] 空间spots特征（作为query）
            sc_features: [N_cells, d_model] 单细胞特征（作为key和value）
        
        Returns:
            output: [N_spots, d_model] 融合后的特征
        """
        N_spots, d_model = spatial_features.shape
        N_cells = sc_features.shape[0]
        
        # 生成query, key, value
        Q = self.q_proj(spatial_features)  # [N_spots, d_model]
        K = self.k_proj(sc_features)  # [N_cells, d_model]
        V = self.v_proj(sc_features)  # [N_cells, d_model]
        
        # 重塑为多头形式
        Q = Q.view(N_spots, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, N_spots, head_dim]
        K = K.view(N_cells, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, N_cells, head_dim]
        V = V.view(N_cells, self.num_heads, self.head_dim).transpose(0, 1)  # [num_heads, N_cells, head_dim]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(self.head_dim)  # [num_heads, N_spots, N_cells]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, V)  # [num_heads, N_spots, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous()  # [N_spots, num_heads, head_dim]
        attn_output = attn_output.view(N_spots, d_model)  # [N_spots, d_model]
        
        # 输出投影
        output = self.out_proj(attn_output)
        output = self.norm(output + spatial_features)  # 残差连接
        
        return output


class SpatialTransformerEncoder(nn.Module):
    """空间Transformer编码器：捕获长距离空间依赖关系"""
    
    def __init__(self, d_model, num_heads=8, dim_feedforward=2048, dropout=0.1, num_layers=2):
        super(SpatialTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, features):
        """
        Args:
            features: [N, d_model] 输入特征
        
        Returns:
            output: [N, d_model] 输出特征
        """
        # Transformer需要[batch, seq_len, d_model]格式
        x = features.unsqueeze(0)  # [1, N, d_model]
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        output = x.squeeze(0)  # [N, d_model]
        
        return output


class SINCANet(nn.Module):
    """SINCA-Net主网络"""
    
    def __init__(
        self,
        num_known_genes,
        num_unknown_genes,
        num_total_genes,
        d_model=256,
        num_heads=8,
        num_transformer_layers=2,
        dim_feedforward=2048,
        dropout=0.1
    ):
        super(SINCANet, self).__init__()
        
        self.d_model = d_model
        self.num_known_genes = num_known_genes
        self.num_unknown_genes = num_unknown_genes
        self.num_total_genes = num_total_genes
        
        # 输入投影层
        self.known_gene_proj = nn.Linear(num_known_genes, d_model)
        self.sc_gene_proj = nn.Linear(num_total_genes, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 空间卷积模块
        self.spatial_conv = SpatialConvolutionModule(d_model)
        
        # 跨模态注意力
        self.cross_modal_attn = CrossModalAttention(d_model, num_heads, dropout)
        
        # 空间Transformer编码器
        self.transformer_encoder = SpatialTransformerEncoder(
            d_model, num_heads, dim_feedforward, dropout, num_transformer_layers
        )
        
        # 输出投影层
        self.unknown_gene_proj = nn.Linear(d_model, num_unknown_genes)
        self.known_gene_refine_proj = nn.Linear(d_model, num_known_genes)  # 可选：精修已知基因
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, known_genes, coords, sc_data):
        """
        Args:
            known_genes: [N_spots, num_known_genes] 已知基因表达
            coords: [N_spots, 2] 空间坐标
            sc_data: [N_cells, num_total_genes] 单细胞参考数据
        
        Returns:
            unknown_genes: [N_spots, num_unknown_genes] 预测的未知基因表达
            refined_known_genes: [N_spots, num_known_genes] 精修后的已知基因表达（可选）
        """
        N_spots = known_genes.shape[0]
        device = known_genes.device
        
        # 1. 投影已知基因表达
        spatial_features = self.known_gene_proj(known_genes)  # [N_spots, d_model]
        
        # 2. 添加位置编码
        pos_enc = self.pos_encoding(coords)  # [N_spots, d_model]
        spatial_features = spatial_features + pos_enc
        
        # 3. 空间卷积模块
        spatial_features = self.spatial_conv(spatial_features, coords)
        spatial_features = self.dropout(spatial_features)
        
        # 4. 投影单细胞数据
        sc_features = self.sc_gene_proj(sc_data)  # [N_cells, d_model]
        
        # 5. 跨模态注意力
        spatial_features = self.cross_modal_attn(spatial_features, sc_features)
        spatial_features = self.dropout(spatial_features)
        
        # 6. 空间Transformer编码器
        spatial_features = self.transformer_encoder(spatial_features)
        spatial_features = self.dropout(spatial_features)
        
        # 7. 输出投影
        unknown_genes = self.unknown_gene_proj(spatial_features)  # [N_spots, num_unknown_genes]
        refined_known_genes = self.known_gene_refine_proj(spatial_features)  # [N_spots, num_known_genes]
        
        return unknown_genes, refined_known_genes
