"""
SINCA-Net: Spatial Inpainting Network with Cross-Modal Attention
用于空间转录组基因插补的深度学习网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InputProjection(nn.Module):
    """输入投影层 - 支持模态嵌入和模态特定适配"""
    def __init__(self, num_genes, d_model, use_modality_embedding=True, use_modality_adapter=True):
        super(InputProjection, self).__init__()
        self.d_model = d_model
        self.use_modality_embedding = use_modality_embedding
        self.use_modality_adapter = use_modality_adapter
        
        # 共用投影层
        self.projection = nn.Linear(num_genes, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        # 模态嵌入（用于区分空间和单细胞数据）
        if use_modality_embedding:
            # 0: 空间转录组, 1: 单细胞
            self.modality_embedding = nn.Embedding(2, d_model)
        
        # 模态特定适配层（轻量级，用于学习模态特定的特征变换）
        if use_modality_adapter:
            # 空间转录组适配器
            self.spatial_adapter = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 4, d_model)
            )
            # 单细胞适配器
            self.sc_adapter = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 4, d_model)
            )
            # 适配器归一化
            self.adapter_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, modality='spatial'):
        """
        Args:
            x: [batch_size, num_spots/cells, num_genes]
            modality: 'spatial' 或 'sc'，指定数据模态
        """
        # 共用投影
        x = self.projection(x)  # [batch_size, num_spots/cells, d_model]
        
        # 添加模态嵌入
        if self.use_modality_embedding:
            modality_id = 0 if modality == 'spatial' else 1
            modality_emb = self.modality_embedding(
                torch.tensor(modality_id, device=x.device)
            )  # [d_model]
            x = x + modality_emb.unsqueeze(0).unsqueeze(0)  # 广播到所有位置
        
        # 模态特定适配
        if self.use_modality_adapter:
            if modality == 'spatial':
                adapter_output = self.spatial_adapter(x)
            else:  # modality == 'sc'
                adapter_output = self.sc_adapter(x)
            # 残差连接 + 归一化
            x = self.adapter_norm(x + adapter_output)
        else:
            x = self.norm(x)
        
        return x


class SpatialConvModule(nn.Module):
    """空间卷积模块 - 捕获局部空间模式"""
    def __init__(self, d_model, kernel_size=3):
        super(SpatialConvModule, self).__init__()
        self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.norm1 = nn.BatchNorm2d(d_model)
        self.norm2 = nn.BatchNorm2d(d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x_spatial):
        # x_spatial: [batch_size, d_model, H, W]
        residual = x_spatial
        x = self.conv1(x_spatial)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        x = self.relu(x)
        return x


class CrossModalAttention(nn.Module):
    """跨模态注意力机制 - 融合单细胞参考信息"""
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, spatial_features, sc_features):
        """
        Args:
            spatial_features: [batch_size, num_spots, d_model] 空间转录组特征
            sc_features: [batch_size, num_cells, d_model] 单细胞RNA特征
        Returns:
            attended_features: [batch_size, num_spots, d_model] 跨模态注意力后的特征
        """
        batch_size = spatial_features.size(0)
        
        # 计算query, key, value
        Q = self.query(spatial_features)  # [batch_size, num_spots, d_model]
        K = self.key(sc_features)  # [batch_size, num_cells, d_model]
        V = self.value(sc_features)  # [batch_size, num_cells, d_model]
        
        # 多头注意力
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, num_spots, head_dim]
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, num_cells, head_dim]
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, num_cells, head_dim]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, num_heads, num_spots, num_cells]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attended = torch.matmul(attn_weights, V)  # [batch_size, num_heads, num_spots, head_dim]
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [batch_size, num_spots, d_model]
        
        # 输出投影和残差连接
        output = self.out_proj(attended)
        output = self.norm(output + spatial_features)
        
        return output


class SpatialTransformerEncoder(nn.Module):
    """空间Transformer编码器 - 捕获长距离空间依赖"""
    def __init__(self, d_model, num_heads=8, d_ff=2048, dropout=0.1, num_layers=6):
        super(SpatialTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: [batch_size, num_spots, d_model]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 自注意力
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = self.dropout(x)
        x = x + residual
        
        # 前馈网络
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + residual
        
        return x


class OutputLayer(nn.Module):
    """输出层"""
    def __init__(self, d_model, num_genes):
        super(OutputLayer, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, num_genes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x: [batch_size, num_spots, d_model]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SINCANet(nn.Module):
    """
    SINCA-Net: Spatial Inpainting Network with Cross-Modal Attention
    
    网络架构:
    输入投影层 → 空间卷积模块(可选) → 跨模态注意力(可选) → 
    空间Transformer编码器(可选) → 输出层
    """
    def __init__(
        self,
        num_genes,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
        use_spatial_conv=True,
        use_cross_modal_attention=True,
        use_transformer=True,
        grid_size=None  # (H, W) 用于空间卷积的网格大小
    ):
        super(SINCANet, self).__init__()
        self.num_genes = num_genes
        self.d_model = d_model
        self.use_spatial_conv = use_spatial_conv
        self.use_cross_modal_attention = use_cross_modal_attention
        self.use_transformer = use_transformer
        self.grid_size = grid_size
        
        # 输入投影层（支持模态嵌入和适配器）
        self.input_proj = InputProjection(
            num_genes, d_model,
            use_modality_embedding=True,  # 启用模态嵌入
            use_modality_adapter=True     # 启用模态适配器
        )
        
        # 空间卷积模块
        if use_spatial_conv:
            self.spatial_conv = SpatialConvModule(d_model)
            self.spatial_unflatten = None  # 将在forward中动态设置
        
        # 跨模态注意力
        if use_cross_modal_attention:
            self.cross_modal_attn = CrossModalAttention(d_model, num_heads, dropout)
        
        # 空间Transformer编码器
        if use_transformer:
            self.transformer_encoder = SpatialTransformerEncoder(
                d_model, num_heads, d_ff, dropout, num_layers
            )
        
        # 输出层
        self.output_layer = OutputLayer(d_model, num_genes)
        
    def _create_spatial_grid(self, coords, H, W):
        """
        将空间坐标转换为2D网格
        Args:
            coords: [num_spots, 2] (x, y) 坐标
            H, W: 网格高度和宽度
        Returns:
            grid: [num_spots] 网格索引
        """
        # 归一化坐标到[0, 1]
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        x_norm = (coords[:, 0] - x_min) / (x_max - x_min + 1e-8)
        y_norm = (coords[:, 1] - y_min) / (y_max - y_min + 1e-8)
        
        # 转换为网格索引
        grid_x = (x_norm * (W - 1)).long().clamp(0, W - 1)
        grid_y = (y_norm * (H - 1)).long().clamp(0, H - 1)
        
        return grid_y * W + grid_x
    
    def forward(self, spatial_expr, spatial_coords=None, sc_expr=None, sc_cluster=None):
        """
        Args:
            spatial_expr: [batch_size, num_spots, num_genes] 或 [num_spots, num_genes] 空间转录组表达
            spatial_coords: [batch_size, num_spots, 2] 或 [num_spots, 2] 空间坐标 (可选)
            sc_expr: [batch_size, num_cells, num_genes] 或 [num_cells, num_genes] 单细胞RNA表达 (可选)
            sc_cluster: [batch_size, num_cells] 或 [num_cells] 单细胞聚类标签 (可选)
        Returns:
            imputed_expr: [batch_size, num_spots, num_genes] 或 [num_spots, num_genes] 插补后的表达
        """
        # 处理没有batch维度的情况
        if spatial_expr.dim() == 2:
            spatial_expr = spatial_expr.unsqueeze(0)
            add_batch_dim = True
        else:
            add_batch_dim = False
        
        if spatial_coords is not None and spatial_coords.dim() == 2:
            spatial_coords = spatial_coords.unsqueeze(0)
        
        if sc_expr is not None and sc_expr.dim() == 2:
            sc_expr = sc_expr.unsqueeze(0)
        
        if sc_cluster is not None and sc_cluster.dim() == 1:
            sc_cluster = sc_cluster.unsqueeze(0)
        
        batch_size, num_spots, num_genes = spatial_expr.shape
        
        # 输入投影（空间转录组）
        x = self.input_proj(spatial_expr, modality='spatial')  # [batch_size, num_spots, d_model]
        
        # 空间卷积模块
        if self.use_spatial_conv and spatial_coords is not None:
            # 确定网格大小（限制最大尺寸以节省内存）
            if self.grid_size is None:
                # 自动计算网格大小，但限制最大尺寸
                max_grid_size = 256  # 限制最大网格尺寸
                grid_dim = min(int(math.sqrt(num_spots) * 2), max_grid_size)
                H = grid_dim
                W = grid_dim
            else:
                H, W = self.grid_size
            
            # 为每个样本创建2D网格表示
            batch_features_2d = []
            for b in range(batch_size):
                coords = spatial_coords[b]  # [num_spots, 2]
                grid_indices = self._create_spatial_grid(coords, H, W)  # [num_spots]
                
                # 创建2D特征图
                feature_2d = torch.zeros(H * W, self.d_model, device=x.device, dtype=x.dtype)
                feature_2d[grid_indices] = x[b]  # 将特征放入对应位置
                feature_2d = feature_2d.view(H, W, self.d_model).permute(2, 0, 1)  # [d_model, H, W]
                
                batch_features_2d.append(feature_2d)
                
                # 及时释放中间变量
                del feature_2d, grid_indices
            
            # 堆叠并应用卷积
            x_2d = torch.stack(batch_features_2d, dim=0)  # [batch_size, d_model, H, W]
            del batch_features_2d  # 释放列表内存
            
            x_2d = self.spatial_conv(x_2d)  # [batch_size, d_model, H, W]
            
            # 转换回序列形式
            x_2d = x_2d.permute(0, 2, 3, 1)  # [batch_size, H, W, d_model]
            x = x_2d.reshape(batch_size, H * W, self.d_model)
            del x_2d  # 释放2D特征图内存
            
            # 只保留有效spot的特征
            if self.grid_size is None or True:  # 总是重新映射以节省内存
                # 重新映射回原始spot顺序
                x_resampled = []
                for b in range(batch_size):
                    coords = spatial_coords[b]
                    grid_indices = self._create_spatial_grid(coords, H, W)
                    x_resampled.append(x[b, grid_indices])
                    del grid_indices
                x = torch.stack(x_resampled, dim=0)
                del x_resampled
        
        # 跨模态注意力
        if self.use_cross_modal_attention and sc_expr is not None:
            # 对单细胞表达进行投影
            # 如果单细胞数据太大，可以随机采样以减少内存使用
            # sc_expr形状: [batch_size, num_cells, num_genes]
            max_cells = 5000  # 限制最大细胞数
            if sc_expr.shape[1] > max_cells:
                # 随机采样max_cells个细胞（在细胞维度上采样）
                num_cells_to_use = max_cells
                indices = torch.randperm(sc_expr.shape[1], device=sc_expr.device)[:num_cells_to_use]
                sc_expr = sc_expr[:, indices, :]  # [batch_size, max_cells, num_genes]
                if sc_cluster is not None:
                    sc_cluster = sc_cluster[:, indices]
            
            sc_proj = self.input_proj(sc_expr, modality='sc')  # [batch_size, num_cells, d_model]
            x = self.cross_modal_attn(x, sc_proj)
            del sc_proj  # 释放内存
        
        # 空间Transformer编码器
        if self.use_transformer:
            x = self.transformer_encoder(x)
        
        # 输出层
        imputed_expr = self.output_layer(x)  # [batch_size, num_spots, num_genes]
        
        # 如果输入没有batch维度，移除batch维度
        if add_batch_dim:
            imputed_expr = imputed_expr.squeeze(0)  # [num_spots, num_genes]
        
        return imputed_expr


def create_model(num_genes, config=None):
    """
    创建SINCA-Net模型的便捷函数
    
    Args:
        num_genes: 基因数量
        config: 配置字典，包含模型超参数
    
    Returns:
        model: SINCA-Net模型实例
    """
    if config is None:
        config = {}
    
    default_config = {
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 2048,
        'dropout': 0.1,
        'use_spatial_conv': True,
        'use_cross_modal_attention': True,
        'use_transformer': True,
        'grid_size': None
    }
    
    default_config.update(config)
    
    model = SINCANet(num_genes=num_genes, **default_config)
    return model



if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    # 设置参数
    num_genes = 2000  # 基因数量
    num_spots = 1000  # 空间spot数量
    num_cells = 3000  # 单细胞数量
    batch_size = 1

    # 创建模型
    print("创建模型...")
    model_config = {
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 4,
        'd_ff': 1024,
        'dropout': 0.1,
        'use_spatial_conv': True,
        'use_cross_modal_attention': True,
        'use_transformer': True,
        'grid_size': None
    }
    model = create_model(num_genes, config=model_config)
    model = model.to(device)
    model.eval()  # 设置为评估模式

    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}\n")

    # 生成示例数据
    print("生成示例数据...")
    # 空间转录组表达数据
    spatial_expr = torch.randn(batch_size, num_spots, num_genes).to(device)
    print(f"空间转录组表达形状: {spatial_expr.shape}")

    # 空间坐标
    spatial_coords = torch.rand(batch_size, num_spots, 2).to(device) * 100  # 坐标范围[0, 100]
    print(f"空间坐标形状: {spatial_coords.shape}")

    # 单细胞RNA表达数据
    sc_expr = torch.randn(batch_size, num_cells, num_genes).to(device)
    print(f"单细胞RNA表达形状: {sc_expr.shape}")

    # 单细胞聚类标签（可选）
    sc_cluster = torch.randint(0, 10, (batch_size, num_cells)).to(device)
    print(f"单细胞聚类标签形状: {sc_cluster.shape}\n")

    # 调用网络进行前向传播
    print("进行前向传播...")
    with torch.no_grad():
        output = model(
            spatial_expr=spatial_expr,
            spatial_coords=spatial_coords,
            sc_expr=sc_expr,
            sc_cluster=sc_cluster
        )

    print(f"输出形状: {output.shape}")
    print(f"输出统计信息:")
    print(f"  最小值: {output.min().item():.4f}")
    print(f"  最大值: {output.max().item():.4f}")
    print(f"  平均值: {output.mean().item():.4f}")
    print(f"  标准差: {output.std().item():.4f}\n")

    # 测试不使用单细胞数据的情况
    print("测试不使用单细胞数据的情况...")
    with torch.no_grad():
        output_no_sc = model(
            spatial_expr=spatial_expr,
            spatial_coords=spatial_coords,
            sc_expr=None,
            sc_cluster=None
        )
    print(f"输出形状: {output_no_sc.shape}\n")

    # 测试不使用空间坐标的情况
    print("测试不使用空间坐标的情况...")
    with torch.no_grad():
        output_no_coords = model(
            spatial_expr=spatial_expr,
            spatial_coords=None,
            sc_expr=sc_expr,
            sc_cluster=sc_cluster
        )
    print(f"输出形状: {output_no_coords.shape}\n")

    print("=" * 60)
    print("网络调用演示完成！")
    print("=" * 60)