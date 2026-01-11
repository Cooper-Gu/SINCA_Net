"""
推理脚本
用于对空间转录组数据进行插补推理，输出扩展到全部基因
"""

import os
import argparse
import torch
import numpy as np
import json
from tqdm import tqdm

from models.sinca_net import create_model
from data.dataloader import create_dataloader
from utils.utils import (
    load_checkpoint, get_device,
    expand_predictions_to_full_genes
)
import scanpy as sc


def inference(
    model_path,
    dataset_dir,
    output_path,
    normalize='log1p',
    use_common_genes=True,
    mask_ratio=0.0  # 推理时不进行掩码
):
    """
    对空间转录组数据进行推理，输出扩展到全部基因
    
    Args:
        model_path: 模型检查点路径
        dataset_dir: 数据集目录
        output_path: 输出文件路径（.h5ad格式，推荐）
        normalize: 归一化方法
        use_common_genes: 是否使用共同基因（推理时输入）
        mask_ratio: 掩码比例（推理时设为0）
    """
    device = get_device()
    print(f"使用设备: {device}")
    
    # 数据路径
    spatial_data_path = os.path.join(dataset_dir, 'Spatial_count.h5ad')
    sc_data_path = os.path.join(dataset_dir, 'scRNA_count_cluster.h5ad')
    
    if not os.path.exists(sc_data_path):
        sc_data_path = None
        print("警告: 未找到单细胞数据，将不使用跨模态注意力")
    
    # 创建数据加载器
    print("加载数据...")
    dataloader, dataset = create_dataloader(
        spatial_data_path=spatial_data_path,
        sc_data_path=sc_data_path,
        mask_ratio=mask_ratio,
        normalize=normalize if normalize != 'none' else None,
        use_common_genes=use_common_genes,
        batch_size=1,
        shuffle=False,
        random_seed=42
    )
    
    num_genes = dataset.num_genes
    print(f"模型输入基因数量: {num_genes}")
    
    # 检查是否有原始基因信息
    if hasattr(dataset, 'original_spatial_genes'):
        print(f"空间转录组原始基因数量: {len(dataset.original_spatial_genes)}")
        if hasattr(dataset, 'common_genes'):
            print(f"共同基因数量: {len(dataset.common_genes)}")
    
    # 加载模型配置
    checkpoint_dir = os.path.dirname(model_path)
    config_path = os.path.join(checkpoint_dir, 'config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        model_config = {
            'd_model': config.get('d_model', 512),
            'num_heads': config.get('num_heads', 8),
            'num_layers': config.get('num_layers', 6),
            'dropout': config.get('dropout', 0.1),
            'use_spatial_conv': config.get('use_spatial_conv', True),
            'use_cross_modal_attention': config.get('use_cross_modal_attention', sc_data_path is not None),
            'use_transformer': config.get('use_transformer', True)
        }
    else:
        # 使用默认配置
        model_config = {
            'd_model': 512,
            'num_heads': 8,
            'num_layers': 6,
            'dropout': 0.1,
            'use_spatial_conv': True,
            'use_cross_modal_attention': sc_data_path is not None,
            'use_transformer': True
        }
    
    # 创建模型
    print("创建模型...")
    model = create_model(num_genes, config=model_config)
    model = model.to(device)
    
    # 加载权重
    checkpoint = load_checkpoint(model, model_path, device=device)
    print(f"模型已加载 (epoch: {checkpoint.get('epoch', 'unknown')})")
    
    # 推理
    print("开始推理...")
    model.eval()
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        spatial_expr = batch['spatial_expr'].to(device)
        spatial_coords = batch['spatial_coords'].to(device)
        
        inputs = {
            'spatial_expr': spatial_expr,
            'spatial_coords': spatial_coords
        }
        
        if 'sc_expr' in batch and batch['sc_expr'] is not None:
            inputs['sc_expr'] = batch['sc_expr'].to(device)
            if 'sc_cluster' in batch and batch['sc_cluster'] is not None:
                inputs['sc_cluster'] = batch['sc_cluster'].to(device)
        
        # 模型预测（共同基因）
        imputed_expr = model(**inputs)  # [num_spots, num_common_genes]
        imputed_expr_np = imputed_expr.cpu().numpy()
        
        print(f"共同基因预测完成，形状: {imputed_expr_np.shape}")
    
    # 扩展到全部基因
    if hasattr(dataset, 'common_genes') and hasattr(dataset, 'original_spatial_genes'):
        print("扩展到空间转录组的全部基因...")
        
        # 获取原始空间转录组表达（用于填充非共同基因）
        original_expr = None
        if hasattr(dataset, 'original_spatial_expr'):
            original_expr = dataset.original_spatial_expr
            # 如果进行了归一化，需要反归一化（这里假设使用log1p）
            if normalize == 'log1p' and original_expr is not None:
                # 注意：这里假设original_spatial_expr已经是归一化后的
                # 如果需要，可以从原始数据重新加载
                pass
        
        # 扩展到全部基因
        full_predictions, gene_mapping = expand_predictions_to_full_genes(
            imputed_expr_np,
            dataset.common_genes,
            dataset.original_spatial_genes,
            original_spatial_expr=original_expr,
            fill_non_common_with_original=True
        )
        
        print(f"  共同基因数量: {gene_mapping['num_common_genes']}")
        print(f"  全部基因数量: {gene_mapping['num_original_genes']}")
        print(f"  非共同基因数量: {len(gene_mapping['non_common_genes'])}")
        print(f"  扩展后预测形状: {full_predictions.shape}")
        
        # 保存结果
        # 创建AnnData对象保存结果
        result_adata = sc.AnnData(X=full_predictions)
        result_adata.var_names = dataset.original_spatial_genes
        result_adata.obs_names = dataset.spatial_adata.obs_names[:full_predictions.shape[0]]
        
        # 添加空间坐标
        if hasattr(dataset, 'spatial_coords'):
            result_adata.obsm['spatial'] = dataset.spatial_coords
            result_adata.obs['x'] = dataset.spatial_coords[:, 0]
            result_adata.obs['y'] = dataset.spatial_coords[:, 1]
        
        # 添加元数据
        result_adata.uns['inference_info'] = {
            'model_path': model_path,
            'common_genes_count': gene_mapping['num_common_genes'],
            'total_genes_count': gene_mapping['num_original_genes'],
            'non_common_genes_count': len(gene_mapping['non_common_genes']),
            'normalize_method': normalize
        }
        
        # 保存为h5ad文件
        result_adata.write(output_path)
        print(f"\n✓ 推理完成！结果已保存到: {output_path}")
        
        # 同时保存基因映射信息
        mapping_path = output_path.replace('.h5ad', '_gene_mapping.json')
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(gene_mapping, f, indent=2, ensure_ascii=False)
        print(f"  基因映射信息已保存到: {mapping_path}")
        
        # 保存numpy格式（可选）
        npy_path = output_path.replace('.h5ad', '.npy')
        np.save(npy_path, full_predictions)
        print(f"  NumPy格式已保存到: {npy_path}")
        
    else:
        # 如果没有基因映射信息，直接保存共同基因的预测结果
        print("警告: 无法获取基因映射信息，保存共同基因的预测结果")
        result_adata = sc.AnnData(X=imputed_expr_np)
        if hasattr(dataset, 'spatial_genes'):
            result_adata.var_names = dataset.spatial_genes
        if hasattr(dataset, 'spatial_coords'):
            result_adata.obsm['spatial'] = dataset.spatial_coords
        
        result_adata.write(output_path)
        print(f"\n✓ 推理完成！结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='SINCA-Net推理脚本 - 输出扩展到全部基因')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='数据集目录路径 (例如: dataset/Dataset1)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出文件路径 (推荐.h5ad格式，例如: results/imputed_spatial.h5ad)')
    parser.add_argument('--normalize', type=str, default='log1p',
                        choices=['log1p', 'minmax', 'zscore', 'none'],
                        help='归一化方法 (默认: log1p，应与训练时一致)')
    parser.add_argument('--use_common_genes', action='store_true', default=True,
                        help='使用共同基因（推理时输入，默认启用）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 执行推理
    inference(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        output_path=args.output_path,
        normalize=args.normalize,
        use_common_genes=args.use_common_genes,
        mask_ratio=0.0
    )


if __name__ == '__main__':
    main()
