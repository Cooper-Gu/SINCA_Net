"""
推理脚本：对空间转录组数据进行基因插补
"""

import os
import argparse
import torch
import numpy as np
import scanpy as sc
from tqdm import tqdm

from models import SINCANet
from data import SpatialTranscriptomicsDataset
from utils import load_checkpoint


def inference(model, dataset, device, batch_size=32):
    """推理：预测未知基因表达"""
    model.eval()
    
    # 获取所有数据
    known_genes, coords, sc_data = dataset.get_all_data()
    known_genes = known_genes.to(device)
    coords = coords.to(device)
    sc_data = sc_data.to(device)
    
    # 批量推理
    all_pred_unknown = []
    all_pred_known_refined = []
    
    num_spots = known_genes.shape[0]
    num_batches = (num_spots + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc='推理中'):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_spots)
            
            batch_known = known_genes[start_idx:end_idx]
            batch_coords = coords[start_idx:end_idx]
            
            # 前向传播
            pred_unknown, pred_known_refined = model(batch_known, batch_coords, sc_data)
            
            all_pred_unknown.append(pred_unknown.cpu().numpy())
            all_pred_known_refined.append(pred_known_refined.cpu().numpy())
    
    # 合并所有批次
    pred_unknown = np.concatenate(all_pred_unknown, axis=0)
    pred_known_refined = np.concatenate(all_pred_known_refined, axis=0)
    
    return pred_unknown, pred_known_refined


def denormalize(data, normalize_method, original_data=None):
    """反归一化数据"""
    if normalize_method == 'log1p':
        # log1p的反函数是expm1
        return np.expm1(data)
    elif normalize_method == 'standardize':
        # 需要原始数据的均值和标准差
        if original_data is not None:
            mean = original_data.mean(axis=0, keepdims=True)
            std = original_data.std(axis=0, keepdims=True) + 1e-8
            return data * std + mean
        else:
            return data
    else:
        return data


def main():
    parser = argparse.ArgumentParser(description='使用SINCA-Net进行基因插补推理')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--dataset_dir', type=str, required=True, help='数据集目录')
    parser.add_argument('--output_path', type=str, required=True, help='输出文件路径（.h5ad格式）')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--normalize', type=str, default='log1p', choices=['log1p', 'standardize', None], help='归一化方法')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--use_all_genes_as_known', action='store_true', help='使用所有基因作为已知基因（用于完整插补）')
    parser.add_argument('--n_splits', type=int, default=10, help='K折交叉验证的折数（用于确定基因划分）')
    parser.add_argument('--fold', type=int, default=0, help='使用第几折（0-9）')
    
    args = parser.parse_args()
    
    # 加载模型检查点
    print(f"加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # 从检查点获取模型配置
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        d_model = model_config.get('d_model', 256)
        num_heads = model_config.get('num_heads', 8)
        num_transformer_layers = model_config.get('num_transformer_layers', 2)
    else:
        d_model = 256
        num_heads = 8
        num_transformer_layers = 2
    
    # 加载数据集
    print(f"加载数据集: {args.dataset_dir}")
    
    if args.use_all_genes_as_known:
        # 使用所有基因作为已知基因（用于完整插补所有基因）
        # 这种情况下，我们需要知道总基因数
        temp_dataset = SpatialTranscriptomicsDataset(
            dataset_dir=args.dataset_dir,
            known_gene_indices=list(range(100)),  # 临时
            unknown_gene_indices=[],
            normalize=args.normalize,
            use_ground_truth=False
        )
        num_total_genes = temp_dataset.num_total_genes
        
        # 创建数据集，所有基因作为已知基因
        dataset = SpatialTranscriptomicsDataset(
            dataset_dir=args.dataset_dir,
            known_gene_indices=list(range(num_total_genes)),
            unknown_gene_indices=[],
            normalize=args.normalize,
            use_ground_truth=False
        )
        
        # 对于完整插补，我们需要预测所有基因
        # 但模型是为特定数量的未知基因设计的
        # 这里我们假设模型可以处理这种情况，或者我们需要重新训练
        # 为了简化，我们使用K折划分的方式
        from data import create_kfold_splits
        splits, _ = create_kfold_splits(args.dataset_dir, n_splits=args.n_splits, normalize=args.normalize)
        train_gene_indices, val_gene_indices = splits[args.fold]
        
        # 使用训练基因作为已知，验证基因作为未知
        dataset = SpatialTranscriptomicsDataset(
            dataset_dir=args.dataset_dir,
            known_gene_indices=train_gene_indices,
            unknown_gene_indices=val_gene_indices,
            normalize=args.normalize,
            use_ground_truth=False
        )
        
        num_known_genes = len(train_gene_indices)
        num_unknown_genes = len(val_gene_indices)
    else:
        # 使用K折划分
        from data import create_kfold_splits
        splits, temp_dataset = create_kfold_splits(
            args.dataset_dir, n_splits=args.n_splits, normalize=args.normalize
        )
        train_gene_indices, val_gene_indices = splits[args.fold]
        
        dataset = SpatialTranscriptomicsDataset(
            dataset_dir=args.dataset_dir,
            known_gene_indices=train_gene_indices,
            unknown_gene_indices=val_gene_indices,
            normalize=args.normalize,
            use_ground_truth=False
        )
        
        num_known_genes = len(train_gene_indices)
        num_unknown_genes = len(val_gene_indices)
        num_total_genes = temp_dataset.num_total_genes
    
    print(f"已知基因数: {num_known_genes}, 未知基因数: {num_unknown_genes}")
    
    # 创建模型
    model = SINCANet(
        num_known_genes=num_known_genes,
        num_unknown_genes=num_unknown_genes,
        num_total_genes=dataset.num_total_genes,
        d_model=d_model,
        num_heads=num_heads,
        num_transformer_layers=num_transformer_layers
    ).to(args.device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    print("模型权重已加载")
    
    # 推理
    print("\n开始推理...")
    pred_unknown, pred_known_refined = inference(model, dataset, args.device, args.batch_size)
    
    # 反归一化
    if args.normalize == 'log1p':
        pred_unknown = denormalize(pred_unknown, args.normalize)
        pred_known_refined = denormalize(pred_known_refined, args.normalize)
    
    # 构建完整的基因表达矩阵
    # 合并已知基因（使用精修版本）和未知基因
    known_genes_data = dataset.spatial_expr[:, dataset.known_gene_indices]
    if args.normalize == 'log1p':
        known_genes_data = denormalize(known_genes_data, args.normalize)
    
    # 创建完整的基因表达矩阵
    full_expr = np.zeros((dataset.num_spots, dataset.num_total_genes))
    
    # 填充已知基因（使用精修版本或原始版本）
    for i, gene_idx in enumerate(dataset.known_gene_indices):
        # 可以选择使用精修版本或原始版本
        # 这里使用精修版本
        full_expr[:, gene_idx] = pred_known_refined[:, i]
        # 或者使用原始版本：
        # full_expr[:, gene_idx] = known_genes_data[:, i]
    
    # 填充未知基因
    for i, gene_idx in enumerate(dataset.unknown_gene_indices):
        full_expr[:, gene_idx] = pred_unknown[:, i]
    
    # 创建AnnData对象
    print("创建输出AnnData对象...")
    output_adata = sc.AnnData(X=full_expr)
    output_adata.var_names = dataset.gene_names
    output_adata.obs_names = dataset.spatial_adata.obs_names
    
    # 添加空间坐标
    output_adata.obsm['spatial'] = dataset.coords
    output_adata.obs['x'] = dataset.coords[:, 0]
    output_adata.obs['y'] = dataset.coords[:, 1]
    
    # 保存
    print(f"保存结果到: {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    output_adata.write(args.output_path)
    
    print(f"\n推理完成！")
    print(f"  输出文件: {args.output_path}")
    print(f"  数据形状: {output_adata.shape} (spots × genes)")
    print(f"  包含基因数: {len(output_adata.var_names)}")


if __name__ == '__main__':
    main()
