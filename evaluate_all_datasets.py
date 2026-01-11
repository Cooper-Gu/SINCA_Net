"""
批量评估所有数据集
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob

from models import SINCANet
from data import create_dataloader, create_kfold_splits
from utils import load_checkpoint, calculate_all_metrics, save_metrics


def evaluate_dataset(model, dataloader, device):
    """评估单个数据集"""
    model.eval()
    
    all_pred_unknown = []
    all_true_unknown = []
    
    with torch.no_grad():
        for batch in dataloader:
            known_genes, coords, unknown_genes, sc_data = batch
            
            known_genes = known_genes.to(device)
            coords = coords.to(device)
            unknown_genes = unknown_genes.to(device)
            sc_data = sc_data.to(device)
            
            # 前向传播
            pred_unknown, _ = model(known_genes, coords, sc_data)
            
            # 收集预测和真实值
            all_pred_unknown.append(pred_unknown.cpu().numpy())
            all_true_unknown.append(unknown_genes.cpu().numpy())
    
    # 合并所有批次
    all_pred_unknown = np.concatenate(all_pred_unknown, axis=0)
    all_true_unknown = np.concatenate(all_true_unknown, axis=0)
    
    # 计算指标
    metrics = calculate_all_metrics(all_pred_unknown, all_true_unknown)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='批量评估SINCA-Net模型')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--dataset_root', type=str, required=True, help='数据集根目录')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--n_splits', type=int, default=10, help='K折交叉验证的折数')
    parser.add_argument('--fold', type=int, default=0, help='使用第几折（0-9）')
    parser.add_argument('--normalize', type=str, default='log1p', choices=['log1p', 'standardize', None], help='归一化方法')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # 查找所有数据集目录
    dataset_dirs = []
    for item in os.listdir(args.dataset_root):
        item_path = os.path.join(args.dataset_root, item)
        if os.path.isdir(item_path) and item.startswith('Dataset'):
            spatial_file = os.path.join(item_path, 'Spatial_count.h5ad')
            scrna_file = os.path.join(item_path, 'scRNA_count_cluster.h5ad')
            if os.path.exists(spatial_file) and os.path.exists(scrna_file):
                dataset_dirs.append(item_path)
    
    dataset_dirs = sorted(dataset_dirs)
    print(f"找到 {len(dataset_dirs)} 个数据集")
    
    # 存储所有结果
    all_results = []
    
    # 评估每个数据集
    for dataset_dir in tqdm(dataset_dirs, desc='评估数据集'):
        dataset_name = os.path.basename(dataset_dir)
        print(f"\n评估数据集: {dataset_name}")
        
        try:
            # 创建K折划分
            splits, temp_dataset = create_kfold_splits(
                dataset_dir, n_splits=args.n_splits, normalize=args.normalize
            )
            
            # 使用指定的fold
            train_gene_indices, val_gene_indices = splits[args.fold]
            
            # 创建数据加载器
            val_loader, val_dataset = create_dataloader(
                dataset_dir=dataset_dir,
                known_gene_indices=train_gene_indices,
                unknown_gene_indices=val_gene_indices,
                batch_size=args.batch_size,
                shuffle=False,
                normalize=args.normalize,
                use_ground_truth=True
            )
            
            # 创建模型
            num_known_genes = len(train_gene_indices)
            num_unknown_genes = len(val_gene_indices)
            num_total_genes = temp_dataset.num_total_genes
            
            model = SINCANet(
                num_known_genes=num_known_genes,
                num_unknown_genes=num_unknown_genes,
                num_total_genes=num_total_genes,
                d_model=d_model,
                num_heads=num_heads,
                num_transformer_layers=num_transformer_layers
            ).to(args.device)
            
            # 加载模型权重
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 评估
            metrics = evaluate_dataset(model, val_loader, args.device)
            
            # 保存结果
            result = {
                'Dataset': dataset_name,
                'PCC_mean': metrics['PCC_mean'],
                'SSIM': metrics['SSIM'],
                'RMAE': metrics['RMAE'],
                'JS': metrics['JS'],
                'ACC': metrics['ACC']
            }
            all_results.append(result)
            
            # 保存单个数据集的详细指标
            output_file = os.path.join(args.output_dir, f'{dataset_name}_fold{args.fold}_metrics.json')
            save_metrics(metrics, output_file)
            
            print(f"  PCC: {metrics['PCC_mean']:.4f}, SSIM: {metrics['SSIM']:.4f}, RMAE: {metrics['RMAE']:.6f}")
            
        except Exception as e:
            print(f"  错误: {str(e)}")
            result = {
                'Dataset': dataset_name,
                'PCC_mean': np.nan,
                'SSIM': np.nan,
                'RMAE': np.nan,
                'JS': np.nan,
                'ACC': np.nan,
                'Error': str(e)
            }
            all_results.append(result)
    
    # 保存汇总结果
    df = pd.DataFrame(all_results)
    summary_file = os.path.join(args.output_dir, f'all_datasets_fold{args.fold}_summary.csv')
    df.to_csv(summary_file, index=False)
    print(f"\n汇总结果已保存到: {summary_file}")
    
    # 打印统计信息
    print("\n评估统计:")
    print(f"  数据集数量: {len(all_results)}")
    if len(df) > 0:
        print(f"  平均PCC: {df['PCC_mean'].mean():.4f} ± {df['PCC_mean'].std():.4f}")
        print(f"  平均SSIM: {df['SSIM'].mean():.4f} ± {df['SSIM'].std():.4f}")
        print(f"  平均RMAE: {df['RMAE'].mean():.6f} ± {df['RMAE'].std():.6f}")
        print(f"  平均JS: {df['JS'].mean():.4f} ± {df['JS'].std():.4f}")
        print(f"  平均ACC: {df['ACC'].mean():.4f} ± {df['ACC'].std():.4f}")


if __name__ == '__main__':
    main()
