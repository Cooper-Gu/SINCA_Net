"""
单数据集评估脚本
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from models import SINCANet
from data import create_dataloader, create_kfold_splits
from utils import load_checkpoint, calculate_all_metrics, save_metrics


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    
    all_pred_unknown = []
    all_true_unknown = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
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
    
    return metrics, all_pred_unknown, all_true_unknown


def main():
    parser = argparse.ArgumentParser(description='评估SINCA-Net模型')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--dataset_dir', type=str, required=True, help='数据集目录')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--n_splits', type=int, default=10, help='K折交叉验证的折数')
    parser.add_argument('--fold', type=int, default=0, help='使用第几折（0-9）')
    parser.add_argument('--normalize', type=str, default='log1p', choices=['log1p', 'standardize', None], help='归一化方法')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型检查点获取配置信息
    print(f"加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # 创建K折划分
    print("创建K折交叉验证划分...")
    splits, temp_dataset = create_kfold_splits(args.dataset_dir, n_splits=args.n_splits, normalize=args.normalize)
    
    # 使用指定的fold
    train_gene_indices, val_gene_indices = splits[args.fold]
    print(f"使用第 {args.fold} 折")
    print(f"训练基因数: {len(train_gene_indices)}, 验证基因数: {len(val_gene_indices)}")
    
    # 创建数据加载器
    print("创建数据加载器...")
    val_loader, val_dataset = create_dataloader(
        dataset_dir=args.dataset_dir,
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
    
    # 从检查点获取模型配置（如果存在）
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        d_model = model_config.get('d_model', 256)
        num_heads = model_config.get('num_heads', 8)
        num_transformer_layers = model_config.get('num_transformer_layers', 2)
    else:
        # 使用默认配置
        d_model = 256
        num_heads = 8
        num_transformer_layers = 2
    
    print(f"创建模型: 已知基因={num_known_genes}, 未知基因={num_unknown_genes}, 总基因数={num_total_genes}")
    
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
    print("模型权重已加载")
    
    # 评估
    print("\n开始评估...")
    metrics, pred_unknown, true_unknown = evaluate(model, val_loader, args.device)
    
    # 打印结果
    print("\n评估结果:")
    print(f"  PCC (mean): {metrics['PCC_mean']:.4f}")
    print(f"  SSIM: {metrics['SSIM']:.4f}")
    print(f"  RMAE: {metrics['RMAE']:.6f}")
    print(f"  JS: {metrics['JS']:.4f}")
    print(f"  ACC: {metrics['ACC']:.4f}")
    
    # 保存结果
    dataset_name = os.path.basename(args.dataset_dir.rstrip('/\\'))
    output_file = os.path.join(args.output_dir, f'{dataset_name}_fold{args.fold}_metrics.json')
    save_metrics(metrics, output_file)
    print(f"\n指标已保存到: {output_file}")
    
    # 保存预测结果
    pred_file = os.path.join(args.output_dir, f'{dataset_name}_fold{args.fold}_predictions.npy')
    true_file = os.path.join(args.output_dir, f'{dataset_name}_fold{args.fold}_ground_truth.npy')
    np.save(pred_file, pred_unknown)
    np.save(true_file, true_unknown)
    print(f"预测结果已保存到: {pred_file}")
    print(f"真实值已保存到: {true_file}")


if __name__ == '__main__':
    main()
