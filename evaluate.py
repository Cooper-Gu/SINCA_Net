"""
评估脚本
用于评估训练好的SINCA-Net模型在单个数据集上的性能
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json

from models.sinca_net import create_model
from data.dataloader import create_dataloader
from utils.utils import (
    load_checkpoint, get_device, save_metrics, load_metrics,
    expand_predictions_to_full_genes
)
from utils.metrics import (
    compute_all_metrics, compute_metrics_per_gene,
    pearson_correlation_coefficient, structural_similarity_index,
    root_mean_absolute_error, cosine_similarity
)


def evaluate(model, dataloader, device, save_preds=False, output_dir=None, 
             expand_to_full_genes=False, dataset=None):
    """
    评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        save_preds: 是否保存预测结果
        output_dir: 输出目录
        expand_to_full_genes: 是否扩展到空间转录组的全部基因
        dataset: 数据集对象（用于获取基因映射信息）
    
    Returns:
        metrics: 评估指标字典
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='评估中'):
            # 获取数据
            spatial_expr = batch['spatial_expr'].to(device)
            spatial_coords = batch['spatial_coords'].to(device)
            target = batch['target'].to(device)
            mask = batch['mask'].to(device)
            
            # 准备输入
            inputs = {
                'spatial_expr': spatial_expr,
                'spatial_coords': spatial_coords
            }
            
            # 如果有单细胞数据，添加到输入
            if 'sc_expr' in batch and batch['sc_expr'] is not None:
                inputs['sc_expr'] = batch['sc_expr'].to(device)
                if 'sc_cluster' in batch and batch['sc_cluster'] is not None:
                    inputs['sc_cluster'] = batch['sc_cluster'].to(device)
            
            # 前向传播
            pred = model(**inputs)
            
            # 收集预测、目标和掩码（移到CPU以节省GPU内存）
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            all_preds.append(pred_np)
            all_targets.append(target_np)
            all_masks.append(mask_np)
    
    # 合并所有batch
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    # 只对掩码位置计算指标（mask=True表示缺失位置，需要预测）
    # 对于整体指标，展平后只取掩码位置
    masked_preds_flat = all_preds[all_masks]
    masked_targets_flat = all_targets[all_masks]
    
    # 整体指标（对所有掩码位置）
    metrics = compute_all_metrics(masked_targets_flat, masked_preds_flat, prefix='')
    
    # 按基因计算指标（对每个基因，只考虑该基因被掩码的位置）
    # 保持 [num_spots, num_genes] 形状，但只对掩码位置计算
    metrics_per_gene = {}
    num_genes = all_preds.shape[1]
    for key in ['PCC', 'SSIM', 'RMAE', 'COSSIM']:
        metrics_per_gene[key] = []
    
    for i in range(num_genes):
        # 获取该基因的掩码位置
        gene_mask = all_masks[:, i]
        if np.sum(gene_mask) == 0:
            # 如果该基因没有掩码位置，跳过或使用默认值
            metrics_per_gene['PCC'].append(0.0)
            metrics_per_gene['SSIM'].append(0.0)
            metrics_per_gene['RMAE'].append(0.0)
            metrics_per_gene['COSSIM'].append(0.0)
            continue
        
        gene_pred = all_preds[gene_mask, i]
        gene_target = all_targets[gene_mask, i]
        
        # 计算该基因的指标
        metrics_per_gene['PCC'].append(pearson_correlation_coefficient(gene_target, gene_pred))
        try:
            metrics_per_gene['SSIM'].append(structural_similarity_index(gene_target, gene_pred))
        except:
            metrics_per_gene['SSIM'].append(0.0)
        metrics_per_gene['RMAE'].append(root_mean_absolute_error(gene_target, gene_pred))
        metrics_per_gene['COSSIM'].append(cosine_similarity(gene_target, gene_pred))
    for key, values in metrics_per_gene.items():
        metrics[f'{key}_mean'] = np.mean(values)
        metrics[f'{key}_std'] = np.std(values)
        metrics[f'{key}_per_gene'] = values.tolist()
    
    # 保存预测结果
    if save_preds and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 如果扩展到全部基因
        if expand_to_full_genes and dataset is not None:
            # 检查是否有共同基因映射信息
            if hasattr(dataset, 'common_genes') and hasattr(dataset, 'original_spatial_genes'):
                print("扩展到空间转录组的全部基因...")
                # 获取原始空间转录组表达（用于填充非共同基因）
                original_expr = None
                if hasattr(dataset, 'original_spatial_expr'):
                    original_expr = dataset.original_spatial_expr
                
                # 扩展到全部基因
                full_predictions, gene_mapping = expand_predictions_to_full_genes(
                    all_preds,
                    dataset.common_genes,
                    dataset.original_spatial_genes,
                    original_spatial_expr=original_expr,
                    fill_non_common_with_original=True
                )
                
                # 保存扩展后的预测结果
                np.save(os.path.join(output_dir, 'predictions_full_genes.npy'), full_predictions)
                
                # 保存基因映射信息
                import json
                with open(os.path.join(output_dir, 'gene_mapping.json'), 'w', encoding='utf-8') as f:
                    json.dump(gene_mapping, f, indent=2, ensure_ascii=False)
                
                print(f"  共同基因数量: {gene_mapping['num_common_genes']}")
                print(f"  全部基因数量: {gene_mapping['num_original_genes']}")
                print(f"  非共同基因数量: {len(gene_mapping['non_common_genes'])}")
                print(f"  已保存扩展后的预测结果: predictions_full_genes.npy")
                
                # 同时保存共同基因的预测结果（用于对比）
                np.save(os.path.join(output_dir, 'predictions_common_genes.npy'), all_preds)
            else:
                print("警告: 无法获取基因映射信息，保存共同基因的预测结果")
                np.save(os.path.join(output_dir, 'predictions.npy'), all_preds)
        else:
            # 保存共同基因的预测结果
            np.save(os.path.join(output_dir, 'predictions.npy'), all_preds)
        
        np.save(os.path.join(output_dir, 'targets.npy'), all_targets)
        print(f"预测结果已保存到: {output_dir}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='评估SINCA-Net模型')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='数据集目录路径 (例如: dataset/Dataset1)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='输出目录路径 (默认: evaluation_results)')
    parser.add_argument('--mask_ratio', type=float, default=0.1,
                        help='掩码比例 (默认: 0.1)')
    parser.add_argument('--normalize', type=str, default='log1p',
                        choices=['log1p', 'minmax', 'zscore', 'none'],
                        help='归一化方法 (默认: log1p)')
    parser.add_argument('--use_common_genes', action='store_true',
                        help='只使用共同基因')
    parser.add_argument('--save_preds', action='store_true',
                        help='保存预测结果')
    parser.add_argument('--expand_to_full_genes', action='store_true',
                        help='扩展到空间转录组的全部基因（推理时使用）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 创建输出目录
    dataset_name = os.path.basename(args.dataset_dir.rstrip('/\\'))
    output_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据路径
    spatial_data_path = os.path.join(args.dataset_dir, 'Spatial_count.h5ad')
    sc_data_path = os.path.join(args.dataset_dir, 'scRNA_count_cluster.h5ad')
    
    if not os.path.exists(sc_data_path):
        sc_data_path = None
        print("警告: 未找到单细胞数据")
    
    # 创建数据加载器
    print("加载数据...")
    dataloader, dataset = create_dataloader(
        spatial_data_path=spatial_data_path,
        sc_data_path=sc_data_path,
        mask_ratio=args.mask_ratio,
        normalize=args.normalize if args.normalize != 'none' else None,
        use_common_genes=args.use_common_genes,
        batch_size=1,
        shuffle=False,
        random_seed=args.seed
    )
    
    num_genes = dataset.num_genes
    print(f"基因数量: {num_genes}")
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    
    # 从检查点加载配置（如果存在）
    checkpoint_dir = os.path.dirname(args.model_path)
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
    
    from models.sinca_net import create_model
    model = create_model(num_genes, config=model_config)
    model = model.to(device)
    
    # 加载权重
    checkpoint = load_checkpoint(model, args.model_path, device=device)
    print(f"模型已加载 (epoch: {checkpoint.get('epoch', 'unknown')})")
    
    # 评估
    print("开始评估...")
    metrics = evaluate(
        model, dataloader, device,
        save_preds=args.save_preds,
        output_dir=output_dir if args.save_preds else None,
        expand_to_full_genes=args.expand_to_full_genes,
        dataset=dataset
    )
    
    # 保存指标
    metrics_path = os.path.join(output_dir, 'metrics.json')
    save_metrics(metrics, metrics_path)
    
    # 打印结果
    print("\n" + "="*60)
    print("评估结果:")
    print("="*60)
    print(f"数据集: {dataset_name}")
    print(f"\n整体指标:")
    print(f"  PCC:  {metrics.get('PCC', 0):.6f}")
    print(f"  SSIM: {metrics.get('SSIM', 0):.6f}")
    print(f"  RMAE: {metrics.get('RMAE', 0):.6f}")
    print(f"  COSSIM: {metrics.get('COSSIM', 0):.6f}")
    
    print(f"\n按基因平均指标:")
    print(f"  PCC_mean:  {metrics.get('PCC_mean', 0):.6f} ± {metrics.get('PCC_std', 0):.6f}")
    print(f"  SSIM_mean: {metrics.get('SSIM_mean', 0):.6f} ± {metrics.get('SSIM_std', 0):.6f}")
    print(f"  RMAE_mean: {metrics.get('RMAE_mean', 0):.6f} ± {metrics.get('RMAE_std', 0):.6f}")
    print(f"  COSSIM_mean: {metrics.get('COSSIM_mean', 0):.6f} ± {metrics.get('COSSIM_std', 0):.6f}")
    
    print(f"\n指标已保存到: {metrics_path}")
    print("="*60)


if __name__ == '__main__':
    main()
