"""
批量评估脚本
用于评估训练好的SINCA-Net模型在所有数据集上的性能
"""

import os
import argparse
import json
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from models.sinca_net import create_model
from data.dataloader import create_dataloader
from utils.utils import (
    load_checkpoint, get_device, save_metrics
)
from utils.metrics import (
    compute_all_metrics, compute_metrics_per_gene,
    pearson_correlation_coefficient, structural_similarity_index,
    root_mean_absolute_error, cosine_similarity
)


def main():
    parser = argparse.ArgumentParser(description='批量评估SINCA-Net模型')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--dataset_root', type=str, default='dataset',
                        help='数据集根目录 (默认: dataset)')
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
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--start_idx', type=int, default=1,
                        help='起始数据集编号 (默认: 1)')
    parser.add_argument('--end_idx', type=int, default=45,
                        help='结束数据集编号 (默认: 45)')
    
    args = parser.parse_args()
    
    # 设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型配置
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
            'use_cross_modal_attention': config.get('use_cross_modal_attention', True),
            'use_transformer': config.get('use_transformer', True)
        }
    else:
        model_config = {
            'd_model': 512,
            'num_heads': 8,
            'num_layers': 6,
            'dropout': 0.1,
            'use_spatial_conv': True,
            'use_cross_modal_attention': True,
            'use_transformer': True
        }
    
    # 存储所有数据集的指标
    all_metrics = []
    dataset_names = []
    
    def evaluate_model(model, dataloader, device, save_preds=False, output_dir=None):
        """评估模型"""
        model.eval()
        all_preds = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='评估中', leave=False):
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
            np.save(os.path.join(output_dir, 'predictions.npy'), all_preds)
            np.save(os.path.join(output_dir, 'targets.npy'), all_targets)
        
        return metrics
    
    # 遍历所有数据集
    print("开始批量评估...")
    print(f"数据集范围: Dataset{args.start_idx} - Dataset{args.end_idx}")
    
    for i in tqdm(range(args.start_idx, args.end_idx + 1), desc='处理数据集'):
        dataset_name = f'Dataset{i}'
        dataset_dir = os.path.join(args.dataset_root, dataset_name)
        
        # 检查数据集是否存在
        spatial_data_path = os.path.join(dataset_dir, 'Spatial_count.h5ad')
        if not os.path.exists(spatial_data_path):
            print(f"\n跳过 {dataset_name}: 文件不存在")
            continue
        
        sc_data_path = os.path.join(dataset_dir, 'scRNA_count_cluster.h5ad')
        if not os.path.exists(sc_data_path):
            sc_data_path = None
        
        try:
            # 创建数据加载器
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
            
            # 创建模型（每个数据集可能有不同的基因数量）
            model = create_model(num_genes, config=model_config)
            model = model.to(device)
            
            # 加载权重（需要处理不同基因数量的情况）
            try:
                checkpoint = load_checkpoint(model, args.model_path, device=device)
                
                # 评估
                output_dir = os.path.join(args.output_dir, dataset_name)
                metrics = evaluate_model(
                    model, dataloader, device,
                    save_preds=args.save_preds,
                    output_dir=output_dir if args.save_preds else None
                )
                
                # 保存单个数据集的指标
                metrics_path = os.path.join(output_dir, 'metrics.json')
                os.makedirs(output_dir, exist_ok=True)
                save_metrics(metrics, metrics_path)
                
                # 添加到总体结果
                metrics['dataset'] = dataset_name
                all_metrics.append(metrics)
                dataset_names.append(dataset_name)
                
                print(f"\n✓ {dataset_name} 评估完成")
                print(f"  PCC: {metrics.get('PCC', 0):.6f}, "
                      f"SSIM: {metrics.get('SSIM', 0):.6f}, "
                      f"RMAE: {metrics.get('RMAE', 0):.6f}, "
                      f"COSSIM: {metrics.get('COSSIM', 0):.6f}")
                
            except Exception as e:
                print(f"\n✗ {dataset_name} 模型加载失败: {e}")
                continue
                
        except Exception as e:
            print(f"\n✗ {dataset_name} 数据处理失败: {e}")
            continue
    
    # 汇总结果
    if len(all_metrics) > 0:
        # 创建汇总表
        summary_data = {
            'Dataset': dataset_names,
            'PCC': [m.get('PCC', 0) for m in all_metrics],
            'SSIM': [m.get('SSIM', 0) for m in all_metrics],
            'RMAE': [m.get('RMAE', 0) for m in all_metrics],
            'COSSIM': [m.get('COSSIM', 0) for m in all_metrics],
            'PCC_mean': [m.get('PCC_mean', 0) for m in all_metrics],
            'SSIM_mean': [m.get('SSIM_mean', 0) for m in all_metrics],
            'RMAE_mean': [m.get('RMAE_mean', 0) for m in all_metrics],
            'COSSIM_mean': [m.get('COSSIM_mean', 0) for m in all_metrics],
        }
        
        df = pd.DataFrame(summary_data)
        
        # 保存为CSV
        csv_path = os.path.join(args.output_dir, 'summary.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n汇总结果已保存到: {csv_path}")
        
        # 保存为JSON
        json_path = os.path.join(args.output_dir, 'summary.json')
        save_metrics({'results': all_metrics}, json_path)
        print(f"详细结果已保存到: {json_path}")
        
        # 打印统计信息
        print("\n" + "="*60)
        print("总体统计:")
        print("="*60)
        print(f"成功评估的数据集数量: {len(all_metrics)}")
        print(f"\n平均指标 (跨所有数据集):")
        print(f"  PCC:  {df['PCC'].mean():.6f} ± {df['PCC'].std():.6f}")
        print(f"  SSIM: {df['SSIM'].mean():.6f} ± {df['SSIM'].std():.6f}")
        print(f"  RMAE: {df['RMAE'].mean():.6f} ± {df['RMAE'].std():.6f}")
        print(f"  COSSIM: {df['COSSIM'].mean():.6f} ± {df['COSSIM'].std():.6f}")
        
        print(f"\n按基因平均指标 (跨所有数据集):")
        print(f"  PCC_mean:  {df['PCC_mean'].mean():.6f} ± {df['PCC_mean'].std():.6f}")
        print(f"  SSIM_mean: {df['SSIM_mean'].mean():.6f} ± {df['SSIM_mean'].std():.6f}")
        print(f"  RMAE_mean: {df['RMAE_mean'].mean():.6f} ± {df['RMAE_mean'].std():.6f}")
        print(f"  COSSIM_mean: {df['COSSIM_mean'].mean():.6f} ± {df['COSSIM_mean'].std():.6f}")
        print("="*60)
    else:
        print("\n没有成功评估的数据集!")


if __name__ == '__main__':
    main()
