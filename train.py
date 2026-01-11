"""
训练脚本
用于训练SINCA-Net模型
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from models.sinca_net import create_model
from data.dataloader import create_dataloader, split_dataset, get_batch_by_mode
from utils.utils import (
    set_seed, save_checkpoint, load_checkpoint, 
    get_device, count_parameters, print_model_info,
    save_metrics
)
from utils.metrics import compute_all_metrics


class Tee:
    """同时输出到控制台和文件的类"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # 确保立即写入
    
    def flush(self):
        for f in self.files:
            f.flush()


def train_epoch(model, dataset, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    
    # 清空CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 获取训练数据（使用训练基因）
    batch = get_batch_by_mode(dataset, mode='train')
    
    # 获取数据
    spatial_expr = batch['spatial_expr'].to(device)
    spatial_coords = batch['spatial_coords'].to(device)
    target = batch['target'].to(device)
    mask = batch['mask'].to(device)
    gene_mask = batch['gene_mask'].to(device)  # 训练基因掩码
    
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
    optimizer.zero_grad()
    
    # 使用梯度缩放来节省内存
    with torch.amp.autocast('cuda', enabled=False):  # 暂时禁用混合精度，如果需要可以启用
        pred = model(**inputs)
        
        # 只对训练基因和掩码位置计算损失
        # 创建组合掩码：既要在掩码位置，又要是训练基因
        combined_mask = mask & gene_mask.unsqueeze(0)
        pred_masked = pred * combined_mask.float()
        target_masked = target * combined_mask.float()
        loss = criterion(pred_masked, target_masked)
    
    # 反向传播
    loss.backward()
    
    # 梯度裁剪，防止梯度爆炸
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # 保存损失值（在删除变量之前）
    train_loss_value = loss.item()
    
    # 清理缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 释放不需要的变量
    del pred, loss, pred_masked, target_masked, combined_mask
    if 'sc_expr' in inputs:
        del inputs['sc_expr']
    
    return train_loss_value


def validate(model, dataset, criterion, device):
    """
    验证模型
    
    Args:
        model: 模型
        dataset: 数据集对象
        criterion: 损失函数
        device: 设备
    """
    model.eval()
    
    # 设置固定的epoch用于验证（使用epoch 0的掩码以确保一致性）
    dataset.set_epoch(0)
    
    # 清空CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 获取验证数据（使用验证基因）
    batch = get_batch_by_mode(dataset, mode='val')
    
    with torch.no_grad():
        # 获取数据
        spatial_expr = batch['spatial_expr'].to(device)
        spatial_coords = batch['spatial_coords'].to(device)
        target = batch['target'].to(device)
        mask = batch['mask'].to(device)
        gene_mask = batch['gene_mask'].to(device)  # 验证基因掩码
        
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
        
        # 只对验证基因和掩码位置计算损失和指标
        # 创建组合掩码：既要在掩码位置，又要是验证基因
        combined_mask = mask & gene_mask.unsqueeze(0)
        pred_masked = pred * combined_mask.float()
        target_masked = target * combined_mask.float()
        loss = criterion(pred_masked, target_masked)
        
        # 收集预测和目标（只对掩码位置，立即移到CPU）
        pred_np = pred_masked.cpu().numpy()
        target_np = target_masked.cpu().numpy()
        
        # 释放GPU内存
        del pred, pred_masked, target_masked, combined_mask
        if 'sc_expr' in inputs:
            del inputs['sc_expr']
    
    # 最终清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 计算评估指标
    metrics = compute_all_metrics(target_np, pred_np, prefix='val_')
    
    return loss.item(), metrics


def main():
    parser = argparse.ArgumentParser(description='训练SINCA-Net模型')
    parser.add_argument('--dataset_dir', type=str, default='dataset/Dataset2',
                        help='数据集目录路径 (例如: dataset/Dataset1)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='输出目录路径 (默认: outputs)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (默认: 100)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch大小 (默认: 1)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率 (默认: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减 (默认: 1e-5)')
    parser.add_argument('--mask_ratio', type=float, default=0.1,
                        help='掩码比例 (默认: 0.1)')
    parser.add_argument('--d_model', type=int, default=256,
                        help='模型维度 (默认: 256，降低以节省内存)')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='注意力头数 (默认: 8)')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Transformer层数 (默认: 4，降低以节省内存)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout比例 (默认: 0.1)')
    parser.add_argument('--normalize', type=str, default='log1p',
                        choices=['log1p', 'minmax', 'zscore', 'none'],
                        help='归一化方法 (默认: log1p)')
    parser.add_argument('--use_common_genes', action='store_true',
                        help='只使用共同基因')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例 (默认: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例 (默认: 0.1)')
    parser.add_argument('--disable_spatial_conv', action='store_true',
                        help='禁用空间卷积模块以节省内存')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 清空CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"sinca_net_{timestamp}"
    exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 保存配置
    config = vars(args)
    config['device'] = str(device)
    with open(os.path.join(exp_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # TensorBoard
    logs_dir = os.path.join(exp_dir, 'logs')
    writer = SummaryWriter(logs_dir)
    
    # 创建日志文件并设置同时输出到控制台和文件
    log_file_path = os.path.join(logs_dir, 'training.log')
    log_file = open(log_file_path, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)
    
    # 数据路径
    spatial_data_path = os.path.join(args.dataset_dir, 'Spatial_count.h5ad')
    sc_data_path = os.path.join(args.dataset_dir, 'scRNA_count_cluster.h5ad')
    
    if not os.path.exists(sc_data_path):
        sc_data_path = None
        print("警告: 未找到单细胞数据，将不使用跨模态注意力")
    
    # 创建数据加载器
    print("加载数据...")
    train_loader, dataset = create_dataloader(
        spatial_data_path=spatial_data_path,
        sc_data_path=sc_data_path,
        mask_ratio=args.mask_ratio,
        normalize=args.normalize if args.normalize != 'none' else None,
        use_common_genes=args.use_common_genes,
        batch_size=args.batch_size,
        shuffle=False,
        random_seed=args.seed
    )
    
    # 获取数据信息
    num_genes = dataset.num_genes
    num_train_genes = dataset.num_train_genes
    num_val_genes = dataset.num_val_genes
    print(f"总基因数量: {num_genes}")
    print(f"训练基因数量: {num_train_genes}")
    print(f"验证基因数量: {num_val_genes}")
    
    # 创建模型
    print("创建模型...")
    model_config = {
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'use_spatial_conv': not args.disable_spatial_conv,
        'use_cross_modal_attention': sc_data_path is not None,
        'use_transformer': True
    }
    model = create_model(num_genes, config=model_config)
    model = model.to(device)
    print_model_info(model)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        print(f"从检查点恢复训练: {args.resume}")
        checkpoint = load_checkpoint(model, args.resume, optimizer, device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # 训练循环
    try:
        print("开始训练...")
        for epoch in range(start_epoch, args.epochs):
            # 每个epoch前清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 设置当前epoch，使每个epoch使用不同的掩码
            dataset.set_epoch(epoch)
            
            # 训练（使用训练基因）
            train_loss = train_epoch(model, dataset, optimizer, criterion, device, epoch)
            
            # 验证前清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 验证（使用验证基因）
            val_loss, val_metrics = validate(model, dataset, criterion, device)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 记录到TensorBoard
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            for metric_name, metric_value in val_metrics.items():
                writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # 打印日志
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  训练损失: {train_loss:.6f}")
            print(f"  验证损失: {val_loss:.6f}")
            print(f"  验证指标:")
            for metric_name, metric_value in val_metrics.items():
                print(f"    {metric_name}: {metric_value:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(exp_dir, 'best_model.pth')
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    best_model_path,
                    best_val_loss=best_val_loss,
                    val_metrics=val_metrics
                )
                print(f"  ✓ 保存最佳模型 (验证损失: {val_loss:.6f})")
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(exp_dir, f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(
                    model, optimizer, epoch, val_loss,
                    checkpoint_path,
                    best_val_loss=best_val_loss,
                    val_metrics=val_metrics
                )
        
        writer.close()
        print(f"\n训练完成! 模型保存在: {exp_dir}")
    finally:
        # 确保恢复标准输出并关闭日志文件
        sys.stdout = original_stdout
        if log_file and not log_file.closed:
            log_file.close()


if __name__ == '__main__':
    main()
