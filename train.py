"""
训练脚本
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from models import SINCANet
from data import create_dataloader, create_kfold_splits
from utils import set_seed, save_checkpoint, create_output_dir, save_metrics


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        known_genes, coords, unknown_genes, sc_data = batch
        
        known_genes = known_genes.to(device)
        coords = coords.to(device)
        unknown_genes = unknown_genes.to(device)
        sc_data = sc_data.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        pred_unknown, pred_known_refined = model(known_genes, coords, sc_data)
        
        # 计算损失
        loss_unknown = criterion(pred_unknown, unknown_genes)
        loss_known = criterion(pred_known_refined, known_genes)
        loss = loss_unknown + 0.1 * loss_known  # 主要损失 + 辅助损失
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_pred_unknown = []
    all_true_unknown = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            known_genes, coords, unknown_genes, sc_data = batch
            
            known_genes = known_genes.to(device)
            coords = coords.to(device)
            unknown_genes = unknown_genes.to(device)
            sc_data = sc_data.to(device)
            
            # 前向传播
            pred_unknown, pred_known_refined = model(known_genes, coords, sc_data)
            
            # 计算损失
            loss_unknown = criterion(pred_unknown, unknown_genes)
            loss_known = criterion(pred_known_refined, known_genes)
            loss = loss_unknown + 0.1 * loss_known
            
            total_loss += loss.item()
            num_batches += 1
            
            # 收集预测和真实值
            all_pred_unknown.append(pred_unknown.cpu().numpy())
            all_true_unknown.append(unknown_genes.cpu().numpy())
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # 计算指标
    all_pred_unknown = np.concatenate(all_pred_unknown, axis=0)
    all_true_unknown = np.concatenate(all_true_unknown, axis=0)
    
    from utils.metrics import calculate_all_metrics
    metrics = calculate_all_metrics(all_pred_unknown, all_true_unknown)
    
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description='训练SINCA-Net模型')
    parser.add_argument('--dataset_dir', type=str, default='dataset/Dataset1', help='数据集目录')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')
    parser.add_argument('--num_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--num_transformer_layers', type=int, default=2, help='Transformer层数')
    parser.add_argument('--n_splits', type=int, default=10, help='K折交叉验证的折数')
    parser.add_argument('--fold', type=int, default=0, help='使用第几折（0-9）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--normalize', type=str, default='log1p', choices=['log1p', 'standardize', None], help='归一化方法')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    output_dir = create_output_dir(args.output_dir)
    print(f"输出目录: {output_dir}")
    
    # 创建K折划分
    print("创建K折交叉验证划分...")
    splits, temp_dataset = create_kfold_splits(args.dataset_dir, n_splits=args.n_splits, normalize=args.normalize)
    
    # 使用指定的fold
    train_gene_indices, val_gene_indices = splits[args.fold]
    print(f"使用第 {args.fold} 折")
    print(f"训练基因数: {len(train_gene_indices)}, 验证基因数: {len(val_gene_indices)}")
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, train_dataset = create_dataloader(
        dataset_dir=args.dataset_dir,
        known_gene_indices=train_gene_indices,
        unknown_gene_indices=val_gene_indices,
        batch_size=args.batch_size,
        shuffle=True,
        normalize=args.normalize,
        use_ground_truth=True
    )
    
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
    
    print(f"创建模型: 已知基因={num_known_genes}, 未知基因={num_unknown_genes}, 总基因数={num_total_genes}")
    
    model = SINCANet(
        num_known_genes=num_known_genes,
        num_unknown_genes=num_unknown_genes,
        num_total_genes=num_total_genes,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_transformer_layers=args.num_transformer_layers
    ).to(args.device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # 模型配置（用于保存）
    model_config = {
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'num_transformer_layers': args.num_transformer_layers,
        'num_known_genes': num_known_genes,
        'num_unknown_genes': num_unknown_genes,
        'num_total_genes': num_total_genes
    }
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    # 训练循环
    best_val_loss = float('inf')
    best_metrics = None
    
    print("\n开始训练...")
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device, epoch)
        
        # 验证
        val_loss, val_metrics = validate(model, val_loader, criterion, args.device)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Metrics/PCC', val_metrics['PCC_mean'], epoch)
        writer.add_scalar('Metrics/SSIM', val_metrics['SSIM'], epoch)
        writer.add_scalar('Metrics/RMAE', val_metrics['RMAE'], epoch)
        writer.add_scalar('Metrics/JS', val_metrics['JS'], epoch)
        writer.add_scalar('Metrics/ACC', val_metrics['ACC'], epoch)
        
        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Val PCC: {val_metrics['PCC_mean']:.4f}")
        print(f"  Val SSIM: {val_metrics['SSIM']:.4f}")
        print(f"  Val RMAE: {val_metrics['RMAE']:.6f}")
        print(f"  Val JS: {val_metrics['JS']:.4f}")
        print(f"  Val ACC: {val_metrics['ACC']:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, best_model_path, best_metrics, model_config)
            print(f"  ✓ 保存最佳模型 (Val Loss: {val_loss:.6f})")
        
        # 定期保存检查点
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path, val_metrics, model_config)
    
    writer.close()
    
    # 保存最终指标
    if best_metrics:
        metrics_path = os.path.join(output_dir, 'best_metrics.json')
        save_metrics(best_metrics, metrics_path)
    
    print(f"\n训练完成！最佳模型保存在: {output_dir}")


if __name__ == '__main__':
    main()
