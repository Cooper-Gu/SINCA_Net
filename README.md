# SINCA-Net: Spatial Inpainting Network with Cross-Modal Attention

用于空间转录组基因插补的深度学习网络

## 网络简介

SINCA-Net是一个创新的深度学习网络，用于空间转录组数据的基因表达插补。该网络借鉴了图像修复（inpainting）的思想，将空间转录组数据视为"图像"，每个spot是像素，基因表达是通道。

### 核心创新点

1. **图像修复思想**：将缺失基因视为需要修复的图像区域，使用深度学习进行插补(重点)
2. **空间卷积模块**：将空间坐标转换为2D网格，使用卷积神经网络捕获局部空间模式
3. **跨模态注意力机制**：融合空间转录组和单细胞RNA数据，利用单细胞数据作为参考信息
4. **空间Transformer编码器**：使用Transformer捕获长距离空间依赖关系

## 模型结构
输入: 空间已知基因表达矩阵 [N_spots, num_known_genes] + 空间坐标 [N_spots, 2] + 单细胞参考数据 [N_cells, num_total_genes]
↓
已知基因空间表达投影 → 高维特征 [N_spots, d_model]
坐标信息投影 → 位置编码 [N_spots, d_model]
↓
空间卷积模块 → 捕获局部空间模式
(将空间坐标转换为2D网格，应用卷积操作)
↓
跨模态注意力模块 → 融合单细胞参考信息
(空间spots作为query，单细胞作为key和value)
↓
空间Transformer编码器 → 捕获长距离依赖
(多头自注意力机制)
↓
输出：未知基因特征投影 → 基因表达空间 [N_spots, num_unknown_genes]
可选：同时输出已知基因的"精修"版本

## 损失函数
- 主要损失：未知基因表达的MSE（需要有部分ground truth用于训练）
- 辅助损失：已知基因重建损失
- 正则化：基因关系矩阵的生物学合理性约束

## 安装依赖

```bash
pip install torch torchvision torch-geometric
pip install scanpy pandas numpy scikit-learn scipy
pip install tensorboard tqdm
```

## 使用方法

### 1. 训练模型

```bash
python train.py --dataset_dir dataset/Dataset1 --output_dir outputs --epochs 100
```

### 2. 评估单个数据集

```bash
python evaluate.py --model_path outputs/sinca_net_xxx/best_model.pth --dataset_dir dataset/Dataset1 --output_dir evaluation_results
```

### 3. 批量评估所有数据集

```bash
python evaluate_all_datasets.py --model_path outputs/sinca_net_xxx/best_model.pth --dataset_root dataset --output_dir evaluation_results
```

## 评估指标

- **PCC**: 皮尔逊相关系数
- **SSIM**: 结构相似性指数
- **RMAE**: 均方根误差
- **JS**: Jaccard Similarity，杰卡德相似性
- **ACC**: Accuracy Score，准确度评分，使用相对值排名方法来评估生成数据的质量

### 4. 推理得到插补基因

```bash
python inference.py --model_path outputs/sinca_net_xxx/best_model.pth --dataset_dir dataset/Dataset1 --output_path results/imputed_spatial_full_genes.h5ad --normalize log1p
```
**输出文件**：
- `imputed_spatial_full_genes.h5ad`：包含全部基因的插补结果（AnnData格式）


## 数据集格式

数据集所在目录：`dataset/Dataset{number}/`
每个数据集目录包含：
- `Spatial_count.h5ad`: 空间转录组数据（AnnData格式，包含位置信息）
- `scRNA_count_cluster.h5ad`: 单细胞RNA数据（AnnData格式，包含聚类信息“cluster"）

## 项目结构

```
.
├── models/
│   └── sinca_net.py          # SINCA-Net网络定义
├── data/
│   └── dataloader.py         # 数据加载器
├── utils/
│   ├── metrics.py            # 评估指标
│   └── utils.py              # 工具函数
├── train.py                  # 训练脚本
├── evaluate.py               # 单数据集评估脚本
├── evaluate_all_datasets.py  # 批量评估脚本
└── README.md                 # 本文档
```

## 引用

如果您使用了SINCA-Net，请引用：

```bibtex
@article{sinca_net_2026,
  title={SINCA-Net: Spatial Inpainting Network with Cross-Modal Attention for Spatial Transcriptomics Gene Imputation},
  author={Your Name},
  journal={IJCAI},
  year={2026}
}
```

