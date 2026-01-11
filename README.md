# SINCA-Net: Spatial Inpainting Network with Cross-Modal Attention for Spatial Transcriptomics Gene Imputation

A Deep Learning Network for Spatial Transcriptomics Gene Imputation

## Network Introduction

SINCA-Net is an innovative deep learning network for gene expression imputation in spatial transcriptomics data. The network draws inspiration from image inpainting, treating spatial transcriptomics data as an image, where each spot is a pixel and gene expressions are channels.

## Installation Dependencies

```bash
pip install -r requirements.txt
```

## Usage Instructions

### 1. Training the Model

```bash
python train.py --dataset_dir dataset/Dataset1 --output_dir outputs --epochs 100
```

### 2. Evaluating a Single Dataset

```bash
python evaluate.py --model_path outputs/sinca_net_xxx/best_model.pth --dataset_dir dataset/Dataset1 --output_dir evaluation_results
```

### 3. Batch Evaluating All Datasets

```bash
python evaluate_all_datasets.py --model_path outputs/sinca_net_xxx/best_model.pth --dataset_root dataset --output_dir evaluation_results
```

## Evaluation Metrics

- **PCC**: Pearson Correlation Coefficient
- **SSIM**: Structural Similarity Index
- **RMAE**: Root Mean Absolute Error
- **JS**: Jaccard Similarity
- **ACC**:  Accuracy Score, using relative ranking methods to assess the quality of generated data

### 4. Inference for Imputing Genes

```bash
python inference.py --model_path outputs/sinca_net_xxx/best_model.pth --dataset_dir dataset/Dataset1 --output_path results/imputed_spatial_full_genes.h5ad --normalize log1p
```
**Output File**：
- `imputed_spatial_full_genes.h5ad`：Contains the complete gene imputation results (in AnnData format).


## Dataset Format

Dataset directory: `dataset/Dataset{number}/`
Each dataset directory contains:
- `Spatial_count.h5ad`: Spatial transcriptomics data (AnnData format, includes location information).
- `scRNA_count_cluster.h5ad`: Single-cell RNA data (AnnData format, includes clustering information "cluster").

## 项目结构

```
.
├── models/
│   └── sinca_net.py          # SINCA-Net network definition
├── data/
│   └── dataloader.py         # Data loader
├── utils/
│   ├── metrics.py            # Evaluation metrics
│   └── utils.py              # Utility functions
├── train.py                  # Training script
├── evaluate.py               # Single-dataset evaluation script
├── evaluate_all_datasets.py  # Batch evaluation script
└── README.md                 # This document
```

## 引用

If you use SINCA-Net, please cite:

```bibtex
@article{sinca_net_2026,
  title={SINCA-Net: Spatial Inpainting Network with Cross-Modal Attention for Spatial Transcriptomics Gene Imputation},
  author={Chengzhi Gui, De-shuang Huang},
  journal={xxxx},
  year={2026}
}
```

