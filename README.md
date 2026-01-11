# SINCA-Net: Spatial Inpainting Network with Cross-Modal Attention for Spatial Transcriptomics Gene Imputation

A deep learning network for spatial transcriptomics gene imputation.

## Network Overview

SINCA-Net is an innovative deep learning network designed for gene expression imputation in spatial transcriptomics data. The network draws inspiration from image inpainting, treating spatial transcriptomics data as an "image," where each spot is a pixel and gene expression constitutes the channels.

### Core Innovations

1. **Image Inpainting Concept**：Treats missing genes as image regions requiring inpainting, utilizing deep learning for imputation.
2. **Spatial Convolution Module**：Transforms spatial coordinates into a 2D grid, employing Convolutional Neural Networks (CNNs) to capture local spatial patterns.
3. **Cross-Modal Attention Mechanism**：Fuses spatial transcriptomics and single-cell RNA data, leveraging single-cell data as reference information.
4. **Spatial Transformer Encoder**：Uses Transformer architecture to capture long-range spatial dependencies.

## Installation & Dependencies

```bash
pip install torch torchvision torch-geometric
pip install scanpy pandas numpy scikit-learn scipy
pip install tensorboard tqdm
```

## Usage

### 1. Train the Model

```bash
python train.py --dataset_dir dataset/Dataset1 --output_dir outputs --epochs 100
```

### 2. Evaluate on a Single Dataset

```bash
python evaluate.py --model_path outputs/sinca_net_xxx/best_model.pth --dataset_dir dataset/Dataset1 --output_dir evaluation_results
```

### 3. Batch Evaluate on All Datasets

```bash
python evaluate_all_datasets.py --model_path outputs/sinca_net_xxx/best_model.pth --dataset_root dataset --output_dir evaluation_results
```

## Evaluation Metrics

- **PCC**: Pearson Correlation Coefficient
- **SSIM**: Structural Similarity Index Measure
- **RMAE**: Root Mean Squared Error
- **JS**: Jaccard Similarity
- **ACC**: Accuracy Score, assesses the quality of generated data using a relative value ranking method

### 4. Run Inference to Obtain Imputed Genes

```bash
python inference.py --model_path outputs/sinca_net_xxx/best_model.pth --dataset_dir dataset/Dataset1 --output_path results/imputed_spatial_full_genes.h5ad --normalize log1p
```
**Output File**：
- `imputed_spatial_full_genes.h5ad`：Contains the imputation results for all genes (in AnnData format).


## 数据集格式

Dataset directories are located at：`dataset/Dataset{number}/`
Each dataset directory contains:
- `Spatial_count.h5ad`: Spatial Transcriptomics Data (AnnData Format, including location information)
- `scRNA_count_cluster.h5ad`: Single-cell RNA data (AnnData format, includes the "cluster" annotation for cell type information)

## 项目结构

```
.
├── models/
│   └── sinca_net.py          # SINCA-Net model definition
├── data/
│   └── dataloader.py         # Data loader
├── utils/
│   ├── metrics.py            # Evaluation metrics
│   └── utils.py              # Utility functions
├── train.py                  # Training script
├── evaluate.py               # Single-dataset evaluation script
├── evaluate_all_datasets.py  # Batch evaluation script
└── README.md                 # This documentation file
```

## 引用

If you have used SINCA-Net, please cite:

```bibtex
@article{sinca_net_2026,
  title={SINCA-Net: Spatial Inpainting Network with Cross-Modal Attention for Spatial Transcriptomics Gene Imputation},
  author={Your Name},
  journal={IJCAI},
  year={2026}
}
```

