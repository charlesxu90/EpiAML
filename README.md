# EpiAML: Epigenomic AML Classifier with Deep Learning

**State-of-the-art deep learning model for acute myeloid leukemia (AML) subtype classification from DNA methylation profiles.**

EpiAML combines cutting-edge machine learning technologies:
- **1D-CNN backbone** with residual connections for robust feature extraction
- **Multi-head self-attention** for capturing long-range dependencies
- **Supervised contrastive learning** for improved class separation
- **Feature ordering** via clustering for spatial locality in convolutions

## Architecture Overview

```
Input (CpG Methylation Features)
    ↓
Input Dropout (optional, for sparse feature learning)
    ↓
1D-CNN Backbone (Residual Blocks with configurable stride)
    ↓
Multi-Head Self-Attention (optional)
    ↓
Global Average Pooling
    ↓
├─→ Projection Head → Contrastive Loss
└─→ Classification Head → Cross-Entropy Loss
```

### Model Improvements (v2)

✓ **Input Dropout**: High dropout (0.99) enables MLP-style sparse feature learning  
✓ **Configurable Stride**: Minimal pooling preserves more information, aggressive pooling faster  
✓ **Optional Attention**: Disabled by default for methylation data (often hurts performance)  
✓ **Backward Compatible**: All new parameters optional, loads old models seamlessly

## Installation

### Prerequisites

```bash
conda create -p ./env -c conda-forge python=3.10
conda activate ./env
```

### Install Dependencies

```bash
# Core packages
pip install numpy pandas scikit-learn h5py tqdm

# PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# TensorBoard for monitoring
pip install tensorboard

# Visualization
pip install matplotlib seaborn

# GPU-accelerated clustering (optional)
pip install cuml-cu11  # For CUDA 11

# SciPy for hierarchical clustering
pip install scipy
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### 1. Create a debug dataset for efficient evalution

```bash
# Create a small dataset for debugging purpose
python create_debug_data.py --input ../pytorch_marlin/data/training_data.h5 --output data/training_data_debug.h5
```

### 2. Feature selection with Shapley analysis
```bash
python select_informative_features.py  --train_file  ../pytorch_marlin/data/training_data.h5  --model_path ../pytorch_marlin/output/marlin_model.pt   --device cuda:0 --output_dir ./feature_selection_shap  --n_features 1000  --prefilter_topk 5000

python select_informative_features.py --train_file  data/training_data_debug.h5 --model_path ../pytorch_marlin/output_sample/marlin_model.pt --device cuda:1 --output_dir ./feature_selection_shap_sample  --n_features 1000  --prefilter_topk 5000

python select_informative_features.py  --train_file  ../pytorch_marlin/data/training_data.h5  --model_path ../pytorch_marlin/output/marlin_model.pt   --device cuda:1 --output_dir ./feature_selection_shap_5k  --n_features 5000  --prefilter_topk 10000 --shap_samples 500 --shap_background 100

python select_informative_features.py  --train_file  ../pytorch_marlin/data/training_data.h5  --model_path ../pytorch_marlin/output/marlin_model.pt   --device cuda:1 --output_dir ./feature_selection_shap_10k  --n_features 10000  --prefilter_topk 100000 --shap_samples 500 --shap_background 100

python select_informative_features.py  --train_file  ../pytorch_marlin/data/training_data.h5  --model_path ../pytorch_marlin/output/marlin_model.pt   --device cuda:1 --output_dir ./feature_selection_shap_50k  --n_features 50000  --prefilter_topk 1000000 --shap_samples 500 --shap_background 100
```

### 3. Create small dataset with selected features
```bash
python create_filtered_dataset.py --input_data ../pytorch_marlin/data/training_data.h5 --feature_selection_dir ./feature_selection_shap_sample --output_data ./feature_selection_shap_sample/training_data_top1000.h5 --n_features 1000

python create_filtered_dataset.py --input_data ../pytorch_marlin/data/training_data.h5 --feature_selection_dir ./feature_selection_shap --output_data ./feature_selection_shap/training_data_top1000.h5 --n_features 1000

python create_filtered_dataset.py --input_data ../pytorch_marlin/data/training_data.h5 --feature_selection_dir ./feature_selection_shap_5k --output_data ./feature_selection_shap_5k/training_data_top5000.h5 --n_features 5000

python create_filtered_dataset.py --input_data ../pytorch_marlin/data/training_data.h5 --feature_selection_dir ./feature_selection_shap_10k --output_data ./feature_selection_shap_10k/training_data_top10000.h5 --n_features 10000

python create_filtered_dataset.py --input_data ../pytorch_marlin/data/training_data.h5 --feature_selection_dir ./feature_selection_shap_50k --output_data ./feature_selection_shap_50k/training_data_top50000.h5 --n_features 50000
```

### 2. Train EpiAML Model with selected features

**Recommended: MLP-style with sparse feature learning (best for methylation data):**
```bash
```