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
```
### 2. Train EpiAML Model

**Recommended: MLP-style with sparse feature learning (best for methylation data):**
```bash
python src/train.py \
  --train_file ../pytorch_marlin/data/training_data.h5 \
  --output_dir ./output_mlp_style \
  --input_dropout 0.99 \
  --stride_config minimal \
  --no_attention \
  --dropout 0.3 \
  --label_smoothing 0.1 \
  --mixup_alpha 0.2 \
  --early_stopping 30 \
  --epochs 500 \
  --batch_size 32
```

**Original CNN configuration (faster but may lose information):**
```bash
python src/train.py \
  --train_file ../pytorch_marlin/data/training_data.h5 \
  --output_dir ./output_cnn \
  --stride_config aggressive \
  --base_channels 64 \
  --num_blocks 4 \
  --dropout 0.3 \
  --epochs 500
```

**Hyperparameter search (generates parallel commands):**
```bash
bash prepare_parallel_hpo.sh
parallel -j 10 < train_model_hpo.sh
python summarize_results.py --root output_debug_search --out hpo_summary.csv
```



**Key Parameters:**

*Model Architecture:*
- `--input_dropout`: Input-level dropout (0.0-0.99) for sparse feature learning (default: 0.0, try 0.99 for MLP-style)
- `--stride_config`: Pooling strategy - `minimal` (preserves info) or `aggressive` (faster, default: minimal)
- `--base_channels`: Base channels for CNN (default: 64)
- `--num_blocks`: Number of CNN residual blocks (default: 4)
- `--no_attention`: Disable attention mechanism (recommended for methylation data)
- `--dropout`: CNN layer dropout (default: 0.3)

*Training:*
- `--feature_order`: Path to ordered features (from clustering)
- `--contrastive_weight`: Weight for contrastive loss (0.0-1.0, default: 0.5)
- `--temperature`: Temperature for contrastive learning (default: 0.07)
- `--samples_per_class`: Samples per class for upsampling (default: 50)
- `--label_smoothing`: Label smoothing factor (default: 0.1)
- `--mixup_alpha`: Mixup augmentation strength (default: 0.2)
- `--early_stopping`: Patience for early stopping (default: 30 epochs)

### 3. Monitor Training

```bash
tensorboard --logdir ./output/logs
```

Open http://localhost:6006 in your browser to view:
- Training/validation loss curves
- Accuracy and F1 scores
- Contrastive vs classification loss
- Learning rate schedule

### 4. Make Predictions

**Batch Prediction:**

```bash
python src/predict.py \
    --model_path ./output/best_model.pt \
    --input_file test_data.h5 \
    --output_file predictions.csv \
    --feature_order ./cluster_output/feature_order.npy \
    --class_mapping ./output/class_mapping.csv \
    --batch_size 32 \
    --device cuda
```

**Single Sample Prediction:**

```bash
python src/predict.py \
    --model_path ./output/best_model.pt \
    --input_file single_sample.csv \
    --single \
    --feature_order ./cluster_output/feature_order.npy \
    --class_mapping ./output/class_mapping.csv
```

**Extract Feature Embeddings:**

```bash
python src/predict.py \
    --model_path ./output/best_model.pt \
    --input_file data.h5 \
    --extract_embeddings \
    --output_file embeddings.npy \
    --feature_order ./cluster_output/feature_order.npy
```

## Model Architecture Details

### 1D-CNN Backbone

**Aggressive Stride Configuration (faster, more downsampling):**
```python
Input (batch, n_features) → (batch, 1, n_features)
    ↓ Conv1d(1, 64, kernel=7, stride=2) + BN + ReLU
    ↓ MaxPool1d(kernel=3, stride=2)
    ↓ ResidualBlock(64, 64, stride=2)
    ↓ ResidualBlock(64, 128, stride=2)
    ↓ ResidualBlock(128, 256, stride=2)
    ↓ ResidualBlock(256, 512, stride=2)
Output (batch, 512, reduced_length)
# For 357K features: 357K → ~5.5K → 128 embedding
```

**Minimal Stride Configuration (preserves more info, recommended):**
```python
Input (batch, n_features) → (batch, 1, n_features)
    ↓ Conv1d(1, 64, kernel=7, stride=1) + BN + ReLU
    ↓ (no maxpool)
    ↓ ResidualBlock(64, 64, stride=1)
    ↓ ResidualBlock(64, 128, stride=1)
    ↓ ResidualBlock(128, 256, stride=2)
    ↓ ResidualBlock(256, 512, stride=2)
Output (batch, 512, reduced_length)
# For 357K features: 357K → ~89K → 128 embedding
```

**With Input Dropout (0.99):**
```python
Input (batch, n_features) → (batch, 1, n_features)
    ↓ Dropout(p=0.99)  # Sparse feature learning like MLP
    ↓ [CNN backbone as above]
```

Each **ResidualBlock** contains:
- Conv1d → BatchNorm → ReLU → Dropout
- Conv1d → BatchNorm
- Skip connection with projection if needed
- ReLU activation

### Multi-Head Self-Attention

- **Number of heads**: 8 (default)
- **Mechanism**: Scaled dot-product attention
- **Residual connection**: Attention output added to input
- **Purpose**: Capture long-range dependencies between CpG sites

### Loss Function

**Combined Loss:**
```
Total Loss = α * L_contrastive + β * L_classification
```

Where:
- **L_contrastive**: Supervised contrastive loss (pulls same class together, pushes different classes apart)
- **L_classification**: Cross-entropy loss for classification
- **α** (contrastive_weight): Default 0.5
- **β** (classification_weight): Fixed at 1.0

**Contrastive Loss Benefits:**
1. Better feature representations
2. Improved generalization
3. Enhanced class separation in embedding space
4. Robust to class imbalance

## Output Files

### After Training

```
output/
├── best_model.pt              # Best model (highest validation accuracy)
├── final_model.pt             # Model from last epoch
├── checkpoint_epoch_*.pt      # Periodic checkpoints
├── config.json                # Training configuration
├── training_history.json      # Loss/accuracy per epoch
├── class_mapping.csv          # Class indices and names
└── logs/                      # TensorBoard logs
```

### After Prediction

**predictions.csv:**
```csv
sample_id,predicted_class,predicted_class_name,confidence,true_label,correct,prob_class_0,prob_class_1,...
0,5,AML_subtype_5,0.9823,5,1,0.0012,0.0034,...
1,12,AML_subtype_12,0.8567,12,1,0.0023,0.0045,...
```

## Advanced Usage

### Custom Model Architecture

```python
from src.model import EpiAMLModel

model = EpiAMLModel(
    input_size=357340,
    num_classes=42,
    base_channels=128,      # Increase for more capacity
    num_blocks=6,           # More blocks = deeper network
    use_attention=True,
    num_heads=16,           # More attention heads
    projection_dim=256,     # Larger projection space
    dropout=0.2             # Stronger regularization
)
```

### Training Without Feature Ordering

If you don't have clustered features, you can train without feature ordering:

```bash
python src/train.py \
    --train_file data.h5 \
    --output_dir ./output \
    --epochs 500
    # Omit --feature_order
```

### Data Augmentation

Augmentation is enabled by default. Control with:

```bash
python src/train.py \
    --train_file data.h5 \
    --flip_percent 0.15 \      # Flip 15% of CpG sites
    --no_augment               # Disable augmentation
```

### Learning Rate Schedule

Uses **Cosine Annealing** by default:
- Starts at `learning_rate`
- Decreases smoothly to `eta_min=1e-6`
- Period = total epochs

### Class Imbalance Handling

**Upsampling** (default):
```bash
--samples_per_class 50  # Upsample each class to 50 samples
```

**No upsampling:**
```bash
--samples_per_class 0
```

## Model Performance Tips

### For Best Accuracy

1. **Use feature ordering** from clustering
2. **Enable attention** mechanism
3. **Tune contrastive weight** (0.3-0.7 typically works well)
4. **Use data augmentation**
5. **Train longer** (500-1000 epochs)
6. **Use validation split** to prevent overfitting

### For Faster Training

1. Increase `--batch_size` (if GPU memory allows)
2. Reduce `--num_blocks` (e.g., 3 instead of 4)
3. Use smaller `--base_channels` (e.g., 32 or 48)
4. Disable attention with `--no_attention`

### For Better Generalization

1. Increase `--weight_decay` (e.g., 1e-4)
2. Use stronger augmentation (`--flip_percent 0.2`)
3. Increase dropout in model (modify code)
4. Use larger validation split (`--val_split 0.3`)

## Comparison with MARLIN

| Feature | MARLIN | EpiAML |
|---------|--------|--------|
| Architecture | Fully-connected (256→128) | 1D-CNN + Attention |
| Dropout | 99% on input | Standard dropout (10-20%) |
| Loss | Cross-entropy only | Contrastive + Cross-entropy |
| Feature Order | Random | Clustered (optional) |
| Attention | None | Multi-head self-attention |
| Parameters | ~92M | ~5-20M (configurable) |
| Training Time | 2-4 hours | 1-3 hours (GPU) |
| Augmentation | 10% CpG flip | 10% CpG flip (optional) |

**EpiAML Advantages:**
- Learns hierarchical features via CNN
- Better handles spatial locality (with feature ordering)
- Contrastive learning improves class separation
- More parameter-efficient
- Attention captures long-range dependencies

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
--batch_size 16

# Reduce model size
--base_channels 32 --num_blocks 3

# Disable attention
--no_attention
```

### Poor Validation Accuracy

1. Check for overfitting (train acc >> val acc):
   - Increase `--weight_decay`
   - Use `--flip_percent` augmentation
   - Reduce model capacity

2. Check for underfitting (both train and val acc low):
   - Increase model capacity (`--base_channels`, `--num_blocks`)
   - Train longer (`--epochs`)
   - Reduce `--weight_decay`

3. Tune hyperparameters:
   - Try different `--contrastive_weight` (0.3, 0.5, 0.7)
   - Adjust `--learning_rate` (1e-5 to 1e-3)
   - Experiment with `--temperature` (0.05-0.1)

### Slow Training

1. Use GPU: `--device cuda`
2. Increase batch size: `--batch_size 64`
3. Reduce num_workers if CPU bottleneck (modify code)
4. Use feature ordering to improve convergence

## Testing the Implementation

### Test Model Architecture

```bash
cd src
python model.py
```

### Test Contrastive Loss

```bash
cd src
python contrastive_loss.py
```

### Test Data Utilities

```bash
cd src
python data_utils.py
```

## Citation

If you use EpiAML in your research, please cite:

```
@software{epiaml2024,
  title={EpiAML: Epigenomic AML Classifier with Deep Learning},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-repo]/EpiAML}
}
```

And cite the original MARLIN paper:

```
Hovestadt V, et al. (2024)
MARLIN: Methylation- and AI-guided Rapid Leukemia Subtype Inference
[Add publication details when available]
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project follows the same license as the original MARLIN project.

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email].

---

**Happy Training! 🚀**
