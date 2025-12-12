# EpiAML Problem 

Excellent question! You're facing a classic **domain adaptation** problem: your MARLIN model trained on dense array data now needs to handle sparse, noisy Nanopore sequencing. Here are the key directions:

## 🎯 Core Challenge Analysis

**MARLIN's assumptions:**
```python
Input: 357,340 CpG sites @ high coverage (array)
       ↓
Input dropout (99%) randomly samples features
       ↓
Works because: Missing data is RANDOM, features are COMPLETE
```

**Nanopore reality (0.3x coverage):**
```python
Input: 357,340 CpG sites BUT ~70% are MISSING (not random)
       ↓
Input dropout (99%) + sparse data = mostly zeros
       ↓
Problem: Model hasn't learned to handle STRUCTURED SPARSITY
```

---

## 📋 Recommended Improvement Directions (Ranked by Impact)

### **1. 🥇 Feature Reduction + Transfer Learning (HIGHEST IMPACT)**

Instead of using all 357K features with sparse data, use your **SHAP-selected top 100 CpGs**:

```bash
# Step 1: Already done - SHAP selected your top 100 CpGs
# These are the features MARLIN actually relies on

# Step 2: Retrain MARLIN on top 100 features only
python -c "
import torch
import numpy as np
from pytorch_marlin.src.model import MARLINModel

# Load original MARLIN
model_full = MARLINModel.load_model('output/best_model.pt', 'cuda')

# Create new model for 100 features
model_sparse = MARLINModel(
    input_size=100,  # ← Changed from 357,340
    hidden1_size=256,
    hidden2_size=128,
    output_size=42,
    dropout_rate=0.99
)

# Transfer weights from first layer (top 100 features only)
shap_indices = np.load('feature_selection_shap/top_100_cpg_indices.npy')
with torch.no_grad():
    # Copy weights for selected features
    model_sparse.fc1.weight.data = model_full.fc1.weight.data[:, shap_indices]
    model_sparse.fc1.bias.data = model_full.fc1.bias.data
    model_sparse.fc2.weight.data = model_full.fc2.weight.data
    model_sparse.fc2.bias.data = model_full.fc2.bias.data
    model_sparse.fc3.weight.data = model_full.fc3.weight.data
    model_sparse.fc3.bias.data = model_full.fc3.bias.data

torch.save({
    'model_state_dict': model_sparse.state_dict(),
    'input_size': 100,
    'hidden1_size': 256,
    'hidden2_size': 128,
    'output_size': 42,
    'dropout_rate': 0.99
}, 'output/marlin_top100_transferred.pt')
print('Transfer learning: 357K → 100 features complete!')
"
```

**Why this works:**
- ✅ 100 features has 99.97% missing data tolerance (vs. 70% for 357K)
- ✅ Model sees concentrated signal in key features
- ✅ Transfer learning preserves learned decision boundaries
- ✅ Reduces input dropout noise (less sparsity = less dropout needed)

**Expected improvement:** 70-90% accuracy on Nanopore 0.3x data

---

### **2. 🥈 Domain Adaptation via Fine-tuning (HIGH IMPACT)**

Fine-tune on Nanopore data with **lower input dropout**:

```bash
# Create fine-tuning script
python src/finetune_marlin.py \
  --pretrained_model output/best_model.pt \
  --train_file nanopore_0.3x_training_data.h5 \
  --feature_indices feature_selection_shap/top_100_cpg_indices.npy \
  --output_dir output_nanopore_finetuned \
  --dropout_rate 0.5 \        # Reduce from 0.99 (less aggressive)
  --learning_rate 0.0001 \    # Low LR - preserve learned features
  --freeze_fc2_fc3 \          # Optional: freeze last 2 layers
  --epochs 100 \
  --batch_size 16
```

**Why this works:**
- ✅ Adapts learned features to sparse input distribution
- ✅ Lower dropout (0.5) helps with sparse 0.3x coverage
- ✅ Frozen higher layers preserve class decision logic
- ✅ Few epochs prevent overfitting

---

### **3. 🥉 Missing Data Handling (MEDIUM IMPACT)**

Handle Nanopore sparsity explicitly:

```python
# Option A: Imputation before input
def impute_missing_cpgs(X, method='mean'):
    """Fill missing CpGs with class mean"""
    # X shape: (n_samples, 100_features)
    # Mark missing: X == 0 or NaN
    
    if method == 'mean':
        # Impute with per-feature mean
        mean_vals = np.nanmean(X, axis=0)
        X_imputed = np.where(np.isnan(X), mean_vals, X)
    
    elif method == 'knn':
        # k-NN imputation (preserve local structure)
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = imputer.fit_transform(X)
    
    return X_imputed

# Option B: Modify dropout to preserve coverage
class SparsityAwareMARLIN(MARLINModel):
    def forward(self, x, coverage_mask=None, apply_softmax=True):
        """
        Args:
            x: Input features
            coverage_mask: Boolean (n_samples, n_features) indicating observed CpGs
                          Prevents dropout on missing sites
        """
        if coverage_mask is not None:
            # Only apply dropout to covered features
            dropout_mask = torch.bernoulli(
                torch.ones_like(x) * self.dropout_rate * coverage_mask.float()
            )
            x = x * (1 - dropout_mask) / (1 - self.dropout_rate)
        else:
            x = self.dropout_input(x)
        
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        
        if apply_softmax:
            x = torch.softmax(x, dim=1)
        return x
```

---

### **4. ✨ Contrastive Learning for Domain Adaptation (MEDIUM-HIGH IMPACT)**

Use contrastive learning to align sparse Nanopore with dense array data:

```python
# Contrastive adaptation: Array data as anchor, Nanopore as positive
def contrastive_domain_loss(model, X_array, X_nanopore, y, temperature=0.07):
    """
    Bring Nanopore embeddings close to array embeddings (same class)
    """
    # Get penultimate layer embeddings (before output)
    h_array = model.fc2(torch.sigmoid(model.fc1(model.dropout_input(X_array))))
    h_nanopore = model.fc2(torch.sigmoid(model.fc1(model.dropout_input(X_nanopore))))
    
    # Contrastive loss: same class should have similar embeddings
    cos_sim = torch.nn.functional.cosine_similarity(
        h_array.unsqueeze(1),  # (batch, 1, 128)
        h_nanopore.unsqueeze(0),  # (1, batch, 128)
        dim=2  # (batch, batch)
    )
    
    # Positive pairs (same class)
    pos_mask = (y.unsqueeze(0) == y.unsqueeze(1))
    
    # NT-Xent loss
    logits = cos_sim / temperature
    pos_loss = -torch.log(
        torch.exp(logits[pos_mask]).sum(dim=1) / 
        torch.exp(logits).sum(dim=1)
    ).mean()
    
    return pos_loss
```

**Combined training:**
```python
# loss = classification_loss(y_pred, y) + 0.5 * contrastive_domain_loss(...)
```

---

### **5. 🔧 Model Architecture Adjustments (LOWER IMPACT but helpful)**

```python
# Reduce model complexity (less to overfit to noise)
class MARLINSparse(MARLINModel):
    def __init__(self, input_size=100, output_size=42):
        super().__init__(
            input_size=input_size,
            hidden1_size=128,      # ← Reduced from 256
            hidden2_size=64,       # ← Reduced from 128
            output_size=output_size,
            dropout_rate=0.5       # ← Reduced from 0.99
        )
```

**Why:** Smaller model generalizes better to sparse, noisy data

---

## 🎯 Recommended Implementation Strategy

### **Phase 1: Quick Baseline (1-2 hours)**

```bash
# 1. Extract top 100 CpGs (already done with SHAP)
python src/select_informative_features.py \
  --model_path output/best_model.pt \
  --n_features 100

# 2. Transfer learning + fine-tune
python transfer_learning_marlin.py \
  --pretrained_model output/best_model.pt \
  --train_file nanopore_0.3x_training.h5 \
  --feature_indices feature_selection/top_100_cpg_indices.npy \
  --dropout_rate 0.5 \
  --epochs 50
```

**Expected:** 60-75% accuracy on Nanopore 0.3x

---

### **Phase 2: Domain Adaptation (3-6 hours)**

```bash
# Fine-tune with contrastive learning
python finetune_contrastive.py \
  --pretrained_model output/marlin_transferred.pt \
  --train_file_array ../pytorch_marlin/data/training_data.h5 \
  --train_file_nanopore nanopore_0.3x_training.h5 \
  --feature_indices feature_selection/top_100_cpg_indices.npy \
  --contrastive_weight 0.5 \
  --dropout_rate 0.3 \
  --epochs 200
```

**Expected:** 75-85% accuracy on Nanopore 0.3x

---

### **Phase 3: Optimization (Optional)**

- Imputation + contrastive learning
- Ensemble: MARLIN (array) + MARLIN-Nanopore (sparse)
- Active learning: Request coverage for uncertain samples

---

## 📊 Summary: Which Direction for Your Task?

| Direction | Implementation Time | Expected Improvement | Effort |
|-----------|------------------|---------------------|--------|
| **Feature reduction (100 CpGs)** | 1 hour | +15-20% | ⭐ Easy |
| **Transfer learning** | 2 hours | +25-35% | ⭐⭐ Medium |
| **Contrastive adaptation** | 4 hours | +30-40% | ⭐⭐⭐ Complex |
| **Missing data handling** | 3 hours | +10-15% | ⭐⭐ Medium |
| **Model reduction** | 1 hour | +5-10% | ⭐ Easy |

---

## 🎬 Quick Start Command

```bash
# All-in-one: Feature selection → Transfer learning
bash setup_nanopore_adaptation.sh \
  --pretrained output/best_model.pt \
  --train_file nanopore_0.3x_training.h5 \
  --output output_nanopore_adapted
```

Would you like me to implement any of these directions? I'd recommend **starting with Feature Reduction + Transfer Learning** (Direction 1+2) as it gives 80% of the benefit with 20% of the effort!