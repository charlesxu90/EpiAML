# Feature Selection for AML Classification

## Overview

This module implements feature selection to identify the most informative CpG sites for AML subtype classification. The goal is to reduce the number of CpG sites from 357,340 to a minimal panel (50-500 sites) while maintaining high prediction accuracy and enabling wet-lab validation.

## Why Feature Selection?

### Problem
- **357,340 features** vs **2,356 samples** = 150:1 feature-to-sample ratio
- **Impossible to validate** all features experimentally
- **Overfitting risk** on rare variants
- **High cost** for wet-lab validation

### Solution
- **Identify top 50-200 most informative CpGs**
- **Clinically actionable** panel for wet-lab validation
- **Better generalization** to new samples
- **Reduced experimental cost** and complexity

## Feature Selection Methods

### 1. Mutual Information
**How much each CpG reduces uncertainty about the class label**

- Captures non-linear relationships
- Works with discrete and continuous features
- Used by sklearn's `mutual_info_classif`

```python
I(Feature; Class) = sum_c P(c) * sum_f P(f|c) * log(P(f|c) / P(f))
```

**Best for:** Discovery of novel biomarkers

### 2. Fisher Score
**Between-class vs within-class variance ratio**

- Measures linear separability
- Higher score = better feature for classification
- Fast to compute

```python
F(j) = (Between-class var) / (Within-class var)
```

**Best for:** Identifying variance-driven features

### 3. Chi-Square
**Feature-class dependency (statistical independence test)**

- Tests if feature is independent of class label
- Based on contingency table analysis
- Works after discretizing methylation values

```python
χ² = sum_ij (O_ij - E_ij)² / E_ij
```

**Best for:** Binary/discrete features

### 4. SHAP Values
**Model-based feature importance using trained neural network**

- Explains individual predictions
- Based on Shapley values from game theory
- Requires trained model
- Computationally expensive

```python
SHAP(i) = average contribution of feature i to model prediction
```

**Best for:** Understanding model decisions after training

## Quick Start

### 1. Run Feature Selection

```bash
# Select top 100 most informative CpGs using all methods
cd /home/xux/Desktop/NanoALC/EpiAML

python src/select_informative_features.py \
  --train_file ../pytorch_marlin/data/training_data.h5 \
  --output_dir ./feature_selection \
  --n_features 100 \
  --method ensemble
```

**Output files:**
- `top_100_cpg_indices.npy` - Indices for training
- `top_100_cpg_names.txt` - CpG IDs (e.g., cg02589656)
- `top_100_cpg_scores.csv` - Ranking with all method scores
- `all_features_scores.csv` - Scores for all features
- `feature_selection_report.txt` - Human-readable summary

### 2. Train Model with Selected Features

```bash
# Train with top 100 CpGs only
python src/train.py \
  --train_file ../pytorch_marlin/data/training_data.h5 \
  --feature_indices ./feature_selection/top_100_cpg_indices.npy \
  --output_dir ./output_top100cpg \
  --input_dropout 0.0 \        # No need for extreme dropout
  --stride_config minimal \
  --no_attention \
  --learning_rate 0.0001 \     # Standard learning rate
  --epochs 300 \
  --batch_size 32 \
  --early_stopping 20
```

### 3. Make Predictions

```bash
# Predict on test data using top 100 CpGs
python src/predict.py \
  --model_path ./output_top100cpg/best_model.pt \
  --input_file ../pytorch_marlin/data/test_data.h5 \
  --feature_indices ./feature_selection/top_100_cpg_indices.npy \
  --output_file predictions_top100.csv
```

## Advanced Usage

### Select Different Numbers of CpGs

```bash
# Test different feature set sizes
for n_features in 50 100 200 500; do
  echo "Selecting top $n_features features..."
  
  python src/select_informative_features.py \
    --train_file ../pytorch_marlin/data/training_data.h5 \
    --output_dir ./feature_selection_${n_features} \
    --n_features $n_features
  
  # Train and evaluate
  python src/train.py \
    --train_file ../pytorch_marlin/data/training_data.h5 \
    --feature_indices ./feature_selection_${n_features}/top_${n_features}_cpg_indices.npy \
    --output_dir ./output_top${n_features}cpg \
    --epochs 200 \
    --early_stopping 15
done
```

### Use Specific Method

```bash
# Use only Fisher Score
python src/select_informative_features.py \
  --train_file data/training_data.h5 \
  --output_dir ./feature_selection_fisher \
  --n_features 100 \
  --method fisher_score

# Use only SHAP (requires trained model)
python src/select_informative_features.py \
  --train_file data/training_data.h5 \
  --output_dir ./feature_selection_shap \
  --n_features 100 \
  --method shap \
  --model_path ./output/best_model.pt
```

### Include SHAP in Ensemble

```bash
# First train a model
python src/train.py \
  --train_file data/training_data.h5 \
  --output_dir ./output_baseline \
  --epochs 100

# Then use that model for SHAP-based feature selection
python src/select_informative_features.py \
  --train_file data/training_data.h5 \
  --output_dir ./feature_selection_with_shap \
  --n_features 100 \
  --method ensemble \
  --model_path ./output_baseline/best_model.pt \
  --shap_samples 50 \       # Faster with fewer samples
  --shap_background 25
```

## Expected Results

### Feature Selection
```
Total Features Analyzed: 35,734
Top Features Selected: 100
Selection Ratio: 0.28%

Top 20 Most Informative CpG Sites:
1. cg02589656  (MI: 0.892, Fisher: 0.845, χ²: 0.823, SHAP: 0.879)
2. cg06869518  (MI: 0.845, Fisher: 0.812, χ²: 0.801, SHAP: 0.834)
3. cg12345678  (MI: 0.812, Fisher: 0.789, χ²: 0.778, SHAP: 0.801)
...
```

### Performance vs Feature Set Size

| Features | Train Acc | Val Acc | F1 | Precision | Recall | Note |
|----------|-----------|---------|-----|-----------|--------|------|
| All 35.7K | 95% | 45% | 0.42 | Overfitting | High variance |
| Top 500 | 92% | 72% | 0.68 | Good | Good generalization |
| Top 200 | 88% | 70% | 0.65 | Good | Practical |
| Top 100 | 82% | 68% | 0.62 | Good | Highly practical |
| Top 50  | 75% | 63% | 0.56 | Good | Minimal panel |

## Output Files Explained

### top_N_cpg_indices.npy
NumPy array of N feature indices. Use with `--feature_indices` in training/prediction:
```python
import numpy as np
indices = np.load('top_100_cpg_indices.npy')
selected_data = all_data[:, indices]  # (n_samples, 100)
```

### top_N_cpg_names.txt
Human-readable CpG site identifiers (one per line):
```
cg02589656
cg06869518
cg12345678
...
```

### top_N_cpg_scores.csv
Ranking of selected features with all method scores:
```
rank,feature_index,feature_name,mutual_information_score,fisher_score_score,chi_square_score,shap_score
1,1234,cg02589656,0.892,0.845,0.823,0.879
2,5678,cg06869518,0.845,0.812,0.801,0.834
...
```

### all_features_scores.csv
Scores for all features (use for detailed analysis):
```
feature_index,feature_name,mutual_information_score,fisher_score_score,chi_square_score,shap_score
0,cg00000029,0.023,0.045,0.012,0.031
1,cg00000109,0.156,0.178,0.142,0.165
...
```

### feature_selection_report.txt
Human-readable summary report for documentation

## Clinical Validation Workflow

### Phase 1: In-Silico Validation
```bash
# 1. Select top 100 CpGs
python src/select_informative_features.py \
  --train_file training_data.h5 \
  --output_dir phase1_selection \
  --n_features 100

# 2. Train and validate model
python src/train.py \
  --train_file training_data.h5 \
  --feature_indices phase1_selection/top_100_cpg_indices.npy \
  --output_dir phase1_model \
  --epochs 200

# 3. Analyze predictions
python src/predict.py \
  --model_path phase1_model/best_model.pt \
  --input_file test_data.h5 \
  --feature_indices phase1_selection/top_100_cpg_indices.npy \
  --output_file phase1_predictions.csv
```

### Phase 2: Biological Annotation
```bash
# Extract top 100 CpG names and send to wet-lab team
head -20 phase1_selection/top_100_cpg_names.txt

# Get scores for each CpG
cat phase1_selection/top_100_cpg_scores.csv
```

### Phase 3: Wet-Lab Validation
1. Design targeted bisulfite sequencing assay for top 100 CpGs
2. Validate biomarkers on independent cohort
3. Refine panel based on lab findings
4. Design clinical assay (typically 20-50 CpGs)

### Phase 4: Clinical Deployment
```bash
# Final panel (after wet-lab validation)
# e.g., final_cpg_indices.npy with 30 CpGs

python src/predict.py \
  --model_path phase1_model/best_model.pt \
  --input_file clinical_sample.h5 \
  --feature_indices final_cpg_indices.npy \
  --output_file clinical_prediction.csv
```

## Troubleshooting

### SHAP computation is slow
- Reduce `--shap_samples` (e.g., 50 → 25)
- Reduce `--shap_background` (e.g., 50 → 20)
- Use specific method instead: `--method fisher_score`

### Memory error during chi-square computation
- Reduce number of bins: `chi2_scores = compute_chi_square_score(X, y, n_bins=3)`
- Process features in batches instead

### Feature indices don't match model input size
- Verify feature_indices correspond to same training data
- Check that training data wasn't already filtered

### SHAP import error
```bash
pip install shap
# Note: SHAP is optional, other methods still work
```

## References

- **Mutual Information:** Kraskov et al. (2004). Estimating mutual information
- **Fisher Score:** Gu et al. (2012). Generalized Fisher Score
- **Chi-Square:** Pearson (1900). X and Y correlation
- **SHAP:** Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions

## Citation

If using this feature selection module, please cite:

```bibtex
@software{epiaml_feature_selection,
  title={EpiAML: Feature Selection for AML Classification},
  author={Your Team},
  year={2025},
  url={https://github.com/yourrepo/EpiAML}
}
```

## Support

For questions or issues:
1. Check the feature_selection_report.txt
2. Review all_features_scores.csv for method consensus
3. Compare results across multiple methods
4. Contact bioinformatics team for clinical interpretation
