#!/bin/bash
# Feature Selection & Training Workflow
# This script demonstrates the complete workflow for AML classification with feature selection

set -e

echo "=========================================================================="
echo "Feature Selection & Training Workflow for AML Classification"
echo "=========================================================================="

# Configuration
TRAIN_DATA="../pytorch_marlin/data/training_data.h5"
OUTPUT_BASE="./experiments"
DEVICE="cuda"

# Step 1: Feature Selection with all methods
echo ""
echo "Step 1: Feature Selection (All Methods)"
echo "=========================================================================="

mkdir -p $OUTPUT_BASE/feature_selection

echo "Running feature selection with all four methods..."
python src/select_informative_features.py \
  --train_file $TRAIN_DATA \
  --output_dir $OUTPUT_BASE/feature_selection \
  --n_features 100 \
  --method ensemble \
  --device $DEVICE

echo "✓ Feature selection complete!"
echo "  Output: $OUTPUT_BASE/feature_selection/"

# Step 2: Compare different feature set sizes
echo ""
echo "Step 2: Train with Different Feature Set Sizes"
echo "=========================================================================="

for n_cpg in 50 100 200; do
  echo ""
  echo "Training with top-$n_cpg CpGs..."
  
  # Select top-N features
  python src/select_informative_features.py \
    --train_file $TRAIN_DATA \
    --output_dir $OUTPUT_BASE/feature_selection_${n_cpg} \
    --n_features $n_cpg \
    --method ensemble \
    --device $DEVICE
  
  # Train model
  python src/train.py \
    --train_file $TRAIN_DATA \
    --feature_indices $OUTPUT_BASE/feature_selection_${n_cpg}/top_${n_cpg}_cpg_indices.npy \
    --output_dir $OUTPUT_BASE/model_top${n_cpg}cpg \
    --input_dropout 0.0 \
    --stride_config minimal \
    --no_attention \
    --learning_rate 0.0001 \
    --epochs 200 \
    --batch_size 32 \
    --early_stopping 20
  
  echo "✓ Completed training with top-$n_cpg CpGs"
done

# Step 3: Make predictions with selected features
echo ""
echo "Step 3: Make Predictions with Selected Features"
echo "=========================================================================="

echo "Making predictions using top-100 CpGs..."
python src/predict.py \
  --model_path $OUTPUT_BASE/model_top100cpg/best_model.pt \
  --input_file $TRAIN_DATA \
  --feature_indices $OUTPUT_BASE/feature_selection_100/top_100_cpg_indices.npy \
  --output_file $OUTPUT_BASE/predictions_top100.csv \
  --device $DEVICE

echo "✓ Predictions saved: $OUTPUT_BASE/predictions_top100.csv"

# Step 4: Summary
echo ""
echo "=========================================================================="
echo "✓ Feature Selection & Training Workflow Complete!"
echo "=========================================================================="
echo ""
echo "Results Summary:"
echo "  Feature selection: $OUTPUT_BASE/feature_selection/"
echo "  Models trained:"
for n_cpg in 50 100 200; do
  echo "    - Top-$n_cpg CpGs: $OUTPUT_BASE/model_top${n_cpg}cpg/"
done
echo "  Predictions: $OUTPUT_BASE/predictions_top100.csv"
echo ""
echo "Next steps:"
echo "  1. Review feature selection report: cat $OUTPUT_BASE/feature_selection/feature_selection_report.txt"
echo "  2. Compare model performance across different feature set sizes"
echo "  3. Analyze predictions with domain experts"
echo "  4. Validate selected CpGs in wet-lab experiments"
