#!/bin/bash
# EpiAML Quick Start Script
# This script demonstrates the complete workflow from feature clustering to prediction

set -e  # Exit on error

echo "========================================"
echo "EpiAML Quick Start Workflow"
echo "========================================"

# Configuration
DATA_FILE="../pytorch_marlin/data/training_data.h5"
CLUSTER_DIR="./cluster_output"
OUTPUT_DIR="./output"
N_CLUSTERS=100
EPOCHS=500
BATCH_SIZE=32

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Training data not found at $DATA_FILE"
    echo "Please update DATA_FILE in this script to point to your data"
    exit 1
fi

echo ""
echo "Step 1: Feature Clustering (optional but recommended)"
echo "========================================"
read -p "Run feature clustering? This may take 10-30 minutes. (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python cluster_cg.py \
        --data "$DATA_FILE" \
        --n_clusters $N_CLUSTERS \
        --output_dir "$CLUSTER_DIR" \
        --gpu

    FEATURE_ORDER_ARG="--feature_order $CLUSTER_DIR/feature_order.npy"
    echo "✓ Feature clustering completed"
else
    echo "Skipping feature clustering (training without ordered features)"
    FEATURE_ORDER_ARG=""
fi

echo ""
echo "Step 2: Train EpiAML Model"
echo "========================================"
echo "Training for $EPOCHS epochs with batch size $BATCH_SIZE"
echo "This may take 1-3 hours on GPU, longer on CPU"
echo ""

python src/train.py \
    --train_file "$DATA_FILE" \
    $FEATURE_ORDER_ARG \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate 1e-4 \
    --contrastive_weight 0.5 \
    --val_split 0.2 \
    --device cuda

echo ""
echo "✓ Training completed!"
echo ""

echo "Step 3: View Training Results"
echo "========================================"
echo "Training logs saved to: $OUTPUT_DIR/logs"
echo ""
echo "To view training curves with TensorBoard:"
echo "  tensorboard --logdir $OUTPUT_DIR/logs"
echo ""

# Check if test data exists
TEST_FILE="../pytorch_marlin/data/test_data.h5"
if [ -f "$TEST_FILE" ]; then
    echo "Step 4: Make Predictions on Test Data"
    echo "========================================"

    python src/predict.py \
        --model_path "$OUTPUT_DIR/best_model.pt" \
        --input_file "$TEST_FILE" \
        --output_file predictions.csv \
        $FEATURE_ORDER_ARG \
        --class_mapping "$OUTPUT_DIR/class_mapping.csv" \
        --device cuda

    echo ""
    echo "✓ Predictions saved to: predictions.csv"
else
    echo "Step 4: Make Predictions"
    echo "========================================"
    echo "No test file found at $TEST_FILE"
    echo ""
    echo "To make predictions, run:"
    echo "  python src/predict.py \\"
    echo "      --model_path $OUTPUT_DIR/best_model.pt \\"
    echo "      --input_file YOUR_TEST_FILE.h5 \\"
    echo "      --output_file predictions.csv \\"
    echo "      $FEATURE_ORDER_ARG \\"
    echo "      --class_mapping $OUTPUT_DIR/class_mapping.csv"
fi

echo ""
echo "========================================"
echo "EpiAML Quick Start Complete!"
echo "========================================"
echo ""
echo "Output files:"
echo "  - Model: $OUTPUT_DIR/best_model.pt"
echo "  - Config: $OUTPUT_DIR/config.json"
echo "  - History: $OUTPUT_DIR/training_history.json"
echo "  - Classes: $OUTPUT_DIR/class_mapping.csv"
if [ -n "$FEATURE_ORDER_ARG" ]; then
    echo "  - Feature order: $CLUSTER_DIR/feature_order.npy"
fi
echo ""
echo "Next steps:"
echo "  1. Review training history in $OUTPUT_DIR/training_history.json"
echo "  2. Visualize training with: tensorboard --logdir $OUTPUT_DIR/logs"
echo "  3. Make predictions on new data with src/predict.py"
echo "  4. Extract embeddings for visualization with --extract_embeddings"
echo ""
