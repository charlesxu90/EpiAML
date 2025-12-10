# Hyperparameter search commands for EpiAML
# Usage: 
#   bash run_train_sele_parms.sh           (sequential)
#   parallel -j 2 < run_train_sele_parms.sh (parallel with 2 jobs)
#   parallel -j 4 < run_train_sele_parms.sh (parallel with 4 jobs)
#
# Note: Use -j 1 for single GPU, adjust based on GPU memory

# Configuration
TRAIN_FILE="../pytorch_marlin/data/training_data.h5"
OUTPUT_BASE="./output_search"

# Create output directory
mkdir -p "$OUTPUT_BASE"

# This function wraps the training command for parallel
train_model() {
    local lr=$1
    local cw=$2
    local do=$3
    local blk=${4:-4}
    local spc=${5:-50}
    local exp_name="lr${lr}_cw${cw}_do${do}_blk${blk}_spc${spc}"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $exp_name"
    python src/train.py \
        --train_file "$TRAIN_FILE" \
        --output_dir "$OUTPUT_BASE/$exp_name" \
        --learning_rate "$lr" \
        --contrastive_weight "$cw" \
        --dropout "$do" \
        --num_blocks "$blk" \
        --samples_per_class "$spc" \
        --epochs 20 \
        --early_stopping 20
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: $exp_name"
}

export -f train_model
export TRAIN_FILE OUTPUT_BASE

# Print header (only in interactive mode)
if [ -t 0 ]; then
    echo "=========================================="
    echo "EpiAML Hyperparameter Search"
    echo "=========================================="
    echo "Data: $TRAIN_FILE"
    echo "Output Base: $OUTPUT_BASE"
    echo "Mode: SEQUENTIAL"
    echo "=========================================="
    echo ""
fi

# Learning rate tests (blk=4, spc=50)
train_model "1e-3" "0.5" "0.3" "4" "50"
train_model "1e-4" "0.5" "0.3" "4" "50"
train_model "1e-5" "0.5" "0.3" "4" "50"

# Contrastive weight tests (blk=4, spc=50)
train_model "1e-4" "0.0" "0.3" "4" "50"
train_model "1e-4" "0.5" "0.3" "4" "50"

# Dropout tests (blk=4, spc=50)
train_model "1e-4" "0.5" "0.3" "4" "50"
train_model "1e-4" "0.5" "0.4" "4" "50"
train_model "1e-4" "0.5" "0.5" "4" "50"
train_model "1e-4" "0.5" "0.6" "4" "50"
train_model "1e-4" "0.5" "0.7" "4" "50"
train_model "1e-4" "0.5" "0.8" "4" "50"
train_model "1e-4" "0.5" "0.9" "4" "50"
train_model "1e-4" "0.5" "0.99" "4" "50"

# Number of blocks tests (spc=50)
train_model "1e-4" "0.5" "0.3" "2" "50"
train_model "1e-4" "0.5" "0.3" "3" "50"
train_model "1e-4" "0.5" "0.3" "4" "50"

# Samples per class tests (blk=4)
train_model "1e-4" "0.5" "0.3" "4" "50"
train_model "1e-4" "0.5" "0.3" "4" "100"
train_model "1e-4" "0.5" "0.3" "4" "200"

# Combined best candidates (based on typical good values)
train_model "1e-4" "0.0" "0.5" "2" "100"
train_model "1e-4" "0.0" "0.7" "2" "100"
train_model "1e-4" "0.0" "0.9" "2" "100"
train_model "1e-5" "0.0" "0.5" "2" "200"
train_model "1e-5" "0.0" "0.7" "2" "200"
train_model "1e-5" "0.5" "0.5" "3" "200"
train_model "1e-5" "0.5" "0.7" "3" "200"
