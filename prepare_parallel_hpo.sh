#!/bin/bash
# prepare_parallel_hpo.sh
# 
# This script generates train_model_hpo.sh which can be executed with:
#   parallel -j 10 < train_model_hpo.sh
#
# Usage:
#   bash prepare_parallel_hpo.sh
#   parallel -j 10 < train_model_hpo.sh

OUTPUT_FILE="train_model_hpo.sh"
# remove if exists
rm -f $OUTPUT_FILE

echo "Generating $OUTPUT_FILE for parallel execution..."


for lr in 1e-3 1e-4 1e-5; do
  for cw in 0.0 0.5; do
    for do in 0.3; do
        for nbl in 2 3 4; do
            exp_name="lr${lr}_cw${cw}_do${do}_blk${nbl}_spc50"
            command="python src/train.py --train_file  data/training_data_debug.h5 --feature_order feature_order_debug/feature_order_indices.npy  --output_dir ./output_debug_search/$exp_name --learning_rate $lr --contrastive_weight $cw --dropout $do --num_blocks $nbl --samples_per_class 50 --epochs 20 --early_stopping 20"
            echo "$command" >> $OUTPUT_FILE
        done
    done
  done
done

echo "Generated $OUTPUT_FILE. You can now run it with:"
echo "  parallel -j 10 < $OUTPUT_FILE"