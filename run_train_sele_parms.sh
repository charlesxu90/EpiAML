# Hyperparameter search commands for EpiAML
# Usage: parallel -j 1 < run_train_sele_parms.sh
# Note: Use -j 1 for single GPU, adjust based on GPU memory

# Learning rate tests
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-3_cw0.5_do0.3_blk4_spc50 --learning_rate 1e-3 --contrastive_weight 0.5 --dropout 0.3 --num_blocks 4 --samples_per_class 50 --epochs 20 --early_stopping 20
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-4_cw0.5_do0.3_blk4_spc50 --learning_rate 1e-4 --contrastive_weight 0.5 --dropout 0.3 --num_blocks 4 --samples_per_class 50 --epochs 20 --early_stopping 20
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-5_cw0.5_do0.3_blk4_spc50 --learning_rate 1e-5 --contrastive_weight 0.5 --dropout 0.3 --num_blocks 4 --samples_per_class 50 --epochs 20 --early_stopping 20

# Contrastive weight tests (with vs without)
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-4_cw0.0_do0.3_blk4_spc50 --learning_rate 1e-4 --contrastive_weight 0.0 --dropout 0.3 --num_blocks 4 --samples_per_class 50 --epochs 20 --early_stopping 20
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-4_cw0.5_do0.3_blk4_spc50 --learning_rate 1e-4 --contrastive_weight 0.5 --dropout 0.3 --num_blocks 4 --samples_per_class 50 --epochs 20 --early_stopping 20

# Dropout tests
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-4_cw0.5_do0.3_blk4_spc50 --learning_rate 1e-4 --contrastive_weight 0.5 --dropout 0.3 --num_blocks 4 --samples_per_class 50 --epochs 20 --early_stopping 20
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-4_cw0.5_do0.4_blk4_spc50 --learning_rate 1e-4 --contrastive_weight 0.5 --dropout 0.4 --num_blocks 4 --samples_per_class 50 --epochs 20 --early_stopping 20
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-4_cw0.5_do0.5_blk4_spc50 --learning_rate 1e-4 --contrastive_weight 0.5 --dropout 0.5 --num_blocks 4 --samples_per_class 50 --epochs 20 --early_stopping 20
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-4_cw0.5_do0.6_blk4_spc50 --learning_rate 1e-4 --contrastive_weight 0.5 --dropout 0.6 --num_blocks 4 --samples_per_class 50 --epochs 20 --early_stopping 20
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-4_cw0.5_do0.7_blk4_spc50 --learning_rate 1e-4 --contrastive_weight 0.5 --dropout 0.7 --num_blocks 4 --samples_per_class 50 --epochs 20 --early_stopping 20
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-4_cw0.5_do0.8_blk4_spc50 --learning_rate 1e-4 --contrastive_weight 0.5 --dropout 0.8 --num_blocks 4 --samples_per_class 50 --epochs 20 --early_stopping 20
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-4_cw0.5_do0.9_blk4_spc50 --learning_rate 1e-4 --contrastive_weight 0.5 --dropout 0.9 --num_blocks 4 --samples_per_class 50 --epochs 20 --early_stopping 20
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-4_cw0.5_do0.99_blk4_spc50 --learning_rate 1e-4 --contrastive_weight 0.5 --dropout 0.99 --num_blocks 4 --samples_per_class 50 --epochs 20 --early_stopping 20

# Number of blocks tests
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-4_cw0.5_do0.3_blk2_spc50 --learning_rate 1e-4 --contrastive_weight 0.5 --dropout 0.3 --num_blocks 2 --samples_per_class 50 --epochs 20 --early_stopping 20
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-4_cw0.5_do0.3_blk3_spc50 --learning_rate 1e-4 --contrastive_weight 0.5 --dropout 0.3 --num_blocks 3 --samples_per_class 50 --epochs 20 --early_stopping 20
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-4_cw0.5_do0.3_blk4_spc50 --learning_rate 1e-4 --contrastive_weight 0.5 --dropout 0.3 --num_blocks 4 --samples_per_class 50 --epochs 20 --early_stopping 20

# Samples per class tests
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-4_cw0.5_do0.3_blk4_spc50 --learning_rate 1e-4 --contrastive_weight 0.5 --dropout 0.3 --num_blocks 4 --samples_per_class 50 --epochs 20 --early_stopping 20
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-4_cw0.5_do0.3_blk4_spc100 --learning_rate 1e-4 --contrastive_weight 0.5 --dropout 0.3 --num_blocks 4 --samples_per_class 100 --epochs 20 --early_stopping 20
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-4_cw0.5_do0.3_blk4_spc200 --learning_rate 1e-4 --contrastive_weight 0.5 --dropout 0.3 --num_blocks 4 --samples_per_class 200 --epochs 20 --early_stopping 20

# Combined best candidates (based on typical good values)
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-4_cw0.0_do0.5_blk2_spc100 --learning_rate 1e-4 --contrastive_weight 0.0 --dropout 0.5 --num_blocks 2 --samples_per_class 100 --epochs 20 --early_stopping 20
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-4_cw0.0_do0.7_blk2_spc100 --learning_rate 1e-4 --contrastive_weight 0.0 --dropout 0.7 --num_blocks 2 --samples_per_class 100 --epochs 20 --early_stopping 20
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-4_cw0.0_do0.9_blk2_spc100 --learning_rate 1e-4 --contrastive_weight 0.0 --dropout 0.9 --num_blocks 2 --samples_per_class 100 --epochs 20 --early_stopping 20
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-5_cw0.0_do0.5_blk2_spc200 --learning_rate 1e-5 --contrastive_weight 0.0 --dropout 0.5 --num_blocks 2 --samples_per_class 200 --epochs 20 --early_stopping 20
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-5_cw0.0_do0.7_blk2_spc200 --learning_rate 1e-5 --contrastive_weight 0.0 --dropout 0.7 --num_blocks 2 --samples_per_class 200 --epochs 20 --early_stopping 20
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-5_cw0.5_do0.5_blk3_spc200 --learning_rate 1e-5 --contrastive_weight 0.5 --dropout 0.5 --num_blocks 3 --samples_per_class 200 --epochs 20 --early_stopping 20
python src/train.py --train_file ../pytorch_marlin/data/training_data.h5 --output_dir ./output_search/lr1e-5_cw0.5_do0.7_blk3_spc200 --learning_rate 1e-5 --contrastive_weight 0.5 --dropout 0.7 --num_blocks 3 --samples_per_class 200 --epochs 20 --early_stopping 20
