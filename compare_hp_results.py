#!/usr/bin/env python3
"""
Compare hyperparameter search results and find optimal parameters.

This script reads all training results from the output directories and
creates a summary showing which parameters perform best.

Usage:
    python compare_hp_results.py --output_dir output_debug_search
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict


def load_training_results(output_dir):
    """Load training results from all subdirectories."""
    results = []
    
    if not os.path.exists(output_dir):
        print(f"Error: Output directory not found: {output_dir}")
        return results
    
    print(f"Scanning for results in: {output_dir}")
    
    for subdir in os.listdir(output_dir):
        subdir_path = os.path.join(output_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        config_path = os.path.join(subdir_path, 'training_config.json')
        history_path = os.path.join(subdir_path, 'training_history.json')
        
        if not os.path.exists(config_path) or not os.path.exists(history_path):
            continue
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            # Extract best metrics
            best_epoch = history.get('best_epoch', -1)
            best_val_f1 = history.get('best_val_f1', 0.0)
            best_val_loss = history.get('best_val_loss', float('inf'))
            final_train_f1 = history['train_f1'][-1] if 'train_f1' in history else 0.0
            final_val_f1 = history['val_f1'][-1] if 'val_f1' in history else 0.0
            
            result = {
                'exp_name': subdir,
                'learning_rate': config.get('learning_rate', 'N/A'),
                'contrastive_weight': config.get('contrastive_weight', 'N/A'),
                'dropout': config.get('dropout', 'N/A'),
                'batch_size': config.get('batch_size', 'N/A'),
                'best_epoch': best_epoch,
                'best_val_f1': best_val_f1,
                'best_val_loss': best_val_loss,
                'final_train_f1': final_train_f1,
                'final_val_f1': final_val_f1,
            }
            results.append(result)
        except Exception as e:
            print(f"  Error loading {subdir}: {e}")
    
    return results


def print_results_summary(results):
    """Print a summary of results."""
    if not results:
        print("No results found!")
        return
    
    df = pd.DataFrame(results)
    
    print(f"\n{'=' * 100}")
    print(f"HYPERPARAMETER SEARCH RESULTS ({len(results)} experiments)")
    print(f"{'=' * 100}")
    
    # Sort by best_val_f1
    df_sorted = df.sort_values('best_val_f1', ascending=False)
    
    print("\nTop 10 Results (sorted by best_val_f1):")
    print("-" * 100)
    print(df_sorted[['exp_name', 'learning_rate', 'contrastive_weight', 'dropout', 
                      'best_epoch', 'best_val_f1', 'final_val_f1']].head(10).to_string(index=False))
    
    # Best result
    best_result = df_sorted.iloc[0]
    print(f"\n{'=' * 100}")
    print("BEST RESULT:")
    print(f"{'=' * 100}")
    print(f"Experiment: {best_result['exp_name']}")
    print(f"Learning Rate: {best_result['learning_rate']}")
    print(f"Contrastive Weight: {best_result['contrastive_weight']}")
    print(f"Dropout: {best_result['dropout']}")
    print(f"Batch Size: {best_result['batch_size']}")
    print(f"Best Validation F1: {best_result['best_val_f1']:.4f} (epoch {int(best_result['best_epoch'])})")
    print(f"Final Validation F1: {best_result['final_val_f1']:.4f}")
    print(f"Final Training F1: {best_result['final_train_f1']:.4f}")
    
    # Analysis by parameter
    print(f"\n{'=' * 100}")
    print("ANALYSIS BY PARAMETER:")
    print(f"{'=' * 100}")
    
    # Learning rate analysis
    print("\nLearning Rate Analysis:")
    lr_groups = df.groupby('learning_rate')['best_val_f1'].agg(['mean', 'std', 'max', 'count'])
    print(lr_groups.to_string())
    
    # Contrastive weight analysis
    print("\nContrastive Weight Analysis:")
    cw_groups = df.groupby('contrastive_weight')['best_val_f1'].agg(['mean', 'std', 'max', 'count'])
    print(cw_groups.to_string())
    
    # Dropout analysis
    print("\nDropout Analysis:")
    do_groups = df.groupby('dropout')['best_val_f1'].agg(['mean', 'std', 'max', 'count'])
    print(do_groups.to_string())
    
    # Save detailed results to CSV
    csv_path = 'hyperparameter_search_results.csv'
    df_sorted.to_csv(csv_path, index=False)
    print(f"\n{'=' * 100}")
    print(f"Detailed results saved to: {csv_path}")
    print(f"{'=' * 100}")
    
    return df_sorted


def main():
    parser = argparse.ArgumentParser(
        description='Compare hyperparameter search results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Analyze hyperparameter search results and find optimal parameters.

Examples:
  python compare_hp_results.py --output_dir output_debug_search
  python compare_hp_results.py --output_dir output_debug_search --top 20
        '''
    )
    
    parser.add_argument('--output_dir', required=True, help='Directory containing training outputs')
    parser.add_argument('--top', type=int, default=10, help='Number of top results to display')
    
    args = parser.parse_args()
    
    results = load_training_results(args.output_dir)
    print_results_summary(results)


if __name__ == '__main__':
    main()
