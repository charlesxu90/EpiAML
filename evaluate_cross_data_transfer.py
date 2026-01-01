#!/usr/bin/env python3
"""
Evaluate cross-data transfer ability of PytorchMARLIN model.
Tests if models trained on array data can classify nanopore data correctly.

This script:
1. Prepares nanopore data with updated labels (using class_id from array label mapping)
2. Evaluates different model configurations (full features, reduced features)
3. Generates evaluation metrics and figures
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import h5py
import torch
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    top_k_accuracy_score
)
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Add pytorch_marlin to path
MARLIN_PATH = Path('/home/xux/Desktop/NanoALC/pytorch_marlin/src')
sys.path.insert(0, str(MARLIN_PATH))

from model import MARLINModel


def load_label_mapping(array_labels_file):
    """Load label mapping from array data labels file."""
    print(f"Loading label mapping from {array_labels_file}...")
    df = pd.read_csv(array_labels_file)

    # Create mapping: class_id -> merged_label
    class_to_label = {}
    for _, row in df.iterrows():
        class_id = int(row['class_id'])
        merged_label = row['merged_label']
        if class_id not in class_to_label:
            class_to_label[class_id] = merged_label

    print(f"  Found {len(class_to_label)} unique classes")
    return class_to_label


def prepare_nanopore_data(
    csv_file,
    labels_file,
    reference_features_file,
    output_h5_file,
    binarize_threshold=0.5
):
    """
    Prepare nanopore data with updated labels.

    Args:
        csv_file: Path to nanopore CSV data
        labels_file: Path to updated labels CSV (with class_id)
        reference_features_file: Path to reference features (for feature ordering)
        output_h5_file: Path to output HDF5 file
        binarize_threshold: Threshold for binarizing methylation values

    Returns:
        Tuple of (data, labels, sample_ids, class_ids)
    """
    import csv as csv_module

    print("\n" + "="*60)
    print("PREPARING NANOPORE DATA")
    print("="*60)

    # Load updated labels
    print(f"\nLoading updated labels from {labels_file}...")
    labels_df = pd.read_csv(labels_file)
    print(f"  Columns: {list(labels_df.columns)}")
    print(f"  Samples in labels: {len(labels_df)}")

    # Create sample_id -> class_id mapping
    sample_to_class = {}
    sample_to_label = {}
    for _, row in labels_df.iterrows():
        sample_id = row['sample_id']
        class_id = int(row['class_id'])
        merged_label = row['merged_label']
        sample_to_class[sample_id] = class_id
        sample_to_label[sample_id] = merged_label

    print(f"  Unique classes: {len(set(sample_to_class.values()))}")

    # Load reference features
    print(f"\nLoading reference features from {reference_features_file}...")
    with open(reference_features_file, 'r') as f:
        reference_features = [line.strip() for line in f]
    n_features = len(reference_features)
    print(f"  Reference features: {n_features:,}")

    # Load nanopore data using fast CSV reader
    print(f"\nLoading nanopore data from {csv_file}...")
    print("  (Using fast CSV reader...)")

    # Read header first
    with open(csv_file, 'r') as f:
        reader = csv_module.reader(f)
        header = next(reader)
        csv_features = header[1:]  # Skip sample_id
        n_samples = sum(1 for _ in reader)

    print(f"  CSV features: {len(csv_features):,}")
    print(f"  Samples: {n_samples}")

    # Create feature index mapping: reference_feature -> csv_column_index
    csv_feature_to_idx = {feat: i+1 for i, feat in enumerate(csv_features)}  # +1 for sample_id offset

    common_features = set(reference_features) & set(csv_features)
    print(f"  Common features: {len(common_features):,}")

    # Create ordered indices for extraction
    ordered_indices = []
    for feat in reference_features:
        if feat in csv_feature_to_idx:
            ordered_indices.append(csv_feature_to_idx[feat])
        else:
            ordered_indices.append(None)  # Missing feature

    # Read full data efficiently
    print("\nReading data and reordering features...")
    data = np.zeros((n_samples, n_features), dtype=np.float32)
    sample_ids = []
    class_ids = []
    merged_labels = []

    with open(csv_file, 'r') as f:
        reader = csv_module.reader(f)
        next(reader)  # Skip header

        for i, row in enumerate(tqdm(reader, total=n_samples, desc="Processing")):
            sample_id = row[0]
            sample_ids.append(sample_id)

            # Get class_id
            if sample_id in sample_to_class:
                class_ids.append(sample_to_class[sample_id])
                merged_labels.append(sample_to_label[sample_id])
            else:
                print(f"  Warning: No label for sample {sample_id}")
                class_ids.append(-1)
                merged_labels.append("Unknown")

            # Extract features in reference order
            for j, idx in enumerate(ordered_indices):
                if idx is not None:
                    val = row[idx]
                    if val.strip():
                        try:
                            data[i, j] = float(val)
                        except ValueError:
                            data[i, j] = 0.0

    sample_ids = np.array(sample_ids)
    class_ids = np.array(class_ids)
    merged_labels = np.array(merged_labels)

    # Compute data statistics before binarization
    print("\nData statistics (before binarization):")
    non_zero_mask = data != 0
    print(f"  Non-zero values: {non_zero_mask.sum():,} ({100*non_zero_mask.sum()/(data.size):.2f}%)")
    print(f"  Zero/missing values: {(~non_zero_mask).sum():,} ({100*(~non_zero_mask).sum()/(data.size):.2f}%)")
    if non_zero_mask.sum() > 0:
        print(f"  Non-zero mean: {data[non_zero_mask].mean():.4f}")
        print(f"  Non-zero std: {data[non_zero_mask].std():.4f}")

    # Binarize data (matching array data preprocessing)
    print(f"\nBinarizing data with threshold {binarize_threshold}...")
    data_binarized = np.where(data >= binarize_threshold, 1.0, -1.0).astype(np.float32)

    # Save to HDF5
    print(f"\nSaving to {output_h5_file}...")
    os.makedirs(os.path.dirname(output_h5_file), exist_ok=True)

    with h5py.File(output_h5_file, 'w') as hf:
        hf.create_dataset('data', data=data_binarized, compression='gzip')
        hf.create_dataset('data_raw', data=data, compression='gzip')  # Also save raw

        dt = h5py.special_dtype(vlen=str)
        # Convert to list of strings for h5py compatibility
        sample_ids_str = [str(s) for s in sample_ids]
        merged_labels_str = [str(s) for s in merged_labels]

        hf.create_dataset('sample_ids', data=sample_ids_str, dtype=dt)
        hf.create_dataset('class_ids', data=class_ids)
        hf.create_dataset('merged_labels', data=merged_labels_str, dtype=dt)
        hf.create_dataset('feature_names', data=[str(f) for f in reference_features], dtype=dt)

        metadata = hf.create_group('metadata')
        metadata.attrs['n_samples'] = n_samples
        metadata.attrs['n_features'] = n_features
        metadata.attrs['binarize_threshold'] = binarize_threshold
        metadata.attrs['n_classes'] = len(set(class_ids))

    print(f"  Saved {n_samples} samples with {n_features:,} features")

    # Print class distribution
    print("\nClass distribution:")
    class_counts = Counter(zip(class_ids, merged_labels))
    for (cid, label), count in sorted(class_counts.items()):
        print(f"  {cid:2d}: {label} ({count})")

    return data_binarized, class_ids, sample_ids, merged_labels


def load_prepared_nanopore_data(h5_file):
    """Load prepared nanopore data from HDF5 file."""
    print(f"\nLoading prepared nanopore data from {h5_file}...")

    with h5py.File(h5_file, 'r') as hf:
        data = hf['data'][:]
        class_ids = hf['class_ids'][:]
        sample_ids = np.array([s.decode() if isinstance(s, bytes) else s for s in hf['sample_ids'][:]])
        merged_labels = np.array([s.decode() if isinstance(s, bytes) else s for s in hf['merged_labels'][:]])
        feature_names = np.array([s.decode() if isinstance(s, bytes) else s for s in hf['feature_names'][:]])

    print(f"  Data shape: {data.shape}")
    print(f"  Unique classes: {len(set(class_ids))}")

    return data, class_ids, sample_ids, merged_labels, feature_names


def load_model(model_path, device='cuda'):
    """Load trained MARLIN model."""
    print(f"\nLoading model from {model_path}...")
    model = MARLINModel.load_model(model_path, device=device)
    model.eval()
    print(f"  Input size: {model.input_size:,}")
    print(f"  Output size: {model.output_size}")
    print(f"  Total parameters: {model.get_num_parameters():,}")
    return model


def load_feature_indices(feature_indices_file):
    """Load feature indices for reduced feature models."""
    print(f"\nLoading feature indices from {feature_indices_file}...")
    indices = np.load(feature_indices_file)
    print(f"  Loaded {len(indices):,} feature indices")
    return indices


def evaluate_model(
    model,
    data,
    true_labels,
    class_to_name,
    feature_indices=None,
    device='cuda',
    batch_size=32
):
    """
    Evaluate model on data.

    Args:
        model: Trained MARLIN model
        data: Input data (N x F)
        true_labels: Ground truth class IDs
        class_to_name: Dict mapping class_id to label name
        feature_indices: Optional indices for reduced feature models
        device: Device for inference
        batch_size: Batch size for inference

    Returns:
        Dictionary with evaluation results
    """
    model.eval()

    # Extract features if using reduced model
    if feature_indices is not None:
        data = data[:, feature_indices]
        print(f"  Using {len(feature_indices):,} features")

    n_samples = data.shape[0]
    n_features = data.shape[1]

    # Check model input size
    if model.input_size != n_features:
        raise ValueError(f"Model expects {model.input_size:,} features, got {n_features:,}")

    # Run inference
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = torch.tensor(data[i:i+batch_size], dtype=torch.float32, device=device)
            probs = model.predict_proba(batch)
            preds = probs.argmax(dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.concatenate(all_preds)

    # Compute metrics
    accuracy = accuracy_score(true_labels, all_preds)
    balanced_acc = balanced_accuracy_score(true_labels, all_preds)

    # Top-k accuracy (for k=3 and k=5) - manual computation for robustness
    # Top-3 accuracy
    top3_indices = np.argsort(all_probs, axis=1)[:, -3:]
    top3_correct = [label in top3 for label, top3 in zip(true_labels, top3_indices)]
    top3_acc = float(np.mean(top3_correct))

    # Top-5 accuracy
    top5_indices = np.argsort(all_probs, axis=1)[:, -5:]
    top5_correct = [label in top5 for label, top5 in zip(true_labels, top5_indices)]
    top5_acc = float(np.mean(top5_correct))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, all_preds, average=None, zero_division=0
    )

    # Macro and weighted averages
    macro_precision = np.mean(precision[support > 0])
    macro_recall = np.mean(recall[support > 0])
    macro_f1 = np.mean(f1[support > 0])

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels, all_preds, average='weighted', zero_division=0
    )

    # Build results
    results = {
        'n_samples': n_samples,
        'n_features': n_features,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'predictions': all_preds,
        'probabilities': all_probs,
        'true_labels': true_labels,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_support': support
    }

    return results


def print_results(results, title, class_to_name):
    """Print evaluation results."""
    print("\n" + "="*60)
    print(title)
    print("="*60)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:           {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Balanced Accuracy:  {results['balanced_accuracy']:.4f} ({results['balanced_accuracy']*100:.2f}%)")
    if results['top3_accuracy'] is not None:
        print(f"  Top-3 Accuracy:     {results['top3_accuracy']:.4f} ({results['top3_accuracy']*100:.2f}%)")
    if results['top5_accuracy'] is not None:
        print(f"  Top-5 Accuracy:     {results['top5_accuracy']:.4f} ({results['top5_accuracy']*100:.2f}%)")

    print(f"\nMacro-Averaged Metrics:")
    print(f"  Precision:  {results['macro_precision']:.4f}")
    print(f"  Recall:     {results['macro_recall']:.4f}")
    print(f"  F1-Score:   {results['macro_f1']:.4f}")

    print(f"\nWeighted-Averaged Metrics:")
    print(f"  Precision:  {results['weighted_precision']:.4f}")
    print(f"  Recall:     {results['weighted_recall']:.4f}")
    print(f"  F1-Score:   {results['weighted_f1']:.4f}")

    # Per-sample predictions
    print(f"\nPer-Sample Predictions:")
    for i, (true, pred) in enumerate(zip(results['true_labels'], results['predictions'])):
        true_name = class_to_name.get(true, f"Class_{true}")
        pred_name = class_to_name.get(pred, f"Class_{pred}")
        correct = "OK" if true == pred else "WRONG"
        prob = results['probabilities'][i, pred]
        print(f"  {i+1:2d}. True: {true_name:30s} | Pred: {pred_name:30s} | Prob: {prob:.4f} | {correct}")


def plot_confusion_matrix(results, class_to_name, output_file, title="Confusion Matrix"):
    """Plot and save confusion matrix."""
    true_labels = results['true_labels']
    pred_labels = results['predictions']

    # Get unique classes present in data
    all_classes = sorted(set(true_labels) | set(pred_labels))
    class_names = [class_to_name.get(c, f"Class_{c}") for c in all_classes]

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=all_classes)

    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    # Plot
    fig, ax = plt.subplots(figsize=(max(12, len(all_classes)*0.5), max(10, len(all_classes)*0.4)))

    sns.heatmap(
        cm_normalized,
        annot=cm,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix to {output_file}")


def plot_per_class_metrics(results, class_to_name, output_file, title="Per-Class Metrics"):
    """Plot per-class precision, recall, and F1 scores."""
    precision = results['per_class_precision']
    recall = results['per_class_recall']
    f1 = results['per_class_f1']
    support = results['per_class_support']

    # Filter to classes with support > 0
    mask = support > 0
    class_ids = np.where(mask)[0]

    class_names = [class_to_name.get(c, f"Class_{c}") for c in class_ids]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(12, len(class_names)*0.8), 6))

    bars1 = ax.bar(x - width, precision[mask], width, label='Precision', color='steelblue')
    bars2 = ax.bar(x, recall[mask], width, label='Recall', color='darkorange')
    bars3 = ax.bar(x + width, f1[mask], width, label='F1-Score', color='forestgreen')

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved per-class metrics to {output_file}")


def plot_comparison(all_results, output_file, title="Model Comparison"):
    """Plot comparison of different models."""
    model_names = list(all_results.keys())
    metrics = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1']
    metric_labels = ['Accuracy', 'Balanced Accuracy', 'Macro F1', 'Weighted F1']

    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    for i, (name, results) in enumerate(all_results.items()):
        values = [results[m] for m in metrics]
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name, color=colors[i])

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison plot to {output_file}")


def plot_topk_accuracy(all_results, output_file, title="Top-K Accuracy Comparison"):
    """Plot top-1, top-3, top-5 accuracy for different models."""
    model_names = list(all_results.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(model_names))
    width = 0.25

    top1 = [all_results[n]['accuracy'] for n in model_names]
    top3 = [all_results[n].get('top3_accuracy', 0) or 0 for n in model_names]
    top5 = [all_results[n].get('top5_accuracy', 0) or 0 for n in model_names]

    bars1 = ax.bar(x - width, top1, width, label='Top-1', color='steelblue')
    bars2 = ax.bar(x, top3, width, label='Top-3', color='darkorange')
    bars3 = ax.bar(x + width, top5, width, label='Top-5', color='forestgreen')

    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved top-k accuracy plot to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate cross-data transfer ability of PytorchMARLIN model'
    )
    parser.add_argument('--prepare_data', action='store_true',
                       help='Prepare nanopore data with updated labels')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation on prepared data')
    parser.add_argument('--output_dir', default='output/cross_data_evaluation',
                       help='Output directory for results')
    parser.add_argument('--device', default='cuda',
                       help='Device for inference (cuda or cpu)')

    args = parser.parse_args()

    # Paths
    BASE_DIR = Path('/home/xux/Desktop/NanoALC')
    DATA_DIR = BASE_DIR / 'data'
    MARLIN_DIR = BASE_DIR / 'pytorch_marlin'

    nanopore_csv = DATA_DIR / 'nanopore_data' / 'NG_nanopore_training_data(1).csv'
    nanopore_labels = DATA_DIR / 'nanopore_data' / 'NG_nanopore_labels(1).csv'
    array_labels = DATA_DIR / 'array_data' / 'labels.csv'
    reference_features = DATA_DIR / 'array_data' / 'reference_features.txt'

    prepared_nanopore_h5 = DATA_DIR / 'nanopore_data' / 'nanopore_prepared.h5'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load class to name mapping
    class_to_name = load_label_mapping(array_labels)

    # Step 1: Prepare data if requested or if not already prepared
    if args.prepare_data or not prepared_nanopore_h5.exists():
        data, class_ids, sample_ids, merged_labels = prepare_nanopore_data(
            csv_file=str(nanopore_csv),
            labels_file=str(nanopore_labels),
            reference_features_file=str(reference_features),
            output_h5_file=str(prepared_nanopore_h5)
        )
    else:
        print(f"\nUsing existing prepared data: {prepared_nanopore_h5}")

    # Step 2: Run evaluation if requested
    if args.evaluate:
        # Load prepared data
        data, class_ids, sample_ids, merged_labels, feature_names = load_prepared_nanopore_data(
            str(prepared_nanopore_h5)
        )

        # Define models to evaluate
        models_to_evaluate = {
            'Full (357K features)': {
                'model_path': MARLIN_DIR / 'results' / 'array_data' / 'marlin_model.pt',
                'feature_indices': None
            },
            '1000 features': {
                'model_path': DATA_DIR / 'feature_selection_shap' / 'training_data_top1000.h5-result' / 'marlin_model.pt',
                'feature_indices': DATA_DIR / 'feature_selection_shap' / 'top_1000_cpg_indices.npy'
            }
        }

        # Check for other reduced feature models
        for n_features in [5000, 10000, 50000]:
            feature_dir = DATA_DIR / f'feature_selection_shap_{n_features//1000}k'
            if feature_dir.exists():
                indices_file = feature_dir / f'top_{n_features}_cpg_indices.npy'
                # Look for trained model
                result_dir = feature_dir / f'training_data_top{n_features}.h5-result'
                model_file = result_dir / 'marlin_model.pt'

                if indices_file.exists() and model_file.exists():
                    models_to_evaluate[f'{n_features} features'] = {
                        'model_path': model_file,
                        'feature_indices': indices_file
                    }

        print("\n" + "="*60)
        print("MODELS TO EVALUATE")
        print("="*60)
        for name, config in models_to_evaluate.items():
            exists = config['model_path'].exists() if config['model_path'] else False
            print(f"  {name}: {'EXISTS' if exists else 'NOT FOUND'}")
            if config['feature_indices']:
                idx_exists = config['feature_indices'].exists()
                print(f"    Feature indices: {'EXISTS' if idx_exists else 'NOT FOUND'}")

        # Run evaluations
        all_results = {}

        for name, config in models_to_evaluate.items():
            model_path = config['model_path']
            feature_indices_path = config['feature_indices']

            if not model_path.exists():
                print(f"\nSkipping {name}: model not found")
                continue

            print(f"\n{'='*60}")
            print(f"EVALUATING: {name}")
            print(f"{'='*60}")

            # Load model
            model = load_model(str(model_path), device=device)

            # Load feature indices if needed
            feature_indices = None
            if feature_indices_path and feature_indices_path.exists():
                feature_indices = load_feature_indices(str(feature_indices_path))

            # Run evaluation
            results = evaluate_model(
                model=model,
                data=data,
                true_labels=class_ids,
                class_to_name=class_to_name,
                feature_indices=feature_indices,
                device=device
            )

            all_results[name] = results

            # Print results
            print_results(results, f"Results: {name}", class_to_name)

            # Plot confusion matrix
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
            plot_confusion_matrix(
                results,
                class_to_name,
                output_dir / f'confusion_matrix_{safe_name}.png',
                title=f'Confusion Matrix - {name}'
            )

            # Plot per-class metrics
            plot_per_class_metrics(
                results,
                class_to_name,
                output_dir / f'per_class_metrics_{safe_name}.png',
                title=f'Per-Class Metrics - {name}'
            )

        # Generate comparison plots
        if len(all_results) > 1:
            plot_comparison(
                all_results,
                output_dir / 'model_comparison.png',
                title='Cross-Data Transfer: Array to Nanopore'
            )

            plot_topk_accuracy(
                all_results,
                output_dir / 'topk_accuracy_comparison.png',
                title='Top-K Accuracy: Array to Nanopore'
            )

        # Save summary to JSON
        summary = {}
        for name, results in all_results.items():
            summary[name] = {
                'n_samples': int(results['n_samples']),
                'n_features': int(results['n_features']),
                'accuracy': float(results['accuracy']),
                'balanced_accuracy': float(results['balanced_accuracy']),
                'top3_accuracy': float(results['top3_accuracy']) if results['top3_accuracy'] else None,
                'top5_accuracy': float(results['top5_accuracy']) if results['top5_accuracy'] else None,
                'macro_precision': float(results['macro_precision']),
                'macro_recall': float(results['macro_recall']),
                'macro_f1': float(results['macro_f1']),
                'weighted_precision': float(results['weighted_precision']),
                'weighted_recall': float(results['weighted_recall']),
                'weighted_f1': float(results['weighted_f1'])
            }

        with open(output_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved evaluation summary to {output_dir / 'evaluation_summary.json'}")

        # Print final summary table
        print("\n" + "="*80)
        print("FINAL SUMMARY: Cross-Data Transfer Evaluation")
        print("="*80)
        print(f"{'Model':<25} {'Acc':>8} {'Bal.Acc':>8} {'Top-3':>8} {'Top-5':>8} {'F1':>8}")
        print("-"*80)
        for name, s in summary.items():
            top3 = f"{s['top3_accuracy']:.4f}" if s['top3_accuracy'] else "N/A"
            top5 = f"{s['top5_accuracy']:.4f}" if s['top5_accuracy'] else "N/A"
            print(f"{name:<25} {s['accuracy']:>8.4f} {s['balanced_accuracy']:>8.4f} {top3:>8} {top5:>8} {s['weighted_f1']:>8.4f}")

        print(f"\nOutput directory: {output_dir}")


if __name__ == '__main__':
    main()
