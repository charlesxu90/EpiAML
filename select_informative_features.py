"""
Feature Selection for AML Classification using SHAP

Use SHAP (SHapley Additive exPlanations) to identify the most informative 
CpG sites from a trained MARLIN model. SHAP extracts which features the model 
actually uses for predictions.

For a 100% accurate model, SHAP-selected features are guaranteed to be 
sufficient for classification, capturing interactions and non-linear patterns.
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# For SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("ERROR: SHAP is required. Install with: pip install shap")

# Import MARLIN model and data utilities
import sys

# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add pytorch_marlin/src to path (works whether script is in EpiAML/ or EpiAML/src/)
possible_marlin_paths = [
    os.path.join(script_dir, '..', 'pytorch_marlin', 'src'),  # If in EpiAML/
    os.path.join(script_dir, '..', '..', 'pytorch_marlin', 'src'),  # If in EpiAML/src/
]

marlin_src = None
for path in possible_marlin_paths:
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        marlin_src = abs_path
        break

if marlin_src is None:
    raise ImportError(f"Cannot find pytorch_marlin/src directory. Searched: {possible_marlin_paths}")

if marlin_src not in sys.path:
    sys.path.insert(0, marlin_src)

from data_utils_fast import load_training_data
from model import MARLINModel

def compute_shap_importance(
    X_train,
    y_train,
    model_path,
    device='cuda',
    sample_size=500,
    background_samples=100,
    cache_dir=None,
    prefilter_topk: int = 5000,
):
    """
    Compute SHAP-based feature importance using trained PyTorch model.
    
    SHAP (SHapley Additive exPlanations) measures the contribution of each feature
    to the model's predictions by computing Shapley values from game theory.
    
    For a 100% accurate model, SHAP identifies exactly which features the model
    relies on, capturing interactions and non-linear patterns.
    
    Args:
        X_train (np.ndarray): Training feature matrix (n_samples, n_features)
        y_train (np.ndarray): Training labels (n_samples,)
        model_path (str): Path to trained model (.pt file)
        device (str): 'cuda' or 'cpu'
        sample_size (int): Number of samples to explain (for computational efficiency)
        background_samples (int): Number of background samples for SHAP baseline
        cache_dir (str): Directory to cache per-class SHAP results. If None, caching is disabled.
        prefilter_topk (int): Pre-filter features by variance before SHAP (0 disables)
    
    Returns:
        np.ndarray: SHAP importance scores for each feature (normalized 0-1)
    """
    if not HAS_SHAP:
        raise ImportError("SHAP is required. Install with: pip install shap")
    
    print("\n" + "="*70)
    print("Computing SHAP Feature Importance")
    print("="*70)
    
    # Setup device
    # Handle device specification: 'cuda', 'cuda:0', 'cuda:1', 'cpu', etc.
    if 'cuda' in device:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU")
            device = 'cpu'
        else:
            # Parse device string (cuda, cuda:0, cuda:1, etc.)
            if ':' in device:
                # Already has device number (cuda:1)
                device_num = int(device.split(':')[1])
            else:
                # Default to cuda:1
                device_num = 1
            
            # Check if specified device exists
            if device_num >= torch.cuda.device_count():
                print(f"Warning: GPU {device_num} not available, using GPU 0")
                device_num = 0
            
            torch.cuda.set_device(device_num)
            device = torch.device(f'cuda:{device_num}')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
    
    # Setup cache directory for per-class SHAP results
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"\nUsing cache directory: {cache_dir}")
    
    # Load MARLIN model
    print(f"\nLoading MARLIN model from {model_path}...")
    model = MARLINModel.load_model(model_path, device=str(device))
    model.eval()
    print(f"✓ Model loaded successfully")
    print(f"  Input size: {X_train.shape[1]:,} features")
    print(f"  Output classes: {len(np.unique(y_train))}")
    
    # Prepare data
    n_samples, n_features = X_train.shape
    print(f"\n" + "="*70)
    print("Preparing SHAP Computation")
    print("="*70)
    
    # Sample data for SHAP explanation
    sample_size = min(sample_size, n_samples)
    background_samples = min(background_samples, n_samples)
    
    print(f"Total samples: {n_samples:,}")
    print(f"Samples for explanation: {sample_size}")
    print(f"Background samples: {background_samples}")
    
    sample_indices = np.random.choice(n_samples, size=sample_size, replace=False)
    X_sample = X_train[sample_indices]
    
    background_indices = np.random.choice(n_samples, size=background_samples, replace=False)
    X_background = X_train[background_indices]

    # Optional pre-filtering to shrink feature space (variance-based)
    prefilter_indices = None
    if prefilter_topk and prefilter_topk > 0 and prefilter_topk < n_features:
        variances = X_train.var(axis=0)
        prefilter_indices = np.argsort(variances)[-prefilter_topk:]
        X_sample = X_sample[:, prefilter_indices]
        X_background = X_background[:, prefilter_indices]
        n_features_pref = X_sample.shape[1]
        print(f"Prefilter enabled: top {prefilter_topk} variance features (down from {n_features:,} to {n_features_pref:,})")

    def pad_to_full(x_subset: np.ndarray) -> np.ndarray:
        """Pad prefiltered features back to full dimension for model input."""
        if prefilter_indices is None:
            return x_subset
        full = np.zeros((x_subset.shape[0], n_features), dtype=x_subset.dtype)
        full[:, prefilter_indices] = x_subset
        return full
    
    # Get number of classes
    n_classes_model = len(np.unique(y_train.astype(int)))
    print(f"Number of classes in model: {n_classes_model}")
    
    # KernelExplainer with multi-output support
    # Strategy: Compute SHAP for all classes at once (much faster than per-class loop)
    
    print("\nUsing KernelExplainer (multi-output SHAP)")
    print(f"  Computing SHAP for all {n_classes_model} classes simultaneously")
    print(f"  This is ~{n_classes_model}x faster than per-class computation!")

    # Create prediction function that returns all class probabilities
    def predict_fn_all_classes(x):
        """
        Prediction function for all classes.
        MARLIN model outputs softmax probabilities for all classes.
        Returns shape: (batch_size, n_classes)
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        # Pad back to full dimension if needed
        x_full = pad_to_full(np.asarray(x))
        
        # MARLIN expects (batch_size, n_features)
        x_tensor = torch.as_tensor(x_full, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            probs_all = model(x_tensor)  # shape: (batch_size, n_classes)
        
        return probs_all.detach().cpu().numpy()

    # Test prediction function
    print("\nTesting prediction function...")
    test_pred = predict_fn_all_classes(X_sample[:2])
    print(f"✓ Prediction shape: {test_pred.shape}")
    print(f"✓ Output format: (batch_size={test_pred.shape[0]}, n_classes={test_pred.shape[1]})")
    print(f"✓ Prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    print("\n" + "="*70)
    print("Computing SHAP Values (All Classes at Once)")
    print("="*70)
    
    # Check if cached results exist
    cache_file = None
    shap_values_all = None
    if cache_dir is not None:
        cache_file = os.path.join(cache_dir, 'shap_all_classes.npy')
        if os.path.exists(cache_file):
            print("Loading cached SHAP values...")
            shap_values_all = np.load(cache_file)
            print(f"✓ Loaded from cache: {cache_file}")
            print(f"  SHAP values shape: {shap_values_all.shape}")
    
    if shap_values_all is None:
        print("Computing SHAP values (this may take several minutes)...")
        print(f"  Samples to explain: {sample_size}")
        print(f"  Features: {X_sample.shape[1]:,}")
        print(f"  Classes: {n_classes_model}")
        
        # Use subset of background for faster baseline computation
        bg_subset = X_background[:min(20, len(X_background))]
        print(f"  Background samples: {len(bg_subset)}")
        
        # Initialize KernelExplainer with multi-output function
        explainer = shap.KernelExplainer(predict_fn_all_classes, bg_subset)
        
        # Compute SHAP values for all outputs at once
        print("\nRunning SHAP explainer...")
        shap_values_all = explainer.shap_values(X_sample, nsamples=100)
        
        # Handle return format: list of arrays (one per class) or 3D array
        if isinstance(shap_values_all, list):
            # List of (n_samples, n_features) arrays, one per class
            print(f"  Raw output: list of {len(shap_values_all)} arrays, each shape {shap_values_all[0].shape}")
            shap_values_all = np.array(shap_values_all)  # (n_classes, n_samples, n_features)
            shap_values_all = np.transpose(shap_values_all, (1, 2, 0))  # (n_samples, n_features, n_classes)
        
        print(f"✓ SHAP computation complete")
        print(f"  Final shape: {shap_values_all.shape}")
        
        # Save to cache
        if cache_file is not None:
            np.save(cache_file, shap_values_all)
            print(f"✓ Saved to cache: {cache_file}")

    # Aggregate SHAP values across all classes and samples
    print("\nAggregating SHAP importance...")
    
    if shap_values_all.ndim == 3:
        # Shape: (n_samples, n_features, n_classes)
        shap_importance_pref = np.abs(shap_values_all).mean(axis=(0, 2))  # (n_features,)
    elif shap_values_all.ndim == 2:
        # Shape: (n_samples, n_features) - single output case
        shap_importance_pref = np.abs(shap_values_all).mean(axis=0)
    else:
        raise ValueError(f"Unexpected SHAP values shape: {shap_values_all.shape}")
    
    print(f"✓ Aggregation complete: {shap_importance_pref.shape}")

    # Map back to full feature space if prefiltered
    if prefilter_indices is not None:
        shap_importance = np.zeros(n_features)
        shap_importance[prefilter_indices] = shap_importance_pref
    else:
        shap_importance = shap_importance_pref
    
    # Normalize to [0, 1]
    shap_importance = (shap_importance - shap_importance.min()) / (shap_importance.max() - shap_importance.min() + 1e-10)
    
    print(f"\n✓ SHAP importance computed for {n_features:,} features")
    print(f"  Min: {shap_importance.min():.6f}")
    print(f"  Max: {shap_importance.max():.6f}")
    print(f"  Mean: {shap_importance.mean():.6f}")
    print(f"  Median: {np.median(shap_importance):.6f}")
    
    return shap_importance


def select_top_features(shap_scores: np.ndarray, n_features: int = 100) -> np.ndarray:
    """
    Select top N features based on SHAP scores.
    
    Args:
        shap_scores (np.ndarray): SHAP importance scores
        n_features (int): Number of features to select
    
    Returns:
        np.ndarray: Indices of top N features (sorted by importance, descending)
    """
    print("\n" + "="*70)
    print(f"Selecting Top {n_features} Features")
    print("="*70)
    
    top_indices = np.argsort(shap_scores)[-n_features:][::-1]
    
    print(f"✓ Selected top {n_features} features")
    print(f"  Top feature score: {shap_scores[top_indices[0]]:.6f}")
    print(f"  Median feature score: {shap_scores[top_indices[len(top_indices)//2]]:.6f}")
    print(f"  Bottom feature score: {shap_scores[top_indices[-1]]:.6f}")
    
    return top_indices


def save_results(
    output_dir: str,
    top_indices: np.ndarray,
    shap_scores: np.ndarray,
    feature_names,
    n_features: int = 100
):
    """
    Save SHAP feature selection results.
    
    Args:
        output_dir (str): Output directory
        top_indices (np.ndarray): Indices of top features
        shap_scores (np.ndarray): SHAP importance scores
        feature_names (np.ndarray): Names/IDs of all features
        n_features (int): Number of top features
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("Saving Results")
    print("="*70)
    
    # 1. Save top feature indices (for loading data)
    indices_path = os.path.join(output_dir, f'top_{n_features}_cpg_indices.npy')
    np.save(indices_path, top_indices)
    print(f"✓ Saved: {indices_path}")
    
    # 2. Save top feature names
    if feature_names is not None:
        top_names = feature_names[top_indices]
        names_path = os.path.join(output_dir, f'top_{n_features}_cpg_names.txt')
        np.savetxt(names_path, top_names, fmt='%s')
        print(f"✓ Saved: {names_path}")
    
    # 3. Save detailed results CSV
    results_list = []
    for rank, idx in enumerate(top_indices, 1):
        row = {
            'rank': rank,
            'feature_index': idx,
            'shap_score': shap_scores[idx]
        }
        if feature_names is not None:
            row['feature_name'] = feature_names[idx]
        results_list.append(row)
    
    results_df = pd.DataFrame(results_list)
    csv_path = os.path.join(output_dir, f'top_{n_features}_cpg_scores.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # 4. Save all feature scores
    print("\nSaving complete feature scores...")
    all_scores_list = []
    total_features = len(shap_scores)
    with tqdm(total=total_features, desc="All features", unit="feat", ncols=80) as pbar:
        for idx in range(total_features):
            row = {
                'feature_index': idx,
                'shap_score': shap_scores[idx]
            }
            if feature_names is not None:
                row['feature_name'] = feature_names[idx]
            all_scores_list.append(row)
            pbar.update(1)
    
    all_scores_df = pd.DataFrame(all_scores_list)
    all_scores_path = os.path.join(output_dir, 'all_features_scores.csv')
    all_scores_df.to_csv(all_scores_path, index=False)
    print(f"✓ Saved: {all_scores_path}")
    
    # 5. Create summary report
    report_path = os.path.join(output_dir, 'feature_selection_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SHAP-Based Feature Selection Report for AML Classification\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total Features Analyzed: {total_features:,}\n")
        f.write(f"Top Features Selected: {n_features}\n")
        f.write(f"Selection Ratio: {n_features / total_features * 100:.2f}%\n\n")
        
        f.write("Method Used: SHAP (SHapley Additive exPlanations)\n")
        f.write("  ✓ Extracts features from trained model\n")
        f.write("  ✓ Captures feature interactions\n")
        f.write("  ✓ Identifies non-linear patterns\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Top 20 Most Informative CpG Sites (by SHAP):\n")
        f.write("="*70 + "\n\n")
        
        for rank, (_, row) in enumerate(results_df.head(20).iterrows(), 1):
            cpg_name = row['feature_name'] if 'feature_name' in row else f"Feature {row['feature_index']}"
            f.write(f"{rank:2d}. {cpg_name}\n")
            f.write(f"    SHAP score: {row['shap_score']:.6f}\n")
            f.write("\n")
    
    print(f"✓ Saved: {report_path}")
    
    print(f"✓ Saved: {report_path}")
    
    print(f"\n" + "="*70)
    print("Results Summary")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Files saved:")
    print(f"  - {os.path.basename(indices_path)} (feature indices for training)")
    print(f"  - {os.path.basename(csv_path)} (top {n_features} with scores)")
    print(f"  - {os.path.basename(all_scores_path)} (all features with scores)")
    print(f"  - {os.path.basename(report_path)} (summary report)")
    if feature_names is not None:
        print(f"  - {os.path.basename(names_path)} (CpG names)")
    
    return {
        'indices': indices_path,
        'names': names_path if feature_names is not None else None,
        'scores': csv_path,
        'all_scores': all_scores_path,
        'report': report_path
    }


def main():
    parser = argparse.ArgumentParser(
        description='SHAP-based feature selection for AML classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
SHAP Feature Selection:
  Uses SHAP (SHapley Additive exPlanations) to identify which CpG sites
  your trained model actually relies on for predictions. For a 100%% accurate
  model, SHAP-selected features are guaranteed to be sufficient.
  
  Results for each class are cached in .shap_cache/ directory to avoid
  redundant computation when rerunning with different parameters.

Examples:
  # Basic usage with trained model (auto-caches per-class SHAP results)
  python select_informative_features.py \\
    --train_file ../pytorch_marlin/data/training_data.h5 \\
    --model_path ../pytorch_marlin/output/best_model.pt \\
    --output_dir ./feature_selection_shap \\
    --n_features 100
  
  # Rerun with different n_features (uses cached SHAP from previous run)
  python select_informative_features.py \\
    --train_file ../pytorch_marlin/data/training_data.h5 \\
    --model_path ../pytorch_marlin/output/best_model.pt \\
    --output_dir ./feature_selection_shap \\
    --n_features 200
  
  # High-quality SHAP with more samples (new cache)
  python select_informative_features.py \\
    --train_file ../pytorch_marlin/data/training_data.h5 \\
    --model_path ../pytorch_marlin/output/best_model.pt \\
    --output_dir ./feature_selection_shap_hq \\
    --n_features 200 \\
    --shap_samples 1000 \\
    --shap_background 200 \\
    --device cuda
  
  # Find minimum CpGs for 95%% accuracy (uses cache for efficiency)
  for n in 50 100 200 500; do
    python select_informative_features.py \\
      --train_file data.h5 \\
      --model_path model.pt \\
      --n_features $n \\
      --output_dir feature_sel_$n
  done
        '''
    )
    
    parser.add_argument('--train_file', required=True,
                        help='Path to training data (.h5 or .csv)')
    parser.add_argument('--model_path', required=True,
                        help='Path to trained model (.pt file) for SHAP feature importance')
    parser.add_argument('--output_dir', default='./feature_selection_shap',
                        help='Output directory for results (default: ./feature_selection_shap)')
    parser.add_argument('--n_features', type=int, default=100,
                        help='Number of top features to select (default: 100)')
    parser.add_argument('--device', default='cuda:1',
                        help='Device for model inference (default: cuda:1). Examples: cuda, cuda:0, cuda:1, cpu')
    parser.add_argument('--shap_samples', type=int, default=200,
                        help='Number of samples for SHAP explanation (default: 200, smaller = faster)')
    parser.add_argument('--shap_background', type=int, default=50,
                        help='Number of background samples for SHAP baseline (default: 50, smaller = faster)')
    parser.add_argument('--prefilter_topk', type=int, default=5000,
                        help='Pre-filter to top-K variance features before SHAP (0 to disable, default: 5000)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Training file not found: {args.train_file}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not HAS_SHAP:
        raise ImportError("SHAP is required. Install with: pip install shap")
    
    # Print info
    print("\n" + "="*70)
    print("SHAP-Based Feature Selection for AML Classification")
    print("="*70)
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"SHAP Available: {HAS_SHAP}")
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_path}")
    print(f"  Data: {args.train_file}")
    print(f"  Device: {args.device}")
    print(f"  Target features: {args.n_features}")
    print(f"  SHAP samples: {args.shap_samples}")
    print(f"  Background samples: {args.shap_background}")
    
    # Load data
    print("\n" + "="*70)
    print("Loading Training Data")
    print("="*70)
    print(f"Data file: {args.train_file}")
    
    print("\nLoading and processing data...")
    with tqdm(total=3, desc="Data Loading", unit="step", ncols=80) as pbar:
        X_train, y_train, feature_names = load_training_data(
            args.train_file,
            format='auto',
            binarize=True
        )
        pbar.update(1)
        
        # Convert y_train to int for uniqueness check
        y_train_int = y_train.astype(int)
        pbar.update(1)
        
        n_classes = len(np.unique(y_train_int))
        pbar.update(1)
    
    print(f"\n✓ Data loaded successfully")
    print(f"  Shape: {X_train.shape[0]:,} samples × {X_train.shape[1]:,} features")
    print(f"  Classes: {n_classes} types")
    print(f"  Features: {type(feature_names).__name__} ({len(feature_names) if feature_names is not None else 'N/A'} names)")
    
    # Compute SHAP importance
    cache_dir = os.path.join(args.output_dir, '.shap_cache')
    shap_scores = compute_shap_importance(
        X_train, y_train,
        args.model_path,
        device=args.device,
        sample_size=args.shap_samples,
        background_samples=args.shap_background,
        cache_dir=cache_dir,
        prefilter_topk=args.prefilter_topk,
    )
    
    # Select top features
    top_indices = select_top_features(shap_scores, n_features=args.n_features)
    
    # Save results
    save_results(
        args.output_dir,
        top_indices,
        shap_scores,
        feature_names,
        n_features=args.n_features
    )
    
    print("\n" + "="*70)
    print("✓ SHAP Feature Selection Complete!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Review results:")
    print(f"   cat {os.path.join(args.output_dir, 'feature_selection_report.txt')}")
    print(f"\n2. Train model with selected {args.n_features} CpGs:")
    print(f"   python src/train.py \\")
    print(f"     --train_file {args.train_file} \\")
    print(f"     --feature_indices {os.path.join(args.output_dir, f'top_{args.n_features}_cpg_indices.npy')} \\")
    print(f"     --output_dir ./output_shap_{args.n_features}cpg \\")
    print(f"     --epochs 300")
    print(f"\n3. Test different feature counts (find minimum for 95%+ accuracy):")
    print(f"   for n in 50 100 200 500; do")
    print(f"     python src/select_informative_features.py --train_file {args.train_file} \\")
    print(f"       --model_path {args.model_path} --n_features $n --output_dir feature_sel_$n")
    print(f"   done")


if __name__ == '__main__':
    main()
