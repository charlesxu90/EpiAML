"""
Create Filtered Dataset from SHAP Feature Selection

This script creates a smaller dataset containing only the features selected
by SHAP analysis, reducing the feature space while preserving discriminative power.
"""

import os
import argparse
import numpy as np
import h5py
from pathlib import Path


def load_full_dataset(data_path):
    """
    Load the full dataset from HDF5 or CSV file.

    Args:
        data_path (str): Path to the original data file

    Returns:
        tuple: (data, labels, feature_names)
    """
    ext = os.path.splitext(data_path)[1].lower()

    print(f"\nLoading full dataset from: {data_path}")

    if ext in ['.h5', '.hdf5']:
        with h5py.File(data_path, 'r') as f:
            data = f['data'][:]
            labels = f['labels'][:] if 'labels' in f else None

            # Handle feature names
            if 'feature_names' in f:
                feature_names = f['feature_names'][:].astype(str)
            else:
                feature_names = None

            # Handle string labels
            if labels is not None and labels.dtype.kind in ['O', 'S', 'U']:
                unique_labels = np.unique(labels)
                label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                labels = np.array([label_to_idx[label] for label in labels])

    elif ext == '.csv':
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for CSV loading. Install with: pip install pandas")

        df = pd.read_csv(data_path)
        labels = df.iloc[:, 0].values

        # Convert string labels to integers if needed
        if labels.dtype == object or labels.dtype.kind in ['S', 'U']:
            unique_labels = np.unique(labels)
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            labels = np.array([label_to_idx[label] for label in labels])

        data = df.iloc[:, 1:].values
        feature_names = np.array(df.columns[1:].tolist())

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    print(f"  Loaded: {data.shape[0]:,} samples × {data.shape[1]:,} features")
    if labels is not None:
        print(f"  Classes: {len(np.unique(labels))} ({np.unique(labels)})")

    return data, labels, feature_names


def load_selected_features(feature_selection_dir, n_features=1000):
    """
    Load selected feature indices from SHAP results.

    Args:
        feature_selection_dir (str): Directory containing SHAP results
        n_features (int): Number of features that were selected

    Returns:
        tuple: (feature_indices, feature_names)
    """
    print(f"\nLoading selected features from: {feature_selection_dir}")

    # Load indices
    indices_path = os.path.join(feature_selection_dir, f'top_{n_features}_cpg_indices.npy')
    if not os.path.exists(indices_path):
        raise FileNotFoundError(f"Feature indices not found: {indices_path}")

    feature_indices = np.load(indices_path)
    print(f"  Loaded {len(feature_indices)} feature indices")

    # Load feature names if available
    names_path = os.path.join(feature_selection_dir, f'top_{n_features}_cpg_names.txt')
    if os.path.exists(names_path):
        feature_names = np.loadtxt(names_path, dtype=str)
        print(f"  Loaded {len(feature_names)} feature names")
    else:
        feature_names = None
        print(f"  Feature names not found, using indices")

    return feature_indices, feature_names


def create_filtered_dataset(data, labels, feature_names_full, feature_indices, feature_names_selected):
    """
    Create filtered dataset with only selected features.

    Args:
        data (np.ndarray): Full dataset
        labels (np.ndarray): Labels
        feature_names_full (np.ndarray): All feature names from original dataset
        feature_indices (np.ndarray): Indices of selected features
        feature_names_selected (np.ndarray): Names of selected features

    Returns:
        tuple: (filtered_data, labels, filtered_feature_names)
    """
    print(f"\nFiltering dataset...")
    print(f"  Original: {data.shape[0]:,} samples × {data.shape[1]:,} features")

    # Filter data
    filtered_data = data[:, feature_indices]

    # Get feature names for filtered dataset
    if feature_names_selected is not None:
        filtered_feature_names = feature_names_selected
    elif feature_names_full is not None:
        filtered_feature_names = feature_names_full[feature_indices]
    else:
        filtered_feature_names = np.array([f"feature_{i}" for i in feature_indices])

    print(f"  Filtered: {filtered_data.shape[0]:,} samples × {filtered_data.shape[1]:,} features")
    print(f"  Reduction: {data.shape[1]:,} → {filtered_data.shape[1]:,} features ({filtered_data.shape[1]/data.shape[1]*100:.2f}%)")

    return filtered_data, labels, filtered_feature_names


def save_filtered_dataset(output_path, data, labels, feature_names, metadata=None):
    """
    Save filtered dataset to HDF5 or CSV file.

    Args:
        output_path (str): Path for output file
        data (np.ndarray): Filtered data
        labels (np.ndarray): Labels
        feature_names (np.ndarray): Feature names
        metadata (dict): Optional metadata to include
    """
    ext = os.path.splitext(output_path)[1].lower()

    print(f"\nSaving filtered dataset to: {output_path}")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    if ext in ['.h5', '.hdf5']:
        with h5py.File(output_path, 'w') as f:
            # Save data and labels
            f.create_dataset('data', data=data, compression='gzip', compression_opts=9)
            if labels is not None:
                f.create_dataset('labels', data=labels)

            # Save feature names
            if feature_names is not None:
                # Convert to bytes for HDF5 compatibility
                feature_names_bytes = [name.encode('utf-8') for name in feature_names]
                f.create_dataset('feature_names', data=feature_names_bytes)

            # Save metadata
            if metadata:
                for key, value in metadata.items():
                    f.attrs[key] = value

            # Add creation info
            f.attrs['created_by'] = 'create_filtered_dataset.py'
            f.attrs['n_samples'] = data.shape[0]
            f.attrs['n_features'] = data.shape[1]
            if labels is not None:
                f.attrs['n_classes'] = len(np.unique(labels))

        print(f"  Saved as HDF5")

    elif ext == '.csv':
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for CSV saving. Install with: pip install pandas")

        # Create DataFrame
        df_data = pd.DataFrame(data, columns=feature_names if feature_names is not None else None)

        if labels is not None:
            df = pd.DataFrame({'label': labels})
            df = pd.concat([df, df_data], axis=1)
        else:
            df = df_data

        df.to_csv(output_path, index=False)
        print(f"  Saved as CSV")

    else:
        raise ValueError(f"Unsupported output format: {ext}")

    print(f"  Dataset shape: {data.shape[0]:,} samples × {data.shape[1]:,} features")
    if labels is not None:
        print(f"  Classes: {len(np.unique(labels))}")


def main():
    parser = argparse.ArgumentParser(
        description='Create filtered dataset from SHAP feature selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Create filtered dataset with top 1000 features
  python create_filtered_dataset.py \\
    --input_data ../pytorch_marlin/data/training_data.h5 \\
    --feature_selection_dir ./feature_selection_shap_sample \\
    --output_data ./data/training_data_top1000.h5 \\
    --n_features 1000

  # Create CSV output instead
  python create_filtered_dataset.py \\
    --input_data ../pytorch_marlin/data/training_data.h5 \\
    --feature_selection_dir ./feature_selection_shap_sample \\
    --output_data ./data/training_data_top1000.csv \\
    --n_features 1000

  # Create multiple filtered datasets with different feature counts
  for n in 500 1000 2000; do
    python create_filtered_dataset.py \\
      --input_data data.h5 \\
      --feature_selection_dir feature_selection_shap \\
      --output_data data_top${n}.h5 \\
      --n_features $n
  done
        '''
    )

    parser.add_argument('--input_data', required=True,
                        help='Path to original full dataset (.h5 or .csv)')
    parser.add_argument('--feature_selection_dir', required=True,
                        help='Directory containing SHAP feature selection results')
    parser.add_argument('--output_data', required=True,
                        help='Path for output filtered dataset (.h5 or .csv)')
    parser.add_argument('--n_features', type=int, default=1000,
                        help='Number of top features to include (default: 1000)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input_data):
        raise FileNotFoundError(f"Input data not found: {args.input_data}")
    if not os.path.exists(args.feature_selection_dir):
        raise FileNotFoundError(f"Feature selection directory not found: {args.feature_selection_dir}")

    print("="*70)
    print("Create Filtered Dataset from SHAP Feature Selection")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Input data: {args.input_data}")
    print(f"  Feature selection: {args.feature_selection_dir}")
    print(f"  Output data: {args.output_data}")
    print(f"  Number of features: {args.n_features}")

    # Load full dataset
    data_full, labels, feature_names_full = load_full_dataset(args.input_data)

    # Load selected features
    feature_indices, feature_names_selected = load_selected_features(
        args.feature_selection_dir,
        n_features=args.n_features
    )

    # Create filtered dataset
    data_filtered, labels_filtered, feature_names_filtered = create_filtered_dataset(
        data_full,
        labels,
        feature_names_full,
        feature_indices,
        feature_names_selected
    )

    # Prepare metadata
    metadata = {
        'source_file': args.input_data,
        'feature_selection_method': 'SHAP',
        'feature_selection_dir': args.feature_selection_dir,
        'original_n_features': data_full.shape[1],
        'selected_n_features': data_filtered.shape[1]
    }

    # Save filtered dataset
    save_filtered_dataset(
        args.output_data,
        data_filtered,
        labels_filtered,
        feature_names_filtered,
        metadata=metadata
    )

    print("\n" + "="*70)
    print("✓ Filtered Dataset Created Successfully!")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Original features: {data_full.shape[1]:,}")
    print(f"  Selected features: {data_filtered.shape[1]:,}")
    print(f"  Reduction: {(1 - data_filtered.shape[1]/data_full.shape[1])*100:.1f}%")
    print(f"  Output file: {args.output_data}")
    print(f"  File size: {os.path.getsize(args.output_data) / 1024 / 1024:.2f} MB")

    print(f"\nNext steps:")
    print(f"1. Train model with filtered dataset:")
    print(f"   python src/train.py \\")
    print(f"     --train_file {args.output_data} \\")
    print(f"     --output_dir ./output_filtered_{args.n_features} \\")
    print(f"     --epochs 300")

    print(f"\n2. Or use feature_order with original dataset:")
    print(f"   python src/train.py \\")
    print(f"     --train_file {args.input_data} \\")
    print(f"     --feature_order {os.path.join(args.feature_selection_dir, f'top_{args.n_features}_cpg_indices.npy')} \\")
    print(f"     --output_dir ./output_filtered_{args.n_features} \\")
    print(f"     --epochs 300")


if __name__ == '__main__':
    main()
