"""
Distribution Comparison: Nanopore Data vs All Data
Checks if datasets follow the same distributions following data_utils_fast.py processing

This script uses HDF5 (.h5) files for fast data loading.
Ensure your data is converted to H5 format before running this script.
"""

import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


def binarize_methylation(beta_values, threshold=0.5):
    """
    Binarize methylation beta values to +1/-1 (same as data_utils_fast.py).

    Args:
        beta_values: Beta values in range [0, 1]
        threshold: Threshold for binarization (default: 0.5)

    Returns:
        Binarized values (+1 for methylated, -1 for unmethylated)
    """
    binarized = np.where(beta_values >= threshold, 1, -1)
    return binarized.astype(np.float32)


def load_nanopore_data(data_dir, binarize=True, threshold=0.5):
    """Load nanopore dataset from H5 file."""
    print("\n" + "="*80)
    print("Loading Nanopore Dataset")
    print("="*80)

    # Load labels
    labels_path = os.path.join(data_dir, "NG_nanopore_labels.csv")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    labels_df = pd.read_csv(labels_path)
    print(f"Labels shape: {labels_df.shape}")
    print(f"Columns: {labels_df.columns.tolist()}")

    # Load from HDF5 (fast loading)
    h5_path = os.path.join(data_dir, "NG_nanopore_training_data.h5")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"HDF5 file not found: {h5_path}\n"
            f"Please convert your CSV data to H5 format first using convert_nanopore_to_h5.py"
        )

    print(f"\nLoading from HDF5: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        data = f['data'][:].astype(np.float32)
        feature_names = f['feature_names'][:].astype(str).tolist()
    print(f"Data shape: {data.shape}")

    # Binarize if requested
    if binarize:
        print(f"Binarizing with threshold={threshold}...")
        data = binarize_methylation(data, threshold=threshold)

    # Get labels (using Diagnosis_lineage as main label)
    labels = labels_df['Diagnosis_lineage'].values
    sample_ids = labels_df['sample_id'].values

    print(f"\nFinal data shape: {data.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    print(f"Label distribution:")
    for label in np.unique(labels):
        count = np.sum(labels == label)
        print(f"  {label}: {count} samples")

    return data, labels, sample_ids, feature_names


def load_all_data(data_dir, binarize=True, threshold=0.5):
    """Load all_data dataset from H5 file."""
    print("\n" + "="*80)
    print("Loading All Data Dataset")
    print("="*80)

    # Load labels
    labels_path = os.path.join(data_dir, "labels.csv")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    labels_df = pd.read_csv(labels_path)
    print(f"Labels shape: {labels_df.shape}")
    print(f"Columns: {labels_df.columns.tolist()}")

    # Load from HDF5 (fast loading)
    h5_path = os.path.join(data_dir, "training_data.h5")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"HDF5 file not found: {h5_path}\n"
            f"Please ensure your data is in H5 format for fast loading."
        )

    print(f"\nLoading from HDF5: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        data = f['data'][:].astype(np.float32)
        feature_names = f['feature_names'][:].astype(str).tolist()
    print(f"Data shape: {data.shape}")

    # Binarize if requested
    if binarize:
        print(f"Binarizing with threshold={threshold}...")
        data = binarize_methylation(data, threshold=threshold)

    # Get labels (using merged_label)
    labels = labels_df['merged_label'].values
    sample_ids = labels_df['sample_id'].values

    print(f"\nFinal data shape: {data.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    print(f"Number of unique labels: {len(np.unique(labels))}")

    return data, labels, sample_ids, feature_names


def align_features(data1, features1, data2, features2, name1="Dataset1", name2="Dataset2"):
    """Align features between two datasets."""
    print("\n" + "="*80)
    print("Aligning Features")
    print("="*80)

    features1_set = set(features1)
    features2_set = set(features2)

    common_features = sorted(features1_set & features2_set)
    only_in_1 = features1_set - features2_set
    only_in_2 = features2_set - features1_set

    print(f"{name1} features: {len(features1)}")
    print(f"{name2} features: {len(features2)}")
    print(f"Common features: {len(common_features)}")
    print(f"Only in {name1}: {len(only_in_1)}")
    print(f"Only in {name2}: {len(only_in_2)}")

    if len(common_features) == 0:
        raise ValueError("No common features found!")

    # Get indices for common features
    idx1 = [features1.index(f) for f in common_features]
    idx2 = [features2.index(f) for f in common_features]

    # Subset data
    data1_aligned = data1[:, idx1]
    data2_aligned = data2[:, idx2]

    print(f"\nAligned {name1} shape: {data1_aligned.shape}")
    print(f"Aligned {name2} shape: {data2_aligned.shape}")

    return data1_aligned, data2_aligned, common_features


def compare_distributions(data1, data2, name1="Nanopore", name2="All Data"):
    """Compare statistical distributions of two datasets."""
    print("\n" + "="*80)
    print("Statistical Distribution Comparison")
    print("="*80)

    # Overall statistics
    print(f"\n{name1}:")
    print(f"  Mean: {np.mean(data1):.4f}")
    print(f"  Std: {np.std(data1):.4f}")
    print(f"  Min: {np.min(data1):.4f}")
    print(f"  Max: {np.max(data1):.4f}")
    print(f"  Median: {np.median(data1):.4f}")

    print(f"\n{name2}:")
    print(f"  Mean: {np.mean(data2):.4f}")
    print(f"  Std: {np.std(data2):.4f}")
    print(f"  Min: {np.min(data2):.4f}")
    print(f"  Max: {np.max(data2):.4f}")
    print(f"  Median: {np.median(data2):.4f}")

    # Value distribution
    print(f"\n{name1} value distribution:")
    unique1, counts1 = np.unique(data1, return_counts=True)
    for val, count in zip(unique1, counts1):
        pct = 100 * count / data1.size
        print(f"  {val:+.1f}: {count:,} ({pct:.2f}%)")

    print(f"\n{name2} value distribution:")
    unique2, counts2 = np.unique(data2, return_counts=True)
    for val, count in zip(unique2, counts2):
        pct = 100 * count / data2.size
        print(f"  {val:+.1f}: {count:,} ({pct:.2f}%)")

    # Sample means
    sample_means1 = np.mean(data1, axis=1)
    sample_means2 = np.mean(data2, axis=1)

    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(sample_means1, sample_means2)
    print(f"\nKolmogorov-Smirnov Test (sample means):")
    print(f"  Statistic: {ks_stat:.4f}")
    print(f"  P-value: {ks_pval:.4e}")
    print(f"  Same distribution: {'Yes' if ks_pval > 0.05 else 'No'} (α=0.05)")

    return {
        'ks_stat': ks_stat,
        'ks_pval': ks_pval,
        'sample_means1': sample_means1,
        'sample_means2': sample_means2
    }


def plot_distributions(data1, data2, labels1, labels2, stats_dict, name1="Nanopore", name2="All Data"):
    """Plot distribution comparisons."""
    print("\n" + "="*80)
    print("Generating Distribution Plots")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Overall value distribution
    ax = axes[0, 0]
    bins = np.linspace(-1.5, 1.5, 50)
    ax.hist(data1.flatten(), bins=bins, alpha=0.6, label=name1, density=True, color='blue')
    ax.hist(data2.flatten(), bins=bins, alpha=0.6, label=name2, density=True, color='orange')
    ax.set_xlabel('Methylation Value')
    ax.set_ylabel('Density')
    ax.set_title('Overall Value Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Sample means distribution
    ax = axes[0, 1]
    ax.hist(stats_dict['sample_means1'], bins=30, alpha=0.6, label=name1, density=True, color='blue')
    ax.hist(stats_dict['sample_means2'], bins=30, alpha=0.6, label=name2, density=True, color='orange')
    ax.set_xlabel('Sample Mean')
    ax.set_ylabel('Density')
    ax.set_title(f'Sample Mean Distribution\nKS p-value: {stats_dict["ks_pval"]:.4e}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Feature-wise means
    ax = axes[1, 0]
    feature_means1 = np.mean(data1, axis=0)
    feature_means2 = np.mean(data2, axis=0)
    ax.scatter(feature_means1, feature_means2, alpha=0.3, s=10)
    lim = [min(feature_means1.min(), feature_means2.min()),
           max(feature_means1.max(), feature_means2.max())]
    ax.plot(lim, lim, 'r--', alpha=0.5, label='y=x')
    ax.set_xlabel(f'{name1} Feature Means')
    ax.set_ylabel(f'{name2} Feature Means')
    ax.set_title('Feature-wise Mean Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Class distribution
    ax = axes[1, 1]
    unique_labels1, counts1 = np.unique(labels1, return_counts=True)
    unique_labels2, counts2 = np.unique(labels2, return_counts=True)

    x1 = np.arange(len(unique_labels1))
    x2 = np.arange(len(unique_labels2))

    width = 0.35
    ax.bar(x1 - width/2, counts1, width, alpha=0.7, label=name1, color='blue')
    ax.set_xlabel('Class')
    ax.set_ylabel('Sample Count')
    ax.set_title('Class Distribution')
    ax.set_xticks(x1)
    ax.set_xticklabels(unique_labels1, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('distribution_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: distribution_comparison.png")
    plt.close()


def plot_combined_scatter(data1, data2, labels1, labels2, name1="Nanopore", name2="All Data", method='pca'):
    """Create scatter plot showing samples from both datasets with source and class."""
    print("\n" + "="*80)
    print(f"Generating Combined Scatter Plot ({method.upper()})")
    print("="*80)

    # Combine datasets
    combined_data = np.vstack([data1, data2])
    combined_labels = np.concatenate([labels1, labels2])
    combined_source = np.array([name1]*len(data1) + [name2]*len(data2))

    print(f"Combined data shape: {combined_data.shape}")
    print(f"Computing {method.upper()} embedding...")

    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(combined_data)
        explained_var = reducer.explained_variance_ratio_
        title_suffix = f"(PC1: {explained_var[0]:.1%}, PC2: {explained_var[1]:.1%})"
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        embedding = reducer.fit_transform(combined_data)
        title_suffix = ""
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: Colored by Source
    ax = axes[0]
    for source in [name1, name2]:
        mask = combined_source == source
        color = 'blue' if source == name1 else 'orange'
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                  alpha=0.6, s=50, label=source, color=color)

    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(f'Samples Colored by Source\n{title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Colored by Class (with Source as marker shape)
    ax = axes[1]

    # Get unique labels
    unique_labels = np.unique(combined_labels)
    n_labels = len(unique_labels)

    # Create color map
    colors = plt.cm.tab20(np.linspace(0, 1, n_labels))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

    # Plot each combination of source and label
    markers = {'Nanopore': 'o', 'All Data': 's'}

    for source in [name1, name2]:
        for label in unique_labels:
            mask = (combined_source == source) & (combined_labels == label)
            if np.sum(mask) > 0:
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                          alpha=0.6, s=50,
                          color=label_to_color[label],
                          marker=markers.get(source, 'o'),
                          label=f'{label} ({source})')

    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(f'Samples Colored by Class\n{title_suffix}')

    # Legend outside plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'combined_scatter_{method}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

    return embedding


def main():
    # Paths
    nanopore_dir = "data/nanopore_data"
    all_data_dir = "data/all_data"

    # Load datasets
    nano_data, nano_labels, nano_ids, nano_features = load_nanopore_data(
        nanopore_dir, binarize=True, threshold=0.5
    )

    all_data, all_labels, all_ids, all_features = load_all_data(
        all_data_dir, binarize=True, threshold=0.5
    )

    # Align features
    nano_data_aligned, all_data_aligned, common_features = align_features(
        nano_data, nano_features,
        all_data, all_features,
        name1="Nanopore", name2="All Data"
    )

    # Compare distributions
    stats_dict = compare_distributions(
        nano_data_aligned, all_data_aligned,
        name1="Nanopore", name2="All Data"
    )

    # Plot distributions
    plot_distributions(
        nano_data_aligned, all_data_aligned,
        nano_labels, all_labels,
        stats_dict,
        name1="Nanopore", name2="All Data"
    )

    # Create combined scatter plots
    print("\nCreating visualization plots...")

    # PCA plot
    pca_embedding = plot_combined_scatter(
        nano_data_aligned, all_data_aligned,
        nano_labels, all_labels,
        name1="Nanopore", name2="All Data",
        method='pca'
    )

    # t-SNE plot (optional, can be slow for large datasets)
    print("\nNote: t-SNE may take a while for large datasets...")
    if nano_data_aligned.shape[0] + all_data_aligned.shape[0] < 5000:
        tsne_embedding = plot_combined_scatter(
            nano_data_aligned, all_data_aligned,
            nano_labels, all_labels,
            name1="Nanopore", name2="All Data",
            method='tsne'
        )
    else:
        print("Skipping t-SNE due to large dataset size (>5000 samples)")

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("\nGenerated files:")
    print("  - distribution_comparison.png")
    print("  - combined_scatter_pca.png")
    if nano_data_aligned.shape[0] + all_data_aligned.shape[0] < 5000:
        print("  - combined_scatter_tsne.png")


if __name__ == "__main__":
    main()
