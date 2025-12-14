import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_h5_data(h5_path):
    with h5py.File(h5_path, 'r') as f:
        X = f['data'][:]
        feature_names = f['feature_names'][:].astype(str)
    return X, feature_names

def print_quantiles(X, name):
    flat = X.flatten()
    quantiles = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    q_vals = np.quantile(flat, quantiles)
    print(f"\n{name} value quantiles:")
    for q, v in zip(quantiles, q_vals):
        print(f"  {q*100:5.1f}%: {v:.4f}")

def plot_value_distributions(X1, X2, name1, name2, out_prefix, bulk_threshold=None, nano_threshold=None):
    plt.figure(figsize=(12,6))
    sns.histplot(X1.flatten(), bins=100, color='blue', label=name1, stat='density', alpha=0.5)
    sns.histplot(X2.flatten(), bins=100, color='orange', label=name2, stat='density', alpha=0.5)

    if bulk_threshold is not None:
        plt.axvline(bulk_threshold, color='blue', linestyle='--', linewidth=2,
                   label=f'{name1} threshold={bulk_threshold}')
    if nano_threshold is not None:
        plt.axvline(nano_threshold, color='orange', linestyle='--', linewidth=2,
                   label=f'{name2} threshold={nano_threshold:.4f}')

    plt.xlabel("Methylation Value")
    plt.ylabel("Density")
    plt.title("Overall Value Distribution (No Binarization)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_value_distribution.png", dpi=200)
    plt.close()

def plot_feature_mean_var(X1, X2, name1, name2, out_prefix):
    means1 = X1.mean(axis=0)
    means2 = X2.mean(axis=0)
    vars1 = X1.var(axis=0)
    vars2 = X2.var(axis=0)

    plt.figure(figsize=(10,5))
    sns.histplot(means1, bins=100, color='blue', label=f"{name1} feature means", stat='density', alpha=0.5)
    sns.histplot(means2, bins=100, color='orange', label=f"{name2} feature means", stat='density', alpha=0.5)
    plt.xlabel("Feature Mean")
    plt.ylabel("Density")
    plt.title("Distribution of Feature Means")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_feature_means.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10,5))
    sns.histplot(vars1, bins=100, color='blue', label=f"{name1} feature variance", stat='density', alpha=0.5)
    sns.histplot(vars2, bins=100, color='orange', label=f"{name2} feature variance", stat='density', alpha=0.5)
    plt.xlabel("Feature Variance")
    plt.ylabel("Density")
    plt.title("Distribution of Feature Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_feature_variance.png", dpi=200)
    plt.close()

def suggest_threshold(X1, X2, bulk_threshold, name1, name2):
    """
    Given a threshold used for bulk data, suggest an equivalent threshold for nanopore.
    The suggestion is based on matching the percentile of the bulk threshold.
    """
    flat1 = X1.flatten()
    flat2 = X2.flatten()

    # Find what percentile the bulk threshold represents
    bulk_percentile = (flat1 < bulk_threshold).mean() * 100

    # Find the corresponding value at that percentile in nanopore data
    nano_threshold = np.percentile(flat2, bulk_percentile)

    print(f"\n{'='*60}")
    print(f"BINARIZATION THRESHOLD ANALYSIS")
    print(f"{'='*60}")
    print(f"\n{name1} threshold: {bulk_threshold}")
    print(f"  This corresponds to the {bulk_percentile:.2f}th percentile in {name1} data")
    print(f"  {bulk_percentile:.2f}% of values are < {bulk_threshold}")
    print(f"  {100-bulk_percentile:.2f}% of values are >= {bulk_threshold}")

    print(f"\nSuggested {name2} threshold: {nano_threshold:.4f}")
    print(f"  This is the {bulk_percentile:.2f}th percentile in {name2} data")
    print(f"  Using this threshold will maintain the same proportion of 0s and 1s")

    # Additional statistics
    print(f"\nAdditional Statistics:")
    print(f"{name1} - Mean: {flat1.mean():.4f}, Median: {np.median(flat1):.4f}, Std: {flat1.std():.4f}")
    print(f"{name2} - Mean: {flat2.mean():.4f}, Median: {np.median(flat2):.4f}, Std: {flat2.std():.4f}")

    return nano_threshold

def main():
    bulk_h5 = "data/array_data/training_data.h5"
    nano_h5 = "data/nanopore_data/NG_nanopore_training_data.h5"
    name1 = "Bulk"
    name2 = "Nanopore"
    out_prefix = "bulk_vs_nanopore"
    bulk_threshold = 0.5

    print(f"Loading {bulk_h5} ...")
    X1, features1 = load_h5_data(bulk_h5)
    print(f"  Shape: {X1.shape}")

    print(f"Loading {nano_h5} ...")
    X2, features2 = load_h5_data(nano_h5)
    print(f"  Shape: {X2.shape}")

    print_quantiles(X1, name1)
    print_quantiles(X2, name2)

    nano_threshold = suggest_threshold(X1, X2, bulk_threshold, name1, name2)

    plot_value_distributions(X1, X2, name1, name2, out_prefix, bulk_threshold, nano_threshold)
    plot_feature_mean_var(X1, X2, name1, name2, out_prefix)

    print("\nPlots saved:")
    print(f"  {out_prefix}_value_distribution.png")
    print(f"  {out_prefix}_feature_means.png")
    print(f"  {out_prefix}_feature_variance.png")

if __name__ == "__main__":
    main()
