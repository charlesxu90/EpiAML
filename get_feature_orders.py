#!/usr/bin/env python3
"""
Feature Ordering for 1D-CNN Input

Reorder methylation features to maximize local similarity for 1D-CNN models.
This script computes pairwise distances between features based on methylation 
patterns and finds optimal orderings where similar features are adjacent.

Three methods provided:
1. Nearest Neighbor (fast greedy TSP)
2. Hierarchical clustering with optimal leaf ordering
3. Minimum Spanning Tree linearization

GPU Support: Uses cuML when available for faster computation.
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import h5py
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from sklearn.decomposition import PCA, IncrementalPCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.sparse import csr_matrix
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

try:
    import cupy as cp
    from cuml.decomposition import PCA as cuPCA
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    HAS_CUML = True
except ImportError:
    HAS_CUML = False

USE_GPU = False


def set_device(use_gpu=True):
    """Set whether to use GPU."""
    global USE_GPU
    USE_GPU = use_gpu and HAS_CUML
    if USE_GPU:
        print("Using GPU (cuML) for computations")
    else:
        print("Using CPU for computations")


def load_feature_data(data_path, max_samples=None):
    """Load methylation data with features as rows."""
    ext = os.path.splitext(data_path)[1].lower()
    
    if ext in ['.h5', '.hdf5']:
        print(f"Loading from HDF5: {data_path}")
        with h5py.File(data_path, 'r') as f:
            data = f['data'][:]
            sample_labels = f['labels'][:].astype(str) if 'labels' in f else None
            feature_names = f['feature_names'][:].astype(str) if 'feature_names' in f else None
            
    elif ext == '.csv':
        print(f"Loading from CSV: {data_path}")
        if not HAS_PANDAS:
            raise ImportError("pandas required for CSV loading")
        
        df = pd.read_csv(data_path, nrows=max_samples)
        sample_labels = df.iloc[:, 0].values
        data = df.iloc[:, 1:].values
        feature_names = np.array(df.columns[1:].tolist())
    else:
        raise ValueError(f"Unsupported format: {ext}")
    
    print(f"  Loaded: {data.shape[0]} samples × {data.shape[1]} features")
    
    if max_samples and data.shape[0] > max_samples:
        print(f"  Subsampling to {max_samples} samples...")
        idx = np.random.choice(data.shape[0], max_samples, replace=False)
        data = data[idx]
        if sample_labels is not None:
            sample_labels = sample_labels[idx]
    
    # Transpose: (n_features, n_samples)
    feature_data = data.T
    print(f"  Feature matrix: {feature_data.shape[0]} features × {feature_data.shape[1]} samples")
    
    return feature_data, feature_names, sample_labels


def reduce_dimensions(feature_data, n_components=50):
    """Reduce feature dimensionality using PCA."""
    n_features, n_samples = feature_data.shape
    n_components = min(n_components, n_samples - 1)
    
    print(f"\nReducing dimensions with PCA to {n_components} components...")
    
    start = time.time()
    
    if USE_GPU and HAS_CUML:
        scaler = cuStandardScaler()
        data_scaled = scaler.fit_transform(feature_data)
        pca = cuPCA(n_components=n_components)
        reduced = pca.fit_transform(data_scaled)
        if hasattr(reduced, 'get'):
            reduced = reduced.get()
        variance = float(sum(pca.explained_variance_ratio_)) * 100
    else:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(feature_data)
        
        if n_features > 10000:
            pca = IncrementalPCA(n_components=n_components, batch_size=10000)
        else:
            pca = PCA(n_components=n_components)
        
        reduced = pca.fit_transform(data_scaled)
        variance = sum(pca.explained_variance_ratio_) * 100
    
    print(f"  PCA completed in {time.time()-start:.1f}s")
    print(f"  Explained variance: {variance:.1f}%")
    
    return reduced, pca


def compute_distance_matrix(feature_repr, metric='correlation', n_neighbors=None):
    """Compute pairwise distances between features."""
    n_features = feature_repr.shape[0]
    
    print(f"\nComputing distance matrix ({metric})...")
    print(f"  Features: {n_features}")
    
    start = time.time()
    
    if n_neighbors and n_features > 10000:
        print(f"  Using sparse k-NN (k={n_neighbors}) for efficiency...")
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)
        nn.fit(feature_repr)
        distances, indices = nn.kneighbors(feature_repr)
        
        rows = np.repeat(np.arange(n_features), n_neighbors)
        cols = indices.flatten()
        vals = distances.flatten()
        dist_matrix = csr_matrix((vals, (rows, cols)), shape=(n_features, n_features))
        dist_matrix = dist_matrix.toarray()
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
    else:
        dist_matrix = pairwise_distances(feature_repr, metric=metric, n_jobs=-1)
    
    dist_matrix = np.nan_to_num(dist_matrix, nan=1.0, posinf=1.0, neginf=0.0)
    
    print(f"  Distance matrix computed in {time.time()-start:.1f}s")
    print(f"  Shape: {dist_matrix.shape}")
    print(f"  Distance range: [{dist_matrix.min():.4f}, {dist_matrix.max():.4f}]")
    
    return dist_matrix


def order_by_nearest_neighbor(dist_matrix, start_idx=None):
    """Order features using nearest neighbor heuristic (greedy TSP)."""
    n_features = dist_matrix.shape[0]
    
    print(f"\nOrdering features with nearest neighbor heuristic...")
    
    start = time.time()
    
    if start_idx is None:
        start_idx = np.argmin(dist_matrix.sum(axis=1))
    
    visited = np.zeros(n_features, dtype=bool)
    order = np.zeros(n_features, dtype=int)
    
    current = start_idx
    order[0] = current
    visited[current] = True
    
    for i in range(1, n_features):
        distances = dist_matrix[current].copy()
        distances[visited] = np.inf
        nearest = np.argmin(distances)
        
        order[i] = nearest
        visited[nearest] = True
        current = nearest
        
        if i % 50000 == 0:
            print(f"    Progress: {i}/{n_features} features ordered...")
    
    print(f"  Nearest neighbor ordering completed in {time.time()-start:.1f}s")
    
    return order


def order_by_hierarchical(dist_matrix):
    """Order features using hierarchical clustering."""
    if not HAS_SCIPY:
        raise ImportError("scipy required for hierarchical clustering")
    
    print(f"\nOrdering features with hierarchical clustering...")
    
    start = time.time()
    
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method='ward')
    ordered_indices = leaves_list(Z)
    
    print(f"  Hierarchical ordering completed in {time.time()-start:.1f}s")
    
    return ordered_indices, Z


def order_by_mst(dist_matrix):
    """Order features using Minimum Spanning Tree linearization."""
    if not HAS_SCIPY:
        raise ImportError("scipy required for MST")
    
    n_features = dist_matrix.shape[0]
    
    print(f"\nOrdering features with MST linearization...")
    
    start = time.time()
    
    print("  Computing minimum spanning tree...")
    mst = minimum_spanning_tree(csr_matrix(dist_matrix))
    mst = mst.toarray()
    mst = mst + mst.T
    
    print("  Linearizing MST with DFS...")
    
    degrees = (mst > 0).sum(axis=1)
    start_node = np.argmax(degrees)
    
    visited = np.zeros(n_features, dtype=bool)
    order = []
    stack = [start_node]
    
    while stack:
        node = stack.pop()
        if visited[node]:
            continue
        visited[node] = True
        order.append(node)
        
        neighbors = np.where(mst[node] > 0)[0]
        neighbors = neighbors[~visited[neighbors]]
        if len(neighbors) > 0:
            sorted_neighbors = neighbors[np.argsort(mst[node, neighbors])[::-1]]
            stack.extend(sorted_neighbors)
    
    order = np.array(order)
    
    print(f"  MST ordering completed in {time.time()-start:.1f}s")
    
    return order


def compute_ordering_quality(dist_matrix, order):
    """Compute quality metrics for a feature ordering."""
    n = len(order)
    
    consecutive_dists = [dist_matrix[order[i], order[i+1]] for i in range(n-1)]
    
    k = min(10, n // 10)
    preserved = 0
    for i in range(n):
        orig_neighbors = set(np.argsort(dist_matrix[order[i]])[:k])
        local_neighbors = set()
        for j in range(max(0, i-k), min(n, i+k+1)):
            if j != i:
                local_neighbors.add(order[j])
        preserved += len(orig_neighbors & local_neighbors)
    
    neighborhood_preservation = preserved / (n * k)
    
    return {
        'mean_consecutive_distance': float(np.mean(consecutive_dists)),
        'median_consecutive_distance': float(np.median(consecutive_dists)),
        'max_consecutive_distance': float(np.max(consecutive_dists)),
        'total_path_length': float(np.sum(consecutive_dists)),
        'neighborhood_preservation': round(neighborhood_preservation, 4)
    }


def plot_ordering_analysis(dist_matrix, order, feature_data, output_dir):
    """Plot analysis of the feature ordering."""
    if not HAS_PLOTTING:
        print("matplotlib not available, skipping plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating ordering analysis plots...")
    
    n = len(order)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    consecutive_dists = [dist_matrix[order[i], order[i+1]] for i in range(n-1)]
    
    window = min(1000, n // 10)
    smoothed = np.convolve(consecutive_dists, np.ones(window)/window, mode='valid')
    
    axes[0, 0].plot(smoothed, linewidth=0.5, alpha=0.8)
    axes[0, 0].set_xlabel('Feature Position', fontsize=12)
    axes[0, 0].set_ylabel('Distance to Next Feature', fontsize=12)
    axes[0, 0].set_title(f'Consecutive Feature Distances (window={window})', fontsize=14)
    axes[0, 0].axhline(np.mean(consecutive_dists), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(consecutive_dists):.4f}')
    axes[0, 0].legend()
    
    axes[0, 1].hist(consecutive_dists, bins=100, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Distance', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Distribution of Consecutive Distances', fontsize=14)
    axes[0, 1].axvline(np.median(consecutive_dists), color='red', linestyle='--',
                       label=f'Median: {np.median(consecutive_dists):.4f}')
    axes[0, 1].legend()
    
    sample_size = min(1000, n)
    sample_idx = np.linspace(0, n-1, sample_size, dtype=int)
    sample_order = order[sample_idx]
    sample_dist = dist_matrix[np.ix_(sample_order, sample_order)]
    
    im = axes[1, 0].imshow(sample_dist, cmap='viridis', aspect='auto')
    axes[1, 0].set_xlabel('Feature (ordered)', fontsize=12)
    axes[1, 0].set_ylabel('Feature (ordered)', fontsize=12)
    axes[1, 0].set_title('Reordered Distance Matrix (sampled)', fontsize=14)
    plt.colorbar(im, ax=axes[1, 0], label='Distance')
    
    sample_data = feature_data[sample_order]
    im2 = axes[1, 1].imshow(sample_data.T, aspect='auto', cmap='RdBu_r', 
                            vmin=0, vmax=1, interpolation='nearest')
    axes[1, 1].set_xlabel('Features (ordered)', fontsize=12)
    axes[1, 1].set_ylabel('Samples', fontsize=12)
    axes[1, 1].set_title('Methylation Values (ordered features)', fontsize=14)
    plt.colorbar(im2, ax=axes[1, 1], label='beta-value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_ordering_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/feature_ordering_analysis.png")
    plt.close()


def save_feature_order(feature_names, ordered_indices, output_dir):
    """Save the ordered feature list in multiple formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    if feature_names is not None:
        ordered_features = feature_names[ordered_indices].tolist()
    else:
        ordered_features = [f"feature_{i}" for i in ordered_indices]
    
    txt_path = os.path.join(output_dir, 'feature_order.txt')
    with open(txt_path, 'w') as f:
        for feat in ordered_features:
            f.write(f"{feat}\n")
    print(f"\nSaved: {txt_path}")
    
    if HAS_PANDAS:
        df = pd.DataFrame({
            'order': range(len(ordered_features)),
            'feature': ordered_features,
            'original_index': ordered_indices
        })
        csv_path = os.path.join(output_dir, 'feature_order.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
    
    npy_path = os.path.join(output_dir, 'feature_order_indices.npy')
    np.save(npy_path, ordered_indices)
    print(f"Saved: {npy_path}")
    
    json_path = os.path.join(output_dir, 'feature_order.json')
    with open(json_path, 'w') as f:
        json.dump(ordered_features, f)
    print(f"Saved: {json_path}")
    
    return ordered_features


def run_ordering_pipeline(data_path, output_dir='./feature_order_output',
                         method='nearest_neighbor', metric='correlation',
                         n_pca_components=50, max_samples=None, random_seed=42):
    """Run the complete feature ordering pipeline."""
    np.random.seed(random_seed)
    
    print("=" * 70)
    print("Feature Ordering for 1D-CNN Input")
    print("=" * 70)
    print(f"Method: {method}")
    print(f"Distance metric: {metric}")
    
    start_total = time.time()
    
    feature_data, feature_names, sample_labels = load_feature_data(
        data_path, max_samples=max_samples
    )
    n_features = feature_data.shape[0]
    
    feature_repr, pca = reduce_dimensions(feature_data, n_components=n_pca_components)
    
    dist_matrix = compute_distance_matrix(
        feature_repr, 
        metric=metric,
        n_neighbors=100 if n_features > 50000 else None
    )
    
    if method == 'nearest_neighbor':
        ordered_indices = order_by_nearest_neighbor(dist_matrix)
    elif method == 'hierarchical':
        ordered_indices, linkage_matrix = order_by_hierarchical(dist_matrix)
    elif method == 'mst':
        ordered_indices = order_by_mst(dist_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    quality = compute_ordering_quality(dist_matrix, ordered_indices)
    print(f"\nOrdering Quality:")
    for k, v in quality.items():
        print(f"  {k}: {v}")
    
    plot_ordering_analysis(dist_matrix, ordered_indices, feature_data, output_dir)
    
    ordered_features = save_feature_order(feature_names, ordered_indices, output_dir)
    
    total_time = time.time() - start_total
    
    results = {
        'method': method,
        'metric': metric,
        'n_features': n_features,
        'n_pca_components': n_pca_components,
        'pca_explained_variance': round(float(sum(pca.explained_variance_ratio_) * 100), 2),
        'quality_metrics': quality,
        'runtime_seconds': round(total_time, 2)
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'ordering_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"Pipeline completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}")
    print(f"\nOutput files in: {output_dir}")
    print("=" * 70)
    
    return {
        'ordered_indices': ordered_indices,
        'ordered_features': ordered_features,
        'quality': quality,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Feature Ordering for 1D-CNN Input',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Reorder methylation features to maximize local similarity for 1D-CNN.

Methods:
  nearest_neighbor - Greedy TSP approximation (fast, good locality)
  hierarchical     - Hierarchical clustering with optimal leaf ordering
  mst              - Minimum spanning tree linearization

Examples:
  python get_feature_orders.py --data training_data.h5 --method nearest_neighbor
  python get_feature_orders.py --data training_data.h5 --metric euclidean
        '''
    )
    
    parser.add_argument('--data', required=True, help='Path to training data')
    parser.add_argument('--output_dir', default='./feature_order_output', help='Output directory')
    parser.add_argument('--method', default='nearest_neighbor',
                       choices=['nearest_neighbor', 'hierarchical', 'mst'])
    parser.add_argument('--metric', default='correlation',
                       choices=['correlation', 'euclidean', 'cosine'])
    parser.add_argument('--n_pca', type=int, default=50)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', action='store_true')
    
    args = parser.parse_args()
    
    set_device(args.gpu)
    
    run_ordering_pipeline(
        data_path=args.data,
        output_dir=args.output_dir,
        method=args.method,
        metric=args.metric,
        n_pca_components=args.n_pca,
        max_samples=args.max_samples,
        random_seed=args.seed
    )


if __name__ == '__main__':
    main()
