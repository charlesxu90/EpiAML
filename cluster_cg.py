"""
CpG Feature Clustering and Reordering for 1D-CNN
Cluster methylation features and reorder them for spatial locality in 1D-CNN models.

This module clusters 357K CpG features based on their methylation patterns across samples,
then outputs an ordered feature list where similar features are adjacent.

GPU Support: Uses cuML (RAPIDS) when available, falls back to scikit-learn on CPU.
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import h5py
from pathlib import Path

# Optional imports with graceful fallback
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Try GPU-accelerated cuML first, fall back to sklearn
try:
    import cupy as cp
    from cuml.cluster import KMeans as cuKMeans
    from cuml.decomposition import PCA as cuPCA
    from cuml.manifold import TSNE as cuTSNE
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    HAS_CUML = True
    print("cuML (GPU) available - using GPU acceleration")
except ImportError:
    HAS_CUML = False

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA, IncrementalPCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Global flag to control GPU usage
USE_GPU = HAS_CUML


def set_device(use_gpu=True):
    """
    Set whether to use GPU or CPU for computations.
    
    Args:
        use_gpu (bool): If True, use GPU (cuML) if available
    """
    global USE_GPU
    if use_gpu and HAS_CUML:
        USE_GPU = True
        print("Using GPU (cuML) for computations")
    else:
        USE_GPU = False
        if use_gpu and not HAS_CUML:
            print("GPU requested but cuML not available, using CPU (sklearn)")
        else:
            print("Using CPU (sklearn) for computations")


def get_device_info():
    """Get information about available compute devices."""
    info = {
        'cuml_available': HAS_CUML,
        'sklearn_available': HAS_SKLEARN,
        'using_gpu': USE_GPU
    }
    if HAS_CUML:
        try:
            import cupy as cp
            info['gpu_name'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            info['gpu_memory_gb'] = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / (1024**3)
        except:
            pass
    return info


def load_feature_data(data_path, max_samples=None):
    """
    Load methylation data with features as rows (for feature clustering).
    
    Each feature (CpG site) becomes a data point, and each sample becomes a dimension.
    This allows clustering features based on their methylation patterns across samples.
    
    Args:
        data_path (str): Path to data file (.h5, .csv)
        max_samples (int): Maximum samples to use as dimensions
    
    Returns:
        tuple: (feature_data, feature_names, sample_labels)
               feature_data shape: (n_features, n_samples)
    """
    ext = os.path.splitext(data_path)[1].lower()
    
    if ext in ['.h5', '.hdf5']:
        print(f"Loading from HDF5: {data_path}")
        with h5py.File(data_path, 'r') as f:
            # Original shape: (n_samples, n_features)
            data = f['data'][:]
            sample_labels = f['labels'][:].astype(str) if 'labels' in f else None
            feature_names = f['feature_names'][:].astype(str) if 'feature_names' in f else None
            
    elif ext == '.csv':
        print(f"Loading from CSV: {data_path}")
        if not HAS_PANDAS:
            raise ImportError("pandas required for CSV loading. Install with: pip install pandas")
        
        df = pd.read_csv(data_path, nrows=max_samples)
        sample_labels = df.iloc[:, 0].values
        data = df.iloc[:, 1:].values
        feature_names = np.array(df.columns[1:].tolist())
        
    else:
        raise ValueError(f"Unsupported format: {ext}")
    
    print(f"  Original shape: {data.shape[0]} samples × {data.shape[1]} features")
    
    # Subsample samples if needed (these become our dimensions)
    if max_samples and data.shape[0] > max_samples:
        print(f"  Subsampling to {max_samples} samples...")
        idx = np.random.choice(data.shape[0], max_samples, replace=False)
        data = data[idx]
        if sample_labels is not None:
            sample_labels = sample_labels[idx]
    
    # Transpose: features become rows, samples become columns
    # Shape: (n_features, n_samples) - e.g., (357340, 42)
    feature_data = data.T
    print(f"  Transposed for feature clustering: {feature_data.shape[0]} features × {feature_data.shape[1]} samples")
    
    return feature_data, feature_names, sample_labels


def reduce_dimensions_pca(feature_data, n_components=50, batch_size=10000):
    """
    Reduce feature dimensionality using PCA for faster clustering.
    Uses GPU (cuML) if available, otherwise falls back to sklearn.
    
    Args:
        feature_data (np.ndarray): Shape (n_features, n_samples)
        n_components (int): Number of PCA components
        batch_size (int): Batch size for incremental PCA (CPU only)
    
    Returns:
        tuple: (reduced_data, pca_model)
    """
    n_features, n_samples = feature_data.shape
    n_components = min(n_components, n_samples - 1)
    
    print(f"\nReducing dimensions with PCA...")
    print(f"  Input: {n_features} features × {n_samples} samples")
    print(f"  Target components: {n_components}")
    
    start = time.time()
    
    if USE_GPU and HAS_CUML:
        print("  Using GPU-accelerated PCA (cuML)")
        # cuML PCA
        scaler = cuStandardScaler()
        data_scaled = scaler.fit_transform(feature_data)
        
        pca = cuPCA(n_components=n_components)
        reduced_data = pca.fit_transform(data_scaled)
        
        # Convert back to numpy if needed
        if hasattr(reduced_data, 'get'):
            reduced_data = reduced_data.get()
        
        variance_explained = float(sum(pca.explained_variance_ratio_)) * 100
        
    else:
        if not HAS_SKLEARN:
            raise ImportError("Neither cuML nor scikit-learn available")
        
        print("  Using CPU PCA (sklearn)")
        # Standardize each feature (row)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(feature_data)
        
        # Use Incremental PCA for memory efficiency with many features
        if n_features > batch_size:
            print(f"  Using Incremental PCA (batch_size={batch_size})")
            pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        else:
            pca = PCA(n_components=n_components)
        
        reduced_data = pca.fit_transform(data_scaled)
        variance_explained = sum(pca.explained_variance_ratio_) * 100
    
    elapsed = time.time() - start
    print(f"  PCA completed in {elapsed:.1f}s")
    print(f"  Output: {reduced_data.shape[0]} features × {reduced_data.shape[1]} components")
    print(f"  Explained variance: {variance_explained:.1f}%")
    
    return reduced_data, pca


def cluster_features_kmeans(feature_repr, n_clusters=100, batch_size=1024, random_state=42):
    """
    Cluster features using K-Means.
    Uses GPU (cuML) if available, otherwise falls back to sklearn MiniBatchKMeans.
    
    Args:
        feature_repr (np.ndarray): PCA-reduced feature data (n_features, n_components)
        n_clusters (int): Number of clusters
        batch_size (int): Mini-batch size (CPU only)
        random_state (int): Random seed
    
    Returns:
        tuple: (cluster_labels, cluster_centers, kmeans_model)
    """
    print(f"\nClustering features with K-Means...")
    print(f"  n_clusters={n_clusters}")
    
    start = time.time()
    
    if USE_GPU and HAS_CUML:
        print("  Using GPU-accelerated K-Means (cuML)")
        kmeans = cuKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=300
        )
        cluster_labels = kmeans.fit_predict(feature_repr)
        cluster_centers = kmeans.cluster_centers_
        
        # Convert back to numpy if needed
        if hasattr(cluster_labels, 'get'):
            cluster_labels = cluster_labels.get()
        if hasattr(cluster_centers, 'get'):
            cluster_centers = cluster_centers.get()
        
        inertia = float(kmeans.inertia_)
        
    else:
        if not HAS_SKLEARN:
            raise ImportError("Neither cuML nor scikit-learn available")
        
        print("  Using CPU MiniBatch K-Means (sklearn)")
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=random_state,
            n_init=3,
            max_iter=300
        )
        cluster_labels = kmeans.fit_predict(feature_repr)
        cluster_centers = kmeans.cluster_centers_
        inertia = kmeans.inertia_
    
    elapsed = time.time() - start
    print(f"  Clustering completed in {elapsed:.1f}s")
    print(f"  Inertia: {inertia:.2f}")
    
    return cluster_labels, kmeans.cluster_centers_, kmeans


def order_clusters_hierarchically(cluster_centers):
    """
    Order clusters using hierarchical clustering to ensure similar clusters are adjacent.
    
    Args:
        cluster_centers (np.ndarray): Cluster centroids (n_clusters, n_components)
    
    Returns:
        np.ndarray: Ordered cluster indices
    """
    if not HAS_SCIPY:
        print("  scipy not available, using original cluster order")
        return np.arange(len(cluster_centers))
    
    print(f"\nOrdering {len(cluster_centers)} clusters hierarchically...")
    
    start = time.time()
    # Hierarchical clustering of cluster centers
    Z = linkage(cluster_centers, method='ward')
    
    # Get optimal leaf ordering
    cluster_order = leaves_list(Z)
    elapsed = time.time() - start
    
    print(f"  Hierarchical ordering completed in {elapsed:.1f}s")
    
    return cluster_order


def order_features_within_clusters(feature_repr, cluster_labels, cluster_order):
    """
    Order features within each cluster by distance to cluster center,
    then concatenate in cluster order.
    
    Args:
        feature_repr (np.ndarray): PCA-reduced features (n_features, n_components)
        cluster_labels (np.ndarray): Cluster assignments for each feature
        cluster_order (np.ndarray): Order of clusters
    
    Returns:
        np.ndarray: Ordered feature indices
    """
    print(f"\nOrdering features within clusters...")
    
    ordered_indices = []
    
    for cluster_id in cluster_order:
        # Get features in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_feature_indices = np.where(cluster_mask)[0]
        
        if len(cluster_feature_indices) == 0:
            continue
        
        # Get feature representations for this cluster
        cluster_features = feature_repr[cluster_feature_indices]
        
        # Compute cluster center
        center = cluster_features.mean(axis=0)
        
        # Order by distance to center (closest first)
        distances = np.linalg.norm(cluster_features - center, axis=1)
        sorted_idx = np.argsort(distances)
        
        # Add to ordered list
        ordered_indices.extend(cluster_feature_indices[sorted_idx])
    
    ordered_indices = np.array(ordered_indices)
    print(f"  Ordered {len(ordered_indices)} features")
    
    return ordered_indices


def compute_tsne(feature_repr, n_samples=10000, perplexity=30, random_state=42):
    """
    Compute t-SNE embedding for visualization.
    Uses GPU (cuML) if available, otherwise falls back to sklearn.
    
    Args:
        feature_repr (np.ndarray): PCA-reduced features (n_features, n_components)
        n_samples (int): Maximum samples for t-SNE (subsample if larger)
        perplexity (int): t-SNE perplexity
        random_state (int): Random seed
    
    Returns:
        tuple: (tsne_embedding, sample_indices)
    """
    n_features = feature_repr.shape[0]
    
    # Subsample if too many features
    if n_features > n_samples:
        print(f"\nSubsampling {n_samples} features for t-SNE (from {n_features})...")
        sample_idx = np.random.choice(n_features, n_samples, replace=False)
        data_sample = feature_repr[sample_idx]
    else:
        sample_idx = np.arange(n_features)
        data_sample = feature_repr
    
    print(f"\nComputing t-SNE embedding...")
    print(f"  n_features={len(sample_idx)}, perplexity={perplexity}")
    
    start = time.time()
    
    if USE_GPU and HAS_CUML:
        print("  Using GPU-accelerated t-SNE (cuML)")
        tsne = cuTSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=1000,
            learning_rate=200.0
        )
        tsne_embedding = tsne.fit_transform(data_sample)
        
        # Convert back to numpy if needed
        if hasattr(tsne_embedding, 'get'):
            tsne_embedding = tsne_embedding.get()
    else:
        if not HAS_SKLEARN:
            raise ImportError("Neither cuML nor scikit-learn available")
        
        print("  Using CPU t-SNE (sklearn)")
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=1000,
            init='pca',
            learning_rate='auto'
        )
        tsne_embedding = tsne.fit_transform(data_sample)
    
    elapsed = time.time() - start
    print(f"  t-SNE completed in {elapsed:.1f}s")
    
    return tsne_embedding, sample_idx


def plot_tsne_clusters(tsne_embedding, cluster_labels, sample_idx, output_dir, 
                       feature_names=None, max_clusters_legend=20):
    """
    Plot t-SNE visualization of feature clusters.
    
    Args:
        tsne_embedding (np.ndarray): t-SNE coordinates (n_samples, 2)
        cluster_labels (np.ndarray): Full cluster labels
        sample_idx (np.ndarray): Indices of sampled features
        output_dir (str): Output directory
        feature_names (np.ndarray): Feature names
        max_clusters_legend (int): Max clusters to show in legend
    """
    if not HAS_PLOTTING:
        print("matplotlib not available, skipping plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get cluster labels for sampled features
    labels_sample = cluster_labels[sample_idx]
    unique_clusters = np.unique(labels_sample)
    n_clusters = len(unique_clusters)
    
    print(f"\nPlotting t-SNE visualization...")
    print(f"  {len(sample_idx)} features, {n_clusters} clusters")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. t-SNE colored by cluster
    scatter = axes[0].scatter(
        tsne_embedding[:, 0], 
        tsne_embedding[:, 1],
        c=labels_sample,
        cmap='tab20',
        alpha=0.6,
        s=3
    )
    axes[0].set_xlabel('t-SNE 1', fontsize=12)
    axes[0].set_ylabel('t-SNE 2', fontsize=12)
    axes[0].set_title(f'Feature Clusters (t-SNE)\n{n_clusters} clusters, {len(sample_idx)} features', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label('Cluster ID', fontsize=10)
    
    # 2. Cluster size distribution
    cluster_sizes = [np.sum(cluster_labels == c) for c in range(cluster_labels.max() + 1)]
    axes[1].hist(cluster_sizes, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].axvline(np.median(cluster_sizes), color='red', linestyle='--', 
                    label=f'Median: {np.median(cluster_sizes):.0f}')
    axes[1].axvline(np.mean(cluster_sizes), color='orange', linestyle='--', 
                    label=f'Mean: {np.mean(cluster_sizes):.0f}')
    axes[1].set_xlabel('Cluster Size (# features)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Cluster Size Distribution', fontsize=14)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_tsne_clusters.png'), dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_dir}/feature_tsne_clusters.png")
    plt.close()
    
    # 3. Larger t-SNE plot with better resolution
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        tsne_embedding[:, 0], 
        tsne_embedding[:, 1],
        c=labels_sample,
        cmap='Spectral',
        alpha=0.5,
        s=2
    )
    ax.set_xlabel('t-SNE 1', fontsize=14)
    ax.set_ylabel('t-SNE 2', fontsize=14)
    ax.set_title(f'CpG Feature Clustering (t-SNE)\n{n_clusters} clusters from {len(cluster_labels)} features', fontsize=16)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster ID', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_tsne_large.png'), dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/feature_tsne_large.png")
    plt.close()


def plot_feature_order_heatmap(feature_data, ordered_indices, feature_names, 
                               cluster_labels, output_dir, n_features_plot=1000):
    """
    Plot heatmap showing feature ordering quality.
    
    Args:
        feature_data (np.ndarray): Original feature data (n_features, n_samples)
        ordered_indices (np.ndarray): Ordered feature indices
        feature_names (np.ndarray): Feature names
        cluster_labels (np.ndarray): Cluster assignments
        output_dir (str): Output directory
        n_features_plot (int): Number of features to show in heatmap
    """
    if not HAS_PLOTTING:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nPlotting feature order heatmap...")
    
    # Sample features evenly across the ordering
    n_features = len(ordered_indices)
    if n_features > n_features_plot:
        step = n_features // n_features_plot
        plot_idx = ordered_indices[::step][:n_features_plot]
    else:
        plot_idx = ordered_indices
    
    # Get data for plotting
    data_plot = feature_data[plot_idx]
    labels_plot = cluster_labels[plot_idx]
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), 
                             gridspec_kw={'height_ratios': [1, 8]})
    
    # 1. Cluster assignment bar
    cmap = plt.cm.get_cmap('tab20')
    cluster_colors = [cmap(l % 20) for l in labels_plot]
    for i, color in enumerate(cluster_colors):
        axes[0].axvspan(i, i+1, color=color, alpha=0.8)
    axes[0].set_xlim(0, len(plot_idx))
    axes[0].set_yticks([])
    axes[0].set_title('Cluster Assignments (ordered features)', fontsize=12)
    
    # 2. Methylation heatmap
    im = axes[1].imshow(data_plot.T, aspect='auto', cmap='RdBu_r', 
                        vmin=0, vmax=1, interpolation='nearest')
    axes[1].set_xlabel('Features (ordered by cluster)', fontsize=12)
    axes[1].set_ylabel('Samples', fontsize=12)
    axes[1].set_title('Methylation Values (ordered features)', fontsize=12)
    
    cbar = plt.colorbar(im, ax=axes[1], shrink=0.8)
    cbar.set_label('Methylation β-value', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_order_heatmap.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/feature_order_heatmap.png")
    plt.close()


def save_feature_order(feature_names, ordered_indices, cluster_labels, output_dir):
    """
    Save the ordered feature list.
    
    Args:
        feature_names (np.ndarray): Original feature names
        ordered_indices (np.ndarray): Ordered feature indices
        cluster_labels (np.ndarray): Cluster assignments
        output_dir (str): Output directory
    
    Returns:
        list: Ordered feature names
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create ordered feature list
    if feature_names is not None:
        ordered_features = feature_names[ordered_indices].tolist()
    else:
        ordered_features = [f"feature_{i}" for i in ordered_indices]
    
    ordered_clusters = cluster_labels[ordered_indices].tolist()
    
    # Save as simple text file (one feature per line)
    txt_path = os.path.join(output_dir, 'feature_order.txt')
    with open(txt_path, 'w') as f:
        for feat in ordered_features:
            f.write(f"{feat}\n")
    print(f"\nSaved ordered feature list: {txt_path}")
    
    # Save as CSV with cluster info
    if HAS_PANDAS:
        df = pd.DataFrame({
            'order': range(len(ordered_features)),
            'feature': ordered_features,
            'original_index': ordered_indices,
            'cluster': ordered_clusters
        })
        csv_path = os.path.join(output_dir, 'feature_order_with_clusters.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved detailed order CSV: {csv_path}")
    
    # Save as numpy array (indices only)
    npy_path = os.path.join(output_dir, 'feature_order_indices.npy')
    np.save(npy_path, ordered_indices)
    print(f"Saved order indices: {npy_path}")
    
    # Save as JSON
    json_path = os.path.join(output_dir, 'feature_order.json')
    with open(json_path, 'w') as f:
        json.dump(ordered_features, f)
    print(f"Saved order JSON: {json_path}")
    
    return ordered_features


def run_feature_clustering_pipeline(
    data_path,
    output_dir='./cluster_output',
    n_clusters=100,
    n_pca_components=50,
    tsne_samples=10000,
    tsne_perplexity=30,
    max_samples=None,
    random_seed=42
):
    """
    Run the complete feature clustering and reordering pipeline.
    
    Pipeline:
    1. Load data (features as rows, samples as columns)
    2. Reduce dimensions with PCA
    3. Cluster features with K-Means
    4. Order clusters hierarchically
    5. Order features within clusters
    6. Visualize with t-SNE
    7. Save ordered feature list
    
    Args:
        data_path (str): Path to training data (.h5 or .csv)
        output_dir (str): Output directory
        n_clusters (int): Number of feature clusters
        n_pca_components (int): PCA components for clustering
        tsne_samples (int): Max features for t-SNE
        tsne_perplexity (int): t-SNE perplexity
        max_samples (int): Max samples to use
        random_seed (int): Random seed
    
    Returns:
        dict: Results including ordered features and metrics
    """
    np.random.seed(random_seed)
    
    print("=" * 70)
    print("CpG Feature Clustering and Reordering for 1D-CNN")
    print("=" * 70)
    
    start_total = time.time()
    
    # Step 1: Load data
    feature_data, feature_names, sample_labels = load_feature_data(
        data_path, 
        max_samples=max_samples
    )
    n_features = feature_data.shape[0]
    
    # Step 2: PCA dimensionality reduction
    feature_repr, pca = reduce_dimensions_pca(
        feature_data, 
        n_components=n_pca_components
    )
    
    # Step 3: Cluster features
    cluster_labels, cluster_centers, kmeans = cluster_features_kmeans(
        feature_repr,
        n_clusters=n_clusters,
        random_state=random_seed
    )
    
    # Step 4: Order clusters hierarchically
    cluster_order = order_clusters_hierarchically(cluster_centers)
    
    # Step 5: Order features within clusters
    ordered_indices = order_features_within_clusters(
        feature_repr, 
        cluster_labels, 
        cluster_order
    )
    
    # Step 6: Compute t-SNE for visualization
    tsne_embedding, tsne_sample_idx = compute_tsne(
        feature_repr,
        n_samples=tsne_samples,
        perplexity=tsne_perplexity,
        random_state=random_seed
    )
    
    # Step 7: Plot results
    plot_tsne_clusters(
        tsne_embedding, 
        cluster_labels, 
        tsne_sample_idx, 
        output_dir,
        feature_names
    )
    
    plot_feature_order_heatmap(
        feature_data,
        ordered_indices,
        feature_names,
        cluster_labels,
        output_dir
    )
    
    # Step 8: Save ordered feature list
    ordered_features = save_feature_order(
        feature_names,
        ordered_indices,
        cluster_labels,
        output_dir
    )
    
    # Compute metrics
    try:
        if len(np.unique(cluster_labels)) > 1:
            sil_idx = np.random.choice(len(feature_repr), min(10000, len(feature_repr)), replace=False)
            sil_score = silhouette_score(feature_repr[sil_idx], cluster_labels[sil_idx])
        else:
            sil_score = 0.0
    except:
        sil_score = 0.0
    
    # Save summary results
    total_time = time.time() - start_total
    
    results = {
        'n_features': n_features,
        'n_clusters': n_clusters,
        'n_pca_components': n_pca_components,
        'pca_explained_variance': round(float(sum(pca.explained_variance_ratio_) * 100), 2),
        'silhouette_score': round(sil_score, 4),
        'cluster_sizes': {
            'min': int(min([np.sum(cluster_labels == c) for c in range(n_clusters)])),
            'max': int(max([np.sum(cluster_labels == c) for c in range(n_clusters)])),
            'mean': round(float(np.mean([np.sum(cluster_labels == c) for c in range(n_clusters)])), 1)
        },
        'runtime_seconds': round(total_time, 2),
        'random_seed': random_seed
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'clustering_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save cluster labels
    np.save(os.path.join(output_dir, 'cluster_labels.npy'), cluster_labels)
    
    # Save t-SNE embedding
    np.save(os.path.join(output_dir, 'tsne_embedding.npy'), tsne_embedding)
    np.save(os.path.join(output_dir, 'tsne_sample_indices.npy'), tsne_sample_idx)
    
    print(f"\n{'=' * 70}")
    print(f"Pipeline completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}")
    print(f"\nResults Summary:")
    print(f"  Total features: {n_features:,}")
    print(f"  Clusters: {n_clusters}")
    print(f"  PCA explained variance: {results['pca_explained_variance']:.1f}%")
    print(f"  Silhouette score: {sil_score:.4f}")
    print(f"\nOutput files in: {output_dir}")
    print(f"  - feature_order.txt         (ordered feature list)")
    print(f"  - feature_order.json        (JSON format)")
    print(f"  - feature_order_indices.npy (numpy indices)")
    print(f"  - feature_tsne_clusters.png (t-SNE visualization)")
    print(f"  - feature_order_heatmap.png (ordering quality)")
    print("=" * 70)
    
    return {
        'ordered_indices': ordered_indices,
        'ordered_features': ordered_features,
        'cluster_labels': cluster_labels,
        'feature_repr': feature_repr,
        'tsne_embedding': tsne_embedding,
        'results': results
    }


def find_optimal_clusters(data_path, output_dir='./cluster_output',
                         k_range=(20, 300, 20), n_pca_components=50,
                         max_samples=None, random_seed=42):
    """
    Find optimal number of clusters using silhouette analysis.
    
    Args:
        data_path (str): Path to training data
        output_dir (str): Output directory
        k_range (tuple): (min_k, max_k, step)
        n_pca_components (int): PCA components
        max_samples (int): Maximum samples
        random_seed (int): Random seed
    
    Returns:
        dict: Optimal k analysis results
    """
    np.random.seed(random_seed)
    
    print("=" * 70)
    print("Finding Optimal Number of Feature Clusters")
    print("=" * 70)
    
    # Load and transform data
    feature_data, feature_names, _ = load_feature_data(data_path, max_samples)
    feature_repr, pca = reduce_dimensions_pca(feature_data, n_components=n_pca_components)
    
    # Test different k values
    k_values = list(range(k_range[0], k_range[1], k_range[2]))
    inertias = []
    silhouettes = []
    
    print(f"\nTesting k values: {k_values}")
    
    # Sample for faster silhouette computation
    n_sample = min(10000, len(feature_repr))
    sample_idx = np.random.choice(len(feature_repr), n_sample, replace=False)
    
    for k in k_values:
        print(f"  k={k}...", end=' ', flush=True)
        start = time.time()
        
        labels, _, kmeans = cluster_features_kmeans(feature_repr, n_clusters=k, random_state=random_seed)
        inertias.append(kmeans.inertia_)
        
        sil = silhouette_score(feature_repr[sample_idx], labels[sample_idx])
        silhouettes.append(sil)
        
        print(f"silhouette={sil:.4f} ({time.time()-start:.1f}s)")
    
    # Find optimal k
    optimal_idx = np.argmax(silhouettes)
    optimal_k = k_values[optimal_idx]
    
    print(f"\nOptimal k by silhouette score: {optimal_k}")
    
    # Plot
    if HAS_PLOTTING:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Inertia', fontsize=12)
        axes[0].set_title('Elbow Method', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(k_values, silhouettes, 'go-', linewidth=2, markersize=8)
        axes[1].axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Analysis', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'optimal_k_analysis.png'), dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_dir}/optimal_k_analysis.png")
        plt.close()
    
    results = {
        'k_values': k_values,
        'inertias': [float(x) for x in inertias],
        'silhouettes': [float(x) for x in silhouettes],
        'optimal_k': int(optimal_k),
        'optimal_silhouette': float(silhouettes[optimal_idx])
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'optimal_k_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='CpG Feature Clustering and Reordering for 1D-CNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
This tool clusters CpG features based on their methylation patterns across samples,
then outputs an ordered feature list where similar features are adjacent.
This ordering is ideal for 1D-CNN models that can learn local feature patterns.

GPU Support:
  Uses cuML (RAPIDS) for GPU acceleration when available.
  Install with: pip install cuml-cu11  (for CUDA 11)
  Falls back to sklearn (CPU) if cuML is not installed.

Examples:
  # Cluster features using GPU (if available)
  python cluster_cg.py --data training_data.h5 --n_clusters 100 --gpu

  # Force CPU-only mode
  python cluster_cg.py --data training_data.h5 --n_clusters 100 --cpu

  # Find optimal number of clusters
  python cluster_cg.py --data training_data.h5 --find_optimal --gpu

  # Custom settings
  python cluster_cg.py --data training_data.csv --n_clusters 200 --n_pca 100

Output files:
  - feature_order.txt           Ordered feature names (one per line)
  - feature_order.json          Ordered features in JSON format
  - feature_order_indices.npy   Numpy array of ordered indices
  - feature_order_with_clusters.csv  Full details with cluster info
  - feature_tsne_clusters.png   t-SNE visualization
  - feature_order_heatmap.png   Heatmap showing ordering quality
        '''
    )
    
    parser.add_argument('--data', required=True,
                       help='Path to training data (.h5 or .csv)')
    parser.add_argument('--output_dir', default='./cluster_output',
                       help='Output directory (default: ./cluster_output)')
    parser.add_argument('--n_clusters', type=int, default=100,
                       help='Number of feature clusters (default: 100)')
    parser.add_argument('--n_pca', type=int, default=50,
                       help='PCA components (default: 50)')
    parser.add_argument('--tsne_samples', type=int, default=10000,
                       help='Max features for t-SNE plot (default: 10000)')
    parser.add_argument('--tsne_perplexity', type=int, default=30,
                       help='t-SNE perplexity (default: 30)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max samples to use (default: all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # GPU/CPU selection
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument('--gpu', action='store_true',
                             help='Use GPU acceleration (cuML) if available')
    device_group.add_argument('--cpu', action='store_true',
                             help='Force CPU-only mode (sklearn)')
    
    # Optimal k finding
    parser.add_argument('--find_optimal', action='store_true',
                       help='Find optimal number of clusters')
    parser.add_argument('--k_min', type=int, default=20,
                       help='Minimum k for optimal search (default: 20)')
    parser.add_argument('--k_max', type=int, default=300,
                       help='Maximum k for optimal search (default: 300)')
    parser.add_argument('--k_step', type=int, default=20,
                       help='Step size for k search (default: 20)')
    
    args = parser.parse_args()
    
    # Set device (GPU or CPU)
    if args.cpu:
        set_device(use_gpu=False)
    elif args.gpu:
        set_device(use_gpu=True)
    else:
        # Auto-detect: use GPU if available
        set_device(use_gpu=HAS_CUML)
    
    # Print device info
    device_info = get_device_info()
    print(f"\nDevice configuration:")
    print(f"  cuML available: {device_info['cuml_available']}")
    print(f"  sklearn available: {device_info['sklearn_available']}")
    print(f"  Using GPU: {device_info['using_gpu']}")
    if device_info.get('gpu_name'):
        print(f"  GPU: {device_info['gpu_name']} ({device_info.get('gpu_memory_gb', 0):.1f} GB)")
    
    if args.find_optimal:
        find_optimal_clusters(
            data_path=args.data,
            output_dir=args.output_dir,
            k_range=(args.k_min, args.k_max, args.k_step),
            n_pca_components=args.n_pca,
            max_samples=args.max_samples,
            random_seed=args.seed
        )
    else:
        run_feature_clustering_pipeline(
            data_path=args.data,
            output_dir=args.output_dir,
            n_clusters=args.n_clusters,
            n_pca_components=args.n_pca,
            tsne_samples=args.tsne_samples,
            tsne_perplexity=args.tsne_perplexity,
            max_samples=args.max_samples,
            random_seed=args.seed
        )


if __name__ == '__main__':
    main()
