#!/usr/bin/env python3
"""
Feature Ordering for 1D-CNN Input - MEMORY OPTIMIZED

Reorder methylation features to maximize local similarity for 1D-CNN models.
This script computes pairwise distances between features based on methylation 
patterns and finds optimal orderings where similar features are adjacent.

OOM FIXES:
1. Sparse k-NN distance matrix (avoids 357K×357K full matrix)
2. IncrementalPCA for memory-efficient dimensionality reduction
3. Sampling-based quality metrics
4. Optional skip plotting for memory constraints
5. GPU support with automatic CPU fallback
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import h5py
import pickle
from pathlib import Path
import gc
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import cupy as cp
    from cuml.decomposition import PCA as cuPCA
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    from cuml.metrics import pairwise_distances as cu_pairwise_distances
    USE_GPU_AVAILABLE = True
except ImportError:
    USE_GPU_AVAILABLE = False

USE_GPU = False


def set_device(use_gpu=True):
    """Set whether to use GPU."""
    global USE_GPU
    USE_GPU = use_gpu and USE_GPU_AVAILABLE
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


def reduce_dimensions(feature_data, n_components=50, batch_size=None):
    """Reduce feature dimensionality using PCA with memory efficiency."""
    n_features, n_samples = feature_data.shape
    n_components = min(n_components, n_samples - 1)
    
    print(f"\nReducing dimensions with PCA to {n_components} components...")
    print(f"  Input shape: {feature_data.shape} (features × samples)")
    
    start = time.time()
    
    if USE_GPU:
        try:
            print("  Using GPU (cuML)...")
            scaler = cuStandardScaler()
            data_scaled = scaler.fit_transform(feature_data)
            pca = cuPCA(n_components=n_components)
            reduced = pca.fit_transform(data_scaled)
            if hasattr(reduced, 'get'):
                reduced = reduced.get()
            variance = float(sum(pca.explained_variance_ratio_)) * 100
        except Exception as e:
            print(f"  GPU PCA failed ({e}), falling back to CPU...")
            set_device(False)
    
    if not USE_GPU:
        scaler = StandardScaler()
        print("  Scaling features...")
        data_scaled = scaler.fit_transform(feature_data)
        
        # Use IncrementalPCA for large feature counts to save memory
        if n_features > 5000:
            if batch_size is None:
                batch_size = max(1000, min(5000, n_samples // 5))
            print(f"  Using IncrementalPCA with batch_size={batch_size}...")
            pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        else:
            pca = PCA(n_components=n_components)
        
        reduced = pca.fit_transform(data_scaled)
        variance = sum(pca.explained_variance_ratio_) * 100
    
    print(f"  PCA completed in {time.time()-start:.1f}s")
    print(f"  Explained variance: {variance:.1f}%")
    print(f"  Reduced shape: {reduced.shape}")
    gc.collect()
    
    return reduced, pca


def compute_distance_matrix(feature_repr, metric='correlation', n_neighbors=None, use_gpu=None):
    """Compute pairwise distances efficiently using sparse k-NN."""
    global USE_GPU
    if use_gpu is None:
        use_gpu = USE_GPU
    
    n_features = feature_repr.shape[0]
    
    print(f"\nComputing distance matrix ({metric})...")
    print(f"  Features: {n_features}")
    if use_gpu:
        print(f"  Using GPU (cuML) for distance computation")
    
    start = time.time()
    
    # Automatically use k-NN for large feature counts
    if n_neighbors is None:
        if n_features > 100000:
            n_neighbors = min(50, max(10, n_features // 10000))
        elif n_features > 50000:
            n_neighbors = min(100, max(20, n_features // 5000))
        elif n_features > 10000:
            n_neighbors = min(200, max(50, n_features // 2000))
        else:
            n_neighbors = None
    
    # Use sparse k-NN for large feature counts
    if n_neighbors and n_features > 5000:
        print(f"  Using sparse k-NN (k={n_neighbors}) to minimize memory...")
        print(f"  Building kNN index with metric '{metric}'...")
        
        # Limit k to available features
        k = min(n_neighbors, n_features - 1)
        
        # Use GPU-accelerated kNN if available and requested
        if use_gpu and USE_GPU_AVAILABLE:
            try:
                print(f"  Using GPU (cuML) for kNN computation...")
                # Convert to CuPy array
                feature_repr_gpu = cp.asarray(feature_repr)
                
                # cuML NearestNeighbors
                nn = cuNearestNeighbors(n_neighbors=k, metric=metric)
                nn.fit(feature_repr_gpu)
                
                print(f"  Computing {n_features:,} × {k} nearest neighbors on GPU...")
                distances, indices = nn.kneighbors(feature_repr_gpu)
                
                # Convert back to numpy
                distances = cp.asnumpy(distances)
                indices = cp.asnumpy(indices)
                
                # Free GPU memory
                del feature_repr_gpu
                cp.get_default_memory_pool().free_all_blocks()
                
            except Exception as e:
                print(f"  GPU kNN failed ({e}), falling back to CPU...")
                use_gpu = False
        
        if not use_gpu or not USE_GPU_AVAILABLE:
            # Choose algorithm based on metric (ball_tree doesn't support correlation)
            if metric == 'correlation':
                # Use brute force for correlation or convert to cosine on normalized data
                print(f"  Note: Using 'brute' algorithm for correlation metric...")
                nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1, algorithm='brute')
            elif metric in ['euclidean', 'manhattan', 'chebyshev', 'minkowski']:
                nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1, algorithm='ball_tree')
            else:
                # For other metrics, let sklearn choose the best algorithm
                nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1, algorithm='auto')
            
            print(f"  Fitting kNN index on {n_features:,} features...")
            nn.fit(feature_repr)
            
            print(f"  Computing {n_features:,} × {k} nearest neighbors...")
            # Process in batches with progress bar
            batch_size = 10000
            distances_list = []
            indices_list = []
            for i in tqdm(range(0, n_features, batch_size), desc="  kNN queries", unit="batch"):
                end_i = min(i + batch_size, n_features)
                dist_batch, ind_batch = nn.kneighbors(feature_repr[i:end_i])
                distances_list.append(dist_batch)
                indices_list.append(ind_batch)
            distances = np.vstack(distances_list)
            indices = np.vstack(indices_list)
        
        print(f"  Converting to dense matrix...")
        rows = np.repeat(np.arange(n_features), distances.shape[1])
        cols = indices.flatten()
        vals = distances.flatten()
        
        # Create sparse matrix
        dist_matrix_sparse = csr_matrix((vals, (rows, cols)), shape=(n_features, n_features))
        
        # Symmetrize in sparse format
        print(f"  Symmetrizing matrix (sparse)...")
        dist_matrix_sparse = dist_matrix_sparse + dist_matrix_sparse.T
        dist_matrix_sparse.data /= 2.0
        
        # Keep as sparse - convert to LIL format for efficient row access
        dist_matrix = dist_matrix_sparse.tolil()
        
        del dist_matrix_sparse
        gc.collect()
        
        print(f"  Distance matrix (sparse) created in {time.time()-start:.1f}s")
        print(f"  Shape: {dist_matrix.shape}, NNZ: {dist_matrix.nnz:,}")
        print(f"  Sparsity: {100.0 * (1 - dist_matrix.nnz / (n_features * n_features)):.1f}%")
    else:
        print(f"  Computing full pairwise distance matrix...")
        
        # Use GPU for full distance matrix if available
        if use_gpu and USE_GPU_AVAILABLE:
            try:
                print(f"  Using GPU (cuML) for pairwise distances...")
                feature_repr_gpu = cp.asarray(feature_repr)
                dist_matrix_gpu = cu_pairwise_distances(feature_repr_gpu, metric=metric)
                dist_matrix = cp.asnumpy(dist_matrix_gpu)
                
                # Free GPU memory
                del feature_repr_gpu, dist_matrix_gpu
                cp.get_default_memory_pool().free_all_blocks()
                
            except Exception as e:
                print(f"  GPU pairwise distances failed ({e}), falling back to CPU...")
                dist_matrix = pairwise_distances(feature_repr, metric=metric, n_jobs=-1)
        else:
            dist_matrix = pairwise_distances(feature_repr, metric=metric, n_jobs=-1)
        
        print(f"  Distance matrix computed in {time.time()-start:.1f}s")
        print(f"  Shape: {dist_matrix.shape}, dtype: {dist_matrix.dtype}")
        print(f"  Memory used: ~{dist_matrix.nbytes / 1e9:.2f} GB")
    
    # Handle both sparse and dense matrices
    if isinstance(dist_matrix, csr_matrix) or isinstance(dist_matrix, csr_matrix.__bases__[0]):
        # Sparse matrix - just clean diagonal
        dist_matrix = dist_matrix.tocsr()
        dist_matrix.setdiag(0)
        if hasattr(dist_matrix, 'eliminate_zeros'):
            dist_matrix.eliminate_zeros()
    else:
        # Dense matrix
        dist_matrix = np.nan_to_num(dist_matrix, nan=1.0, posinf=1.0, neginf=0.0)
        np.fill_diagonal(dist_matrix, 0)
    
    return dist_matrix


def order_by_nearest_neighbor(dist_matrix, start_idx=None, use_gpu=None):
    """Order features using nearest neighbor heuristic (greedy TSP)."""
    global USE_GPU
    if use_gpu is None:
        use_gpu = USE_GPU
    
    n_features = dist_matrix.shape[0]
    
    print(f"\nOrdering {n_features:,} features with nearest neighbor heuristic...")
    print(f"  Distance matrix type: {type(dist_matrix).__name__}")
    
    # Convert to CSR for efficient row access
    is_sparse = isinstance(dist_matrix, (csr_matrix,)) or hasattr(dist_matrix, 'tolil')
    if is_sparse:
        dist_matrix = dist_matrix.tocsr()
        print(f"  Using sparse matrix access (~{dist_matrix.nnz / 1e6:.1f}M nnz)")
    
    start_time = time.time()
    
    # GPU acceleration for both dense and sparse matrices
    if use_gpu and USE_GPU_AVAILABLE:
        try:
            if is_sparse:
                print(f"  Using GPU (CuPy) with sparse matrix operations...")
                # For sparse: GPU overhead per-iteration is too high, only use GPU for initial find
                # Convert visited to GPU once, reuse it
                visited_gpu = cp.zeros(n_features, dtype=bool)
                
                if start_idx is None:
                    print(f"  Finding best start node...")
                    start_idx = np.argmin(np.array(dist_matrix.sum(axis=1)).flatten())
                
                print(f"  Starting from feature {start_idx}")
                
                visited = np.zeros(n_features, dtype=bool)
                order = np.zeros(n_features, dtype=int)
                
                current = start_idx
                order[0] = current
                visited[current] = True
                visited_gpu[current] = True
                
                # For sparse matrices, CPU is actually faster due to low GPU transfer overhead per row
                # GPU transfer cost >> sparse row extraction + argmin cost
                # So we use CPU path but keep same interface
                print(f"  Note: CPU path faster for sparse matrices (minimal GPU transfer overhead)")
                for i in tqdm(range(1, n_features), desc="  Building feature order", unit="features"):
                    # Get row from sparse matrix (CPU is faster for this)
                    distances = dist_matrix.getrow(current).toarray().flatten().copy()
                    distances[visited] = np.inf
                    nearest = np.argmin(distances)
                    
                    order[i] = nearest
                    visited[nearest] = True
                    current = nearest
                
                # Free GPU memory
                del visited_gpu
                cp.get_default_memory_pool().free_all_blocks()
                
            else:
                print(f"  Using GPU (CuPy) for dense matrix operations...")
                dist_matrix_gpu = cp.asarray(dist_matrix)
                
                if start_idx is None:
                    print(f"  Finding best start node on GPU...")
                    start_idx = int(cp.argmin(cp.sum(dist_matrix_gpu, axis=1)))
                
                print(f"  Starting from feature {start_idx}")
                
                visited_gpu = cp.zeros(n_features, dtype=bool)
                order = np.zeros(n_features, dtype=int)
                
                current = start_idx
                order[0] = current
                visited_gpu[current] = True
                
                for i in tqdm(range(1, n_features), desc="  Building feature order (GPU)", unit="features"):
                    distances = dist_matrix_gpu[current].copy()
                    distances[visited_gpu] = cp.inf
                    nearest = int(cp.argmin(distances))
                    
                    order[i] = nearest
                    visited_gpu[nearest] = True
                    current = nearest
                
                # Free GPU memory
                del dist_matrix_gpu, visited_gpu
                cp.get_default_memory_pool().free_all_blocks()
        
        except Exception as e:
            print(f"  GPU ordering failed ({e}), falling back to CPU...")
            use_gpu = False
    
    # CPU-only path
    if not use_gpu:
        if start_idx is None:
            print(f"  Finding best start node...")
            if is_sparse:
                # For sparse, compute sum differently
                start_idx = np.argmin(np.array(dist_matrix.sum(axis=1)).flatten())
            else:
                start_idx = np.argmin(dist_matrix.sum(axis=1))
        
        print(f"  Starting from feature {start_idx}")
        
        visited = np.zeros(n_features, dtype=bool)
        order = np.zeros(n_features, dtype=int)
        
        current = start_idx
        order[0] = current
        visited[current] = True
        
        for i in tqdm(range(1, n_features), desc="  Building feature order", unit="features"):
            # Get distances from current feature
            if is_sparse:
                distances = dist_matrix.getrow(current).toarray().flatten().copy()
            else:
                distances = dist_matrix[current].copy()
            
            distances[visited] = np.inf
            nearest = np.argmin(distances)
            
            order[i] = nearest
            visited[nearest] = True
            current = nearest
    
    total_time = time.time() - start_time
    print(f"  Nearest neighbor ordering completed in {total_time:.1f}s")
    
    return order


def order_by_hierarchical(dist_matrix):
    """Order features using hierarchical clustering."""
    n_features = dist_matrix.shape[0]
    if n_features > 50000:
        print(f"  WARNING: Hierarchical clustering with {n_features} features may be very slow/memory intensive")
    
    print(f"\nOrdering features with hierarchical clustering...")
    
    start = time.time()
    
    print("  Converting to condensed distance matrix...")
    condensed = squareform(dist_matrix, checks=False)
    
    print("  Computing linkage...")
    Z = linkage(condensed, method='ward')
    
    print("  Extracting leaf order...")
    ordered_indices = leaves_list(Z)
    
    print(f"  Hierarchical ordering completed in {time.time()-start:.1f}s")
    
    return ordered_indices, Z


def order_by_mst(dist_matrix):
    """Order features using Minimum Spanning Tree linearization."""
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


def compute_ordering_quality(dist_matrix, order, sample_size=10000):
    """Compute quality metrics for a feature ordering (sampled to save memory)."""
    n = len(order)
    
    print(f"\nComputing ordering quality metrics...")
    
    # Determine if sparse or dense
    is_sparse = isinstance(dist_matrix, (csr_matrix,)) or hasattr(dist_matrix, 'getrow')
    
    # Compute consecutive distances
    print(f"  Computing consecutive distances...")
    if is_sparse:
        consecutive_dists = []
        for i in range(min(1000, n-1)):  # Sample for sparse efficiency
            d = dist_matrix[order[i], order[i+1]]
            if isinstance(d, np.matrix):
                d = float(d)
            consecutive_dists.append(d)
        # Estimate for full
        consecutive_dists = consecutive_dists * ((n-1) // len(consecutive_dists) + 1)
    else:
        consecutive_dists = [dist_matrix[order[i], order[i+1]] for i in range(n-1)]
    
    # Sample for neighborhood preservation computation
    if n > sample_size:
        print(f"  Sampling {sample_size:,}/{n:,} features for neighborhood assessment...")
        sample_indices = np.random.choice(n, sample_size, replace=False)
    else:
        sample_indices = np.arange(n)
    
    k = min(10, max(1, n // 1000))
    preserved = 0
    checked = 0
    
    print(f"  Computing neighborhood preservation (k={k})...")
    
    for idx in tqdm(sample_indices, desc="  Checking neighborhoods", unit="features"):
        i = idx
        # Get k nearest neighbors in original space
        if is_sparse:
            # For sparse, get the k smallest non-zero values in row
            row = dist_matrix.getrow(order[i])
            _, cols = row.nonzero()
            row_vals = row.data
            if len(row_vals) > k:
                top_k_idx = np.argsort(row_vals)[:k]
                orig_neighbors = set(cols[top_k_idx])
            else:
                orig_neighbors = set(cols)
        else:
            orig_neighbors = set(np.argsort(dist_matrix[order[i]])[:k])
        
        # Get local neighbors in ordering
        local_neighbors = set()
        for j in range(max(0, i-k), min(n, i+k+1)):
            if j != i:
                local_neighbors.add(order[j])
        
        preserved += len(orig_neighbors & local_neighbors)
        checked += 1
    
    neighborhood_preservation = preserved / (checked * k) if checked > 0 else 0.0
    
    metrics = {
        'mean_consecutive_distance': float(np.mean(consecutive_dists[:1000])),
        'median_consecutive_distance': float(np.median(consecutive_dists[:1000])),
        'max_consecutive_distance': float(np.max(consecutive_dists[:1000])),
        'total_path_length': float(np.sum(consecutive_dists[:1000])),
        'neighborhood_preservation': round(neighborhood_preservation, 4)
    }
    
    print(f"\n  Ordering Quality:")
    for key, val in metrics.items():
        print(f"    {key}: {val}")
    
    return metrics


def save_sparse_cache(dist_matrix, order, output_dir, prefix='distance_cache'):
    """Save sparse distance matrix and ordering to disk."""
    cache_dir = os.path.join(output_dir, '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"\nSaving sparse metrics to cache...")
    
    # Save sparse distance matrix
    dist_path = os.path.join(cache_dir, f'{prefix}_matrix.npz')
    if isinstance(dist_matrix, (csr_matrix,)) or hasattr(dist_matrix, 'nnz'):
        dist_csr = dist_matrix.tocsr() if hasattr(dist_matrix, 'tocsr') else dist_matrix
        save_npz(dist_path, dist_csr)
        print(f"  Saved sparse distance matrix: {dist_path}")
    else:
        np.save(dist_path, dist_matrix)
        print(f"  Saved dense distance matrix: {dist_path}")
    
    # Save ordering
    order_path = os.path.join(cache_dir, f'{prefix}_order.npy')
    np.save(order_path, order)
    print(f"  Saved feature order: {order_path}")
    
    # Save cache metadata
    metadata_path = os.path.join(cache_dir, f'{prefix}_metadata.json')
    metadata = {
        'n_features': len(order),
        'is_sparse': isinstance(dist_matrix, (csr_matrix,)) or hasattr(dist_matrix, 'nnz'),
        'timestamp': time.time(),
        'dist_path': dist_path,
        'order_path': order_path
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved cache metadata: {metadata_path}")
    
    return cache_dir


def load_sparse_cache(output_dir, prefix='distance_cache'):
    """Load sparse distance matrix and ordering from cache."""
    cache_dir = os.path.join(output_dir, '.cache')
    metadata_path = os.path.join(cache_dir, f'{prefix}_metadata.json')
    
    if not os.path.exists(metadata_path):
        return None, None, None
    
    print(f"\nLoading sparse metrics from cache...")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load distance matrix
    dist_path = metadata['dist_path']
    if dist_path.endswith('.npz'):
        dist_matrix = load_npz(dist_path)
    else:
        dist_matrix = np.load(dist_path)
    
    # Load ordering
    order_path = metadata['order_path']
    order = np.load(order_path)
    
    print(f"  Loaded sparse distance matrix: {dist_path}")
    print(f"  Loaded feature order: {order_path}")
    print(f"  Cache created at: {time.ctime(metadata['timestamp'])}")
    
    return dist_matrix, order, metadata


def plot_ordering_analysis(dist_matrix, order, feature_data, output_dir):
    """Plot analysis of the feature ordering with distance distributions."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating ordering analysis plots...")
    
    n = len(order)
    is_sparse = isinstance(dist_matrix, (csr_matrix,)) or hasattr(dist_matrix, 'getrow')
    
    # Compute consecutive distances efficiently
    print(f"  Computing consecutive distances...")
    consecutive_dists = []
    for i in range(n-1):
        if is_sparse:
            d = dist_matrix[order[i], order[i+1]]
            if isinstance(d, np.matrix):
                d = float(d.item())
            else:
                d = float(d)
        else:
            d = float(dist_matrix[order[i], order[i+1]])
        consecutive_dists.append(d)
    consecutive_dists = np.array(consecutive_dists)
    
    # Sample distances for distribution analysis
    print(f"  Sampling distances for distribution analysis...")
    if is_sparse:
        # For sparse matrix, sample random pairs and their distances
        sample_size = min(50000, n * 10)
        sample_distances = []
        for _ in range(sample_size):
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i != j:
                d = dist_matrix[i, j]
                if d != 0:  # Only non-zero entries
                    sample_distances.append(float(d))
        sample_distances = np.array(sample_distances)
    else:
        # For dense matrix, get upper triangle
        indices = np.triu_indices(n, k=1)
        sample_distances = dist_matrix[indices]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Consecutive distances over position with smoothing
    ax1 = fig.add_subplot(gs[0, :2])
    window = min(1000, max(100, len(consecutive_dists) // 100))
    smoothed = np.convolve(consecutive_dists, np.ones(window)/window, mode='valid')
    ax1.plot(range(len(smoothed)), smoothed, linewidth=1.5, alpha=0.8, color='blue')
    ax1.fill_between(range(len(smoothed)), smoothed, alpha=0.3, color='blue')
    ax1.set_xlabel('Feature Position', fontsize=12)
    ax1.set_ylabel('Distance to Next Feature', fontsize=12)
    ax1.set_title(f'Consecutive Feature Distances (smoothed, window={window})', fontsize=13, fontweight='bold')
    ax1.axhline(float(np.mean(consecutive_dists)), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(consecutive_dists):.4f}')
    ax1.axhline(float(np.median(consecutive_dists)), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(consecutive_dists):.4f}')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Consecutive distances stats box
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    stats_text = f"""CONSECUTIVE DISTANCES
    
Mean:      {np.mean(consecutive_dists):.6f}
Median:    {np.median(consecutive_dists):.6f}
Std Dev:   {np.std(consecutive_dists):.6f}
Min:       {np.min(consecutive_dists):.6f}
Max:       {np.max(consecutive_dists):.6f}
Q25:       {np.percentile(consecutive_dists, 25):.6f}
Q75:       {np.percentile(consecutive_dists, 75):.6f}
Sum:       {np.sum(consecutive_dists):.2f}
Count:     {len(consecutive_dists):,}"""
    ax2.text(0.1, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 3: Histogram of consecutive distances
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(consecutive_dists, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Distance', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Distribution: Consecutive Distances', fontsize=12, fontweight='bold')
    ax3.axvline(float(np.mean(consecutive_dists)), color='red', linestyle='--', linewidth=2, label='Mean')
    ax3.axvline(float(np.median(consecutive_dists)), color='green', linestyle='--', linewidth=2, label='Median')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Log-scale histogram of consecutive distances
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(consecutive_dists, bins=100, color='coral', edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Distance', fontsize=11)
    ax4.set_ylabel('Frequency (log scale)', fontsize=11)
    ax4.set_title('Distribution: Consecutive Distances (log scale)', fontsize=12, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, which='both')
    
    # Plot 5: Histogram of all sampled distances
    ax5 = fig.add_subplot(gs[1, 2])
    if len(sample_distances) > 0:
        ax5.hist(sample_distances, bins=100, color='mediumseagreen', edgecolor='black', alpha=0.7)
        ax5.set_xlabel('Distance', fontsize=11)
        ax5.set_ylabel('Frequency', fontsize=11)
        ax5.set_title(f'Distribution: All Distances\n(sampled, n={len(sample_distances):,})', 
                     fontsize=12, fontweight='bold')
        ax5.axvline(float(np.mean(sample_distances)), color='red', linestyle='--', linewidth=2, label='Mean')
        ax5.axvline(float(np.median(sample_distances)), color='green', linestyle='--', linewidth=2, label='Median')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Cumulative distribution of consecutive distances
    ax6 = fig.add_subplot(gs[2, 0])
    sorted_dists = np.sort(consecutive_dists)
    cumsum = np.arange(1, len(sorted_dists) + 1) / len(sorted_dists)
    ax6.plot(sorted_dists, cumsum, linewidth=2, color='darkblue')
    ax6.fill_between(sorted_dists, cumsum, alpha=0.3, color='darkblue')
    ax6.set_xlabel('Distance', fontsize=11)
    ax6.set_ylabel('Cumulative Probability', fontsize=11)
    ax6.set_title('CDF: Consecutive Distances', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Box plot comparing statistics
    ax7 = fig.add_subplot(gs[2, 1])
    data_to_plot = [consecutive_dists]
    labels_to_use = ['Consecutive']
    if len(sample_distances) > 0:
        data_to_plot.append(sample_distances)
        labels_to_use.append('All Sampled')
    bp = ax7.boxplot(data_to_plot, patch_artist=True)
    ax7.set_xticklabels(labels_to_use)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen'][:len(bp['boxes'])]):
        patch.set_facecolor(color)
    ax7.set_ylabel('Distance', fontsize=11)
    ax7.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Plot 8: Distance matrix visualization (sampled)
    ax8 = fig.add_subplot(gs[2, 2])
    sample_size_vis = min(500, n)
    sample_idx = np.linspace(0, n-1, sample_size_vis, dtype=int)
    sample_order_idx = order[sample_idx]
    
    if is_sparse:
        # For sparse, create a small dense submatrix
        sample_dist_dense = dist_matrix[sample_order_idx][:, sample_order_idx].toarray()
    else:
        sample_dist_dense = dist_matrix[np.ix_(sample_order_idx, sample_order_idx)]
    
    im = ax8.imshow(sample_dist_dense, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax8.set_xlabel('Feature (ordered)', fontsize=11)
    ax8.set_ylabel('Feature (ordered)', fontsize=11)
    ax8.set_title(f'Distance Matrix Heatmap\n(sampled {sample_size_vis}×{sample_size_vis})', 
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax8, label='Distance')
    
    # Add overall title
    fig.suptitle(f'Feature Ordering Analysis: {n:,} features ordered with nearest-neighbor heuristic', 
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(os.path.join(output_dir, 'feature_ordering_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/feature_ordering_analysis.png")
    plt.close()
    
    # Create additional distance distribution summary
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Violin plot
    parts = axes[0, 0].violinplot([consecutive_dists], positions=[0], widths=0.7, 
                                   showmeans=True, showmedians=True)
    axes[0, 0].set_ylabel('Distance', fontsize=11)
    axes[0, 0].set_title('Violin Plot: Consecutive Distances', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks([0])
    axes[0, 0].set_xticklabels(['Consecutive'])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Distance percentiles
    percentiles = np.linspace(0, 100, 21)
    percentile_vals = np.percentile(consecutive_dists, percentiles)
    axes[0, 1].plot(percentiles, percentile_vals, marker='o', linewidth=2, markersize=6, color='darkgreen')
    axes[0, 1].set_xlabel('Percentile', fontsize=11)
    axes[0, 1].set_ylabel('Distance', fontsize=11)
    axes[0, 1].set_title('Percentile Plot: Consecutive Distances', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distance autocorrelation-like plot (consecutive pairs)
    if len(consecutive_dists) > 1:
        axes[1, 0].scatter(consecutive_dists[:-1], consecutive_dists[1:], alpha=0.3, s=10)
        axes[1, 0].set_xlabel('Distance[i]', fontsize=11)
        axes[1, 0].set_ylabel('Distance[i+1]', fontsize=11)
        axes[1, 0].set_title('Consecutive Distance Correlation', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = np.corrcoef(consecutive_dists[:-1], consecutive_dists[1:])[0, 1]
        axes[1, 0].text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=axes[1, 0].transAxes,
                       fontsize=11, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Summary statistics table
    axes[1, 1].axis('off')
    summary_stats = [
        ['Metric', 'Value'],
        ['Total Features', f'{n:,}'],
        ['Consecutive Pairs', f'{len(consecutive_dists):,}'],
        ['Mean Distance', f'{np.mean(consecutive_dists):.6f}'],
        ['Median Distance', f'{np.median(consecutive_dists):.6f}'],
        ['Std Deviation', f'{np.std(consecutive_dists):.6f}'],
        ['Min Distance', f'{np.min(consecutive_dists):.6f}'],
        ['Max Distance', f'{np.max(consecutive_dists):.6f}'],
        ['Total Path Length', f'{np.sum(consecutive_dists):.2f}'],
        ['Distance Range', f'{np.max(consecutive_dists) - np.min(consecutive_dists):.6f}'],
    ]
    
    table = axes[1, 1].table(cellText=summary_stats, cellLoc='left', loc='center',
                            colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_stats)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    axes[1, 1].set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    fig2.suptitle('Distance Distribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distance_distribution_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/distance_distribution_analysis.png")
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
                         n_pca_components=50, max_samples=None, random_seed=42,
                         skip_plotting=False, use_cache=True):
    """Run the complete feature ordering pipeline with memory efficiency."""
    np.random.seed(random_seed)
    
    print("=" * 70)
    print("Feature Ordering for 1D-CNN Input (Memory Optimized)")
    print("=" * 70)
    print(f"Method: {method}")
    print(f"Distance metric: {metric}")
    print(f"PCA components: {n_pca_components}")
    print(f"Skip plotting: {skip_plotting}")
    print(f"Use cache: {use_cache}")
    
    start_total = time.time()
    
    # Try to load from cache
    if use_cache:
        dist_matrix, order, metadata = load_sparse_cache(output_dir, prefix=f'{method}_{metric}')
        if dist_matrix is not None and order is not None:
            print(f"\nUsing cached distance matrix and order from previous run")
            feature_data, feature_names, sample_labels = load_feature_data(
                data_path, max_samples=max_samples
            )
            
            quality = compute_ordering_quality(dist_matrix, order, sample_size=10000)
            
            if not skip_plotting:
                try:
                    plot_ordering_analysis(dist_matrix, order, feature_data, output_dir)
                except Exception as e:
                    print(f"  Warning: Plotting failed ({e}), skipping visualization")
            
            ordered_features = save_feature_order(feature_names, order, output_dir)
            
            total_time = time.time() - start_total
            
            results = {
                'method': method,
                'metric': metric,
                'n_features': len(order),
                'n_pca_components': n_pca_components,
                'pca_explained_variance': metadata.get('pca_explained_variance', 'N/A') if metadata else 'N/A',
                'quality_metrics': quality,
                'runtime_seconds': round(total_time, 2),
                'loaded_from_cache': True
            }
            
            with open(os.path.join(output_dir, 'ordering_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved: {output_dir}/ordering_results.json")
            
            print(f"\n{'=' * 70}")
            print(f"Pipeline completed in {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"{'=' * 70}")
            print(f"\nOutput files in: {output_dir}")
            print("=" * 70)
            
            return {
                'ordered_indices': order,
                'ordered_features': ordered_features,
                'quality': quality,
                'results': results
            }
    
    # Full computation path
    feature_data, feature_names, sample_labels = load_feature_data(
        data_path, max_samples=max_samples
    )
    n_features = feature_data.shape[0]
    print(f"Total features: {n_features:,}")
    
    feature_repr, pca = reduce_dimensions(feature_data, n_components=n_pca_components)
    
    dist_matrix = compute_distance_matrix(
        feature_repr, 
        metric=metric,
        n_neighbors=None,  # Auto-computed in function
        use_gpu=USE_GPU
    )
    
    print(f"\nPerforming feature ordering with method '{method}'...")
    if method == 'nearest_neighbor':
        order = order_by_nearest_neighbor(dist_matrix, use_gpu=USE_GPU)
    elif method == 'hierarchical':
        if n_features > 50000:
            print(f"  WARNING: Hierarchical clustering may be very slow for {n_features:,} features.")
            print(f"  Consider using 'nearest_neighbor' or 'mst' instead.")
        order, linkage_matrix = order_by_hierarchical(dist_matrix)
    elif method == 'mst':
        order = order_by_mst(dist_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Save sparse cache before computing quality metrics
    if use_cache:
        save_sparse_cache(dist_matrix, order, output_dir, 
                         prefix=f'{method}_{metric}')
    
    quality = compute_ordering_quality(dist_matrix, order, sample_size=10000)
    
    if not skip_plotting:
        try:
            plot_ordering_analysis(dist_matrix, order, feature_data, output_dir)
        except Exception as e:
            print(f"  Warning: Plotting failed ({e}), skipping visualization")
    
    ordered_features = save_feature_order(feature_names, order, output_dir)
    
    total_time = time.time() - start_total
    
    results = {
        'method': method,
        'metric': metric,
        'n_features': n_features,
        'n_pca_components': n_pca_components,
        'pca_explained_variance': round(float(sum(pca.explained_variance_ratio_) * 100), 2),
        'quality_metrics': quality,
        'runtime_seconds': round(total_time, 2),
        'loaded_from_cache': False
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'ordering_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output_dir}/ordering_results.json")
    
    print(f"\n{'=' * 70}")
    print(f"Pipeline completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}")
    print(f"\nOutput files in: {output_dir}")
    print("=" * 70)
    
    return {
        'ordered_indices': order,
        'ordered_features': ordered_features,
        'quality': quality,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Feature Ordering for 1D-CNN Input (Memory Optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Reorder methylation features to maximize local similarity for 1D-CNN.

OOM FIXES:
- Sparse k-NN for distance computation (avoids full 357K×357K matrix)
- IncrementalPCA for memory-efficient dimensionality reduction
- Sampling-based quality metrics
- Optional visualization skip (--skip_plotting)

Methods:
  nearest_neighbor - Greedy TSP approximation (RECOMMENDED - fast, low memory)
  hierarchical     - Hierarchical clustering with optimal leaf ordering
  mst              - Minimum spanning tree linearization

Examples:
  python get_feature_orders.py --data training_data.h5 --method nearest_neighbor
  python get_feature_orders.py --data training_data.h5 --skip_plotting
  python get_feature_orders.py --data training_data.h5 --gpu --method nearest_neighbor
        '''
    )
    
    parser.add_argument('--data', required=True, help='Path to training data')
    parser.add_argument('--output_dir', default='./feature_order_output', help='Output directory')
    parser.add_argument('--method', default='nearest_neighbor',
                       choices=['nearest_neighbor', 'hierarchical', 'mst'],
                       help='Ordering method (nearest_neighbor recommended for large datasets)')
    parser.add_argument('--metric', default='correlation',
                       choices=['correlation', 'euclidean', 'cosine'])
    parser.add_argument('--n_pca', type=int, default=50,
                       help='Number of PCA components for distance calculation')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max samples to load (use for memory testing)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', action='store_true', help='Use GPU (cuML) if available')
    parser.add_argument('--skip_plotting', action='store_true', 
                       help='Skip visualization to save memory')
    parser.add_argument('--no_cache', action='store_true',
                       help='Do not use cached distance matrix/order (recompute from scratch)')
    
    args = parser.parse_args()
    
    set_device(args.gpu)
    
    run_ordering_pipeline(
        data_path=args.data,
        output_dir=args.output_dir,
        method=args.method,
        metric=args.metric,
        n_pca_components=args.n_pca,
        max_samples=args.max_samples,
        random_seed=args.seed,
        skip_plotting=args.skip_plotting,
        use_cache=not args.no_cache
    )


if __name__ == '__main__':
    main()
