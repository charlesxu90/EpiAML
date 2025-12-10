#!/usr/bin/env python3
"""
Create a debug dataset with 10% of features from the original data.

This script is useful for quick testing and debugging without waiting for
the full 357K feature processing. It samples 10% of features uniformly
and saves to a new HDF5 file.

Usage:
    python create_debug_data.py --input training_data.h5 --output training_data_debug.h5
    python create_debug_data.py --input training_data.h5 --output training_data_debug.h5 --fraction 0.05
"""

import os
import argparse
import h5py
import numpy as np
from pathlib import Path


def create_debug_data(input_path, output_path, fraction=0.1, random_seed=42):
    """Create a debug dataset with a subset of features."""
    np.random.seed(random_seed)
    
    print("=" * 70)
    print("Creating Debug Dataset")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Feature fraction: {fraction * 100:.1f}%")
    print(f"Random seed: {random_seed}")
    
    # Read input data
    print(f"\nLoading input data...")
    with h5py.File(input_path, 'r') as f:
        data = f['data'][:]
        labels = f['labels'][:] if 'labels' in f else None
        feature_names = f['feature_names'][:] if 'feature_names' in f else None
        
        n_samples, n_features = data.shape
        print(f"  Input shape: {n_samples} samples × {n_features} features")
    
    # Sample features
    n_debug_features = max(1, int(n_features * fraction))
    print(f"\nSampling {n_debug_features:,} features ({fraction*100:.1f}%)...")
    
    feature_indices = np.sort(np.random.choice(n_features, n_debug_features, replace=False))
    debug_data = data[:, feature_indices]
    
    if feature_names is not None:
        debug_feature_names = feature_names[feature_indices]
    else:
        debug_feature_names = None
    
    print(f"  Output shape: {debug_data.shape[0]} samples × {debug_data.shape[1]} features")
    print(f"  Sampled feature indices range: [{feature_indices.min()}, {feature_indices.max()}]")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Write output data
    print(f"\nSaving debug data to HDF5...")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('data', data=debug_data, compression='gzip', compression_opts=4)
        
        if labels is not None:
            f.create_dataset('labels', data=labels)
        
        if debug_feature_names is not None:
            f.create_dataset('feature_names', data=debug_feature_names)
        
        # Store feature indices as dataset (not attribute - attributes have size limits)
        f.create_dataset('feature_indices', data=feature_indices)
        
        # Store metadata about sampling
        f.attrs['original_n_features'] = n_features
        f.attrs['debug_n_features'] = n_debug_features
        f.attrs['fraction'] = fraction
        f.attrs['feature_fraction_percent'] = fraction * 100
        f.attrs['random_seed'] = random_seed
        
        print(f"  Saved: {output_path}")
        print(f"  Compression: gzip (level 4)")
    
    # Print file sizes
    input_size = os.path.getsize(input_path) / (1024**3)
    output_size = os.path.getsize(output_path) / (1024**3)
    
    print(f"\nFile sizes:")
    print(f"  Input:  {input_size:.2f} GB")
    print(f"  Output: {output_size:.2f} GB")
    print(f"  Reduction: {(1 - output_size/input_size)*100:.1f}%")
    
    print(f"\n{'=' * 70}")
    print(f"Debug dataset created successfully!")
    print(f"{'=' * 70}")
    print(f"\nTo access feature indices in the output file:")
    print(f"  with h5py.File('{output_path}', 'r') as f:")
    print(f"      feature_indices = f['feature_indices'][:]")
    
    return {
        'input_path': input_path,
        'output_path': output_path,
        'original_shape': (n_samples, n_features),
        'debug_shape': debug_data.shape,
        'feature_indices': feature_indices,
        'input_size_gb': input_size,
        'output_size_gb': output_size
    }


def main():
    parser = argparse.ArgumentParser(
        description='Create a debug dataset with 10% of features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Create a smaller debug dataset by sampling features uniformly from the full dataset.
This is useful for quick testing and debugging without processing 357K features.

Examples:
  # Default 10% sampling
  python create_debug_data.py --input training_data.h5 --output training_data_debug.h5
  
  # 5% sampling
  python create_debug_data.py --input training_data.h5 --output training_data_debug.h5 --fraction 0.05
  
  # 20% sampling
  python create_debug_data.py --input training_data.h5 --output training_data_debug.h5 --fraction 0.20
        '''
    )
    
    parser.add_argument('--input', required=True, help='Path to input HDF5 file')
    parser.add_argument('--output', required=True, help='Path to output HDF5 file')
    parser.add_argument('--fraction', type=float, default=0.1,
                       help='Fraction of features to sample (default: 0.1 = 10%%)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate fraction
    if not 0 < args.fraction <= 1:
        print(f"Error: fraction must be between 0 and 1, got {args.fraction}")
        return
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        return
    
    create_debug_data(
        input_path=args.input,
        output_path=args.output,
        fraction=args.fraction,
        random_seed=args.seed
    )


if __name__ == '__main__':
    main()
