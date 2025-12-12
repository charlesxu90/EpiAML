#!/usr/bin/env python3
"""
Convert nanopore CSV to HDF5 with same feature order as all_data
Reorders columns to match reference_features.txt for fair comparison
"""

import csv
import sys
import os
import argparse
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm


def load_reference_features(reference_file):
    """Load reference feature order from all_data."""
    print(f"Loading reference feature order from {reference_file}...")
    with open(reference_file, 'r') as f:
        features = [line.strip() for line in f]
    print(f"  Loaded {len(features):,} reference features")
    return features


def load_labels(labels_file, label_column='Diagnosis_WHO4'):
    """Load labels from CSV file."""
    print(f"Loading labels from {labels_file}...")
    labels_df = pd.read_csv(labels_file)
    print(f"  Available columns: {list(labels_df.columns)}")
    print(f"  Using label column: {label_column}")

    # Create mapping from sample_id to label
    sample_to_label = {}
    for _, row in labels_df.iterrows():
        sample_id = row['sample_id']
        label = row[label_column]
        sample_to_label[sample_id] = label

    print(f"  Loaded {len(sample_to_label):,} labels")
    unique_labels = set(sample_to_label.values())
    print(f"  Unique labels: {len(unique_labels)}")
    for label in sorted(unique_labels):
        count = sum(1 for v in sample_to_label.values() if v == label)
        print(f"    {label}: {count}")

    return sample_to_label


def csv_to_hdf5_reordered(csv_file, h5_file, reference_features, sample_to_label,
                          compression='gzip', chunk_size=100):
    """
    Convert nanopore CSV to HDF5 with reordered features to match reference.

    Args:
        csv_file (str): Input CSV file (nanopore data)
        h5_file (str): Output HDF5 file
        reference_features (list): Ordered list of feature names from all_data
        sample_to_label (dict): Mapping from sample_id to label
        compression (str): Compression algorithm ('gzip', 'lzf', or None)
        chunk_size (int): Number of rows to process at once
    """
    print(f"\nConverting nanopore CSV to HDF5 (with feature reordering)")
    print(f"  Input: {csv_file}")
    print(f"  Output: {h5_file}")
    print(f"  Compression: {compression}")
    print(f"  Chunk size: {chunk_size} rows")

    # Get file size
    file_size_mb = os.path.getsize(csv_file) / (1024**2)
    print(f"  Input file size: {file_size_mb:.2f} MB")

    # First pass: read header and create column mapping
    print("\nReading header and creating column mapping...")
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)

        # Count total samples
        n_samples = sum(1 for _ in reader)

    print(f"  Original columns: {len(header):,}")
    print(f"  Samples: {n_samples:,}")

    # Create mapping from reference feature to column index in CSV
    # Assume first column is sample_id, rest are features
    csv_feature_to_idx = {}
    for i, col_name in enumerate(header[1:], start=1):  # Skip sample_id
        csv_feature_to_idx[col_name] = i

    # Check feature overlap
    reference_set = set(reference_features)
    csv_set = set(csv_feature_to_idx.keys())

    common_features = reference_set & csv_set
    missing_in_csv = reference_set - csv_set
    extra_in_csv = csv_set - reference_set

    print(f"\nFeature alignment:")
    print(f"  Reference features: {len(reference_features):,}")
    print(f"  CSV features: {len(csv_feature_to_idx):,}")
    print(f"  Common features: {len(common_features):,}")
    print(f"  Missing in CSV: {len(missing_in_csv):,}")
    print(f"  Extra in CSV: {len(extra_in_csv):,}")

    # Create ordered column indices for extraction
    # For each reference feature, get its index in CSV (or None if missing)
    ordered_indices = []
    for feat in reference_features:
        if feat in csv_feature_to_idx:
            ordered_indices.append(csv_feature_to_idx[feat])
        else:
            ordered_indices.append(None)  # Will fill with 0.0

    n_features = len(reference_features)

    # Remove existing output file if it exists
    if os.path.exists(h5_file):
        os.remove(h5_file)

    # Create HDF5 file and datasets
    print("\nCreating HDF5 file...")
    with h5py.File(h5_file, 'w') as hf:
        # Create datasets
        data_dataset = hf.create_dataset(
            'data',
            shape=(n_samples, n_features),
            dtype='float32',
            compression=compression,
            compression_opts=4 if compression == 'gzip' else None,
            chunks=(min(chunk_size, n_samples), n_features)
        )

        # Use variable-length string type for labels
        dt = h5py.special_dtype(vlen=str)
        labels_dataset = hf.create_dataset(
            'labels',
            shape=(n_samples,),
            dtype=dt
        )

        # Store sample IDs
        sample_ids_dataset = hf.create_dataset(
            'sample_ids',
            shape=(n_samples,),
            dtype=dt
        )

        # Store feature names in reference order
        hf.create_dataset(
            'feature_names',
            data=[str(name) for name in reference_features],
            dtype=dt
        )

        # Add metadata
        metadata = hf.create_group('metadata')
        metadata.attrs['n_samples'] = n_samples
        metadata.attrs['n_features'] = n_features
        metadata.attrs['common_features'] = len(common_features)
        metadata.attrs['missing_features'] = len(missing_in_csv)

        # Second pass: read data and write to HDF5
        print("\nReading and converting data...")
        row_idx = 0
        samples_without_labels = []

        with open(csv_file, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header

            chunk_data = []
            chunk_labels = []
            chunk_sample_ids = []

            # Use tqdm for progress bar
            with tqdm(total=n_samples, unit='rows', desc='Converting') as pbar:
                for row in reader:
                    # Extract sample_id
                    sample_id = row[0]

                    # Get label
                    if sample_id in sample_to_label:
                        label = sample_to_label[sample_id]
                    else:
                        label = "Unknown"
                        samples_without_labels.append(sample_id)

                    # Extract and reorder features
                    features_float = []
                    for idx in ordered_indices:
                        if idx is None:
                            # Feature missing in CSV, use 0.0
                            features_float.append(0.0)
                        else:
                            # Get value from CSV
                            value = row[idx]
                            try:
                                features_float.append(float(value) if value.strip() else 0.0)
                            except ValueError:
                                features_float.append(0.0)

                    chunk_data.append(features_float)
                    chunk_labels.append(label)
                    chunk_sample_ids.append(sample_id)

                    # Write chunk when full
                    if len(chunk_data) >= chunk_size:
                        # Write chunks
                        data_dataset[row_idx:row_idx+len(chunk_data)] = np.array(chunk_data, dtype=np.float32)
                        labels_dataset[row_idx:row_idx+len(chunk_labels)] = chunk_labels
                        sample_ids_dataset[row_idx:row_idx+len(chunk_sample_ids)] = chunk_sample_ids

                        row_idx += len(chunk_data)
                        pbar.update(len(chunk_data))

                        # Clear chunks
                        chunk_data = []
                        chunk_labels = []
                        chunk_sample_ids = []

                # Write remaining data
                if chunk_data:
                    data_dataset[row_idx:row_idx+len(chunk_data)] = np.array(chunk_data, dtype=np.float32)
                    labels_dataset[row_idx:row_idx+len(chunk_labels)] = chunk_labels
                    sample_ids_dataset[row_idx:row_idx+len(chunk_sample_ids)] = chunk_sample_ids
                    pbar.update(len(chunk_data))

        # Update metadata with class count
        all_labels = labels_dataset[:]
        unique_labels = len(set(all_labels))
        metadata.attrs['n_classes'] = unique_labels
        print(f"  Unique classes in dataset: {unique_labels}")

        if samples_without_labels:
            print(f"  Warning: {len(samples_without_labels)} samples without labels (marked as 'Unknown')")

    # Print file size comparison
    h5_size_mb = os.path.getsize(h5_file) / (1024**2)
    print(f"\n✓ Conversion complete!")
    print(f"  Input (CSV): {file_size_mb:.2f} MB")
    print(f"  Output (HDF5): {h5_size_mb:.2f} MB")
    print(f"  Size reduction: {(1 - h5_size_mb/file_size_mb)*100:.1f}%")
    print(f"  Output file: {h5_file}")
    print(f"\n✓ Feature order matches reference (all_data)")
    print(f"  Common features: {len(common_features):,}/{len(reference_features):,} ({100*len(common_features)/len(reference_features):.1f}%)")


def verify_hdf5(h5_file):
    """Verify HDF5 file and print summary."""
    print(f"\nVerifying HDF5 file: {h5_file}")

    with h5py.File(h5_file, 'r') as hf:
        print(f"  Datasets: {list(hf.keys())}")

        if 'data' in hf:
            data_shape = hf['data'].shape
            print(f"  Data shape: {data_shape[0]:,} samples × {data_shape[1]:,} features")

        if 'labels' in hf:
            labels = hf['labels'][:]
            unique_labels = set(labels)
            print(f"  Labels: {len(labels):,} samples, {len(unique_labels)} unique classes")
            for label in sorted(unique_labels):
                count = sum(1 for l in labels if l == label)
                print(f"    {label}: {count}")

        if 'sample_ids' in hf:
            sample_ids = hf['sample_ids'][:]
            print(f"  Sample IDs: {len(sample_ids):,}")

        if 'feature_names' in hf:
            n_features = len(hf['feature_names'])
            print(f"  Feature names: {n_features:,}")
            print(f"  First 5 features: {list(hf['feature_names'][:5])}")

        if 'metadata' in hf:
            print(f"  Metadata:")
            for key, value in hf['metadata'].attrs.items():
                print(f"    {key}: {value}")

    print("✓ Verification successful!")


def main():
    parser = argparse.ArgumentParser(
        description='Convert nanopore CSV to HDF5 with feature reordering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Convert nanopore data with default paths
  python convert_nanopore_to_h5.py

  # With custom paths
  python convert_nanopore_to_h5.py \\
    --input data/nanopore_data/NG_nanopore_training_data.csv \\
    --output data/nanopore_data/nanopore_training_data.h5 \\
    --reference data/all_data/reference_features.txt \\
    --labels data/nanopore_data/NG_nanopore_labels.csv

  # With custom label column
  python convert_nanopore_to_h5.py --label_column Diagnosis_lineage

Why reorder features:
  - Ensures same feature order as all_data for fair comparison
  - Handles missing features by filling with 0.0
  - Preserves sample IDs for traceability
        '''
    )

    parser.add_argument('--input',
                        default='data/nanopore_data/NG_nanopore_training_data(1).csv',
                        help='Input nanopore CSV file')
    parser.add_argument('--output',
                        default='data/nanopore_data/nanopore_training_data.h5',
                        help='Output HDF5 file')
    parser.add_argument('--reference',
                        default='data/all_data/reference_features.txt',
                        help='Reference features file (for feature order)')
    parser.add_argument('--labels',
                        default='data/nanopore_data/NG_nanopore_labels.csv',
                        help='Labels CSV file')
    parser.add_argument('--label_column',
                        default='Diagnosis_WHO4',
                        help='Name of label column (default: Diagnosis_WHO4)')
    parser.add_argument('--compression',
                        default='gzip',
                        choices=['gzip', 'lzf', 'none'],
                        help='Compression algorithm (default: gzip)')
    parser.add_argument('--chunk_size',
                        type=int,
                        default=100,
                        help='Rows per chunk (default: 100)')
    parser.add_argument('--verify',
                        action='store_true',
                        help='Verify HDF5 file after conversion')

    args = parser.parse_args()

    # Check if files exist
    for file_path in [args.input, args.reference, args.labels]:
        if not os.path.exists(file_path):
            print(f"✗ Error: File not found: {file_path}")
            sys.exit(1)

    # Load reference features
    reference_features = load_reference_features(args.reference)

    # Load labels
    sample_to_label = load_labels(args.labels, args.label_column)

    # Convert
    compression = None if args.compression == 'none' else args.compression

    try:
        csv_to_hdf5_reordered(
            csv_file=args.input,
            h5_file=args.output,
            reference_features=reference_features,
            sample_to_label=sample_to_label,
            compression=compression,
            chunk_size=args.chunk_size
        )

        # Verify if requested
        if args.verify:
            verify_hdf5(args.output)

        print(f"\n✓ Ready for comparison with all_data!")
        print(f"  - Both datasets now have same feature order")
        print(f"  - Load with: h5py.File('{args.output}', 'r')")

    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
