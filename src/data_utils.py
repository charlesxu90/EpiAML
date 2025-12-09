"""
Data utilities for EpiAML

Handles loading, preprocessing, and batching of methylation data.
Supports both ordered (from clustering) and unordered CpG features.
"""

import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


def load_training_data(data_path, format='auto', binarize=True, feature_order=None):
    """
    Load training data from file.

    Args:
        data_path (str): Path to data file (.h5 or .csv)
        format (str): File format ('h5', 'csv', or 'auto')
        binarize (bool): Whether to binarize methylation values (>=0.5 -> 1, <0.5 -> -1)
        feature_order (str or np.ndarray): Path to feature order file or array of indices

    Returns:
        tuple: (data, labels, feature_names)
               data shape: (n_samples, n_features)
    """
    ext = os.path.splitext(data_path)[1].lower()

    if format == 'auto':
        format = 'h5' if ext in ['.h5', '.hdf5'] else 'csv'

    if format == 'h5':
        print(f"Loading data from HDF5: {data_path}")
        with h5py.File(data_path, 'r') as f:
            data = f['data'][:]
            labels = f['labels'][:] if 'labels' in f else None
            feature_names = f['feature_names'][:].astype(str) if 'feature_names' in f else None

            # Handle string labels
            if labels is not None and labels.dtype.kind in ['O', 'S', 'U']:
                # Convert string labels to integers
                unique_labels = np.unique(labels)
                label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                labels = np.array([label_to_idx[label] for label in labels])

    elif format == 'csv':
        print(f"Loading data from CSV: {data_path}")
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
        raise ValueError(f"Unsupported format: {format}")

    print(f"  Loaded: {data.shape[0]} samples × {data.shape[1]} features")

    if labels is not None:
        print(f"  Classes: {len(np.unique(labels))} ({np.unique(labels)})")

    # Apply feature ordering if provided
    if feature_order is not None:
        if isinstance(feature_order, str):
            # Load from file
            ext = os.path.splitext(feature_order)[1].lower()
            if ext == '.npy':
                order_indices = np.load(feature_order)
            elif ext == '.json':
                import json
                with open(feature_order, 'r') as f:
                    ordered_features = json.load(f)
                # Find indices
                order_indices = np.array([np.where(feature_names == f)[0][0]
                                         for f in ordered_features if f in feature_names])
            elif ext == '.txt':
                with open(feature_order, 'r') as f:
                    ordered_features = [line.strip() for line in f]
                order_indices = np.array([np.where(feature_names == f)[0][0]
                                         for f in ordered_features if f in feature_names])
            else:
                raise ValueError(f"Unsupported feature order format: {ext}")
        else:
            order_indices = feature_order

        print(f"  Applying feature ordering: {len(order_indices)} features")
        data = data[:, order_indices]
        if feature_names is not None:
            feature_names = feature_names[order_indices]

    # Binarize if requested
    if binarize:
        print("  Binarizing: β >= 0.5 → +1, β < 0.5 → -1")
        data = np.where(data >= 0.5, 1.0, -1.0)

    return data, labels, feature_names


class MethylationDataset(Dataset):
    """
    PyTorch Dataset for methylation data with memory-efficient lazy loading.

    Args:
        data (np.ndarray): Methylation data of shape (n_samples, n_features)
        labels (np.ndarray): Labels of shape (n_samples,)
        transform (callable): Optional transform to apply to samples
        lazy_load (bool): If True, don't convert all data to tensors at once (saves memory)
    """
    def __init__(self, data, labels, transform=None, lazy_load=True):
        if lazy_load:
            # Keep data as numpy array for memory efficiency
            # Only convert to tensor when accessing individual samples
            self.data = data
            self.data_is_numpy = True
        else:
            # Original behavior: convert all to tensor upfront
            self.data = torch.FloatTensor(data)
            self.data_is_numpy = False

        self.labels = torch.LongTensor(labels) if labels is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert to tensor only when accessed
        if self.data_is_numpy:
            # Copy the data to avoid non-writable tensor warnings
            # This is needed when data is memory-mapped (read-only)
            sample = torch.from_numpy(self.data[idx].copy()).float()
        else:
            sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        if self.labels is not None:
            return sample, self.labels[idx]
        else:
            return sample


class MethylationDataAugmentation:
    """
    Data augmentation for methylation data.

    Args:
        flip_percent (float): Percentage of CpG sites to randomly flip
        noise_std (float): Standard deviation of Gaussian noise
    """
    def __init__(self, flip_percent=0.1, noise_std=0.0):
        self.flip_percent = flip_percent
        self.noise_std = noise_std

    def __call__(self, sample):
        """
        Apply augmentation to a sample.

        Args:
            sample (torch.Tensor): Input sample of shape (n_features,)

        Returns:
            torch.Tensor: Augmented sample
        """
        if self.flip_percent > 0:
            # Random CpG flipping
            n_flip = int(self.flip_percent * len(sample))
            flip_indices = torch.randperm(len(sample))[:n_flip]
            sample[flip_indices] = -sample[flip_indices]

        if self.noise_std > 0:
            # Gaussian noise
            noise = torch.randn_like(sample) * self.noise_std
            sample = sample + noise

        return sample


def upsample_data(data, labels, samples_per_class=50, random_seed=42, cache_dir=None):
    """
    Upsample data to have equal samples per class.
    Optionally cache to disk for memory efficiency.

    Args:
        data (np.ndarray): Data of shape (n_samples, n_features)
        labels (np.ndarray): Labels of shape (n_samples,)
        samples_per_class (int): Number of samples per class
        random_seed (int): Random seed
        cache_dir (str): If provided, cache upsampled data to disk and use memory mapping

    Returns:
        tuple: (upsampled_data, upsampled_labels)
    """
    np.random.seed(random_seed)

    unique_classes = np.unique(labels)

    # Calculate output shape
    n_samples = len(unique_classes) * samples_per_class
    n_features = data.shape[1]

    print(f"Upsampling to {samples_per_class} samples per class...")

    # If cache_dir provided, use disk-based approach
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_data_path = os.path.join(cache_dir, 'upsampled_data.npy')
        cache_labels_path = os.path.join(cache_dir, 'upsampled_labels.npy')

        # Create memory-mapped array
        upsampled_data = np.lib.format.open_memmap(
            cache_data_path, mode='w+',
            dtype=data.dtype, shape=(n_samples, n_features)
        )
        upsampled_labels = np.zeros(n_samples, dtype=labels.dtype)

        idx = 0
        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]
            sampled_indices = np.random.choice(
                cls_indices, size=samples_per_class, replace=True
            )
            upsampled_data[idx:idx+samples_per_class] = data[sampled_indices]
            upsampled_labels[idx:idx+samples_per_class] = cls
            idx += samples_per_class

        # Flush to disk
        del upsampled_data

        # Reload as read-only memory map
        upsampled_data = np.load(cache_data_path, mmap_mode='r')
        np.save(cache_labels_path, upsampled_labels)

        print(f"  Upsampled data cached to disk: {cache_dir}")
        print(f"  {len(unique_classes)} classes × {samples_per_class} = {len(upsampled_labels)} samples")

    else:
        # Original in-memory approach
        upsampled_data = []
        upsampled_labels = []

        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]
            sampled_indices = np.random.choice(
                cls_indices, size=samples_per_class, replace=True
            )
            cls_data = data[sampled_indices]
            upsampled_data.append(cls_data)
            upsampled_labels.extend([cls] * samples_per_class)

        upsampled_data = np.vstack(upsampled_data)
        upsampled_labels = np.array(upsampled_labels)

        print(f"  Upsampled: {len(unique_classes)} classes × {samples_per_class} = {len(upsampled_labels)} samples")

    return upsampled_data, upsampled_labels


def create_data_loaders(
    train_data,
    train_labels,
    val_data=None,
    val_labels=None,
    batch_size=32,
    num_workers=4,
    augment_train=True,
    flip_percent=0.1
):
    """
    Create data loaders for training and validation.

    Args:
        train_data (np.ndarray): Training data
        train_labels (np.ndarray): Training labels
        val_data (np.ndarray): Validation data (optional)
        val_labels (np.ndarray): Validation labels (optional)
        batch_size (int): Batch size
        num_workers (int): Number of data loading workers
        augment_train (bool): Whether to augment training data
        flip_percent (float): Percentage for data augmentation

    Returns:
        tuple: (train_loader, val_loader or None)
    """
    # Training dataset
    train_transform = MethylationDataAugmentation(flip_percent=flip_percent) if augment_train else None
    train_dataset = MethylationDataset(train_data, train_labels, transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Validation dataset
    val_loader = None
    if val_data is not None and val_labels is not None:
        val_dataset = MethylationDataset(val_data, val_labels, transform=None)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader


def split_train_val(data, labels, val_split=0.2, random_seed=42):
    """
    Split data into training and validation sets.

    Args:
        data (np.ndarray): Data of shape (n_samples, n_features)
        labels (np.ndarray): Labels of shape (n_samples,)
        val_split (float): Validation split ratio
        random_seed (int): Random seed

    Returns:
        tuple: (train_data, train_labels, val_data, val_labels)
    """
    np.random.seed(random_seed)

    n_samples = len(data)
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * (1 - val_split))

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_data = data[train_indices]
    train_labels = labels[train_indices]
    val_data = data[val_indices]
    val_labels = labels[val_indices]

    print(f"Train/Val split: {len(train_data)}/{len(val_data)} samples")

    return train_data, train_labels, val_data, val_labels


def save_class_mapping(labels, output_path):
    """
    Save class mapping to CSV.

    Args:
        labels (np.ndarray): Integer labels
        output_path (str): Output file path
    """
    import pandas as pd

    unique_labels = np.unique(labels)
    mapping = pd.DataFrame({
        'class_index': unique_labels,
        'class_name': [f'Class_{i}' for i in unique_labels]
    })

    mapping.to_csv(output_path, index=False)
    print(f"Saved class mapping to {output_path}")


if __name__ == '__main__':
    print("=" * 70)
    print("Data Utilities Test")
    print("=" * 70)

    # Create dummy data
    n_samples = 100
    n_features = 1000
    n_classes = 5

    dummy_data = np.random.rand(n_samples, n_features)
    dummy_labels = np.random.randint(0, n_classes, n_samples)

    print(f"\nDummy Data:")
    print(f"  Shape: {dummy_data.shape}")
    print(f"  Labels: {len(np.unique(dummy_labels))} classes")

    # Test dataset
    print(f"\nTesting MethylationDataset:")
    dataset = MethylationDataset(dummy_data, dummy_labels)
    print(f"  Dataset length: {len(dataset)}")
    sample, label = dataset[0]
    print(f"  Sample shape: {sample.shape}, Label: {label}")

    # Test data augmentation
    print(f"\nTesting Data Augmentation:")
    augment = MethylationDataAugmentation(flip_percent=0.1)
    sample_aug = augment(sample.clone())
    print(f"  Original sample range: [{sample.min():.2f}, {sample.max():.2f}]")
    print(f"  Augmented sample range: [{sample_aug.min():.2f}, {sample_aug.max():.2f}]")

    # Test upsampling
    print(f"\nTesting Upsampling:")
    up_data, up_labels = upsample_data(dummy_data, dummy_labels, samples_per_class=30)
    print(f"  Upsampled shape: {up_data.shape}")
    print(f"  Class distribution: {np.bincount(up_labels)}")

    # Test train/val split
    print(f"\nTesting Train/Val Split:")
    train_data, train_labels, val_data, val_labels = split_train_val(
        dummy_data, dummy_labels, val_split=0.2
    )

    # Test data loaders
    print(f"\nTesting Data Loaders:")
    train_loader, val_loader = create_data_loaders(
        train_data, train_labels,
        val_data, val_labels,
        batch_size=16,
        num_workers=0
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    batch_data, batch_labels = next(iter(train_loader))
    print(f"  Batch data shape: {batch_data.shape}")
    print(f"  Batch labels shape: {batch_labels.shape}")

    print("\n" + "=" * 70)
    print("Data utilities test completed successfully!")
    print("=" * 70)
