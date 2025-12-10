"""
EpiAML Training Script

Train EpiAML model with combined supervised contrastive loss and classification loss.
"""

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from model import EpiAMLModel
from contrastive_loss import CombinedLoss
from data_utils import (
    load_training_data,
    create_data_loaders,
    upsample_data,
    split_train_val,
    save_class_mapping
)


def mixup_data(x, y, alpha=0.2):
    """
    Apply mixup augmentation to a batch.

    Args:
        x: Input data
        y: Labels
        alpha: Mixup interpolation strength

    Returns:
        mixed_x, y_a, y_b, lam: Mixed inputs and label pairs with mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, mixup_alpha=0.0):
    """Train for one epoch with optional mixup augmentation."""
    model.train()

    running_loss = 0.0
    running_contr_loss = 0.0
    running_class_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Apply mixup augmentation if enabled
        if mixup_alpha > 0:
            data, labels_a, labels_b, lam = mixup_data(data, labels, mixup_alpha)

            # Forward pass
            logits, projections = model(data, return_projection=True)

            # Compute mixed loss
            loss_a, contr_loss_a, class_loss_a = criterion(logits, projections, labels_a)
            loss_b, contr_loss_b, class_loss_b = criterion(logits, projections, labels_b)
            loss = lam * loss_a + (1 - lam) * loss_b
            contr_loss = lam * contr_loss_a + (1 - lam) * contr_loss_b
            class_loss = lam * class_loss_a + (1 - lam) * class_loss_b

            # For accuracy, use original labels (labels_a)
            labels = labels_a
        else:
            # Forward pass
            logits, projections = model(data, return_projection=True)

            # Compute loss
            loss, contr_loss, class_loss = criterion(logits, projections, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * data.size(0)
        running_contr_loss += contr_loss.item() * data.size(0)
        running_class_loss += class_loss.item() * data.size(0)

        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    epoch_loss = running_loss / total
    epoch_contr_loss = running_contr_loss / total
    epoch_class_loss = running_class_loss / total
    epoch_acc = 100.0 * correct / total

    epoch_precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    epoch_recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    epoch_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    return {
        'loss': epoch_loss,
        'contr_loss': epoch_contr_loss,
        'class_loss': epoch_class_loss,
        'accuracy': epoch_acc,
        'precision': epoch_precision,
        'recall': epoch_recall,
        'f1': epoch_f1
    }


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()

    running_loss = 0.0
    running_contr_loss = 0.0
    running_class_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)

            # Forward pass
            logits, projections = model(data, return_projection=True)

            # Compute loss
            loss, contr_loss, class_loss = criterion(logits, projections, labels)

            # Statistics
            running_loss += loss.item() * data.size(0)
            running_contr_loss += contr_loss.item() * data.size(0)
            running_class_loss += class_loss.item() * data.size(0)

            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    val_loss = running_loss / total
    val_contr_loss = running_contr_loss / total
    val_class_loss = running_class_loss / total
    val_acc = 100.0 * correct / total

    val_precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    val_recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    val_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

    return {
        'loss': val_loss,
        'contr_loss': val_contr_loss,
        'class_loss': val_class_loss,
        'accuracy': val_acc,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1
    }


def train_epiaml(
    train_file,
    output_dir,
    feature_order=None,
    feature_indices=None,
    epochs=500,
    batch_size=32,
    learning_rate=1e-4,
    weight_decay=1e-4,  # Increased from 1e-5 for more regularization
    val_split=0.2,
    samples_per_class=50,
    augment=True,
    flip_percent=0.1,
    contrastive_weight=0.5,
    temperature=0.07,
    base_channels=64,
    num_blocks=4,
    use_attention=False,  # Default OFF - often hurts methylation data
    input_dropout=0.0,  # Set to 0.99 for MLP-style sparse feature learning
    stride_config='minimal',  # 'minimal' (preserves info) or 'aggressive' (faster)
    device='cuda',
    random_seed=42,
    save_every=50,
    early_stopping_patience=30,  # Stop if no improvement for 30 epochs
    dropout=0.3,  # Increased dropout for regularization
    label_smoothing=0.1,  # Label smoothing to prevent overconfidence
    mixup_alpha=0.2  # Mixup augmentation strength
):
    """
    Train EpiAML model.

    Args:
        train_file (str): Path to training data (.h5 or .csv)
        output_dir (str): Output directory
        feature_order (str): Path to feature order file (optional)
        feature_indices (str): Path to selected feature indices (.npy, optional for feature selection)
        epochs (int): Number of epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay for regularization
        val_split (float): Validation split ratio
        samples_per_class (int): Samples per class for upsampling
        augment (bool): Whether to use data augmentation
        flip_percent (float): Percentage of CpGs to flip for augmentation
        contrastive_weight (float): Weight for contrastive loss
        temperature (float): Temperature for contrastive loss
        base_channels (int): Base channels for CNN
        num_blocks (int): Number of CNN blocks
        use_attention (bool): Whether to use attention
        input_dropout (float): Input-level dropout for sparse feature learning (0.0-0.99)
        stride_config (str): Pooling strategy - 'minimal' or 'aggressive'
        device (str): Device ('cuda' or 'cpu')
        random_seed (int): Random seed
        save_every (int): Save checkpoint every N epochs
        dropout (float): Dropout rate for CNN layers
        label_smoothing (float): Label smoothing factor
        mixup_alpha (float): Mixup augmentation alpha
        early_stopping_patience (int): Early stopping patience
    """
    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Device configuration
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data from {train_file}...")
    data, labels, feature_names = load_training_data(
        train_file,
        format='auto',
        binarize=True,
        feature_order=feature_order
    )

    print(f"Loaded: {data.shape[0]} samples × {data.shape[1]} features")
    print(f"Classes: {len(np.unique(labels))}")
    
    # Apply feature selection if indices provided
    if feature_indices is not None and os.path.exists(feature_indices):
        print(f"\nApplying feature selection from {feature_indices}...")
        selected_indices = np.load(feature_indices)
        data = data[:, selected_indices]
        if feature_names is not None:
            feature_names = feature_names[selected_indices]
        print(f"After feature selection: {data.shape[0]} samples × {data.shape[1]} features")
        print(f"  Selected {len(selected_indices)} most informative CpGs")

    # Split train/val
    if val_split > 0:
        train_data, train_labels, val_data, val_labels = split_train_val(
            data, labels, val_split=val_split, random_seed=random_seed
        )
    else:
        train_data, train_labels = data, labels
        val_data, val_labels = None, None

    # Upsample training data (with disk caching for memory efficiency)
    if samples_per_class > 0:
        cache_dir = os.path.join(output_dir, 'cache')
        train_data, train_labels = upsample_data(
            train_data, train_labels,
            samples_per_class=samples_per_class,
            random_seed=random_seed,
            cache_dir=cache_dir  # Enable disk caching
        )

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data, train_labels,
        val_data, val_labels,
        batch_size=batch_size,
        num_workers=4,
        augment_train=augment,
        flip_percent=flip_percent
    )

    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_data)} samples")
    if val_data is not None:
        print(f"  Validation: {len(val_data)} samples")
    print(f"  Batches per epoch: {len(train_loader)}")

    # Initialize model
    input_size = data.shape[1]
    num_classes = len(np.unique(labels))

    model = EpiAMLModel(
        input_size=input_size,
        num_classes=num_classes,
        base_channels=base_channels,
        num_blocks=num_blocks,
        use_attention=use_attention,
        dropout=dropout,  # Use configurable dropout
        input_dropout=input_dropout,  # Sparse feature learning
        stride_config=stride_config  # Pooling strategy
    ).to(device)

    epiaml_params = model.get_num_parameters()
    print(f"\nEpiAML Model Configuration:")
    print(f"  Architecture: 1D-CNN + Attention + Contrastive Learning")
    print(f"  Input size: {input_size:,}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Base channels: {base_channels}")
    print(f"  CNN blocks: {num_blocks}")
    print(f"  Use attention: {use_attention}")
    print(f"  Input dropout: {input_dropout} {'(sparse feature learning)' if input_dropout > 0.5 else ''}")
    print(f"  Stride config: {stride_config}")
    print(f"  CNN dropout: {dropout}")
    print(f"  Total parameters: {epiaml_params:,}")
    print(f"  Model size: ~{epiaml_params * 4 / (1024**2):.2f} MB")

    print(f"\n{'='*70}\n")

    # Loss and optimizer (with label smoothing for regularization)
    criterion = CombinedLoss(
        contrastive_weight=contrastive_weight,
        classification_weight=1.0,
        temperature=temperature,
        label_smoothing=label_smoothing
    )
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # TensorBoard writer
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))

    # Training
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    print(f"Contrastive weight: {contrastive_weight}, Temperature: {temperature}")
    print(f"Model config: stride_config={stride_config}, input_dropout={input_dropout}")
    print(f"Regularization: dropout={dropout}, label_smoothing={label_smoothing}, mixup_alpha={mixup_alpha}")
    print(f"Early stopping patience: {early_stopping_patience}")

    start_time = time.time()
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        # Train (with mixup augmentation)
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, mixup_alpha=mixup_alpha)

        # Validate
        if val_loader is not None:
            val_metrics = validate(model, val_loader, criterion, device)
        else:
            val_metrics = None

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/train_contrastive', train_metrics['contr_loss'], epoch)
        writer.add_scalar('Loss/train_classification', train_metrics['class_loss'], epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Precision/train', train_metrics['precision'], epoch)
        writer.add_scalar('Recall/train', train_metrics['recall'], epoch)
        writer.add_scalar('F1/train', train_metrics['f1'], epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)

        if val_metrics:
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            writer.add_scalar('Precision/val', val_metrics['precision'], epoch)
            writer.add_scalar('Recall/val', val_metrics['recall'], epoch)
            writer.add_scalar('F1/val', val_metrics['f1'], epoch)

        # Print progress for every epoch (single line format)
        elapsed = time.time() - start_time
        log_line = (f'Epoch {epoch}/{epochs} ({elapsed:.1f}s) '
                   f'Train: {train_metrics["loss"]:.4f} '
                   f'({train_metrics["contr_loss"]:.4f} + {train_metrics["class_loss"]:.4f}), '
                   f'Acc {train_metrics["accuracy"]:.2f}%')

        if val_metrics:
            log_line += (f'; Val: {val_metrics["loss"]:.4f}, '
                        f'Acc {val_metrics["accuracy"]:.2f}%, '
                        f'Prec {val_metrics["precision"]:.4f}, '
                        f'Recall {val_metrics["recall"]:.4f}')

        # Will append best model indicator later if needed
        best_model_indicator = ''

        # Save history
        history_entry = {
            'epoch': epoch,
            'train': train_metrics,
            'lr': current_lr
        }
        if val_metrics:
            history_entry['val'] = val_metrics
        history.append(history_entry)

        # Save best model and check for early stopping
        if val_metrics:
            # Track best model by validation loss (more stable than accuracy)
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Save best model by accuracy
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_model_path = os.path.join(output_dir, 'best_model.pt')
                model.save_model(best_model_path)
                best_model_indicator = ' [BEST]'

        # Print the complete log line
        print(log_line + best_model_indicator)

        # Early stopping check
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(f'\n*** Early stopping triggered! No improvement in validation loss for {early_stopping_patience} epochs ***')
            break

        # Save checkpoint
        if epoch % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
            model.save_model(checkpoint_path)

    # Save final model
    total_time = time.time() - start_time
    print(f'\n{"="*70}')
    print(f'Training completed in {total_time:.1f}s ({total_time/60:.1f} minutes)')
    print(f'{"="*70}')

    final_model_path = os.path.join(output_dir, 'final_model.pt')
    model.save_model(final_model_path)
    print(f'Final model saved to {final_model_path}')

    # Print final metrics summary
    print(f'\nFinal Training Metrics:')
    print(f'  Accuracy:  {train_metrics["accuracy"]:.2f}%')
    print(f'  Precision: {train_metrics["precision"]:.4f}')
    print(f'  Recall:    {train_metrics["recall"]:.4f}')
    print(f'  F1 Score:  {train_metrics["f1"]:.4f}')

    if val_metrics:
        print(f'\nBest Validation Metrics:')
        print(f'  Accuracy:  {best_val_acc:.2f}%')
        print(f'\nFinal Validation Metrics:')
        print(f'  Accuracy:  {val_metrics["accuracy"]:.2f}%')
        print(f'  Precision: {val_metrics["precision"]:.4f}')
        print(f'  Recall:    {val_metrics["recall"]:.4f}')
        print(f'  F1 Score:  {val_metrics["f1"]:.4f}')

    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save configuration
    config = {
        'train_file': train_file,
        'feature_order': feature_order,
        'input_size': input_size,
        'num_classes': num_classes,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'val_split': val_split,
        'samples_per_class': samples_per_class,
        'contrastive_weight': contrastive_weight,
        'temperature': temperature,
        'base_channels': base_channels,
        'num_blocks': num_blocks,
        'use_attention': use_attention,
        'input_dropout': input_dropout,
        'stride_config': stride_config,
        'dropout': dropout,
        'label_smoothing': label_smoothing,
        'mixup_alpha': mixup_alpha,
        'training_time': round(total_time, 2),
        'best_val_accuracy': round(best_val_acc, 4) if val_metrics else None,
        'final_train_accuracy': round(train_metrics['accuracy'], 4),
        'random_seed': random_seed
    }

    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Save class mapping
    save_class_mapping(labels, os.path.join(output_dir, 'class_mapping.csv'))

    writer.close()

    print(f'\nOutput files saved to: {output_dir}')
    print(f'  - best_model.pt (best validation accuracy)')
    print(f'  - final_model.pt (last epoch)')
    print(f'  - config.json (training configuration)')
    print(f'  - training_history.json (metrics per epoch)')
    print(f'  - class_mapping.csv (class labels)')
    print(f'  - logs/ (TensorBoard logs)')

    return model, history


def main():
    parser = argparse.ArgumentParser(
        description='Train EpiAML model with contrastive learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
EpiAML combines 1D-CNN backbone, multi-head attention, and supervised
contrastive learning for accurate leukemia subtype classification.

Examples:
  # Recommended: MLP-style with sparse features (99% input dropout)
  python train.py --train_file ../pytorch_marlin/data/training_data.h5 \\
                  --output_dir ./output_mlp_style \\
                  --input_dropout 0.99 --stride_config minimal --no_attention \\
                  --epochs 500 --batch_size 32
  
  # With feature selection (fewer CpG sites)
  python train.py --train_file ../pytorch_marlin/data/training_data.h5 \\
                  --feature_indices ./feature_selection/top_100_cpg_indices.npy \\
                  --output_dir ./output_top100cpg \\
                  --input_dropout 0.0 --stride_config minimal --no_attention \\
                  --epochs 300 --batch_size 32
  
  # Original CNN configuration
  python train.py --train_file ../pytorch_marlin/data/training_data.h5 \\
                  --output_dir ./output_cnn \\
                  --stride_config aggressive \\
                  --epochs 500 --batch_size 32

Monitor training:
  tensorboard --logdir ./output/logs
        '''
    )

    parser.add_argument('--train_file', required=True,
                        help='Path to training data (.h5 or .csv)')
    parser.add_argument('--output_dir', default='./output',
                        help='Output directory (default: ./output)')
    parser.add_argument('--feature_order', default=None,
                        help='Path to feature order file (.npy, .json, .txt)')
    parser.add_argument('--feature_indices', default=None,
                        help='Path to selected feature indices (.npy, from feature selection)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs (default: 500)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split (default: 0.2)')

    # Data augmentation
    parser.add_argument('--samples_per_class', type=int, default=50,
                        help='Samples per class for upsampling (default: 50)')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--flip_percent', type=float, default=0.1,
                        help='CpG flip percentage for augmentation (default: 0.1)')

    # Loss parameters
    parser.add_argument('--contrastive_weight', type=float, default=0.5,
                        help='Weight for contrastive loss (default: 0.5)')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for contrastive loss (default: 0.07)')

    # Model architecture
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base channels for CNN (default: 64)')
    parser.add_argument('--num_blocks', type=int, default=4,
                        help='Number of CNN blocks (default: 4)')
    parser.add_argument('--no_attention', action='store_true',
                        help='Disable attention mechanism')
    parser.add_argument('--input_dropout', type=float, default=0.0,
                        help='Input-level dropout for sparse feature learning (0.0-0.99, default: 0.0)')
    parser.add_argument('--stride_config', type=str, default='minimal',
                        choices=['minimal', 'aggressive'],
                        help='Pooling strategy: minimal (preserves info) or aggressive (faster, default: minimal)')

    # Regularization (to prevent overfitting)
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor (default: 0.1)')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='Mixup augmentation alpha (default: 0.2, 0 to disable)')
    parser.add_argument('--early_stopping', type=int, default=30,
                        help='Early stopping patience in epochs (default: 30, 0 to disable)')

    # Other
    parser.add_argument('--device', default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save checkpoint every N epochs (default: 50)')

    args = parser.parse_args()

    train_epiaml(
        train_file=args.train_file,
        output_dir=args.output_dir,
        feature_order=args.feature_order,
        feature_indices=args.feature_indices,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        samples_per_class=args.samples_per_class,
        augment=not args.no_augment,
        flip_percent=args.flip_percent,
        contrastive_weight=args.contrastive_weight,
        temperature=args.temperature,
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
        use_attention=not args.no_attention,
        input_dropout=args.input_dropout,
        stride_config=args.stride_config,
        device=args.device,
        random_seed=args.seed,
        save_every=args.save_every,
        early_stopping_patience=args.early_stopping,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha
    )


if __name__ == '__main__':
    main()
