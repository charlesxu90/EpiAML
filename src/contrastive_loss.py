"""
Supervised Contrastive Loss for EpiAML

Implements supervised contrastive learning to learn better representations
by pulling together samples from the same class and pushing apart samples
from different classes in the embedding space.

Reference:
    Khosla et al. "Supervised Contrastive Learning" NeurIPS 2020
    https://arxiv.org/abs/2004.11362
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Pulls together embeddings from the same class and pushes apart
    embeddings from different classes.

    Args:
        temperature (float): Temperature scaling parameter
        contrast_mode (str): 'all' or 'one' - whether to contrast all positives or just one
        base_temperature (float): Base temperature for normalization
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Compute supervised contrastive loss.

        Args:
            features (torch.Tensor): Normalized feature vectors of shape (batch_size, projection_dim)
            labels (torch.Tensor): Ground truth labels of shape (batch_size,)

        Returns:
            torch.Tensor: Scalar loss value
        """
        device = features.device
        batch_size = features.shape[0]

        # Ensure features are normalized
        features = F.normalize(features, dim=1)

        # Reshape labels for broadcasting
        labels = labels.contiguous().view(-1, 1)

        # Create mask for positive pairs (same class)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute similarity matrix: (batch_size, batch_size)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Create mask to exclude self-contrast (diagonal)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)

        # Mask out self-contrast and get positive samples
        mask = mask * logits_mask

        # Compute log probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood over positive samples
        # Only consider pairs where mask > 0
        mask_sum = mask.sum(1)

        # Avoid division by zero for classes with only one sample
        mask_sum = torch.clamp(mask_sum, min=1.0)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # Loss is negative log-likelihood
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for EpiAML training.

    Combines supervised contrastive loss with cross-entropy classification loss.

    Args:
        contrastive_weight (float): Weight for contrastive loss
        classification_weight (float): Weight for classification loss
        temperature (float): Temperature for contrastive loss
        label_smoothing (float): Label smoothing factor (0.0 = no smoothing)
    """
    def __init__(self, contrastive_weight=0.5, classification_weight=1.0, temperature=0.07, label_smoothing=0.0):
        super(CombinedLoss, self).__init__()
        self.contrastive_weight = contrastive_weight
        self.classification_weight = classification_weight

        self.contrastive_loss = SupervisedContrastiveLoss(temperature=temperature)
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, projections, labels):
        """
        Compute combined loss.

        Args:
            logits (torch.Tensor): Classification logits of shape (batch_size, num_classes)
            projections (torch.Tensor): Normalized projections of shape (batch_size, projection_dim)
            labels (torch.Tensor): Ground truth labels of shape (batch_size,)

        Returns:
            tuple: (total_loss, contrastive_loss, classification_loss)
        """
        # Contrastive loss
        contr_loss = self.contrastive_loss(projections, labels)

        # Classification loss
        class_loss = self.classification_loss(logits, labels)

        # Combined loss
        total_loss = (
            self.contrastive_weight * contr_loss +
            self.classification_weight * class_loss
        )

        return total_loss, contr_loss, class_loss


class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning (alternative to contrastive loss).

    Args:
        margin (float): Margin for triplet loss
        mining (str): 'hard' or 'semi-hard' or 'all'
    """
    def __init__(self, margin=1.0, mining='hard'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mining = mining

    def forward(self, embeddings, labels):
        """
        Compute triplet loss.

        Args:
            embeddings (torch.Tensor): Feature embeddings of shape (batch_size, embedding_dim)
            labels (torch.Tensor): Ground truth labels of shape (batch_size,)

        Returns:
            torch.Tensor: Scalar loss value
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)

        # Create masks for positives and negatives
        labels = labels.view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(device)
        mask_negative = 1.0 - mask_positive

        # Remove diagonal (self-pairs)
        mask_positive = mask_positive - torch.eye(batch_size).to(device)

        if self.mining == 'hard':
            # Hard negative mining: select hardest negative for each anchor
            hardest_negative_dist, _ = (distances + 1e5 * mask_positive).min(dim=1)

            # Hard positive mining: select hardest positive for each anchor
            hardest_positive_dist, _ = (distances * mask_positive + 1e5 * (1 - mask_positive)).max(dim=1)

            # Triplet loss
            loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
            loss = loss.mean()

        elif self.mining == 'semi-hard':
            # Semi-hard negative mining
            # Select negatives that are within margin but farther than positive
            mask_semi_hard = mask_negative.clone()
            for i in range(batch_size):
                pos_dists = distances[i] * mask_positive[i]
                max_pos = pos_dists.max()

                neg_mask = (distances[i] > max_pos) & (distances[i] < max_pos + self.margin)
                mask_semi_hard[i] = mask_semi_hard[i] * neg_mask.float()

            # Compute loss for valid triplets
            loss_mat = F.relu(distances.unsqueeze(1) - distances.unsqueeze(2) + self.margin)
            loss_mat = loss_mat * mask_positive.unsqueeze(2) * mask_semi_hard.unsqueeze(1)

            num_triplets = (loss_mat > 0).float().sum()
            if num_triplets > 0:
                loss = loss_mat.sum() / (num_triplets + 1e-8)
            else:
                loss = torch.tensor(0.0, device=device)

        else:  # 'all'
            # Use all valid triplets
            loss_mat = F.relu(distances.unsqueeze(1) - distances.unsqueeze(2) + self.margin)
            loss_mat = loss_mat * mask_positive.unsqueeze(2) * mask_negative.unsqueeze(1)

            num_triplets = (loss_mat > 0).float().sum()
            if num_triplets > 0:
                loss = loss_mat.sum() / (num_triplets + 1e-8)
            else:
                loss = torch.tensor(0.0, device=device)

        return loss


if __name__ == '__main__':
    print("=" * 70)
    print("Contrastive Loss Test")
    print("=" * 70)

    # Test supervised contrastive loss
    batch_size = 16
    projection_dim = 128
    num_classes = 5

    # Create dummy data
    features = torch.randn(batch_size, projection_dim)
    features = F.normalize(features, dim=1)
    labels = torch.randint(0, num_classes, (batch_size,))

    print(f"\nTest Data:")
    print(f"  Features shape: {features.shape}")
    print(f"  Labels: {labels}")
    print(f"  Unique classes: {labels.unique()}")

    # Test supervised contrastive loss
    print(f"\nTesting Supervised Contrastive Loss:")
    scl = SupervisedContrastiveLoss(temperature=0.07)
    loss_scl = scl(features, labels)
    print(f"  Loss: {loss_scl.item():.4f}")

    # Test triplet loss
    print(f"\nTesting Triplet Loss:")
    for mining in ['hard', 'all']:
        tl = TripletLoss(margin=1.0, mining=mining)
        loss_tl = tl(features, labels)
        print(f"  Loss ({mining} mining): {loss_tl.item():.4f}")

    # Test combined loss
    print(f"\nTesting Combined Loss:")
    logits = torch.randn(batch_size, num_classes)
    combined = CombinedLoss(contrastive_weight=0.5, classification_weight=1.0)
    total_loss, contr_loss, class_loss = combined(logits, features, labels)
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Contrastive loss: {contr_loss.item():.4f}")
    print(f"  Classification loss: {class_loss.item():.4f}")

    print("\n" + "=" * 70)
    print("Contrastive loss test completed successfully!")
    print("=" * 70)
