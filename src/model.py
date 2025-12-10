"""
EpiAML: Epigenomic AML Classifier with 1D-CNN and Contrastive Learning

State-of-the-art model architecture combining:
1. 1D-CNN backbone with residual connections for feature extraction
2. Multi-head self-attention for capturing long-range dependencies (optional, disabled by default)
3. Supervised contrastive learning for better class separation
4. Feature-ordered input (from CpG clustering) for spatial locality

Model Architecture:
- Input: Ordered CpG methylation features (357,340 or clustered features)
- Optional input dropout (0-99%) for sparse feature selection (mimics MLP success)
- 1D-CNN blocks with residual connections and batch normalization
- Configurable stride: 'minimal' (preserves info) or 'aggressive' (faster but more pooling)
- Optional multi-head attention layer for global context
- Projection head for contrastive learning
- Classification head for final predictions

Key Improvements over Original:
1. ✅ Input dropout (set to 0.99 to match MLP's feature selection philosophy)
2. ✅ Configurable stride: 'minimal' reduces information bottleneck
3. ✅ Attention disabled by default (often hurts methylation data performance)
4. ✅ Preserved support for both configurations via stride_config parameter

Usage Examples:
    # Recommended: Minimal pooling with high input dropout
    model = EpiAMLModel(input_dropout=0.99, stride_config='minimal', use_attention=False)
    
    # Original aggressive pooling (faster but less accurate)
    model = EpiAMLModel(stride_config='aggressive')
    
    # With attention (use cautiously - may hurt performance)
    model = EpiAMLModel(use_attention=True, stride_config='minimal')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with batch normalization and dropout.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Convolution kernel size
        stride (int): Convolution stride
        dropout (float): Dropout rate
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.1):
        super(ResidualBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)

        return out


class MultiHeadSelfAttention1D(nn.Module):
    """
    Multi-head self-attention for 1D sequences.

    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiHeadSelfAttention1D, self).__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, length)

        Returns:
            torch.Tensor: Output tensor of shape (batch, channels, length)
        """
        batch_size, channels, length = x.shape

        # Reshape to (batch, length, channels) for attention
        x = x.transpose(1, 2)  # (batch, length, channels)

        # Linear projection to Q, K, V
        qkv = self.qkv(x)  # (batch, length, 3*channels)
        qkv = qkv.reshape(batch_size, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, length, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        out = attn @ v  # (batch, num_heads, length, head_dim)
        out = out.transpose(1, 2).reshape(batch_size, length, channels)

        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)

        # Reshape back to (batch, channels, length)
        out = out.transpose(1, 2)

        return out


class CNN1DBackbone(nn.Module):
    """
    1D-CNN backbone with residual blocks for methylation feature extraction.

    Args:
        input_size (int): Number of input features
        base_channels (int): Base number of channels
        num_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        stride_config (str): 'aggressive' (stride=2, faster but more info loss) or 
                           'minimal' (stride=1, preserves more information)
    """
    def __init__(self, input_size, base_channels=64, num_blocks=4, dropout=0.1, stride_config='minimal'):
        super(CNN1DBackbone, self).__init__()

        self.input_size = input_size
        self.stride_config = stride_config

        # Configure stride based on strategy
        if stride_config == 'aggressive':
            # Original aggressive pooling: fast but loses information
            initial_stride = 2
            use_maxpool = True
            block_strides = [2 if i > 0 else 1 for i in range(num_blocks)]
        elif stride_config == 'minimal':
            # Minimal pooling: preserves more information
            initial_stride = 1
            use_maxpool = False  # Skip maxpool to preserve features
            # Only stride on last few blocks for modest downsampling
            block_strides = [2 if i >= num_blocks - 2 else 1 for i in range(num_blocks)]
        else:
            raise ValueError(f"Unknown stride_config: {stride_config}. Use 'aggressive' or 'minimal'.")

        # Initial convolution
        self.conv1 = nn.Conv1d(1, base_channels, kernel_size=7, stride=initial_stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Optional maxpool (only for aggressive mode)
        if use_maxpool:
            self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.Identity()  # No-op

        # Residual blocks with configurable strides
        self.blocks = nn.ModuleList()
        channels = base_channels
        for i in range(num_blocks):
            stride = block_strides[i]
            out_channels = channels * 2 if stride > 1 else channels
            self.blocks.append(ResidualBlock1D(channels, out_channels,
                                               kernel_size=3, stride=stride, dropout=dropout))
            channels = out_channels

        self.output_channels = channels

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input of shape (batch, input_size)

        Returns:
            torch.Tensor: Features of shape (batch, channels, reduced_length)
        """
        # Add channel dimension: (batch, input_size) -> (batch, 1, input_size)
        x = x.unsqueeze(1)

        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        return x


class EpiAMLModel(nn.Module):
    """
    EpiAML: State-of-the-art epigenomic AML classifier.

    Combines 1D-CNN backbone, multi-head attention, and contrastive learning
    for accurate leukemia subtype classification from DNA methylation profiles.

    Args:
        input_size (int): Number of input CpG features
        num_classes (int): Number of output classes
        base_channels (int): Base channels for CNN
        num_blocks (int): Number of CNN residual blocks
        use_attention (bool): Whether to use attention mechanism
        num_heads (int): Number of attention heads
        projection_dim (int): Dimension for contrastive projection head
        dropout (float): Dropout rate for CNN layers
        input_dropout (float): Input-level dropout for feature selection (like MLP's 99% dropout)
        stride_config (str): Pooling strategy - 'aggressive' (stride=2, faster) or 'minimal' (stride=1, preserves info)
        attention_pool_size (int): Target sequence length before attention (for memory efficiency)
    """
    def __init__(
        self,
        input_size=357340,
        num_classes=42,
        base_channels=64,
        num_blocks=4,
        use_attention=False,  # Default OFF - attention often hurts performance on methylation data
        num_heads=8,
        projection_dim=128,
        dropout=0.1,
        input_dropout=0.0,  # Set to 0.99 to mimic MLP's feature selection
        stride_config='minimal',  # 'aggressive' or 'minimal' - controls information bottleneck
        attention_pool_size=128
    ):
        super(EpiAMLModel, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.projection_dim = projection_dim
        self.input_dropout_rate = input_dropout
        self.stride_config = stride_config

        # Input dropout for feature selection (matches MLP's 99% dropout philosophy)
        # This forces the network to learn from random feature subsets
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else None

        # 1D-CNN backbone with configurable stride
        self.backbone = CNN1DBackbone(
            input_size=input_size,
            base_channels=base_channels,
            num_blocks=num_blocks,
            dropout=dropout,
            stride_config=stride_config
        )

        # Aggressive pooling before attention to reduce memory
        # This pools the sequence to a fixed small length before attention
        # Memory savings: ~11,166 length -> 128 length
        # Attention matrix: batch × heads × len × len × 4 bytes
        # Before: 32 × 8 × 11166 × 11166 × 4 = 118.8 GiB (OOM!)
        # After:  32 × 8 × 128 × 128 × 4 = 50 MB (fits easily!)
        if use_attention:
            self.pre_attention_pool = nn.AdaptiveAvgPool1d(attention_pool_size)
            self.attention = MultiHeadSelfAttention1D(
                embed_dim=self.backbone.output_channels,
                num_heads=num_heads,
                dropout=dropout
            )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Feature dimension after pooling
        self.feature_dim = self.backbone.output_channels

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, projection_dim)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, return_features=False, return_projection=False):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            return_features (bool): Whether to return feature embeddings
            return_projection (bool): Whether to return projection for contrastive loss

        Returns:
            torch.Tensor or tuple: Classification logits, optionally with features/projections
        """
        # Apply input dropout for feature selection (sparse feature learning)
        if self.input_dropout is not None:
            x = self.input_dropout(x)

        # Extract features with CNN backbone
        features = self.backbone(x)  # (batch, channels, length)

        # Apply attention if enabled (with aggressive pooling first to save memory)
        if self.use_attention:
            # Pool to small sequence length before attention
            features_pooled = self.pre_attention_pool(features)  # (batch, channels, pool_size)
            # Apply attention on pooled features
            attn_out = self.attention(features_pooled)  # (batch, channels, pool_size)
            # Use attended pooled features
            features = attn_out

        # Global pooling
        pooled = self.global_pool(features).squeeze(-1)  # (batch, channels)

        # Classification
        logits = self.classifier(pooled)

        # Return based on flags
        if return_projection:
            projections = F.normalize(self.projection_head(pooled), dim=1)
            if return_features:
                return logits, pooled, projections
            return logits, projections

        if return_features:
            return logits, pooled

        return logits

    def predict_proba(self, x):
        """
        Predict class probabilities (inference mode).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Probability distribution of shape (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def get_embeddings(self, x):
        """
        Get feature embeddings for visualization or analysis.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Feature embeddings of shape (batch_size, feature_dim)
        """
        self.eval()
        with torch.no_grad():
            _, features = self.forward(x, return_features=True)
            return features

    def get_num_parameters(self):
        """
        Calculate total number of trainable parameters.

        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_model(self, filepath):
        """
        Save model weights and architecture configuration.

        Args:
            filepath (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'use_attention': self.use_attention,
            'projection_dim': self.projection_dim,
            'input_dropout_rate': self.input_dropout_rate,
            'stride_config': self.stride_config
        }, filepath)

    @classmethod
    def load_model(cls, filepath, device='cpu'):
        """
        Load model from saved checkpoint.

        Args:
            filepath (str): Path to the saved model
            device (str): Device to load the model on ('cpu' or 'cuda')

        Returns:
            EpiAMLModel: Loaded model instance
        """
        checkpoint = torch.load(filepath, map_location=device)

        model = cls(
            input_size=checkpoint['input_size'],
            num_classes=checkpoint['num_classes'],
            use_attention=checkpoint.get('use_attention', False),
            projection_dim=checkpoint.get('projection_dim', 128),
            input_dropout=checkpoint.get('input_dropout_rate', 0.0),
            stride_config=checkpoint.get('stride_config', 'minimal')
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model


if __name__ == '__main__':
    # Test model instantiation
    print("=" * 70)
    print("EpiAML Model Architecture Test")
    print("=" * 70)

    # Test with default parameters (minimal pooling, no attention)
    print("\n[1] Testing MINIMAL POOLING configuration (recommended):")
    model_minimal = EpiAMLModel(input_size=357340, num_classes=42, stride_config='minimal')

    print(f"\nModel Configuration:")
    print(f"  Input size: {model_minimal.input_size:,}")
    print(f"  Number of classes: {model_minimal.num_classes}")
    print(f"  Stride config: {model_minimal.stride_config}")
    print(f"  Use attention: {model_minimal.use_attention}")
    print(f"  Input dropout: {model_minimal.input_dropout_rate}")
    print(f"  Feature dimension: {model_minimal.feature_dim}")
    print(f"  Projection dimension: {model_minimal.projection_dim}")
    print(f"  Total parameters: {model_minimal.get_num_parameters():,}")

    # Test with MLP-like high dropout
    print("\n[2] Testing with MLP-STYLE 99% INPUT DROPOUT:")
    model_sparse = EpiAMLModel(input_size=357340, num_classes=42, 
                               stride_config='minimal', input_dropout=0.99)
    print(f"  Input dropout: {model_sparse.input_dropout_rate} (sparse feature learning)")
    print(f"  Total parameters: {model_sparse.get_num_parameters():,}")

    # Test aggressive pooling
    print("\n[3] Testing AGGRESSIVE POOLING configuration (faster, original):")
    model_aggressive = EpiAMLModel(input_size=357340, num_classes=42, stride_config='aggressive')
    print(f"  Stride config: {model_aggressive.stride_config}")
    print(f"  Total parameters: {model_aggressive.get_num_parameters():,}")

    # Use model_minimal for remaining tests
    model = model_minimal

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, model.input_size)

    print(f"\nTest Forward Pass:")
    print(f"  Input shape: {dummy_input.shape}")

    # Test classification output
    logits = model(dummy_input)
    print(f"  Logits shape: {logits.shape}")

    # Test with projection head
    logits, projections = model(dummy_input, return_projection=True)
    print(f"  Projections shape: {projections.shape}")

    # Test with features
    logits, features = model(dummy_input, return_features=True)
    print(f"  Features shape: {features.shape}")

    # Test predictions
    probs = model.predict_proba(dummy_input)
    print(f"  Probabilities shape: {probs.shape}")
    print(f"  Probabilities sum (per sample): {probs.sum(dim=1)}")

    print("\n" + "=" * 70)
    print("Model test completed successfully!")
    print("=" * 70)
