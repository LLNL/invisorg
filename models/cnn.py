"""
Model class for the microbiome maps convolutional neural network.
"""

import torch
from torch import nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------------------------------------------------
#
class MeanActivationMask(nn.Module):
    """
    Non-learnable spatial attention mask based on average activation magnitude.

    Computes a soft attention mask by taking the mean activation across all channels
    at each spatial location, followed by a sigmoid nonlinearity. This acts as a
    deterministic gating mechanism that suppresses low-activation regions and emphasizes
    salient spatial locations without introducing trainable parameters.
    """

    def __init__(self):
        """
        Initializes the MeanActivationMask module.
        """
        super().__init__()

    def forward(self, x):
        """
        Forward pass for computing the mean-based spatial attention mask.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            Tensor: Output tensor of shape (B, C, H, W), modulated by the mask.
        """

        # Note x: (B, C, H, W)
        with torch.no_grad():
            mask = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
            mask = torch.sigmoid(mask)          # Soft mask between 0 and 1
        return x * mask  # Element-wise modulation


class LearnableActivationMask(nn.Module):
    """
    Learnable spatial attention mask using a lightweight 1x1 convolutional bottleneck.

    This module applies two 1x1 convolutions (with a reduction ratio) and a sigmoid
    activation to generate a spatial mask. The output is a learned attention map that
    dynamically weights spatial regions based on their relevance to the task.

    Inspired by channel-spatial attention modules like SE/CBAM.
    """

    def __init__(self, in_channels, reduction=16):
        """
        Initializes the LearnableActivationMask module.

        Args:
            in_channels (int): Number of input channels from the previous convolutional layer.
            reduction (int): Bottleneck reduction ratio for the internal hidden layer. Default is 16.
        """

        super().__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, kernel_size=1),
            nn.Sigmoid()  # Output mask ∈ (0, 1)
        )

    def forward(self, x):
        """
        Forward pass for computing the learnable attention mask.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            Tensor: Output tensor of shape (B, C, H, W), modulated by the learned mask.
        """

        # Note x: (B, C, H, W)
        mask = self.mask(x)        # (B, 1, H, W)
        return x * mask            # Broadcast multiply


class MmSparseAwareCNN(nn.Module):
    """
    Convolutional neural network for microbiome map classification with sparsity-aware attention masking.

    This architecture extends a 5-block CNN by applying a spatial attention mask after the final convolutional block.
    The mask suppresses low-activation regions and highlights salient areas in the feature maps prior to flattening.
    The attention mask can be either:

    - A deterministic, non-learnable `MeanActivationMask` that uses the mean channel activation per spatial location.
    - A trainable `LearnableActivationMask` that generates a task-adaptive mask via lightweight 1×1 convolutions.

    The model is designed for sparse, structured biological images (e.g., microbiome maps), where local activation
    varies significantly across regions. The attention mask mimics sparse convolution behavior, allowing the model
    to focus its capacity on relevant input areas while ignoring background noise.

    Args:
        num_classes (int): Number of output classes for classification.
        input_size (tuple): Input image dimensions as (C, H, W). Default is (3, 256, 256).
        hidden_d (int): Dimension of the hidden fully connected layer before the output layer. Default is 512.
        use_learnable_mask (bool): If True, use `LearnableActivationMask`; otherwise, use `MeanActivationMask`.

    Inputs:
        x (Tensor): Input tensor of shape (B, C, H, W)

    Returns:
        Tensor: Output tensor of shape (B, num_classes) with values in [0, 1] (after sigmoid)
    """

    def __init__(
        self,
        num_classes: int,
        input_size: tuple = (3, 256, 256),
        hidden_d: int = 512,
        use_learnable_mask: bool = False,
        is_multi_label: bool = False
    ):
        super().__init__()

        self.is_multi_label = is_multi_label

        in_channels = input_size[0]
        dropout_rate = 0.5
        conv_dr = 0.2

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=conv_dr),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=conv_dr),
            nn.MaxPool2d(2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=conv_dr),
            nn.MaxPool2d(2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=conv_dr),
            nn.MaxPool2d(2)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p=conv_dr),
            nn.MaxPool2d(2)
        )

        if use_learnable_mask:
            self.mask_early = LearnableActivationMask(in_channels=32)
            self.mask_late  = LearnableActivationMask(in_channels=256)
        else:
            self.mask_early = MeanActivationMask()
            self.mask_late  = MeanActivationMask()

        # Compute flatten dimension dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)
            x = self.block1(dummy_input)
            x = self.block2(x)
            x = self.mask_early(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.mask_late(x)
            self.flatten_dim = x.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_d),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_d, num_classes)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the sparse‐aware CNN.

        This method applies a sequence of convolutional blocks interleaved with spatial attention masks,
        flattens the resulting feature maps, and then feeds them through the final classifier head.

        Args:
            x (torch.Tensor): Input image batch of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output logits (or probabilities, if a sigmoid/softmax is included)
                        of shape (B, num_classes), corresponding to the predicted class scores
                        for each example in the batch.
        """

        x = self.block1(x)
        x = self.block2(x)
        x = self.mask_early(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.mask_late(x)
        x = torch.flatten(x, 1)

        logits = self.classifier(x)

        if self.is_multi_label:
            return torch.sigmoid(logits)
        else:
            return logits
