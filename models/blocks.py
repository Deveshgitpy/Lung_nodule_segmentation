"""
models/blocks.py
================
Reusable building blocks for the 3D U-Net architecture.

Each block encapsulates a specific functional unit of the network,
making the architecture modular, testable, and easy to swap.

Block hierarchy:
  DoubleConvBlock  → two consecutive conv3d → BN → ReLU operations
  EncoderBlock     → DoubleConvBlock + MaxPool3d (downsampling)
  DecoderBlock     → TransposedConv3d (upsample) + skip concat + DoubleConvBlock
  BottleneckBlock  → DoubleConvBlock + Dropout (no pooling)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):
    """
    Two consecutive 3D convolution operations, each followed by
    Batch Normalisation and ReLU activation.

    Architecture per sub-block:
        Conv3d(in_ch, out_ch, k=3, pad=1) → BN(out_ch) → ReLU(inplace)
    Applied twice, with the first conv handling the channel dimension
    change and the second refining features at constant channel width.

    Args:
        in_channels:  Number of input feature map channels.
        out_channels: Number of output feature map channels.
        bn_momentum:  Momentum for BatchNorm running statistics.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            # First convolution: change channel dimension
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,      # Bias is redundant before BatchNorm
            ),
            nn.BatchNorm3d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),

            # Second convolution: refine features at constant width
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    U-Net encoder block: double-conv → (optionally) max-pool.

    Produces both:
      - `skip`: the feature map *before* pooling (for skip connections)
      - `pooled`: the downsampled feature map *after* pooling

    The separation of skip and pooled outputs allows the model to
    explicitly route high-resolution spatial features to the decoder.

    Args:
        in_channels:  Channels coming in.
        out_channels: Channels after double-conv.
        pool:         Whether to apply MaxPool3d (False for bottleneck).
        bn_momentum:  BatchNorm momentum.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool: bool = True,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()

        self.conv = DoubleConvBlock(in_channels, out_channels, bn_momentum)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2) if pool else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            skip:   Feature map at full resolution (used for skip connection).
            pooled: Downsampled feature map (fed to next encoder level).
                    If pool=False (bottleneck), pooled == skip.
        """
        skip = self.conv(x)
        pooled = self.pool(skip) if self.pool is not None else skip
        return skip, pooled


class DecoderBlock(nn.Module):
    """
    U-Net decoder block: upsample → concatenate skip → double-conv.

    Upsampling is performed via a transposed convolution (learnable),
    which avoids the checkerboard artefacts sometimes seen with simple
    bilinear upsampling followed by a convolution in medical images.

    After upsampling, the skip-connection feature map from the encoder
    is concatenated along the channel axis. This is the core mechanism
    that allows the U-Net to combine coarse semantic features (from the
    bottleneck) with fine spatial features (from the encoder).

    Args:
        in_channels:  Channels of the upsampled (decoder) feature map.
        skip_channels: Channels of the skip connection feature map.
        out_channels: Channels after the double-conv.
        bn_momentum:  BatchNorm momentum.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()

        # Transposed conv halves spatial dims and halves channels
        self.upsample = nn.ConvTranspose3d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )

        # After concatenation: (in_channels // 2) + skip_channels
        self.conv = DoubleConvBlock(
            in_channels // 2 + skip_channels,
            out_channels,
            bn_momentum,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    Decoder feature map from previous (deeper) level.
            skip: Encoder skip-connection feature map at matching resolution.

        Returns:
            Refined feature map at upsampled resolution.
        """
        x = self.upsample(x)

        # Handle potential size mismatches due to odd input dimensions.
        # This is common in volumetric CT where the depth axis can have
        # an odd number of slices after preprocessing.
        if x.shape != skip.shape:
            x = F.interpolate(
                x,
                size=skip.shape[2:],   # Match D, H, W of skip
                mode="trilinear",
                align_corners=False,
            )

        # Concatenate along channel dimension
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class BottleneckBlock(nn.Module):
    """
    U-Net bottleneck: the deepest point of the network, connecting the
    encoder and decoder paths.

    Adds Dropout3d for regularisation, which is especially important
    in the bottleneck where all spatial information has been compressed.

    Args:
        in_channels:   Channels from the final encoder level.
        out_channels:  Bottleneck output channels (typically 2× encoder depth).
        dropout_rate:  Dropout probability applied to the bottleneck output.
        bn_momentum:   BatchNorm momentum.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.2,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()

        self.conv = DoubleConvBlock(in_channels, out_channels, bn_momentum)
        # Dropout3d zeroes entire feature map channels (more aggressive
        # regularisation than per-voxel Dropout, suitable for 3D data)
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.conv(x))


class OutputBlock(nn.Module):
    """
    Final 1×1×1 convolution + Sigmoid activation to produce per-voxel
    nodule probability maps.

    Using a 1×1×1 conv preserves spatial resolution while mapping from
    the decoder feature channels to a single output channel (binary mask).

    Args:
        in_channels: Channels from the final decoder block.
        out_channels: Number of output classes (1 for binary segmentation).
    """

    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        # Sigmoid produces per-voxel probability in [0, 1]
        # During training with BCEWithLogitsLoss, we'd skip Sigmoid,
        # but here we apply it explicitly for compatibility with
        # the combined Dice+BCE loss which handles logits separately.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.conv(x))
