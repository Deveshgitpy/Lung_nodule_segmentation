"""
models/unet.py
==============
3D U-Net architecture for volumetric lung nodule segmentation.

This implementation follows the architecture described in:
  Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
  Image Segmentation," MICCAI 2015.

Extended to 3D for volumetric CT segmentation, following the convention
from Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation
from Sparse Annotation," MICCAI 2016.

Architecture summary (default config):
  Input:      [B, 1,   D,   H,   W]
  Enc-1:      [B, 32,  D,   H,   W]  → pool → [B, 32,  D/2,  H/2,  W/2]
  Enc-2:      [B, 64,  D/2, H/2, W/2] → pool → [B, 64,  D/4,  H/4,  W/4]
  Enc-3:      [B, 128, D/4, H/4, W/4] → pool → [B, 128, D/8,  H/8,  W/8]
  Enc-4:      [B, 256, D/8, H/8, W/8] → pool → [B, 256, D/16, H/16, W/16]
  Bottleneck: [B, 512, D/16, H/16, W/16]
  Dec-4:      [B, 256, D/8,  H/8,  W/8]
  Dec-3:      [B, 128, D/4,  H/4,  W/4]
  Dec-2:      [B, 64,  D/2,  H/2,  W/2]
  Dec-1:      [B, 32,  D,    H,    W]
  Output:     [B, 1,   D,    H,    W]   (sigmoid probabilities)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from models.blocks import BottleneckBlock, DecoderBlock, EncoderBlock, OutputBlock


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric medical image segmentation.

    The model uses an encoder-decoder structure with skip connections.
    The encoder progressively down-samples the input volume while
    increasing feature channels. The decoder up-samples back to the
    original resolution, using skip connections from the encoder to
    recover spatial detail lost during downsampling.

    Args:
        in_channels:         Number of input modality channels (1 for CT).
        out_channels:        Number of output classes (1 for binary nodule mask).
        encoder_channels:    List of feature map sizes at each encoder depth.
                             Default: [32, 64, 128, 256]
        bottleneck_channels: Feature map size at the bottleneck.
                             Default: 512
        dropout_rate:        Dropout probability at the bottleneck.
        bn_momentum:         Momentum for BatchNorm running statistics.

    Example:
        >>> model = UNet3D(in_channels=1, out_channels=1)
        >>> x = torch.randn(2, 1, 64, 64, 64)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([2, 1, 64, 64, 64])
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_channels: Optional[List[int]] = None,
        bottleneck_channels: int = 512,
        dropout_rate: float = 0.2,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_channels = encoder_channels
        self.bottleneck_channels = bottleneck_channels
        self.depth = len(encoder_channels)

        # ── Encoder ─────────────────────────────────────────────────────────
        # Build encoder blocks dynamically based on encoder_channels list.
        # Each EncoderBlock downsamples spatially by 2× via MaxPool.
        self.encoders = nn.ModuleList()

        # First encoder block takes raw input channels
        self.encoders.append(
            EncoderBlock(
                in_channels=in_channels,
                out_channels=encoder_channels[0],
                pool=True,
                bn_momentum=bn_momentum,
            )
        )

        # Remaining encoder blocks double the channels at each level
        for i in range(1, self.depth):
            self.encoders.append(
                EncoderBlock(
                    in_channels=encoder_channels[i - 1],
                    out_channels=encoder_channels[i],
                    pool=True,
                    bn_momentum=bn_momentum,
                )
            )

        # ── Bottleneck ──────────────────────────────────────────────────────
        # Deepest block: most abstract representations, smallest spatial dims.
        # Dropout added for regularisation at the information bottleneck.
        self.bottleneck = BottleneckBlock(
            in_channels=encoder_channels[-1],
            out_channels=bottleneck_channels,
            dropout_rate=dropout_rate,
            bn_momentum=bn_momentum,
        )

        # ── Decoder ─────────────────────────────────────────────────────────
        # Build decoder blocks in reverse order.
        # Each DecoderBlock upsamples by 2× and concatenates the skip connection.
        self.decoders = nn.ModuleList()

        # reversed encoder channels: [256, 128, 64, 32]
        reversed_enc = list(reversed(encoder_channels))

        # First decoder: bottleneck → skip from deepest encoder
        self.decoders.append(
            DecoderBlock(
                in_channels=bottleneck_channels,
                skip_channels=reversed_enc[0],
                out_channels=reversed_enc[0],
                bn_momentum=bn_momentum,
            )
        )

        # Remaining decoder blocks
        for i in range(1, self.depth):
            self.decoders.append(
                DecoderBlock(
                    in_channels=reversed_enc[i - 1],
                    skip_channels=reversed_enc[i],
                    out_channels=reversed_enc[i],
                    bn_momentum=bn_momentum,
                )
            )

        # ── Output layer ────────────────────────────────────────────────────
        # 1×1×1 conv to map features → probability map
        self.output_block = OutputBlock(
            in_channels=encoder_channels[0],
            out_channels=out_channels,
        )

        # ── Weight initialisation ────────────────────────────────────────────
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Kaiming He initialisation for Conv3d layers.
        This is appropriate for ReLU-activated networks and helps avoid
        vanishing/exploding gradients in deep 3D networks.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the 3D U-Net.

        Args:
            x: Input tensor of shape [B, C_in, D, H, W].
               For single-channel CT: [B, 1, D, H, W].

        Returns:
            Probability map of shape [B, C_out, D, H, W].
            For binary segmentation: [B, 1, D, H, W] with values in [0, 1].
        """
        # ── Encoder pass ────────────────────────────────────────────────────
        # Collect skip connections at each level for the decoder.
        skips: List[torch.Tensor] = []
        current = x

        for encoder in self.encoders:
            skip, current = encoder(current)
            skips.append(skip)

        # ── Bottleneck ──────────────────────────────────────────────────────
        current = self.bottleneck(current)

        # ── Decoder pass ────────────────────────────────────────────────────
        # Apply decoder blocks in reverse order of skip connections.
        # skips = [enc1_out, enc2_out, enc3_out, enc4_out]  (coarse → fine)
        # We traverse in reverse:  enc4_out, enc3_out, enc2_out, enc1_out
        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 1)]     # Reverse order
            current = decoder(current, skip)

        # ── Output ──────────────────────────────────────────────────────────
        return self.output_block(current)

    def get_num_parameters(self) -> Dict[str, int]:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def freeze_encoder(self) -> None:
        """
        Freeze encoder weights for transfer learning or fine-tuning scenarios.
        Only the decoder and output layers will be updated during training.
        """
        for encoder in self.encoders:
            for param in encoder.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder weights to allow full fine-tuning."""
        for encoder in self.encoders:
            for param in encoder.parameters():
                param.requires_grad = True


def build_unet_from_config(config: dict) -> UNet3D:
    """
    Factory function to instantiate a UNet3D from a config dictionary.

    Args:
        config: Dictionary with keys matching UNet3D constructor arguments,
                typically loaded from configs/training_config.yaml under
                the 'model' section.

    Returns:
        Instantiated UNet3D model.

    Example:
        >>> import yaml
        >>> with open("configs/training_config.yaml") as f:
        ...     cfg = yaml.safe_load(f)
        >>> model = build_unet_from_config(cfg["model"])
    """
    model_cfg = config.get("model", config)  # Accept full config or model sub-dict

    return UNet3D(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        encoder_channels=model_cfg.get("encoder_channels", [32, 64, 128, 256]),
        bottleneck_channels=model_cfg.get("bottleneck_channels", 512),
        dropout_rate=model_cfg.get("dropout_rate", 0.2),
        bn_momentum=model_cfg.get("bn_momentum", 0.1),
    )


if __name__ == "__main__":
    # Quick architecture sanity check
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        encoder_channels=[32, 64, 128, 256],
        bottleneck_channels=512,
    ).to(device)

    params = model.get_num_parameters()
    print(f"UNet3D — Total params: {params['total']:,} | "
          f"Trainable: {params['trainable']:,}")

    # Test forward pass with a 64³ patch (standard training patch size)
    dummy = torch.randn(2, 1, 64, 64, 64).to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model(dummy)
    elapsed = time.time() - t0

    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"Forward pass time (batch=2, 64³): {elapsed*1000:.1f} ms")
    assert out.shape == dummy.shape, "Output shape mismatch!"
    print("✓ Shape check passed.")
