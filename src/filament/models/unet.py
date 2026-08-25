"""A lightweight multi-task U-Net: mask, spine, and offset heads.

This is a from-scratch, dependency-light implementation (torch + torchvision
only) rather than a wrapper around `segmentation_models_pytorch`, so the three
heads can share one decoder trunk and be trained end to end without fighting
a single-output library abstraction. `timm`/`smp` remain candidates for the
encoder swap in P4's ablations; this module only fixes the *interface*
(`ModelOutput` with three tensors), not the backbone.

Design choices tied directly to the plan:

- Encoder blocks halve resolution four times (1024 -> 64 at the bottleneck),
  matching the 1024x1024 input resolution the plan settled on -- at 512x512 a
  mean filament (~2,250 px at full res) is already only ~140 px and barbs
  become sub-pixel, which the literature review flagged as the likely cause of
  Flat U-Net's 0.69 recall.
- GroupNorm, not BatchNorm: batch sizes on a free-tier T4 at 1024x1024 will be
  small (single digits), where BatchNorm's running statistics are unreliable.
- The offset head outputs a 2-channel field but is NOT passed through any
  bounding nonlinearity (e.g. tanh) -- offsets are supervised as raw (row,
  col) displacement components in `filament.losses.segmentation.offset_loss`,
  and bounding them would need a magnitude convention decided before P1's
  target-generation code exists. Left as a documented open point, not a
  silent design decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["ModelOutput", "FilamentUNet"]


@dataclass
class ModelOutput:
    """The three raw-logit/raw-value outputs of the decoder.

    mask_logits, spine_logits : (B, 1, H, W)   -- pass through sigmoid for probabilities
    offsets                   : (B, 2, H, W)   -- raw (row, col) displacement, unbounded
    """

    mask_logits: torch.Tensor
    spine_logits: torch.Tensor
    offsets: torch.Tensor


def _norm(channels: int, groups: int = 8) -> nn.GroupNorm:
    # GroupNorm requires channels % groups == 0; fall back to per-channel
    # (InstanceNorm-equivalent) if the channel count doesn't divide evenly,
    # which only happens for the smallest configurations in tests.
    g = groups if channels % groups == 0 else 1
    return nn.GroupNorm(g, channels)


class ConvBlock(nn.Module):
    """Two 3x3 convs, each GroupNorm + SiLU, matching input to output channels."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _norm(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """Strided-conv downsample followed by a ConvBlock.

    A strided conv is used instead of maxpool + conv so the downsampling
    itself is learned rather than a fixed operator -- cheap to do and a
    long-standing minor win in dense-prediction encoders.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.norm = _norm(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.block = ConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm(self.down(x)))
        return self.block(x)


class Up(nn.Module):
    """Bilinear upsample, concatenate the matching skip connection, ConvBlock.

    Bilinear + conv is used instead of a transposed conv to avoid the
    checkerboard-artefact tendency of learned upsampling -- a real concern
    here since checkerboarding at the barb scale (a few px) would directly
    corrupt the structure the spine head depends on.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Guard against off-by-one size mismatches from odd input dimensions;
        # crops the larger tensor to match rather than requiring the caller to
        # pad inputs to a power of two.
        if x.shape[-2:] != skip.shape[-2:]:
            dh = skip.shape[-2] - x.shape[-2]
            dw = skip.shape[-1] - x.shape[-1]
            x = nn.functional.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class FilamentUNet(nn.Module):
    """Encoder-decoder with three task-specific 1x1-conv output heads.

    Parameters
    ----------
    in_channels:
        Input channel count. Defaults to 3 to leave room for the geometry
        channels from Edge 3 (r/R_sun, longitude) alongside the raw H-alpha
        intensity, without a separate code path -- callers with fewer
        channels (e.g. grayscale-only during early P1 iteration) just pass
        `in_channels=1`.
    base_channels:
        Channel count after the stem; doubles at each of the four downs,
        matching classical U-Net rather than the "flat" channel schedule of
        Zhu et al. 2025 -- capacity is not the bottleneck we are fighting
        (compute is), and their result shows a flat schedule trades some
        recall for parameter count, which is the wrong trade for this metric.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.stem = ConvBlock(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)

        self.up1 = Up(c * 16, c * 8, c * 8)
        self.up2 = Up(c * 8, c * 4, c * 4)
        self.up3 = Up(c * 4, c * 2, c * 2)
        self.up4 = Up(c * 2, c, c)

        self.mask_head = nn.Conv2d(c, 1, kernel_size=1)
        self.spine_head = nn.Conv2d(c, 1, kernel_size=1)
        self.offset_head = nn.Conv2d(c, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        x0 = self.stem(x)  # H
        x1 = self.down1(x0)  # H/2
        x2 = self.down2(x1)  # H/4
        x3 = self.down3(x2)  # H/8
        x4 = self.down4(x3)  # H/16 (bottleneck)

        y = self.up1(x4, x3)
        y = self.up2(y, x2)
        y = self.up3(y, x1)
        y = self.up4(y, x0)

        return ModelOutput(
            mask_logits=self.mask_head(y),
            spine_logits=self.spine_head(y),
            offsets=self.offset_head(y),
        )
