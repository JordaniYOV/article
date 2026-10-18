"""The only trainable components. Encoders are absent from this module."""
import torch
from torch import nn
from torch.nn import functional as F
from functools import lru_cache
import math


@lru_cache(maxsize=32)
def resize_matrix(source, destination, device, dtype):
    """Fixed align_corners=False bilinear weights, built on CPU without atomics."""
    weights = torch.zeros(destination, source, dtype=torch.float64)
    for output in range(destination):
        position = (output + .5) * source / destination - .5
        left = math.floor(position)
        fraction = position - left
        weights[output, min(source - 1, max(0, left))] += 1 - fraction
        weights[output, min(source - 1, max(0, left + 1))] += fraction
    return weights.to(device=device, dtype=dtype)


def resize_logits(logits, output_size):
    # CUDA interpolate(bilinear) backward is incompatible with strict deterministic
    # algorithms. Separable fixed matrices have the same geometric rule and use
    # deterministic matmul (with CUBLAS_WORKSPACE_CONFIG=:4096:8 set by CLI/worker).
    vertical = resize_matrix(logits.shape[-2], output_size, logits.device, logits.dtype)
    horizontal = resize_matrix(logits.shape[-1], output_size, logits.device, logits.dtype)
    return (vertical @ logits) @ horizontal.T


class CraterHead(nn.Module):
    def __init__(self, channels, family="spatial", hidden=64):
        super().__init__()
        self.family = family
        if family == "linear":
            self.layers = nn.Conv2d(channels, 1, kernel_size=1)
        elif family == "spatial":
            if hidden % 8:
                raise ValueError("GroupNorm requires hidden_channels divisible by 8")
            self.layers = nn.Sequential(
                nn.Conv2d(channels, hidden, 1), nn.GroupNorm(8, hidden), nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.GELU(),
                nn.Conv2d(hidden, 1, 1))
        else:
            raise ValueError("Unknown head family")

    def forward(self, features, output_size=128):
        return resize_logits(self.layers(features), output_size)


def loss_per_image(logits, target, positive_weight, dice_weight=.5):
    # Stable logit BCE; soft area targets preserve fractional small-crater pixels.
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight, reduction="none").mean((1, 2, 3))
    probability = logits.sigmoid()
    intersection = (probability * target).sum((1, 2, 3))
    dice = 1 - (2 * intersection + 1) / (probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dice_weight * dice
