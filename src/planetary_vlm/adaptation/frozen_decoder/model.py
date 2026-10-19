"""Depth taps -> explicit scale pyramid -> top-down fusion; shared unchanged with C."""
from torch import nn
from torch.nn import functional as F


def refinement(width):
    return nn.Sequential(nn.Conv2d(width, width, 3, padding=1, bias=False), nn.GroupNorm(8, width), nn.GELU(),
                         nn.Conv2d(width, width, 3, padding=1, bias=False), nn.GroupNorm(8, width), nn.GELU())


class PyramidDecoder(nn.Module):
    def __init__(self, channels, width=64):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(channels, width, 1) for _ in range(4)])
        self.refinements = nn.ModuleList([refinement(width) for _ in range(4)])
        self.classifier = nn.Conv2d(width, 1, 1)

    def forward(self, features, size):
        if len(features) != 4:
            raise ValueError("Decoder requires four declared encoder-depth taps")
        side = features[0].shape[-1]
        scales = [side * 4, side * 2, side, max(side // 2, 1)]
        maps = [F.interpolate(proj(x), (scale, scale), mode="bilinear", align_corners=False)
                for proj, x, scale in zip(self.projections, features, scales)]
        fused = None
        for index in reversed(range(4)):
            fused = maps[index] if fused is None else maps[index] + F.interpolate(
                fused, maps[index].shape[-2:], mode="bilinear", align_corners=False)
            fused = self.refinements[index](fused)
        return F.interpolate(self.classifier(fused), (size, size), mode="bilinear", align_corners=False)
