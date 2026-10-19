"""No hidden activation, normalization, learned upsampling or spatial convolution."""
from torch import nn
from torch.nn import functional as F


class LinearProbe(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.classifier = nn.Conv2d(channels, 1, 1)

    def forward(self, features, size):
        return F.interpolate(self.classifier(features[-1]), (size, size), mode="bilinear", align_corners=False)
