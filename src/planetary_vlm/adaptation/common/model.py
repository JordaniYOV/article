"""Only model-visible tensors are passed into the encoder."""
import torch
from torch import nn


class System(nn.Module):
    def __init__(self, head, encoder=None, adapted=False):
        super().__init__()
        self.head, self.encoder, self.adapted = head, encoder, adapted

    def train(self, mode=True):
        super().train(mode)
        if self.encoder is not None:
            self.encoder.train(mode if self.adapted else False)
        return self

    def forward(self, x, size):
        if isinstance(x, (list, tuple)):
            if self.adapted:
                raise ValueError("LoRA cannot train on frozen cached features")
            features = x
        elif self.encoder is None:
            raise ValueError("Image input requires the encoder")
        elif self.adapted:
            features = self.encoder(x)
        else:
            with torch.no_grad():
                features = self.encoder(x)
        return self.head(features, size)


def make_head(cfg):
    if cfg["method"] == "linear_probe":
        from ..linear_probe.model import LinearProbe
        return LinearProbe(cfg["encoder"]["channels"])
    from ..frozen_decoder.model import PyramidDecoder
    return PyramidDecoder(cfg["encoder"]["channels"], cfg["decoder"]["width"])


def loss(logits, target, positive_weight, dice_weight):
    from torch.nn import functional as F
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=logits.new_tensor(positive_weight), reduction="none").mean((1, 2, 3))
    p = logits.sigmoid()
    dice = 1 - (2 * (p * target).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return (bce + dice_weight * dice).mean()
