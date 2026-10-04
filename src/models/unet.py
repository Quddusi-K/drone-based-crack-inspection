"""U-Net baseline for binary segmentation. Additional architectures register in MODELS."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32, depth: int = 4):
        super().__init__()
        chs = [base_channels * 2 ** i for i in range(depth + 1)]
        self.encoders = nn.ModuleList()
        c = in_channels
        for ch in chs:
            self.encoders.append(DoubleConv(c, ch))
            c = ch
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.ups.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2))
            self.decoders.append(DoubleConv(chs[i - 1] * 2, chs[i - 1]))
        self.head = nn.Conv2d(chs[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if i < len(self.encoders) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, dec in zip(self.ups, self.decoders):
            x = up(x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)  # logits; apply sigmoid for probabilities


MODELS = {"unet": UNet}


def build_model(name: str = "unet", **kwargs) -> nn.Module:
    if name not in MODELS:
        raise KeyError(f"Unknown model {name}. Available: {list(MODELS)}")
    return MODELS[name](**kwargs)
