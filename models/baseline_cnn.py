"""
Baseline CNN for binary cataract classification.
3-4 conv blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool), global pool -> FC -> 1 output (sigmoid).
"""
import torch
import torch.nn as nn
from torch import Tensor


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU -> MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.pool(x)
        return x


class BaselineCNN(nn.Module):
    """
    Baseline CNN: configurable channels (e.g. 32 -> 64 -> 128 -> 256).
    Input: (B, 3, H, W). Output: (B, 1) logits.
    """

    def __init__(self, in_channels: int = 3, channels: tuple = (32, 64, 128, 256), num_classes: int = 1):
        super().__init__()
        blocks = []
        prev = in_channels
        for ch in channels:
            blocks.append(ConvBlock(prev, ch))
            prev = ch
        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(prev, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
