"""Small CNN for MNIST / CIFAR-10 and torch state_dict ↔ BytesIO helpers."""

from __future__ import annotations

import io
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_to_bytes(state_dict: dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getvalue()


def bytes_to_state_dict(data: bytes, map_location: Optional[str] = None) -> dict[str, torch.Tensor]:
    dev = map_location or "cpu"
    buf = io.BytesIO(data)
    try:
        return torch.load(buf, map_location=torch.device(dev), weights_only=True)
    except TypeError:
        buf.seek(0)
        return torch.load(buf, map_location=torch.device(dev))


def apply_state_dict(model: nn.Module, data: bytes, map_location: Optional[str] = None) -> None:
    sd = bytes_to_state_dict(data, map_location=map_location)
    model.load_state_dict(sd)


class SmallCNN(nn.Module):
    """
    LeNet-style CNN. Use in_channels=1, num_classes=10 for MNIST;
    in_channels=3, num_classes=10 for CIFAR-10.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adapt = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.adapt(self.pool(F.relu(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
