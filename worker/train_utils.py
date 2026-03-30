"""Training steps on a contiguous dataset shard (first 10k MNIST training images)."""

from __future__ import annotations

import io
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import MNIST

def build_mnist_base_10k(root: str = "./data") -> Subset:
    tfm = transforms.ToTensor()
    raw = MNIST(root=root, train=True, download=True, transform=tfm)
    n = min(10_000, len(raw))
    return Subset(raw, list(range(n)))


def train_shard_batch_loop(
    model: nn.Module,
    base_10k: Subset,
    image_start: int,
    image_end: int,
    resume_next_index: int,
    device: torch.device,
    max_steps: int,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> Tuple[bytes, int, int, bool]:
    """
    Train up to ``max_steps`` batches on indices ``[resume_next_index, image_end)`` (absolute 0..10k).
    Returns (weights_bytes, last_consumed_index, batches_run, shard_complete).
    """
    resume_next_index = max(image_start, min(resume_next_index, image_end - 1))
    idx_map = list(range(resume_next_index, image_end))
    if not idx_map:
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return buf.getvalue(), image_end - 1, 0, True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    steps_run = 0
    last_consumed = resume_next_index - 1

    for i in range(0, len(idx_map), batch_size):
        if steps_run >= max_steps:
            break
        chunk = idx_map[i : i + batch_size]
        xs = torch.stack([base_10k[j][0] for j in chunk]).to(device)
        ys = torch.stack([torch.tensor(base_10k[j][1], device=device) for j in chunk])
        optimizer.zero_grad()
        loss = criterion(model(xs), ys)
        loss.backward()
        optimizer.step()
        steps_run += 1
        last_consumed = chunk[-1]

    shard_complete = last_consumed >= (image_end - 1)
    out = io.BytesIO()
    torch.save(model.state_dict(), out)
    return out.getvalue(), last_consumed, steps_run, shard_complete