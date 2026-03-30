"""Tracker-side evaluation for aggregated global models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from shared.models import SmallCNN, apply_state_dict


def _default_archive2_dirs() -> list[Path]:
    # Try repo-relative first, then a common macOS location.
    return [
        Path("archive 2"),
        Path.home() / "Desktop" / "archive 2",
    ]


def _load_fashion_mnist_test_csv(max_rows: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load Fashion-MNIST test split from local CSV.

    Expects `fashion-mnist_test.csv` under `archive 2/` by default.
    Override directory via env var `FASHION_MNIST_CSV_DIR`.
    CSV format: label,pixel0,...,pixel783
    """
    env = os.environ.get("FASHION_MNIST_CSV_DIR")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    candidates.extend(_default_archive2_dirs())

    test_csv: Optional[Path] = None
    for base in candidates:
        p = base.expanduser().resolve() / "fashion-mnist_test.csv"
        if p.exists():
            test_csv = p
            break
    if test_csv is None:
        tried = ", ".join(str((c.expanduser().resolve() / "fashion-mnist_test.csv")) for c in candidates)
        raise FileNotFoundError(
            "could not find Fashion-MNIST test CSV. Tried: "
            f"{tried}. Set FASHION_MNIST_CSV_DIR or add `archive 2/` to the repo root."
        )
    raw = np.loadtxt(
        str(test_csv),
        delimiter=",",
        skiprows=1,
        max_rows=max_rows if max_rows and max_rows > 0 else None,
        dtype=np.float32,
    )
    ys = torch.from_numpy(raw[:, 0].astype(np.int64))
    xs = torch.from_numpy(raw[:, 1:]).reshape(-1, 1, 28, 28) / 255.0
    return xs, ys


@torch.no_grad()
def eval_global_fashion_mnist_test_acc(weights_bytes: bytes, device: str = "cpu", batch_size: int = 512) -> float:
    """Return accuracy % of `SmallCNN` on Fashion-MNIST test CSV."""
    model = SmallCNN(in_channels=1, num_classes=10).to(torch.device(device))
    apply_state_dict(model, weights_bytes, map_location=device)
    model.eval()

    xs, ys = _load_fashion_mnist_test_csv()
    xs = xs.to(torch.device(device))
    ys = ys.to(torch.device(device))

    correct = 0
    total = int(xs.shape[0])
    for i in range(0, total, batch_size):
        xb = xs[i : i + batch_size]
        yb = ys[i : i + batch_size]
        pred = model(xb).argmax(dim=1)
        correct += int((pred == yb).sum().item())
    return 100.0 * correct / max(1, total)

