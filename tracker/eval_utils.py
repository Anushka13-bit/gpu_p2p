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


def _resolve_archive2_dir() -> Path:
    env = os.environ.get("FASHION_MNIST_CSV_DIR")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    candidates.extend(_default_archive2_dirs())
    for c in candidates:
        p = c.expanduser().resolve()
        if p.exists():
            return p
    # Fallback to repo-relative for error messaging below.
    return Path("archive 2").resolve()


def _load_fashion_mnist_train_slice_csv(start_row: int, num_rows: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a slice of Fashion-MNIST *train* CSV.

    Uses 0-indexed row offsets into the CSV data rows (excluding header).
    """
    base = _resolve_archive2_dir()
    train_csv = base / "fashion-mnist_train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"expected Fashion-MNIST train CSV at {train_csv}")
    if start_row < 0 or num_rows <= 0:
        raise ValueError("start_row must be >=0 and num_rows must be >0")

    raw = np.loadtxt(
        str(train_csv),
        delimiter=",",
        skiprows=1 + int(start_row),
        max_rows=int(num_rows),
        dtype=np.float32,
    )
    ys = torch.from_numpy(raw[:, 0].astype(np.int64))
    xs = torch.from_numpy(raw[:, 1:]).reshape(-1, 1, 28, 28) / 255.0
    return xs, ys


def _load_fashion_mnist_test_csv(max_rows: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load Fashion-MNIST test split from local CSV.

    Expects `fashion-mnist_test.csv` under `archive 2/` by default.
    Override directory via env var `FASHION_MNIST_CSV_DIR`.
    CSV format: label,pixel0,...,pixel783
    """
    base = _resolve_archive2_dir()
    test_csv = base / "fashion-mnist_test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"expected Fashion-MNIST test CSV at {test_csv}")
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


@torch.no_grad()
def eval_global_fashion_mnist_val_acc(
    weights_bytes: bytes,
    device: str = "cpu",
    batch_size: int = 512,
    *,
    # 80/10/10 split over a 12_500 sample window: train[0:10k], val[10k:11.25k], (train-heldout-test[11.25k:12.5k])
    val_start_row: int = 10_000,
    val_rows: int = 1_250,
) -> float:
    """Validation accuracy % on a held-out slice of the Fashion-MNIST train CSV."""
    model = SmallCNN(in_channels=1, num_classes=10).to(torch.device(device))
    apply_state_dict(model, weights_bytes, map_location=device)
    model.eval()

    xs, ys = _load_fashion_mnist_train_slice_csv(val_start_row, val_rows)
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

