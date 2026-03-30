"""Training steps on a contiguous dataset shard (first 10k samples of a dataset)."""

from __future__ import annotations

import io
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import FashionMNIST, MNIST


@torch.no_grad()
def eval_accuracy_on_range(
    model: nn.Module,
    base_10k: Subset,
    image_start: int,
    image_end: int,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    """Accuracy % on subset indices [image_start, image_end)."""
    model.eval()
    indices = list(range(image_start, image_end))
    if not indices:
        model.train()
        return 0.0
    correct = 0
    total = 0
    for i in range(0, len(indices), batch_size):
        chunk = indices[i : i + batch_size]
        xs = torch.stack([base_10k[j][0] for j in chunk]).to(device)
        ys = torch.stack([torch.tensor(base_10k[j][1], device=device) for j in chunk])
        pred = model(xs).argmax(dim=1)
        correct += int((pred == ys).sum().item())
        total += len(chunk)
    model.train()
    return 100.0 * correct / max(1, total)


def build_dataset_base_10k(dataset: str = "fashion_mnist", root: str = "./data") -> Subset:
    """
    Download/load a torchvision dataset and return the first 10k train samples as a Subset.

    Supported:
    - fashion_mnist (default)
    - mnist
    """
    ds = (dataset or "").strip().lower().replace("-", "_")
    tfm = transforms.ToTensor()
    if ds in ("fashion_mnist", "fashionmnist", "fmnist"):
        raw = FashionMNIST(root=root, train=True, download=True, transform=tfm)
    elif ds in ("mnist",):
        raw = MNIST(root=root, train=True, download=True, transform=tfm)
    else:
        raise ValueError(f"unknown dataset={dataset!r} (supported: fashion_mnist, mnist)")
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
    local_epochs: int = 1,
    batch_size: int = 64,
    lr: float = 1e-3,
    verbose: bool = False,
    log_steps: bool = False,
    log_prefix: str = "",
) -> Tuple[bytes, int, int, bool, float | None, float | None, float | None, int, int]:
    """
    Train up to ``max_steps`` batches on indices ``[resume_next_index, image_end)`` (absolute 0..10k),
    repeating the slice for ``local_epochs`` passes (local epochs).
    Returns (
      weights_bytes, last_consumed_index, batches_run, shard_complete,
      last_loss, running_train_acc, shard_eval_acc,
      local_epochs_planned, local_epochs_completed
    ).
    """
    resume_next_index = max(image_start, min(resume_next_index, image_end - 1))
    idx_map_base = list(range(resume_next_index, image_end))
    if not idx_map_base:
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return buf.getvalue(), image_end - 1, 0, True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    steps_run = 0
    last_consumed = resume_next_index - 1
    running_correct = 0
    running_total = 0
    local_epochs_planned = max(1, int(local_epochs))
    planned_batches_per_epoch = (len(idx_map_base) + batch_size - 1) // batch_size
    planned_total_batches = planned_batches_per_epoch * local_epochs_planned
    if verbose:
        print(
            f"{log_prefix} slice [{resume_next_index},{image_end})  "
            f"local_epochs={local_epochs_planned}  "
            f"batches_this_call≤{min(max_steps, planned_total_batches)}/{planned_total_batches} (batch_size={batch_size})",
            flush=True,
        )

    epochs_completed = 0
    for ep in range(local_epochs_planned):
        if verbose:
            print(f"{log_prefix}  epoch {ep + 1}/{local_epochs_planned}", flush=True)
        for i in range(0, len(idx_map_base), batch_size):
            if steps_run >= max_steps:
                break
            chunk = idx_map_base[i : i + batch_size]
            xs = torch.stack([base_10k[j][0] for j in chunk]).to(device)
            ys = torch.stack([torch.tensor(base_10k[j][1], device=device) for j in chunk])
            optimizer.zero_grad()
            loss = criterion(model(xs), ys)
            loss.backward()
            optimizer.step()
            steps_run += 1
            last_consumed = chunk[-1]
            with torch.no_grad():
                pred = model(xs).argmax(dim=1)
                running_correct += int((pred == ys).sum().item())
                running_total += len(chunk)
            if verbose:
                ra = 100.0 * running_correct / max(1, running_total)
                if log_steps:
                    print(
                        f"{log_prefix}  step {steps_run}/{min(max_steps, planned_total_batches)}  "
                        f"loss={loss.item():.4f}  running_train_acc={ra:.2f}%",
                        flush=True,
                    )
        if steps_run >= max_steps:
            break
        epochs_completed += 1

    shard_complete = last_consumed >= (image_end - 1)
    last_loss = float(loss.item()) if "loss" in locals() else None
    running_train_acc = 100.0 * running_correct / max(1, running_total) if running_total else None
    shard_eval_acc = None
    if verbose:
        acc = eval_accuracy_on_range(model, base_10k, image_start, image_end, device)
        shard_eval_acc = float(acc)
        print(
            f"{log_prefix} shard eval [{image_start},{image_end}): {acc:.2f}%  "
            f"shard_complete={shard_complete}",
            flush=True,
        )
    else:
        # Even if quiet, still compute once per submit so tracker can display it.
        shard_eval_acc = float(eval_accuracy_on_range(model, base_10k, image_start, image_end, device))
    out = io.BytesIO()
    torch.save(model.state_dict(), out)
    return (
        out.getvalue(),
        last_consumed,
        steps_run,
        shard_complete,
        last_loss,
        running_train_acc,
        shard_eval_acc,
        local_epochs_planned,
        epochs_completed,
    )