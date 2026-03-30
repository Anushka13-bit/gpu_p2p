"""FedAvg over in-memory weight buffers (io.BytesIO / raw bytes)."""

from __future__ import annotations

import io
from typing import List, Tuple

import torch


def fedavg_state_dicts(weight_buffers: List[Tuple[bytes, float]]) -> bytes:
    """
    Load N CPU state_dicts from bytes, execute trust-weighted average of float tensors,
    return a new torch.save blob as raw bytes. No disk I/O.
    Integer / long tensors (e.g. num_batches_tracked) are taken from the first model.
    """
    if not weight_buffers:
        raise ValueError("fedavg_state_dicts requires at least one buffer")
    state_dicts: list[dict[str, torch.Tensor]] = []
    trust_weights: list[float] = []
    for raw, weight in weight_buffers:
        trust_weights.append(weight)
        bio = io.BytesIO(raw)
        try:
            sd = torch.load(bio, map_location=torch.device("cpu"), weights_only=True)
        except TypeError:
            bio.seek(0)
            sd = torch.load(bio, map_location=torch.device("cpu"))
        state_dicts.append(sd)

    keys = list(state_dicts[0].keys())
    averaged: dict[str, torch.Tensor] = {}
    denominator = sum(trust_weights)
    if denominator == 0.0:
        denominator = 1.0

    for key in keys:
        tensors = [sd[key] for sd in state_dicts]
        if not tensors[0].dtype.is_floating_point:
            averaged[key] = tensors[0].clone()
            continue
        out = (tensors[0].float() * trust_weights[0]).clone()
        for i, t in enumerate(tensors[1:], 1):
            out = out + (t.float() * trust_weights[i])
        out = (out / denominator).to(tensors[0].dtype)
        averaged[key] = out

    out_buf = io.BytesIO()
    torch.save(averaged, out_buf)
    return out_buf.getvalue()
