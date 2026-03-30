"""In-RAM checkpoints: global FedAvg weights and per-shard progress."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional

import torch

from shared.protocol import TaskStatus
from shared.models import SmallCNN, weights_to_bytes


@dataclass
class TaskCheckpoint:
    """Latest in-memory weights + progress for a data shard."""

    weights_bytes: Optional[bytes] = None
    last_index: int = -1


@dataclass
class GlobalState:
    """Federated global model bytes (no disk I/O)."""

    round_no: int = 1
    global_weights_bytes: Optional[bytes] = None
    version_label: str = "Global Model v1"
    last_val_acc: Optional[float] = None
    last_test_acc: Optional[float] = None


class StateManager:
    """
    Stores the latest global model and per-task checkpoints entirely in RAM.
    New workers for failed tasks receive the last saved checkpoint bytes.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._global = GlobalState()
        self._per_task: Dict[str, TaskCheckpoint] = {}
        self._ckpt_dir = Path(os.environ.get("GPU_P2P_CHECKPOINT_DIR", "./checkpoints")).resolve()
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _task_ckpt_path(self, task_id: str) -> Path:
        return self._ckpt_dir / f"{task_id}.pt"

    def _global_path(self, round_no: int) -> Path:
        return self._ckpt_dir / f"global_round_{round_no}.pt"

    def ensure_task_slot(self, task_id: str) -> None:
        with self._lock:
            self._per_task.setdefault(task_id, TaskCheckpoint())

    def get_global_bytes(self) -> Optional[bytes]:
        with self._lock:
            return copy.deepcopy(self._global.global_weights_bytes) if self._global.global_weights_bytes else None

    def ensure_initial_global(self) -> bytes:
        """
        Ensure there is a single shared initial model for round 1.

        Without this, each worker would initialize its own random weights and FedAvg would average
        across different random initializations, often collapsing to near-random accuracy.
        """
        with self._lock:
            if self._global.global_weights_bytes is not None:
                return copy.deepcopy(self._global.global_weights_bytes)
            torch.manual_seed(0)
            model = SmallCNN(in_channels=1, num_classes=10)
            raw = weights_to_bytes(model.state_dict())
            self._global.global_weights_bytes = raw
            self._global.round_no = 1
            self._global.version_label = "Global Model v1"
            self._global.last_val_acc = None
            self._global.last_test_acc = None
            try:
                self._global_path(1).write_bytes(raw)
            except OSError:
                pass
            return copy.deepcopy(raw)

    def set_global_bytes(
        self,
        weights_bytes: bytes,
        next_round: int,
        *,
        val_acc: Optional[float] = None,
        test_acc: Optional[float] = None,
    ) -> None:
        with self._lock:
            self._global.global_weights_bytes = weights_bytes
            self._global.round_no = next_round
            self._global.version_label = f"Global Model v{next_round}"
            self._global.last_val_acc = float(val_acc) if val_acc is not None else None
            self._global.last_test_acc = float(test_acc) if test_acc is not None else None
            try:
                self._global_path(next_round).write_bytes(weights_bytes)
            except OSError:
                pass

    def global_round(self) -> int:
        with self._lock:
            return self._global.round_no

    def global_version_label(self) -> str:
        with self._lock:
            return self._global.version_label

    def update_task_checkpoint(self, task_id: str, weights_bytes: bytes, last_index: int) -> None:
        with self._lock:
            self._per_task.setdefault(task_id, TaskCheckpoint())
            self._per_task[task_id].weights_bytes = weights_bytes
            self._per_task[task_id].last_index = last_index
            try:
                self._task_ckpt_path(task_id).write_bytes(weights_bytes)
            except OSError:
                pass

    def get_task_checkpoint_bytes(self, task_id: str) -> Optional[bytes]:
        with self._lock:
            ckpt = self._per_task.get(task_id)
            return copy.deepcopy(ckpt.weights_bytes) if ckpt and ckpt.weights_bytes else None

    def checkpoint_dir(self) -> str:
        return str(self._ckpt_dir)

    def get_task_resume_index(self, task_id: str) -> int:
        with self._lock:
            self._per_task.setdefault(task_id, TaskCheckpoint())
            return self._per_task[task_id].last_index

    def get_weights_for_assignment(self, task_id: str, prior_status: TaskStatus) -> Optional[bytes]:
        """
        Weights blob (torch.save state_dict) appropriate for this assignment.
        ORPHANED → last checkpoint; same-worker resync uses checkpoint if present; else global.
        """
        with self._lock:
            self._per_task.setdefault(task_id, TaskCheckpoint())
            ckpt = self._per_task[task_id]
            if prior_status == TaskStatus.ORPHANED and ckpt.weights_bytes is not None:
                return copy.deepcopy(ckpt.weights_bytes)
            if prior_status in (TaskStatus.IN_PROGRESS, TaskStatus.ASSIGNED) and ckpt.weights_bytes is not None:
                return copy.deepcopy(ckpt.weights_bytes)
            if self._global.global_weights_bytes is not None:
                return copy.deepcopy(self._global.global_weights_bytes)
            return None

    def collect_shard_weights_for_fedavg(self, task_ids: List[str]) -> List[bytes]:
        with self._lock:
            out: List[bytes] = []
            for tid in task_ids:
                ckpt = self._per_task.get(tid)
                if not ckpt or ckpt.weights_bytes is None:
                    raise ValueError(f"missing checkpoint for {tid}")
                out.append(copy.deepcopy(ckpt.weights_bytes))
            return out

    def reset_task_checkpoints(self, task_ids: List[str]) -> None:
        with self._lock:
            for tid in task_ids:
                self._per_task[tid] = TaskCheckpoint()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "round_no": self._global.round_no,
                "version_label": self._global.version_label,
                "has_global_weights": self._global.global_weights_bytes is not None,
                "last_val_acc": self._global.last_val_acc,
                "last_test_acc": self._global.last_test_acc,
                "tasks": {
                    tid: {
                        "has_checkpoint": v.weights_bytes is not None,
                        "last_index": v.last_index,
                    }
                    for tid, v in self._per_task.items()
                },
            }
