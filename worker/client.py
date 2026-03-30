"""Register with tracker, heartbeat, poll tasks, submit in-memory weights."""

from __future__ import annotations

import io
import time
from typing import Any, Mapping, Optional

import requests

from shared.hardware_sniff import sniff_register_tuple
from shared.protocol import (
    HeartbeatRequest,
    LogEvent,
    RegisterRequest,
    RegisterResponse,
    SubmitWeightsMetadata,
    TaskResponse,
)


def encode_task_for_container(task: TaskResponse) -> str:
    """Serialize tracker task + optional weights for ``trainer_wrapper.ContainerTaskPayload``."""
    import json

    if not task.has_task or task.task is None:
        raise ValueError("no task assigned")
    payload = {
        "assignment": task.task.model_dump(mode="json"),
        "weights_b64": task.global_model_bytes_b64,
    }
    return json.dumps(payload)


class TrackerClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def register(
        self,
        gpu_vram_mb: float,
        cpu_count: int,
        host_label: str | None = None,
        hardware_report: Mapping[str, Any] | None = None,
    ) -> RegisterResponse:
        body = RegisterRequest(
            gpu_vram_mb=gpu_vram_mb,
            cpu_count=cpu_count,
            host_label=host_label,
            hardware_report=dict(hardware_report) if hardware_report is not None else None,
        )
        r = self.session.post(
            f"{self.base_url}/register",
            json=body.model_dump(),
            timeout=self.timeout,
        )
        r.raise_for_status()
        return RegisterResponse.model_validate(r.json())

    def heartbeat(self, worker_id: str, task_id: str | None = None) -> bool:
        body = HeartbeatRequest(worker_id=worker_id, task_id=task_id)
        r = self.session.post(
            f"{self.base_url}/heartbeat",
            json=body.model_dump(),
            timeout=self.timeout,
        )
        if r.status_code == 404:
            return False
        r.raise_for_status()
        return bool(r.json().get("ok"))

    def request_task(self, worker_id: str) -> TaskResponse:
        r = self.session.get(f"{self.base_url}/task/{worker_id}", timeout=self.timeout)
        r.raise_for_status()
        return TaskResponse.model_validate(r.json())

    def submit_weights(
        self,
        worker_id: str,
        task_id: str,
        weights_bytes: bytes,
        last_index: int,
        steps_completed: int,
        shard_complete: bool,
        train_loss_last: float | None = None,
        train_acc_running: float | None = None,
        shard_eval_acc: float | None = None,
    ) -> dict[str, Any]:
        meta = SubmitWeightsMetadata(
            worker_id=worker_id,
            task_id=task_id,
            last_index=last_index,
            steps_completed=steps_completed,
            shard_complete=shard_complete,
            train_loss_last=train_loss_last,
            train_acc_running=train_acc_running,
            shard_eval_acc=shard_eval_acc,
        )
        files = {
            "weights_file": ("weights.pt", io.BytesIO(weights_bytes), "application/octet-stream"),
        }
        data = {"meta_json": meta.model_dump_json()}
        r = self.session.post(
            f"{self.base_url}/submit_weights",
            data=data,
            files=files,
            timeout=self.timeout,
        )
        if not r.ok:
            raise RuntimeError(f"submit_weights failed: {r.status_code} {r.text}")
        return r.json()

    def log_event(self, worker_id: str, message: str, level: str = "INFO", task_id: str | None = None, host_label: str | None = None) -> None:
        body = LogEvent(worker_id=worker_id, host_label=host_label, task_id=task_id, level=level, message=message, ts=time.time())
        r = self.session.post(f"{self.base_url}/log", json=body.model_dump(), timeout=self.timeout)
        if not r.ok:
            # Best-effort; don't crash training if logging fails.
            return


def sniff_hardware_defaults() -> tuple[float, int]:
    """``(gpu_vram_mb, cpu_count)`` for ``/register`` — NVIDIA VRAM or Apple unified heuristic."""
    return sniff_register_tuple()
