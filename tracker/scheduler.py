"""Task table: 10_000 images → 5 shards; heartbeat watchdog; assignments."""

from __future__ import annotations

import base64
import time
import uuid
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, List, Optional, Tuple

from shared.protocol import TaskAssignment, TaskResponse, TaskStatus

from .aggregator import fedavg_state_dicts
from .state_manager import StateManager


TOTAL_IMAGES = 10_000
NUM_SHARDS = 5
SHARD_SIZE = TOTAL_IMAGES // NUM_SHARDS
HEARTBEAT_TIMEOUT_SEC = 15.0


def _shard_bounds(idx: int) -> Tuple[int, int]:
    start = idx * SHARD_SIZE
    end = start + SHARD_SIZE
    return start, end


def _resume_next_index(image_start: int, image_end: int, last_consumed: int) -> int:
    if last_consumed < image_start:
        return image_start
    return max(image_start, min(last_consumed + 1, image_end - 1))


@dataclass
class TaskShard:
    task_id: str
    image_start: int
    image_end: int  # exclusive
    status: TaskStatus = TaskStatus.PENDING
    assigned_worker: Optional[str] = None
    last_heartbeat: float = 0.0
    last_reported_index: int = -1


@dataclass
class WorkerRecord:
    worker_id: str
    gpu_vram_mb: float
    cpu_count: int
    host_label: Optional[str] = None
    registered_at: float = field(default_factory=time.time)


class Scheduler:
    def __init__(self, state: StateManager) -> None:
        self._state = state
        self._lock = RLock()
        self._workers: Dict[str, WorkerRecord] = {}
        self._tasks: Dict[str, TaskShard] = {}
        for i in range(NUM_SHARDS):
            tid = f"shard-{i}"
            s, e = _shard_bounds(i)
            self._tasks[tid] = TaskShard(task_id=tid, image_start=s, image_end=e)
            self._state.ensure_task_slot(tid)

    def check_timeouts(self, now: Optional[float] = None) -> List[str]:
        now = now or time.time()
        orphaned: List[str] = []
        with self._lock:
            for tid, t in self._tasks.items():
                if t.status not in (TaskStatus.IN_PROGRESS, TaskStatus.ASSIGNED):
                    continue
                if t.assigned_worker and (now - t.last_heartbeat) > HEARTBEAT_TIMEOUT_SEC:
                    t.status = TaskStatus.ORPHANED
                    t.assigned_worker = None
                    orphaned.append(tid)
        return orphaned

    def register_worker(self, gpu_vram_mb: float, cpu_count: int, host_label: Optional[str]) -> str:
        self.check_timeouts()
        wid = str(uuid.uuid4())
        with self._lock:
            self._workers[wid] = WorkerRecord(
                worker_id=wid,
                gpu_vram_mb=gpu_vram_mb,
                cpu_count=cpu_count,
                host_label=host_label,
            )
        return wid

    def touch_heartbeat(self, worker_id: str, task_id: Optional[str]) -> bool:
        self.check_timeouts()
        now = time.time()
        with self._lock:
            if worker_id not in self._workers:
                return False
            if task_id:
                t = self._tasks.get(task_id)
                if (
                    t
                    and t.assigned_worker == worker_id
                    and t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
                ):
                    t.last_heartbeat = now
                    t.status = TaskStatus.IN_PROGRESS
            return True

    def _pick_task_for_worker(self) -> Optional[TaskShard]:
        for st in (TaskStatus.ORPHANED, TaskStatus.PENDING):
            for t in self._tasks.values():
                if t.status == st:
                    return t
        return None

    def _worker_active_task(self, worker_id: str) -> Optional[TaskShard]:
        for t in self._tasks.values():
            if t.assigned_worker != worker_id:
                continue
            if t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
                return t
        return None

    def request_task(self, worker_id: str) -> TaskResponse:
        self.check_timeouts()
        with self._lock:
            if worker_id not in self._workers:
                return TaskResponse(has_task=False, task=None, global_model_bytes_b64=None)

            existing = self._worker_active_task(worker_id)
            if existing:
                return self._build_task_response_locked(existing, existing.status)

            picked = self._pick_task_for_worker()
            if not picked:
                return TaskResponse(has_task=False, task=None, global_model_bytes_b64=None)

            prior_status = picked.status
            picked.assigned_worker = worker_id
            picked.status = TaskStatus.ASSIGNED
            picked.last_heartbeat = time.time()
            return self._build_task_response_locked(picked, prior_status)

    def _build_task_response_locked(self, t: TaskShard, prior_status: TaskStatus) -> TaskResponse:
        last_idx = self._state.get_task_resume_index(t.task_id)
        resume_next = _resume_next_index(t.image_start, t.image_end, last_idx)
        starting = self._state.get_weights_for_assignment(t.task_id, prior_status)
        b64 = base64.b64encode(starting).decode("ascii") if starting else None
        assign = TaskAssignment(
            task_id=t.task_id,
            image_start=t.image_start,
            image_end=t.image_end,
            exclusive_end=t.image_end,
            round_no=self._state.global_round(),
            resume_next_index=resume_next,
        )
        return TaskResponse(has_task=True, task=assign, global_model_bytes_b64=b64)

    def submit_weights(
        self,
        worker_id: str,
        task_id: str,
        weights_bytes: bytes,
        last_index: int,
        shard_complete: bool,
    ) -> Tuple[bool, str]:
        self.check_timeouts()
        with self._lock:
            t = self._tasks.get(task_id)
            if not t or t.assigned_worker != worker_id:
                return False, "task not assigned to worker"
            if t.status == TaskStatus.COMPLETED:
                return False, "task already completed"

            t.status = TaskStatus.IN_PROGRESS
            t.last_heartbeat = time.time()
            t.last_reported_index = last_index

        self._state.update_task_checkpoint(task_id, weights_bytes, last_index)

        with self._lock:
            t = self._tasks[task_id]
            if shard_complete:
                t.status = TaskStatus.COMPLETED
                t.assigned_worker = None

            if self._all_tasks_completed():
                return self._run_aggregation_locked()

        return True, "checkpoint accepted"

    def _all_tasks_completed(self) -> bool:
        return all(t.status == TaskStatus.COMPLETED for t in self._tasks.values())

    def _run_aggregation_locked(self) -> Tuple[bool, str]:
        try:
            buffers = self._state.collect_shard_weights_for_fedavg(sorted(self._tasks.keys()))
        except ValueError as e:
            return False, str(e)
        merged = fedavg_state_dicts(buffers)
        next_round = self._state.global_round() + 1
        self._state.set_global_bytes(merged, next_round)

        for t in self._tasks.values():
            t.status = TaskStatus.PENDING
            t.assigned_worker = None
            t.last_reported_index = -1
        self._state.reset_task_checkpoints(list(self._tasks.keys()))

        label = self._state.global_version_label()
        return True, f"aggregated to {label}"

    def health_snapshot(self) -> dict:
        self.check_timeouts()
        now = time.time()
        with self._lock:
            tasks = {
                tid: {
                    "status": t.status.value,
                    "worker": t.assigned_worker,
                    "range": [t.image_start, t.image_end],
                    "last_index": t.last_reported_index,
                    "heartbeat_age_sec": (
                        round(now - t.last_heartbeat, 3) if t.last_heartbeat else None
                    ),
                }
                for tid, t in self._tasks.items()
            }
            base = self._state.snapshot()
            return {"workers": len(self._workers), "task_table": tasks, **base}
