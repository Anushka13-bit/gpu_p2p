"""Task table: 10_000 images → 5 shards; heartbeat watchdog; assignments."""

from __future__ import annotations

import base64
import time
import uuid
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

from shared.protocol import TaskAssignment, TaskResponse, TaskStatus

from .aggregator import fedavg_state_dicts
from .state_manager import StateManager


TOTAL_IMAGES = 10_000
NUM_SHARDS = 5
SHARD_SIZE = TOTAL_IMAGES // NUM_SHARDS
# Must exceed worker heartbeat interval (default 3s) so brief jitter does not orphan tasks.
HEARTBEAT_TIMEOUT_SEC = 12.0
REGISTRY_DISPLAY_INTERVAL_SEC = 15.0
WATCHDOG_CHECK_INTERVAL_SEC = 3.0


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
    last_eval_acc: Optional[float] = None
    last_epochs_completed: Optional[int] = None
    last_epochs_planned: Optional[int] = None


@dataclass
class WorkerRecord:
    worker_id: str
    gpu_vram_mb: float
    cpu_count: int
    host_label: Optional[str] = None
    hardware_report: Optional[dict[str, Any]] = None
    registered_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


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

    def register_worker(
        self,
        gpu_vram_mb: float,
        cpu_count: int,
        host_label: Optional[str],
        hardware_report: Optional[dict[str, Any]] = None,
    ) -> str:
        self.check_timeouts()
        wid = str(uuid.uuid4())
        with self._lock:
            now = time.time()
            self._workers[wid] = WorkerRecord(
                worker_id=wid,
                gpu_vram_mb=gpu_vram_mb,
                cpu_count=cpu_count,
                host_label=host_label,
                hardware_report=hardware_report,
                registered_at=now,
                last_seen=now,
            )
        return wid

    def touch_heartbeat(self, worker_id: str, task_id: Optional[str]) -> bool:
        self.check_timeouts()
        now = time.time()
        with self._lock:
            if worker_id not in self._workers:
                return False
            self._workers[worker_id].last_seen = now
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
        shard_eval_acc: Optional[float] = None,
        local_epochs_planned: Optional[int] = None,
        local_epochs_completed: Optional[int] = None,
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
            if shard_eval_acc is not None:
                t.last_eval_acc = float(shard_eval_acc)
            if local_epochs_planned is not None:
                t.last_epochs_planned = int(local_epochs_planned)
            if local_epochs_completed is not None:
                t.last_epochs_completed = int(local_epochs_completed)

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
        nbuf = len(buffers)
        merged = fedavg_state_dicts(buffers)
        next_round = self._state.global_round() + 1
        self._state.set_global_bytes(merged, next_round)

        for t in self._tasks.values():
            t.status = TaskStatus.PENDING
            t.assigned_worker = None
            t.last_reported_index = -1
        self._state.reset_task_checkpoints(list(self._tasks.keys()))

        label = self._state.global_version_label()
        print(
            f"[fedavg] averaged {nbuf} shard weight tensors (element-wise mean of float params) → {label}",
            flush=True,
        )
        return True, f"aggregated to {label}"

    def worker_current_shard(self, worker_id: str) -> Optional[str]:
        with self._lock:
            for tid, t in self._tasks.items():
                if t.assigned_worker == worker_id and t.status in (
                    TaskStatus.ASSIGNED,
                    TaskStatus.IN_PROGRESS,
                ):
                    return tid
            return None

    def _current_shard_locked(self, worker_id: str) -> Optional[str]:
        for tid, t in self._tasks.items():
            if t.assigned_worker == worker_id and t.status in (
                TaskStatus.ASSIGNED,
                TaskStatus.IN_PROGRESS,
            ):
                return tid
        return None

    def registry_snapshot(self, now: Optional[float] = None) -> dict[str, Any]:
        """Counts and rows for terminal / API; active = heartbeat within timeout."""
        now = now or time.time()
        self.check_timeouts()
        with self._lock:
            task_prog: dict[str, dict[str, Any]] = {}
            for tid, t in self._tasks.items():
                denom = max(1, t.image_end - t.image_start)
                if t.last_reported_index < t.image_start:
                    pct = 0.0
                else:
                    pct = 100.0 * (min(t.last_reported_index, t.image_end - 1) - t.image_start + 1) / denom
                task_prog[tid] = {
                    "status": t.status.value,
                    "worker": t.assigned_worker,
                    "range": [t.image_start, t.image_end],
                    "last_index": t.last_reported_index,
                    "progress_pct": round(pct, 2),
                    "eval_acc": t.last_eval_acc,
                    "epochs": (
                        f"{t.last_epochs_completed}/{t.last_epochs_planned}"
                        if t.last_epochs_completed is not None and t.last_epochs_planned is not None
                        else None
                    ),
                }

            rows: list[dict[str, Any]] = []
            active = 0
            for w in self._workers.values():
                age = now - w.last_seen
                is_live = age <= HEARTBEAT_TIMEOUT_SEC
                if is_live:
                    active += 1
                shard = self._current_shard_locked(w.worker_id)
                shard_prog = task_prog.get(shard) if shard else None
                rows.append(
                    {
                        "worker_id": w.worker_id,
                        "host_label": w.host_label,
                        "registered_at": w.registered_at,
                        "last_seen_age_sec": round(age, 1),
                        "alive": is_live,
                        "current_shard": shard,
                        "current_shard_progress_pct": shard_prog.get("progress_pct") if shard_prog else None,
                        "current_shard_last_index": shard_prog.get("last_index") if shard_prog else None,
                        "current_shard_eval_acc": shard_prog.get("eval_acc") if shard_prog else None,
                        "current_shard_epochs": shard_prog.get("epochs") if shard_prog else None,
                    }
                )
            rows.sort(key=lambda r: r["registered_at"])
            on_task = sum(1 for r in rows if r["current_shard"] is not None)
            return {
                "total_nodes": len(self._workers),
                "active_nodes": active,
                "nodes_on_shard": on_task,
                "heartbeat_timeout_sec": HEARTBEAT_TIMEOUT_SEC,
                "nodes": rows,
                "task_table": task_prog,
            }

    def format_registry_terminal(self, now: Optional[float] = None) -> str:
        snap = self.registry_snapshot(now)
        lines = [
            "",
            f"——— node registry @ {time.strftime('%H:%M:%S')} ———",
            f"  total_nodes={snap['total_nodes']}  active_nodes(≤{snap['heartbeat_timeout_sec']:.0f}s since heartbeat)={snap['active_nodes']}  "
            f"nodes_with_assigned_shard={snap['nodes_on_shard']}",
            f"  data: {TOTAL_IMAGES} MNIST indices → {NUM_SHARDS} shards × {SHARD_SIZE} rows (distinct workers take distinct pending shards)",
        ]
        for r in snap["nodes"]:
            shard = r["current_shard"] or "—"
            lbl = repr(r["host_label"])[1:-1] if r["host_label"] else "—"
            alive = "alive" if r["alive"] else "STALE"
            short_id = r["worker_id"][:8]
            prog = r["current_shard_progress_pct"]
            prog_s = f"{prog:.1f}%" if isinstance(prog, (int, float)) else "—"
            ep_s = r.get("current_shard_epochs") or "—"
            acc = r.get("current_shard_eval_acc")
            acc_s = f"{acc:.1f}%" if isinstance(acc, (int, float)) else "—"
            lines.append(
                f"    [{alive}] {short_id}…  host={lbl}  last_seen={r['last_seen_age_sec']}s  "
                f"shard={shard}  progress={prog_s}  epochs={ep_s}  acc={acc_s}"
            )
        lines.append("  shard summary:")
        for tid, info in snap["task_table"].items():
            lines.append(
                f"    {tid}: {info['status']:<10}  worker={info['worker'] or '—':<36}  "
                f"progress={info['progress_pct']:>6}%  last_index={info['last_index']}"
            )
        lines.append("———" * 10)
        return "\n".join(lines)

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
            roster = [
                {
                    "worker_id": w.worker_id,
                    "gpu_vram_mb": w.gpu_vram_mb,
                    "cpu_count": w.cpu_count,
                    "host_label": w.host_label,
                    "registered_at": w.registered_at,
                    "last_seen_age_sec": round(now - w.last_seen, 3),
                    "hardware_report": w.hardware_report,
                }
                for w in self._workers.values()
            ]
            reg = {
                "total_nodes": len(self._workers),
                "active_nodes": sum(1 for w in self._workers.values() if (now - w.last_seen) <= HEARTBEAT_TIMEOUT_SEC),
                "heartbeat_timeout_sec": HEARTBEAT_TIMEOUT_SEC,
            }
            return {
                "workers": len(self._workers),
                "worker_roster": roster,
                "node_registry": reg,
                "task_table": tasks,
                **base,
            }
