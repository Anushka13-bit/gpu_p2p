"""Task table: dataset indices → shards; heartbeat watchdog; assignments."""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

from shared.protocol import TaskAssignment, TaskResponse, TaskStatus

from .aggregator import fedavg_state_dicts
from .eval_utils import eval_global_fashion_mnist_test_acc, eval_global_fashion_mnist_val_acc
from . import learning_credits
from .state_manager import StateManager


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# We shard over a contiguous index range [0, TOTAL_IMAGES).
# For Fashion-MNIST, users often want the full 70k (train 60k + test 10k).
#
# Override via env if you want different sizing:
#   GPU_P2P_TOTAL_IMAGES=60000 GPU_P2P_NUM_SHARDS=12
TOTAL_IMAGES = max(1, _env_int("GPU_P2P_TOTAL_IMAGES", 70_000))
NUM_SHARDS = max(1, _env_int("GPU_P2P_NUM_SHARDS", 10))

# Use ceil division so shards cover the entire [0, TOTAL_IMAGES) range.
SHARD_SIZE = (TOTAL_IMAGES + NUM_SHARDS - 1) // NUM_SHARDS


# If a shard assignee stops heartbeating, ORPHANED after this many seconds (then another worker
# can pick it up). Must be > worker ``--heartbeat-sec`` (default 3); brief CPU/GIL stalls can
# delay heartbeats — use a higher value via env on flaky hosts.
#
# Tune:
#   GPU_P2P_HEARTBEAT_TIMEOUT_SEC=15   # faster reassignment (demo LAN)
#   GPU_P2P_HEARTBEAT_TIMEOUT_SEC=60   # conservative (heavy CPU training)
HEARTBEAT_TIMEOUT_SEC = max(5.0, _env_float("GPU_P2P_HEARTBEAT_TIMEOUT_SEC", 20.0))

REGISTRY_DISPLAY_INTERVAL_SEC = 15.0

# How often the tracker supervisor calls check_timeouts(). Lower = detect orphan slightly sooner
# after the timeout elapses.
WATCHDOG_CHECK_INTERVAL_SEC = max(0.5, _env_float("GPU_P2P_WATCHDOG_INTERVAL_SEC", 1.0))


# Resource gating for shard assignment (to prefer GPU-capable nodes).
# Note: tracker only knows *reported* specs at registration time, not live "available/free VRAM".
GPU_ONLY = _env_int("GPU_P2P_GPU_ONLY", 0) == 1
MIN_VRAM_MB = max(0.0, float(_env_int("GPU_P2P_MIN_VRAM_MB", 1)))
MIN_THREADS = max(1, _env_int("GPU_P2P_MIN_THREADS", 1))
REQUIRE_CUDA_TORCH = _env_int("GPU_P2P_REQUIRE_CUDA_TORCH", 0) == 1


def _worker_meets_resource_requirements(w: "WorkerRecord") -> bool:
    """
    Decide whether the tracker should even assign a shard to this worker.
    Uses the worker's registration-time reported specs.
    """
    if not GPU_ONLY:
        return True

    if w.gpu_vram_mb < MIN_VRAM_MB:
        return False

    if w.cpu_count < MIN_THREADS:
        return False

    if REQUIRE_CUDA_TORCH:
        # hardware_report comes from shared.hardware_sniff.build_hardware_report()
        # and includes cuda_available based on torch.cuda.is_available().
        hw = w.hardware_report or {}
        if bool(hw.get("cuda_available")) is not True:
            return False

    return True


def _shard_bounds(idx: int) -> Tuple[int, int]:
    start = idx * SHARD_SIZE
    end = min(TOTAL_IMAGES, start + SHARD_SIZE)
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
    completed_by_worker_id: Optional[str] = None


@dataclass
class WorkerRecord:
    worker_id: str
    gpu_vram_mb: float
    cpu_count: int
    host_label: Optional[str] = None
    hardware_report: Optional[dict[str, Any]] = None
    registered_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    # Proof-of-Learning credits (tracker-side; in-memory)
    credits_total: float = 0.0
    reputation: float = 50.0
    positive_streak_rounds: int = 0
    credit_events_count: int = 0

    @property
    def compute_tier(self) -> str:
        if self.gpu_vram_mb >= 8000:
            return "Tier 1 (High-End GPU)"
        elif self.gpu_vram_mb >= 2000:
            return "Tier 2 (Standard GPU)"
        else:
            return "Tier 3 (Edge/CPU Fallback)"


class Scheduler:
    def __init__(self, state: StateManager) -> None:
        self._state = state
        self._lock = RLock()
        self._workers: Dict[str, WorkerRecord] = {}
        self._tasks: Dict[str, TaskShard] = {}
        # Demo-friendly default: run a single FedAvg round then stop.
        # Override with GPU_P2P_MAX_FED_ROUNDS=0 for unlimited rounds.
        self._max_fed_rounds = max(0, _env_int("GPU_P2P_MAX_FED_ROUNDS", 1))
        self._earlystop_patience = max(0, _env_int("GPU_P2P_EARLYSTOP_PATIENCE", 3))
        self._earlystop_min_delta = max(0.0, _env_float("GPU_P2P_EARLYSTOP_MIN_DELTA", 0.1))
        self._training_stopped = False
        self._stop_reason: Optional[str] = None
        self._best_val_acc: Optional[float] = None
        self._rounds_without_improve = 0
        # Latest per-worker per-task live progress (sent per epoch).
        self._progress: Dict[tuple[str, str], dict[str, Any]] = {}
        self._credit_events: list[dict[str, Any]] = []
        for i in range(NUM_SHARDS):
            tid = f"shard-{i}"
            s, e = _shard_bounds(i)
            self._tasks[tid] = TaskShard(task_id=tid, image_start=s, image_end=e)
            self._state.ensure_task_slot(tid)
        print(
            "[scheduler] stop policy: "
            f"max_fed_rounds={self._max_fed_rounds or 'unlimited'} "
            f"earlystop_patience={self._earlystop_patience or 'disabled'} "
            f"earlystop_min_delta={self._earlystop_min_delta:.4f}",
            flush=True,
        )
        print(
            f"[scheduler] liveness: heartbeat_timeout={HEARTBEAT_TIMEOUT_SEC:.1f}s "
            f"watchdog_every={WATCHDOG_CHECK_INTERVAL_SEC:.1f}s (set GPU_P2P_HEARTBEAT_TIMEOUT_SEC / "
            f"GPU_P2P_WATCHDOG_INTERVAL_SEC to tune orphan → reschedule latency)",
            flush=True,
        )

    def _append_credit_event(self, worker_id: str, breakdown: learning_credits.CreditBreakdown) -> None:
        d = breakdown.as_dict()
        d["ts"] = time.time()
        d["worker_id"] = worker_id
        self._credit_events.append(d)
        if len(self._credit_events) > 400:
            self._credit_events = self._credit_events[-400:]

    def _apply_credit_breakdown(self, worker_id: str, breakdown: learning_credits.CreditBreakdown) -> None:
        w = self._workers.get(worker_id)
        if not w:
            return
        w.credits_total += breakdown.credits
        w.credit_events_count += 1
        w.reputation = learning_credits.update_reputation(w.reputation, breakdown.credits)
        self._append_credit_event(worker_id, breakdown)

    def credit_snapshot(self) -> dict[str, Any]:
        """Leaderboard + recent PoL events (for GET /credits, dashboard)."""
        with self._lock:
            board = sorted(
                (
                    {
                        "worker_id": w.worker_id,
                        "host_label": w.host_label,
                        "credits_total": round(w.credits_total, 4),
                        "reputation": round(w.reputation, 2),
                        "positive_streak_rounds": w.positive_streak_rounds,
                        "credit_events_count": w.credit_events_count,
                    }
                    for w in self._workers.values()
                ),
                key=lambda r: r["credits_total"],
                reverse=True,
            )
            return {
                "leaderboard": board,
                "recent_events": list(self._credit_events[-80:]),
            }

    def _stop_training_locked(self, reason: str) -> None:
        if self._training_stopped:
            return
        self._training_stopped = True
        self._stop_reason = reason
        print(f"[training] stop requested: {reason}", flush=True)

    def is_training_stopped(self) -> bool:
        with self._lock:
            return self._training_stopped

    def reset_session(self) -> dict[str, Any]:
        """
        Start a new training session on this tracker process: shards → PENDING,
        training gate reopened, global model and checkpoints cleared, roster cleared.

        Workers must register again (new worker_id + ticket).
        """
        with self._lock:
            self._training_stopped = False
            self._stop_reason = None
            self._best_val_acc = None
            self._rounds_without_improve = 0
            self._progress.clear()
            self._credit_events.clear()
            self._workers.clear()
            for t in self._tasks.values():
                t.status = TaskStatus.PENDING
                t.assigned_worker = None
                t.last_heartbeat = 0.0
                t.last_reported_index = -1
                t.last_eval_acc = None
                t.last_epochs_completed = None
                t.last_epochs_planned = None
                t.completed_by_worker_id = None
            tids = list(self._tasks.keys())
            self._state.reset_for_new_session(tids)
        print("[scheduler] session reset: shards PENDING, training_stopped cleared, workers cleared", flush=True)
        return {"ok": True, "shards": tids}

    def update_progress(
        self,
        worker_id: str,
        task_id: str,
        local_epoch: int,
        local_epochs_total: int,
        shard_progress_pct: Optional[float],
        train_acc_running: Optional[float],
        train_loss_last: Optional[float],
        ts: float,
    ) -> None:
        """Called by /progress endpoint; used for live tracker registry rendering."""
        with self._lock:
            self._progress[(worker_id, task_id)] = {
                "local_epoch": int(local_epoch),
                "local_epochs_total": int(local_epochs_total),
                "shard_progress_pct": float(shard_progress_pct) if shard_progress_pct is not None else None,
                "train_acc_running": float(train_acc_running) if train_acc_running is not None else None,
                "train_loss_last": float(train_loss_last) if train_loss_last is not None else None,
                "ts": float(ts),
            }

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
        worker_id: str,
        gpu_vram_mb: float,
        cpu_count: int,
        host_label: Optional[str],
        hardware_report: Optional[dict[str, Any]] = None,
    ) -> str:
        self.check_timeouts()
        with self._lock:
            now = time.time()
            if worker_id in self._workers:
                raise ValueError("worker_id already registered")
            self._workers[worker_id] = WorkerRecord(
                worker_id=worker_id,
                gpu_vram_mb=gpu_vram_mb,
                cpu_count=cpu_count,
                host_label=host_label,
                hardware_report=hardware_report,
                registered_at=now,
                last_seen=now,
            )
        return worker_id

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

    def _pick_task_for_worker(self, worker_id: str) -> Optional[TaskShard]:
        """Prefer higher-rarity shards for higher-reputation workers (more responsibility / upside)."""
        candidates: list[TaskShard] = []
        for st in (TaskStatus.ORPHANED, TaskStatus.PENDING):
            for t in self._tasks.values():
                if t.status == st:
                    candidates.append(t)
        if not candidates:
            return None
        wrec = self._workers.get(worker_id)
        # If we know the worker doesn't satisfy resource gating, don't assign tasks at all.
        if wrec and not _worker_meets_resource_requirements(wrec):
            return None
        rep_val = wrec.reputation if wrec else 50.0
        high_trust = rep_val >= 50.0
        candidates.sort(
            key=lambda t: learning_credits.shard_rarity_multiplier(t.task_id),
            reverse=high_trust,
        )
        return candidates[0]

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

            if self._training_stopped:
                return TaskResponse(has_task=False, task=None, global_model_bytes_b64=None)

            existing = self._worker_active_task(worker_id)
            if existing:
                return self._build_task_response_locked(existing, existing.status)

            picked = self._pick_task_for_worker(worker_id)
            if not picked:
                return TaskResponse(has_task=False, task=None, global_model_bytes_b64=None)

            prior_status = picked.status
            picked.assigned_worker = worker_id
            picked.status = TaskStatus.ASSIGNED
            picked.last_heartbeat = time.time()
            return self._build_task_response_locked(picked, prior_status)

    def _build_task_response_locked(self, t: TaskShard, prior_status: TaskStatus) -> TaskResponse:
        # Ensure all workers start from the same initial global model in round 1.
        self._state.ensure_initial_global()
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
        steps_completed: int = 0,
        train_acc_running: Optional[float] = None,
    ) -> Tuple[bool, str, dict[str, Any]]:
        extras: dict[str, Any] = {}
        self.check_timeouts()
        baseline: Optional[float]
        rep = 50.0
        streak = 0
        with self._lock:
            t = self._tasks.get(task_id)
            if not t or t.assigned_worker != worker_id:
                return False, "task not assigned to worker", {}
            if t.status == TaskStatus.COMPLETED:
                return False, "task already completed", {}

            t.status = TaskStatus.IN_PROGRESS
            t.last_heartbeat = time.time()
            t.last_reported_index = last_index
            if shard_eval_acc is not None:
                t.last_eval_acc = float(shard_eval_acc)
            if local_epochs_planned is not None:
                t.last_epochs_planned = int(local_epochs_planned)
            if local_epochs_completed is not None:
                t.last_epochs_completed = int(local_epochs_completed)
            baseline = self._state.snapshot().get("last_val_acc")
            wrec = self._workers.get(worker_id)
            if wrec:
                rep = wrec.reputation
                streak = wrec.positive_streak_rounds

        self._state.update_task_checkpoint(task_id, weights_bytes, last_index)

        if steps_completed > 0:
            bd = learning_credits.interim_submit_credit(
                baseline_val_acc=baseline,
                shard_eval_acc=shard_eval_acc,
                steps_completed=steps_completed,
                task_id=task_id,
                reputation=rep,
                positive_streak=streak,
                train_acc_running=train_acc_running,
            )
            if bd.phase != "interim_skip":
                with self._lock:
                    self._apply_credit_breakdown(worker_id, bd)
                extras["learning_credit"] = bd.as_dict()

        with self._lock:
            t = self._tasks[task_id]
            if shard_complete:
                t.completed_by_worker_id = worker_id
                t.status = TaskStatus.COMPLETED
                t.assigned_worker = None

            if self._all_tasks_completed():
                ok, msg, agg_extras = self._run_aggregation_locked()
                extras.update(agg_extras)
                return ok, msg, extras

        return True, "checkpoint accepted", extras

    def _all_tasks_completed(self) -> bool:
        return all(t.status == TaskStatus.COMPLETED for t in self._tasks.values())

    def _run_aggregation_locked(self) -> Tuple[bool, str, dict[str, Any]]:
        old_val = self._state.snapshot().get("last_val_acc")
        shard_rows: list[dict[str, Any]] = []
        for tid, t in self._tasks.items():
            if t.completed_by_worker_id:
                shard_rows.append(
                    {
                        "worker_id": t.completed_by_worker_id,
                        "task_id": tid,
                        "eval_acc": t.last_eval_acc,
                    }
                )
        try:
            buffers = self._state.collect_shard_weights_for_fedavg(sorted(self._tasks.keys()))
        except ValueError as e:
            return False, str(e), {}
        nbuf = len(buffers)
        merged = fedavg_state_dicts(buffers)
        next_round = self._state.global_round() + 1
        val_acc: Optional[float] = None
        test_acc: Optional[float] = None
        try:
            val_acc = float(eval_global_fashion_mnist_val_acc(merged, device="cpu"))
            test_acc = float(eval_global_fashion_mnist_test_acc(merged, device="cpu"))
        except Exception as e:
            print(f"[eval] failed: {e!r}", flush=True)
        self._state.set_global_bytes(merged, next_round, val_acc=val_acc, test_acc=test_acc)

        round_summary: dict[str, Any] = {}
        dist = learning_credits.round_pool_distribution(
            old_val_acc=old_val,
            new_val_acc=val_acc,
            shard_rows=shard_rows,
        )
        for wid, breakdown in dist.items():
            if breakdown.phase not in ("round_skip",):
                self._apply_credit_breakdown(wid, breakdown)
            w = self._workers.get(wid)
            if w and breakdown.phase not in ("round_skip",):
                if breakdown.credits > 0.05:
                    w.positive_streak_rounds += 1
                elif breakdown.credits < -0.05:
                    w.positive_streak_rounds = 0
            round_summary[wid] = breakdown.as_dict()

        completed_rounds = max(0, next_round - 1)
        # Early-stop is based on validation accuracy, not test accuracy.
        if val_acc is not None:
            if self._best_val_acc is None:
                self._best_val_acc = val_acc
                self._rounds_without_improve = 0
            else:
                improvement = val_acc - self._best_val_acc
                if improvement >= self._earlystop_min_delta:
                    self._best_val_acc = val_acc
                    self._rounds_without_improve = 0
                else:
                    self._rounds_without_improve += 1

        if self._max_fed_rounds > 0 and completed_rounds >= self._max_fed_rounds:
            self._stop_training_locked(
                f"max_fed_rounds reached ({completed_rounds}/{self._max_fed_rounds})"
            )

        if (
            (not self._training_stopped)
            and self._earlystop_patience > 0
            and val_acc is not None
            and self._best_val_acc is not None
            and self._rounds_without_improve >= self._earlystop_patience
        ):
            self._stop_training_locked(
                "early-stop: no meaningful global val_acc improvement "
                f"for {self._rounds_without_improve} rounds "
                f"(min_delta={self._earlystop_min_delta:.4f})"
            )

        if not self._training_stopped:
            for t in self._tasks.values():
                t.status = TaskStatus.PENDING
                t.assigned_worker = None
                t.last_reported_index = -1
                t.completed_by_worker_id = None
            self._state.reset_task_checkpoints(list(self._tasks.keys()))

        label = self._state.global_version_label()
        print(
            f"[fedavg] averaged {nbuf} shard weight tensors (element-wise mean of float params) → {label}",
            flush=True,
        )
        if val_acc is not None or test_acc is not None:
            vs = f"{val_acc:.2f}%" if val_acc is not None else "n/a"
            ts = f"{test_acc:.2f}%" if test_acc is not None else "n/a"
            print(f"[eval] {label} fashion-mnist val_acc={vs}  test_acc={ts}", flush=True)
        if self._training_stopped and self._stop_reason:
            print(f"[training] terminal condition met; no further tasks will be scheduled ({self._stop_reason})", flush=True)
        agg_extras: dict[str, Any] = {
            "aggregation": True,
            "round_credits": round_summary,
            "old_val_acc": old_val,
            "new_val_acc": val_acc,
            "global_val_delta": (
                round(float(val_acc) - float(old_val), 4)
                if val_acc is not None and old_val is not None
                else None
            ),
        }
        return True, f"aggregated to {label}", agg_extras

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
                live = self._progress.get((w.worker_id, shard)) if shard else None
                rows.append(
                    {
                        "worker_id": w.worker_id,
                        "host_label": w.host_label,
                        "registered_at": w.registered_at,
                        "compute_tier": w.compute_tier,
                        "last_seen_age_sec": round(age, 1),
                        "alive": is_live,
                        "credits_total": round(w.credits_total, 4),
                        "reputation": round(w.reputation, 2),
                        "positive_streak_rounds": w.positive_streak_rounds,
                        "current_shard": shard,
                        "current_shard_progress_pct": shard_prog.get("progress_pct") if shard_prog else None,
                        "current_shard_last_index": shard_prog.get("last_index") if shard_prog else None,
                        "current_shard_eval_acc": shard_prog.get("eval_acc") if shard_prog else None,
                        "current_shard_epochs": shard_prog.get("epochs") if shard_prog else None,
                        "live_epoch": live.get("local_epoch") if live else None,
                        "live_epoch_total": live.get("local_epochs_total") if live else None,
                        "live_progress_pct": live.get("shard_progress_pct") if live else None,
                        "live_train_acc": live.get("train_acc_running") if live else None,
                        "live_ts_age_sec": round(now - live.get("ts"), 1) if live and live.get("ts") else None,
                    }
                )
            rows.sort(key=lambda r: r["registered_at"])
            on_task = sum(1 for r in rows if r["current_shard"] is not None)
            return {
                "total_nodes": len(self._workers),
                "active_nodes": active,
                "nodes_on_shard": on_task,
                "heartbeat_timeout_sec": HEARTBEAT_TIMEOUT_SEC,
                "training_stopped": self._training_stopped,
                "stop_reason": self._stop_reason,
                "best_val_acc": self._best_val_acc,
                "rounds_without_improve": self._rounds_without_improve,
                "learning_credits": self.credit_snapshot(),
                "stop_policy": {
                    "max_fed_rounds": self._max_fed_rounds,
                    "earlystop_patience": self._earlystop_patience,
                    "earlystop_min_delta": self._earlystop_min_delta,
                },
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
        stop_state = "STOPPED" if snap.get("training_stopped") else "running"
        lines.append(
            "  training="
            f"{stop_state} "
            f"policy(max_rounds={snap.get('stop_policy', {}).get('max_fed_rounds', 0) or 'unlimited'}, "
            f"patience={snap.get('stop_policy', {}).get('earlystop_patience', 0) or 'disabled'}, "
            f"min_delta={snap.get('stop_policy', {}).get('earlystop_min_delta', 0.0):.4f})"
        )
        if snap.get("best_val_acc") is not None:
            lines.append(
                f"  best_val_acc={snap['best_val_acc']:.2f}%  rounds_without_improve={snap.get('rounds_without_improve', 0)}"
            )
        if snap.get("training_stopped") and snap.get("stop_reason"):
            lines.append(f"  stop_reason={snap['stop_reason']}")
        for r in snap["nodes"]:
            shard = r["current_shard"] or "—"
            lbl = repr(r["host_label"])[1:-1] if r["host_label"] else "—"
            alive = "alive" if r["alive"] else "STALE"
            short_id = r["worker_id"][:8]
            # Prefer live per-epoch progress if available; else fall back to submit-based progress.
            live_pct = r.get("live_progress_pct")
            prog = live_pct if isinstance(live_pct, (int, float)) else r["current_shard_progress_pct"]
            prog_s = f"{prog:.1f}%" if isinstance(prog, (int, float)) else "—"
            ep_s = r.get("current_shard_epochs") or "—"

            # Epoch live "bar" removed (still tracked internally).
            live_ep = r.get("live_epoch")
            live_total = r.get("live_epoch_total")
            if isinstance(live_ep, int) and isinstance(live_total, int) and live_total > 0:
                ep_live_s = f"{live_ep}/{live_total}"
            else:
                ep_live_s = "—"
            lines.append(
                f"    [{alive}] {short_id}…  host={lbl}  tier={r.get('compute_tier', 'Unknown')}  last_seen={r['last_seen_age_sec']}s  "
                f"shard={shard}  progress={prog_s}  epochs={ep_live_s}"
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
                    "compute_tier": w.compute_tier,
                    "registered_at": w.registered_at,
                    "last_seen_age_sec": round(now - w.last_seen, 3),
                    "hardware_report": w.hardware_report,
                    "credits_total": round(w.credits_total, 4),
                    "reputation": round(w.reputation, 2),
                    "positive_streak_rounds": w.positive_streak_rounds,
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
                "training_stopped": self._training_stopped,
                "stop_reason": self._stop_reason,
                "best_val_acc": self._best_val_acc,
                "rounds_without_improve": self._rounds_without_improve,
                "learning_credits": self.credit_snapshot(),
                "stop_policy": {
                    "max_fed_rounds": self._max_fed_rounds,
                    "earlystop_patience": self._earlystop_patience,
                    "earlystop_min_delta": self._earlystop_min_delta,
                },
                "task_table": tasks,
                **base,
            }
