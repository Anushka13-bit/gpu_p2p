"""
Central control plane — FastAPI tracker.
Run from repo root: PYTHONPATH=. uvicorn tracker.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from shared.protocol import (
    HealthResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    LogEvent,
    ProgressEvent,
    RegisterRequest,
    RegisterResponse,
    SubmitWeightsMetadata,
    TaskResponse,
)

from .scheduler import REGISTRY_DISPLAY_INTERVAL_SEC, WATCHDOG_CHECK_INTERVAL_SEC, Scheduler
from .state_manager import StateManager

state_manager = StateManager()
scheduler = Scheduler(state_manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async def _registry_supervisor():
        last_print = 0.0
        while True:
            await asyncio.sleep(WATCHDOG_CHECK_INTERVAL_SEC)
            orphaned = scheduler.check_timeouts()
            if orphaned:
                print(
                    f"[watchdog] reassigned/orphaned (assigned shard had no heartbeat in time): {orphaned}",
                    flush=True,
                )
            now = time.time()
            if (now - last_print) >= REGISTRY_DISPLAY_INTERVAL_SEC:
                print(scheduler.format_registry_terminal(now), flush=True)
                last_print = now

    task = asyncio.create_task(_registry_supervisor())
    print(
        f"[tracker] supervisor: registry + watchdog every {REGISTRY_DISPLAY_INTERVAL_SEC:.0f}s "
        f"(see scheduler.HEARTBEAT_TIMEOUT_SEC for liveness window)",
        flush=True,
    )
    print(scheduler.format_registry_terminal(time.time()), flush=True)
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="GPU Tracker", lifespan=lifespan)

_LOG_COUNTS: dict[str, int] = {}


@app.post("/register", response_model=RegisterResponse)
async def register(body: RegisterRequest) -> RegisterResponse:
    wid = scheduler.register_worker(
        gpu_vram_mb=body.gpu_vram_mb,
        cpu_count=body.cpu_count,
        host_label=body.host_label,
        hardware_report=body.hardware_report,
    )
    print(
        f"[register] worker_id={wid} gpu_vram_mb={body.gpu_vram_mb} "
        f"cpu_count={body.cpu_count} host_label={body.host_label!r}",
        flush=True,
    )
    if body.hardware_report:
        print(json.dumps(body.hardware_report, indent=2, default=str), flush=True)
    return RegisterResponse(worker_id=wid, message="registered")


@app.post("/heartbeat", response_model=HeartbeatResponse)
async def heartbeat(body: HeartbeatRequest) -> HeartbeatResponse:
    ok = scheduler.touch_heartbeat(body.worker_id, body.task_id)
    if not ok:
        raise HTTPException(status_code=404, detail="unknown worker")
    return HeartbeatResponse(ok=True, server_time=time.time())


@app.get("/task/{worker_id}", response_model=TaskResponse)
async def get_task(worker_id: str) -> TaskResponse:
    return scheduler.request_task(worker_id)


@app.post("/submit_weights")
async def submit_weights(
    meta_json: str = Form(...),
    weights_file: UploadFile = File(...),
) -> JSONResponse:
    try:
        meta = SubmitWeightsMetadata.model_validate_json(meta_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid metadata: {e}") from e

    weights_bytes = await weights_file.read()
    if not weights_bytes:
        raise HTTPException(status_code=400, detail="empty weights buffer")
    ok, msg = scheduler.submit_weights(
        meta.worker_id,
        meta.task_id,
        weights_bytes,
        meta.last_index,
        meta.shard_complete,
        shard_eval_acc=meta.shard_eval_acc,
        local_epochs_planned=meta.local_epochs_planned,
        local_epochs_completed=meta.local_epochs_completed,
    )
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    broadcast: dict[str, Any] | None = None
    if msg.startswith("aggregated"):
        hs = scheduler.health_snapshot()
        broadcast = {
            "round_no": state_manager.global_round(),
            "version": state_manager.global_version_label(),
            "message": msg,
            "training_stopped": hs.get("training_stopped", False),
            "stop_reason": hs.get("stop_reason"),
            "best_test_acc": hs.get("best_test_acc"),
        }

    # Print per-worker training metrics on the tracker terminal.
    if meta.train_loss_last is not None or meta.train_acc_running is not None or meta.shard_eval_acc is not None:
        print(
            f"[metrics] worker={meta.worker_id[:8]}… task={meta.task_id} "
            f"last_index={meta.last_index} steps={meta.steps_completed} done={meta.shard_complete} "
            f"epochs={meta.local_epochs_completed}/{meta.local_epochs_planned} "
            f"loss={meta.train_loss_last} train_acc={meta.train_acc_running} eval_acc={meta.shard_eval_acc}",
            flush=True,
        )

    hs = scheduler.health_snapshot()
    body: dict[str, Any] = {
        "ok": True,
        "detail": msg,
        "checkpoint_dir": state_manager.checkpoint_dir(),
        "training_stopped": hs.get("training_stopped", False),
        "stop_reason": hs.get("stop_reason"),
        "best_test_acc": hs.get("best_test_acc"),
    }
    if broadcast:
        body["aggregation"] = broadcast
    return JSONResponse(body)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    snap = scheduler.health_snapshot()
    return HealthResponse(status="ok", round_no=snap.get("round_no", 1), tasks=snap)


@app.get("/global_model")
async def global_model() -> JSONResponse:
    raw = state_manager.get_global_bytes()
    if raw is None:
        raise HTTPException(status_code=404, detail="no global model yet")
    import base64

    snap = state_manager.snapshot()
    return JSONResponse(
        {
            "round_no": state_manager.global_round(),
            "version_label": state_manager.global_version_label(),
            "weights_b64": base64.b64encode(raw).decode("ascii"),
            "checkpoint_dir": state_manager.checkpoint_dir(),
            "last_test_acc": snap.get("last_test_acc"),
        }
    )


@app.post("/log")
async def log_event(body: LogEvent) -> JSONResponse:
    key = body.worker_id
    _LOG_COUNTS[key] = _LOG_COUNTS.get(key, 0) + 1
    wid = body.worker_id[:8]
    host = body.host_label or "—"
    task = body.task_id or "—"
    print(f"[workerlog] {body.level} host={host} worker={wid}… task={task}: {body.message}", flush=True)
    return JSONResponse({"ok": True, "count": _LOG_COUNTS[key]})


@app.post("/progress")
async def progress_event(body: ProgressEvent) -> JSONResponse:
    scheduler.update_progress(
        worker_id=body.worker_id,
        task_id=body.task_id,
        local_epoch=body.local_epoch,
        local_epochs_total=body.local_epochs_total,
        shard_progress_pct=body.shard_progress_pct,
        train_acc_running=body.train_acc_running,
        train_loss_last=body.train_loss_last,
        ts=body.ts,
    )
    return JSONResponse({"ok": True})


@app.get("/checkpoint/{task_id}")
async def checkpoint(task_id: str) -> Response:
    raw = state_manager.get_task_checkpoint_bytes(task_id)
    if raw is None:
        raise HTTPException(status_code=404, detail="no checkpoint for task_id")
    return Response(content=raw, media_type="application/octet-stream")
