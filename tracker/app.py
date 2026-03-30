"""
Central control plane — FastAPI tracker.
Run from repo root: PYTHONPATH=. uvicorn tracker.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import time
import os
import io
import torch
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, Depends
from fastapi.responses import JSONResponse, Response

from shared.protocol import (
    HealthResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    LogEvent,
    RegisterRequest,
    RegisterResponse,
    SubmitWeightsMetadata,
    TaskResponse,
    SignupRequest,
    SignupResponse,
    BanRequest,
    UnbanRequest,
)

from .scheduler import REGISTRY_DISPLAY_INTERVAL_SEC, WATCHDOG_CHECK_INTERVAL_SEC, Scheduler
from .state_manager import StateManager
from .worker_registry import WorkerRegistry

state_manager = StateManager()
scheduler = Scheduler(state_manager)
registry = WorkerRegistry(os.getenv("REGISTRY_DB", "registry.db"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("ADMIN_TOKEN", "admin-secret-change-me") == "admin-secret-change-me":
        print("WARNING: ADMIN_TOKEN is set to default value. Set ADMIN_TOKEN env variable before production use.", flush=True)
    
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

async def require_auth(x_worker_token: str = Header(..., alias="X-Worker-Token")) -> dict:
    worker = registry.authenticate(x_worker_token)
    if not worker:
        raise HTTPException(status_code=401, detail="unknown token — visit /signup first")
    if worker["status"] == "banned":
        raise HTTPException(status_code=403, detail={"error": "worker is banned", "reason": worker["ban_reason"]})
    return worker

def require_admin(x_admin_token: str = Header(..., alias="X-Admin-Token")) -> None:
    expected = os.getenv("ADMIN_TOKEN", "admin-secret-change-me")
    if x_admin_token != expected:
        raise HTTPException(status_code=403, detail="invalid admin token")

@app.post("/signup", response_model=SignupResponse)
async def signup(body: SignupRequest) -> SignupResponse:
    res = registry.signup(body.name, body.email)
    if not res:
        raise HTTPException(status_code=400, detail="email already registered")
    return SignupResponse(**res)


@app.post("/register", response_model=RegisterResponse)
async def register(body: RegisterRequest, worker: dict = Depends(require_auth)) -> RegisterResponse:
    wid = scheduler.register_worker(
        gpu_vram_mb=body.gpu_vram_mb,
        cpu_count=body.cpu_count,
        host_label=body.host_label,
        hardware_report=body.hardware_report,
    )
    print(
        f"[register] worker_id={wid} gpu_vram_mb={body.gpu_vram_mb} "
        f"cpu_count={body.cpu_count} host_label={body.host_label!r} status={worker['status']}",
        flush=True,
    )
    if body.hardware_report:
        print(json.dumps(body.hardware_report, indent=2, default=str), flush=True)
    return RegisterResponse(worker_id=wid, message="registered")


@app.post("/heartbeat", response_model=HeartbeatResponse)
async def heartbeat(
    body: HeartbeatRequest,
    worker: dict = Depends(require_auth)
) -> HeartbeatResponse:
    ok = scheduler.touch_heartbeat(body.worker_id, body.task_id)
    if not ok:
        raise HTTPException(status_code=404, detail="unknown worker")
    return HeartbeatResponse(ok=True, server_time=time.time())


@app.get("/task/{worker_id}", response_model=TaskResponse)
async def get_task(
    worker_id: str,
    worker: dict = Depends(require_auth)
) -> TaskResponse:
    return scheduler.request_task(worker_id)


def is_validation_failure(weights_bytes: bytes) -> bool:
    try:
        sd = torch.load(io.BytesIO(weights_bytes), map_location="cpu", weights_only=True)
        for t in sd.values():
            if t.dtype.is_floating_point:
                if t.abs().sum().item() == 0.0:
                    return True
                break
        return False
    except Exception:
        return True


@app.post("/submit_weights")
async def submit_weights(
    meta_json: str = Form(...),
    weights_file: UploadFile = File(...),
    worker: dict = Depends(require_auth)
) -> JSONResponse:
    try:
        meta = SubmitWeightsMetadata.model_validate_json(meta_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid metadata: {e}") from e

    weights_bytes = await weights_file.read()
    if not weights_bytes:
        raise HTTPException(status_code=400, detail="empty weights buffer")

    # VALIDATION CHECK
    failed = is_validation_failure(weights_bytes)
    worker_id_real = worker["worker_id"] # Use the real worker_id from DB for the auth
    
    trust_weight = registry.get_trust_weight(worker_id_real)

    if failed:
        signal = registry.record_failure(worker_id_real)
        return JSONResponse({"ok": False, "detail": "validation failed"}, status_code=400)
    else:
        signal = registry.record_success(worker_id_real)

    ok, msg = scheduler.submit_weights(
        meta.worker_id,
        meta.task_id,
        weights_bytes,
        meta.last_index,
        meta.shard_complete,
        shard_eval_acc=meta.shard_eval_acc,
        local_epochs_planned=meta.local_epochs_planned,
        local_epochs_completed=meta.local_epochs_completed,
        trust_weight=trust_weight
    )
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    broadcast: dict[str, Any] | None = None
    if msg.startswith("aggregated"):
        broadcast = {
            "round_no": state_manager.global_round(),
            "version": state_manager.global_version_label(),
            "message": msg,
        }

    if meta.train_loss_last is not None or meta.train_acc_running is not None or meta.shard_eval_acc is not None:
        print(
            f"[metrics] worker={meta.worker_id[:8]}… task={meta.task_id} "
            f"last_index={meta.last_index} steps={meta.steps_completed} done={meta.shard_complete} "
            f"epochs={meta.local_epochs_completed}/{meta.local_epochs_planned} "
            f"loss={meta.train_loss_last} train_acc={meta.train_acc_running} eval_acc={meta.shard_eval_acc}",
            flush=True,
        )

    body: dict[str, Any] = {"ok": True, "detail": msg, "checkpoint_dir": state_manager.checkpoint_dir()}
    if broadcast:
        body["aggregation"] = broadcast
    if signal:
        body["signal"] = signal
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
    return JSONResponse(
        {
            "round_no": state_manager.global_round(),
            "version_label": state_manager.global_version_label(),
            "weights_b64": base64.b64encode(raw).decode("ascii"),
            "checkpoint_dir": state_manager.checkpoint_dir(),
        }
    )


@app.post("/log")
async def log_event(
    body: LogEvent,
    worker: dict = Depends(require_auth)
) -> JSONResponse:
    key = body.worker_id
    _LOG_COUNTS[key] = _LOG_COUNTS.get(key, 0) + 1
    wid = body.worker_id[:8]
    host = body.host_label or "—"
    task = body.task_id or "—"
    print(f"[workerlog] {body.level} host={host} worker={wid}… task={task}: {body.message}", flush=True)
    return JSONResponse({"ok": True, "count": _LOG_COUNTS[key]})


@app.get("/checkpoint/{task_id}")
async def checkpoint(task_id: str) -> Response:
    raw = state_manager.get_task_checkpoint_bytes(task_id)
    if raw is None:
        raise HTTPException(status_code=404, detail="no checkpoint for task_id")
    return Response(content=raw, media_type="application/octet-stream")


@app.post("/admin/ban")
async def admin_ban(body: BanRequest, _: None = Depends(require_admin)) -> JSONResponse:
    registry.ban_worker(body.worker_id, body.reason)
    scheduler.orphan_task_for_worker(body.worker_id)
    return JSONResponse({"banned": True, "worker_id": body.worker_id})


@app.post("/admin/unban")
async def admin_unban(body: UnbanRequest, _: None = Depends(require_admin)) -> JSONResponse:
    registry.unban_worker(body.worker_id)
    return JSONResponse({"unbanned": True, "worker_id": body.worker_id})


@app.get("/admin/workers")
async def admin_workers(_: None = Depends(require_admin)) -> JSONResponse:
    workers = registry.get_all_workers()
    # Mock credit balances since no ledger exists
    for w in workers:
        w["credit_balance"] = 100
    return JSONResponse(workers)
