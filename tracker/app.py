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
from fastapi.responses import JSONResponse

from shared.protocol import (
    HealthResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    RegisterRequest,
    RegisterResponse,
    SubmitWeightsMetadata,
    TaskResponse,
)

from .scheduler import REGISTRY_DISPLAY_INTERVAL_SEC, Scheduler
from .state_manager import StateManager

state_manager = StateManager()
scheduler = Scheduler(state_manager)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async def _registry_supervisor():
        while True:
            await asyncio.sleep(REGISTRY_DISPLAY_INTERVAL_SEC)
            orphaned = scheduler.check_timeouts()
            if orphaned:
                print(
                    f"[watchdog] reassigned/orphaned (assigned shard had no heartbeat in time): {orphaned}",
                    flush=True,
                )
            print(scheduler.format_registry_terminal(), flush=True)

    task = asyncio.create_task(_registry_supervisor())
    print(
        f"[tracker] supervisor: registry + watchdog every {REGISTRY_DISPLAY_INTERVAL_SEC:.0f}s "
        f"(see scheduler.HEARTBEAT_TIMEOUT_SEC for liveness window)",
        flush=True,
    )
    print(scheduler.format_registry_terminal(), flush=True)
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="GPU Tracker", lifespan=lifespan)


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

    body: dict[str, Any] = {"ok": True, "detail": msg}
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

    return JSONResponse(
        {
            "round_no": state_manager.global_round(),
            "version_label": state_manager.global_version_label(),
            "weights_b64": base64.b64encode(raw).decode("ascii"),
        }
    )
