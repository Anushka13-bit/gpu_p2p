"""Pydantic schemas for tracker ↔ worker communication."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    ORPHANED = "ORPHANED"


class RegisterRequest(BaseModel):
    gpu_vram_mb: float = Field(..., ge=0, description="Total GPU VRAM in MB (0 if CPU-only mock)")
    cpu_count: int = Field(..., ge=1)
    host_label: Optional[str] = None
    hardware_report: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional JSON from shared.hardware_sniff.build_hardware_report (full client specs).",
    )
    token: Optional[str] = Field(None, description="Legacy token field.")
    worker_token: Optional[str] = Field(None, description="Pre-shared UUID worker token issued by the tracker admin.")

class SignupRequest(BaseModel):
    name: str
    email: str

class SignupResponse(BaseModel):
    worker_id: str
    worker_token: str
    status: str
    message: str

class WorkerInfo(BaseModel):
    worker_id: str
    name: str
    email: str
    status: str
    shards_completed: int
    failed_validations: int
    credit_balance: int
    joined_at: float

class BanRequest(BaseModel):
    worker_id: str
    reason: str

class UnbanRequest(BaseModel):
    worker_id: str

class RegisterResponse(BaseModel):
    worker_id: str
    message: str


class HeartbeatRequest(BaseModel):
    worker_id: str
    task_id: Optional[str] = None
    token: Optional[str] = Field(default=None, description="Pre-shared UUID worker token (required by tracker auth).")


class HeartbeatResponse(BaseModel):
    ok: bool
    server_time: float


class TaskAssignment(BaseModel):
    task_id: str
    image_start: int
    image_end: int
    exclusive_end: int
    round_no: int
    """Logical training round; bumps after FedAvg."""
    resume_next_index: int = Field(
        ...,
        description="Next dataset row index to consume (absolute, in [image_start, image_end)).",
    )


class TaskResponse(BaseModel):
    has_task: bool
    task: Optional[TaskAssignment] = None
    global_model_bytes_b64: Optional[str] = Field(
        default=None,
        description="Base64-encoded torch.save state_dict; None means initialize randomly.",
    )


class SubmitWeightsMetadata(BaseModel):
    worker_id: str
    task_id: str
    last_index: int = Field(..., ge=-1, description="Last consumed index within [image_start, image_end).")
    steps_completed: int = Field(default=0, ge=0)
    shard_complete: bool = Field(default=False, description="True when this worker finished its shard.")
    # Optional metrics (so tracker can print per-worker training progress).
    train_loss_last: Optional[float] = Field(default=None, description="Last mini-batch loss observed on worker.")
    train_acc_running: Optional[float] = Field(
        default=None, description="Running train accuracy (%) over mini-batches in this submit window."
    )
    shard_eval_acc: Optional[float] = Field(
        default=None, description="Eval accuracy (%) over the whole shard after this submit window."
    )
    local_epochs_planned: Optional[int] = Field(default=None, description="Local epochs planned for this submit window.")
    local_epochs_completed: Optional[int] = Field(
        default=None, description="Local epochs completed in this submit window (may be capped by steps)."
    )


class LogEvent(BaseModel):
    worker_id: str
    host_label: Optional[str] = None
    task_id: Optional[str] = None
    level: str = Field(default="INFO")
    message: str
    ts: float = Field(..., description="Unix timestamp seconds from worker.")


class AggregationBroadcast(BaseModel):
    round_no: int
    completed_tasks: int
    total_tasks: int
    message: str
    detail: Optional[dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    round_no: int
    tasks: dict[str, Any]
