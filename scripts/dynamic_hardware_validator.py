#!/usr/bin/env python3
"""
Dynamic Hardware Validator for decentralized workers.

Features:
- Detect live GPU/CPU/RAM stats (uses psutil + pynvml; falls back to GPUtil).
- Validate against a Job Manifest JSON before starting Docker.
- Monitor VRAM during execution and Safe Kill when usage hits a threshold (default 95%).
- Emit a "Node Health Report" (stdout always; optionally tracker /log if tracker creds are provided).

This is designed to be used by an external "worker launcher" that starts a Docker container after validation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional


def _json_load_manifest(value: str) -> dict[str, Any]:
    value = value.strip()
    if not value:
        raise ValueError("empty manifest")
    if value.startswith("{"):
        return json.loads(value)
    # Assume it's a file path.
    with open(value, "r", encoding="utf-8") as f:
        return json.load(f)


def _now_ts() -> float:
    return time.time()


def _to_mb(bytes_or_mb: float, unit: str) -> float:
    unit_u = (unit or "").strip().upper()
    if unit_u in ("MB", "MIB"):
        return float(bytes_or_mb)
    if unit_u in ("GB", "GIB"):
        return float(bytes_or_mb) * 1024.0
    raise ValueError(f"unknown unit={unit!r} (expected MB or GB)")


def _safe_log(payload: dict[str, Any]) -> None:
    # Make it easy for external launchers to parse in logs.
    print(json.dumps(payload, default=str, ensure_ascii=True), flush=True)


@dataclass
class GpuSnapshot:
    index: int
    model: Optional[str]
    total_vram_mb: float
    used_vram_mb: float
    available_vram_mb: float
    cuda_cores: Optional[int]


@dataclass
class CpuSnapshot:
    physical_cores: int
    logical_threads: int
    cpu_percent: float
    available_threads: int


@dataclass
class RamSnapshot:
    available_bytes: int


def _detect_cpu() -> CpuSnapshot:
    import psutil

    physical = int(psutil.cpu_count(logical=False) or 0)
    logical = int(psutil.cpu_count(logical=True) or 0)
    # Short interval keeps this responsive while still giving a meaningful utilization estimate.
    cpu_percent = float(psutil.cpu_percent(interval=0.2))

    # Approximate "available threads" as threads not currently being used heavily by the system.
    # Example: 16 threads, 80% utilization → ~3 available threads.
    available_threads = int(max(0, round(logical * (1.0 - cpu_percent / 100.0))))
    return CpuSnapshot(
        physical_cores=max(1, physical),
        logical_threads=max(1, logical),
        cpu_percent=cpu_percent,
        available_threads=available_threads,
    )


def _detect_ram() -> RamSnapshot:
    import psutil

    vm = psutil.virtual_memory()
    return RamSnapshot(available_bytes=int(vm.available))


def _detect_gpu_with_pynvml(gpu_index: int) -> GpuSnapshot:
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # NVML returns bytes
        total_mb = float(mem.total) / (1024.0 * 1024.0)
        used_mb = float(mem.used) / (1024.0 * 1024.0)
        avail_mb = float(mem.free) / (1024.0 * 1024.0)
        name_raw = pynvml.nvmlDeviceGetName(handle)
        model = name_raw.decode("utf-8", errors="ignore") if isinstance(name_raw, (bytes, bytearray)) else str(name_raw)

        # CUDA core count is not available across all NVML/pynvml versions.
        cuda_cores: Optional[int] = None
        try:
            # Some pynvml versions expose nvmlDeviceGetCudaCoreCount
            fn = getattr(pynvml, "nvmlDeviceGetCudaCoreCount", None)
            if fn is not None:
                cuda_cores = int(fn(handle))
        except Exception:
            cuda_cores = None

        return GpuSnapshot(
            index=gpu_index,
            model=model,
            total_vram_mb=total_mb,
            used_vram_mb=used_mb,
            available_vram_mb=avail_mb,
            cuda_cores=cuda_cores,
        )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _detect_gpu_with_gputil(gpu_index: int) -> GpuSnapshot:
    import GPUtil

    gpus = GPUtil.getGPUs()
    if not gpus:
        return GpuSnapshot(
            index=gpu_index,
            model=None,
            total_vram_mb=0.0,
            used_vram_mb=0.0,
            available_vram_mb=0.0,
            cuda_cores=None,
        )
    # If index out of range, clamp to last GPU.
    idx = min(max(0, gpu_index), len(gpus) - 1)
    g = gpus[idx]
    total_mb = float(g.memoryTotal)
    used_mb = float(g.memoryUsed)
    avail_mb = max(0.0, total_mb - used_mb)
    return GpuSnapshot(
        index=idx,
        model=getattr(g, "name", None),
        total_vram_mb=total_mb,
        used_vram_mb=used_mb,
        available_vram_mb=avail_mb,
        cuda_cores=None,
    )


def _detect_gpu(gpu_index: int) -> GpuSnapshot:
    try:
        return _detect_gpu_with_pynvml(gpu_index)
    except Exception:
        # Fall back to GPUtil (already in requirements.txt).
        return _detect_gpu_with_gputil(gpu_index)


def build_node_health_report(
    manifest: dict[str, Any],
    *,
    gpu_index: int = 0,
    vram_threshold_pct: float = 95.0,
    stage: str = "PRECHECK",
    error: Optional[str] = None,
) -> dict[str, Any]:
    cpu = _detect_cpu()
    ram = _detect_ram()
    gpu = _detect_gpu(gpu_index)

    used_fraction = (gpu.used_vram_mb / gpu.total_vram_mb) if gpu.total_vram_mb > 0 else 0.0
    report: dict[str, Any] = {
        "stage": stage,
        "ts": _now_ts(),
        "manifest": {
            "min_vram": manifest.get("min_vram"),
            "min_cpu_cores": manifest.get("min_cpu_cores"),
            "min_threads": manifest.get("min_threads"),
        },
        "thresholds": {"safe_vram_usage_pct": vram_threshold_pct},
        "gpu": {
            "index": gpu.index,
            "model": gpu.model,
            "total_vram_mb": round(gpu.total_vram_mb, 2),
            "used_vram_mb": round(gpu.used_vram_mb, 2),
            "available_vram_mb": round(gpu.available_vram_mb, 2),
            "cuda_cores": gpu.cuda_cores,
            "used_fraction": round(used_fraction, 4),
        },
        "cpu": {
            "physical_cores": cpu.physical_cores,
            "logical_threads": cpu.logical_threads,
            "cpu_percent": round(cpu.cpu_percent, 2),
            "available_threads": cpu.available_threads,
        },
        "ram": {
            "available_bytes": ram.available_bytes,
            "available_gb": round(ram.available_bytes / (1024.0**3), 3),
        },
    }
    if error:
        report["error"] = error
    return report


def validate_resources(manifest: dict[str, Any], *, gpu_index: int = 0) -> tuple[bool, dict[str, Any]]:
    """
    Compare live stats against Job Manifest.

    Required manifest keys:
      - min_vram
      - min_cpu_cores
      - min_threads

    Additional optional keys:
      - vram_unit: "GB" (default) or "MB"
    """
    vram_unit = manifest.get("vram_unit", "GB")
    min_vram_mb = _to_mb(float(manifest["min_vram"]), vram_unit)
    min_cpu_cores = int(manifest["min_cpu_cores"])
    min_threads = int(manifest["min_threads"])

    report = build_node_health_report(
        manifest,
        gpu_index=gpu_index,
        stage="VALIDATION",
        vram_threshold_pct=float(manifest.get("safe_vram_usage_pct", 95.0)),
    )
    gpu_avail = float(report["gpu"]["available_vram_mb"])
    cpu_avail_threads = int(report["cpu"]["available_threads"])
    cpu_physical = int(report["cpu"]["physical_cores"])

    # Match prompt: if available_vram < min_vram OR available_threads < min_threads → reject.
    # Also enforce min_cpu_cores to cover the manifest requirement.
    if gpu_avail < min_vram_mb or cpu_avail_threads < min_threads or cpu_physical < min_cpu_cores:
        required = {
            "min_vram_mb": round(min_vram_mb, 2),
            "available_vram_mb": round(gpu_avail, 2),
            "min_cpu_physical_cores": min_cpu_cores,
            "physical_cores": cpu_physical,
            "min_threads": min_threads,
            "available_threads": cpu_avail_threads,
        }
        reason = "Hardware Insufficiency"
        report["decision"] = "RECUSE"
        report["error"] = f"{reason}: {required}"
        return False, report

    report["decision"] = "ACCEPT"
    return True, report


def _maybe_send_to_tracker(
    *,
    tracker_url: Optional[str],
    worker_id: Optional[str],
    worker_ticket: Optional[str],
    task_id: Optional[str],
    report: dict[str, Any],
) -> None:
    if not tracker_url or not worker_id or not worker_ticket:
        return

    # Late import so this script can run without tracker dependencies if you want.
    from worker.client import TrackerClient

    client = TrackerClient(tracker_url)
    client.ticket = worker_ticket
    # Use /log as the generic "Node Health Report" transport.
    client.log_event(
        worker_id=worker_id,
        task_id=task_id,
        level="INFO" if report.get("decision") != "RECUSE" else "ERROR",
        host_label=None,
        message=f"Node Health Report: {json.dumps(report, default=str, ensure_ascii=True)}",
    )


def _monitor_and_maybe_kill(
    *,
    container: Any,
    manifest: dict[str, Any],
    tracker_url: Optional[str],
    worker_id: Optional[str],
    worker_ticket: Optional[str],
    task_id: Optional[str],
    gpu_index: int,
) -> int:
    safe_threshold_pct = float(manifest.get("safe_vram_usage_pct", 95.0))
    monitor_interval_sec = float(manifest.get("monitor_interval_sec", 2.0))

    while True:
        # Reload container state (docker SDK object is cached).
        try:
            container.reload()
        except Exception:
            pass

        # If container already exited, stop monitoring.
        try:
            if getattr(container, "status", None) not in ("running", "created"):
                break
        except Exception:
            break

        # Collect live stats and check "Safe Kill" threshold.
        report = build_node_health_report(
            manifest,
            gpu_index=gpu_index,
            stage="MONITOR",
            vram_threshold_pct=safe_threshold_pct,
        )
        used_fraction = float(report["gpu"]["used_fraction"])

        _safe_log({"type": "health", "stage": "MONITOR", "used_fraction": used_fraction, "report": report})
        _maybe_send_to_tracker(
            tracker_url=tracker_url,
            worker_id=worker_id,
            worker_ticket=worker_ticket,
            task_id=task_id,
            report=report,
        )

        if used_fraction >= (safe_threshold_pct / 100.0):
            # Safe kill: stop container quickly.
            kill_report = dict(report)
            kill_report["stage"] = "SAFE_KILL"
            kill_report["decision"] = "SAFE_KILLED"
            kill_report["error"] = f"Safe Kill: VRAM usage exceeded {safe_threshold_pct}%"

            _safe_log(kill_report)
            _maybe_send_to_tracker(
                tracker_url=tracker_url,
                worker_id=worker_id,
                worker_ticket=worker_ticket,
                task_id=task_id,
                report=kill_report,
            )
            try:
                container.kill()
            except Exception:
                pass
            return 2

        time.sleep(monitor_interval_sec)

    # Container finished normally.
    return 0


def _run_docker_container(
    *,
    image: str,
    command: Optional[str],
    gpus: Optional[str],
    env: dict[str, str],
    volumes: Optional[dict[str, dict[str, str]]],
    worker_ticket: Optional[str],
    tracker_url: Optional[str],
    worker_id: Optional[str],
    task_id: Optional[str],
    manifest: dict[str, Any],
    gpu_index: int,
) -> Any:
    import docker
    from docker.types import DeviceRequest

    client = docker.from_env()

    env2 = dict(env)
    # Optional: pass validator context into the container.
    if tracker_url:
        env2.setdefault("TRACKER_URL", tracker_url)
    if worker_id:
        env2.setdefault("WORKER_ID", worker_id)
    if worker_ticket:
        env2.setdefault("WORKER_TICKET", worker_ticket)
    if task_id:
        env2.setdefault("TASK_ID", task_id)

    device_requests = None
    if gpus is not None or manifest.get("gpu_index") is not None:
        # Prefer manifest/gpu_index when provided.
        idx = int(manifest.get("gpu_index", gpu_index))
        device_requests = [DeviceRequest(count=1, capabilities=[["gpu"]], device_ids=[idx])]

    run_kwargs: dict[str, Any] = dict(
        image=image,
        detach=True,
        remove=True,
        environment=env2,
    )
    if command:
        # docker SDK accepts list[str] or str. Keep it simple: use shell form if string.
        run_kwargs["command"] = command
    if volumes:
        run_kwargs["volumes"] = volumes
    if device_requests is not None:
        run_kwargs["device_requests"] = device_requests

    return client.containers.run(**run_kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Job Manifest JSON string or path to a .json file.")

    ap.add_argument("--docker-image", help="Docker image to run (required unless --dry-run).")
    ap.add_argument("--docker-command", help="Optional command for docker container.")
    ap.add_argument("--gpus", default=None, help='e.g. "device=0" (optional).')

    ap.add_argument("--gpu-index", type=int, default=0, help="GPU index to validate against (default: 0).")
    ap.add_argument("--dry-run", action="store_true", help="Only run validation/health reporting; do not launch Docker.")

    # Optional: send Node Health Reports to the existing tracker via /log.
    ap.add_argument("--tracker-url", default=os.getenv("TRACKER_URL"))
    ap.add_argument("--worker-id", default=os.getenv("WORKER_ID"))
    ap.add_argument("--worker-ticket", default=os.getenv("WORKER_TICKET"))
    ap.add_argument("--task-id", default=None)

    args = ap.parse_args()

    manifest = _json_load_manifest(args.manifest)
    manifest_task_id = manifest.get("task_id") or args.task_id

    # --- Preflight validation ---
    ok, report = validate_resources(manifest, gpu_index=args.gpu_index)
    _safe_log(report)
    _maybe_send_to_tracker(
        tracker_url=args.tracker_url,
        worker_id=args.worker_id,
        worker_ticket=args.worker_ticket,
        task_id=manifest_task_id,
        report=report,
    )

    if not ok:
        # "Kill logic": reject job before starting docker.
        # Exit code 1 so external orchestrators can treat it as a hard failure for the job.
        return 1

    if args.dry_run:
        return 0

    if not args.docker_image:
        raise ValueError("--docker-image is required unless --dry-run is set")

    # --- Run + monitor ---
    monitor_env: dict[str, str] = {
        # Pass-through: container can read this if it wants to.
        "JOB_MANIFEST_JSON": json.dumps(manifest, default=str, ensure_ascii=True),
    }

    # Keep this minimal: volumes/env are job-specific; for now we only pass manifest JSON.
    container = _run_docker_container(
        image=args.docker_image,
        command=args.docker_command,
        gpus=args.gpus,
        env=monitor_env,
        volumes=None,
        worker_ticket=args.worker_ticket,
        tracker_url=args.tracker_url,
        worker_id=args.worker_id,
        task_id=manifest_task_id,
        manifest=manifest,
        gpu_index=args.gpu_index,
    )

    # Initial health report DURING execution.
    start_report = build_node_health_report(
        manifest,
        gpu_index=args.gpu_index,
        stage="START",
        vram_threshold_pct=float(manifest.get("safe_vram_usage_pct", 95.0)),
    )
    _safe_log(start_report)
    _maybe_send_to_tracker(
        tracker_url=args.tracker_url,
        worker_id=args.worker_id,
        worker_ticket=args.worker_ticket,
        task_id=manifest_task_id,
        report=start_report,
    )

    rc = _monitor_and_maybe_kill(
        container=container,
        manifest=manifest,
        tracker_url=args.tracker_url,
        worker_id=args.worker_id,
        worker_ticket=args.worker_ticket,
        task_id=manifest_task_id,
        gpu_index=args.gpu_index,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())

