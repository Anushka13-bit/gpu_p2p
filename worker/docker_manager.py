"""
Run training inside an isolated container with GPU access (NVIDIA Container Toolkit).

Requires Docker daemon and the NVIDIA runtime. Equivalent CLI flags: --gpus all.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import docker
from docker.types import DeviceRequest


def run_training_container(
    image: str,
    tracker_url: str,
    worker_id: str,
    worker_ticket: str | None,
    task_env: Dict[str, str],
    volumes: Optional[Dict[str, Dict[str, str]]] = None,
    gpus: str = "all",
    detach: bool = True,
    remove_on_exit: bool = True,
) -> docker.models.containers.Container:
    """
    ``task_env`` should include serialized task JSON under key ``TASK_JSON`` or
    pass keys expected by ``trainer_wrapper.py`` (TRACKER_URL, WORKER_ID, WORKER_TICKET, TASK_JSON).

    ``volumes`` example for dev: {"/abs/path/project": {"bind": "/app", "mode": "rw"}}
    """
    client = docker.from_env()
    env = {
        "TRACKER_URL": tracker_url,
        "WORKER_ID": worker_id,
        "PYTHONPATH": "/app",
        **task_env,
    }
    if worker_ticket:
        env["WORKER_TICKET"] = worker_ticket
    run_kw: Dict[str, Any] = dict(
        image=image,
        command=["python", "-u", "-m", "worker.trainer_wrapper"],
        environment=env,
        detach=detach,
        remove=remove_on_exit,
        volumes=volumes or {},
    )
    from shared.hardware_sniff import build_hardware_report
    report = build_hardware_report()
    
    if gpus and report.nvidia_vram_mb is not None:
        run_kw["device_requests"] = [DeviceRequest(count=-1, capabilities=[["gpu"]])]
    else:
        env["FALLBACK_MODE"] = "1"

    container = client.containers.run(**run_kw)
    return container


def logs_stream(container: docker.models.containers.Container, **kwargs: Any):
    return container.logs(stream=True, **kwargs)
