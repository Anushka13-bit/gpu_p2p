"""
Hardware detection for /register and dev tools.

- NVIDIA: GPUtil reports discrete VRAM (MiB).
- Apple Silicon: there is no separate GPU VRAM; memory is unified with RAM.
  We expose ``unified_ram_mb`` and an optional heuristic
  ``effective_gpu_budget_mb`` (~55%% of RAM) so ``gpu_vram_mb`` in the API
  is non-zero for schedulers that key off that field.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Optional


def _sysctl_n(name: str) -> Optional[str]:
    if platform.system() != "Darwin":
        return None
    try:
        p = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if p.returncode != 0:
            return None
        return (p.stdout or "").strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def darwin_physical_memory_mb() -> Optional[float]:
    raw = _sysctl_n("hw.memsize")
    if raw is None:
        return None
    try:
        return int(raw) / (1024 * 1024)
    except ValueError:
        return None


def darwin_cpu_brand_string() -> Optional[str]:
    return _sysctl_n("machdep.cpu.brand_string")


def darwin_gpu_chipset_lines() -> list[str]:
    """Human-readable display lines from system_profiler (can take a few seconds)."""
    if platform.system() != "Darwin":
        return []
    try:
        p = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if p.returncode != 0 or not p.stdout:
            return []
        lines: list[str] = []
        for line in p.stdout.splitlines():
            s = line.strip()
            if "Chipset Model:" in s or "Vendor:" in s or "VRAM" in s or "Resolution:" in s:
                lines.append(s)
        return lines or [ln.strip() for ln in p.stdout.splitlines() if ln.strip()][:8]
    except (OSError, subprocess.TimeoutExpired):
        return []


def is_apple_silicon_cpu() -> bool:
    if platform.system() != "Darwin":
        return False
    b = darwin_cpu_brand_string() or ""
    return "Apple" in b


def torch_mps_available() -> Optional[bool]:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return None


def nvidia_gputil_first_vram_mb() -> Optional[float]:
    try:
        import GPUtil

        gpus = GPUtil.getGPUs()
        if gpus:
            return float(gpus[0].memoryTotal)
    except Exception:
        pass
    return None


# Unified-memory heuristic: fraction of system RAM often usable as GPU working set on Apple Silicon.
APPLE_UNIFIED_GPU_BUDGET_FRAC = 0.55


@dataclass
class HardwareReport:
    cpu_count: int
    platform_system: str
    machine: str
    nvidia_vram_mb: Optional[float]
    apple_unified_ram_mb: Optional[float]
    effective_register_gpu_vram_mb: float
    """Value worker sends as ``gpu_vram_mb`` on /register."""
    apple_gpu_lines: list[str]
    cpu_brand: Optional[str]
    mps_available: Optional[bool]
    cuda_available: Optional[bool]
    torch_version: Optional[str]
    register_note: str


def build_hardware_report() -> HardwareReport:
    cpus = max(1, os.cpu_count() or 1)
    sysname = platform.system()
    machine = platform.machine()

    nvidia = nvidia_gputil_first_vram_mb()
    ram = darwin_physical_memory_mb() if sysname == "Darwin" else None
    apple_lines = darwin_gpu_chipset_lines() if sysname == "Darwin" else []
    brand = darwin_cpu_brand_string() if sysname == "Darwin" else None
    mps = torch_mps_available()

    cuda: Optional[bool]
    tv: Optional[str]
    try:
        import torch

        tv = torch.__version__
        cuda = bool(torch.cuda.is_available())
    except Exception:
        tv = None
        cuda = None

    eff: float
    note: str
    if nvidia is not None:
        eff = nvidia
        note = "NVIDIA VRAM from GPUtil (discrete GPU memory)."
    elif sysname == "Darwin" and (is_apple_silicon_cpu() or mps is True) and ram is not None:
        eff = round(ram * APPLE_UNIFIED_GPU_BUDGET_FRAC, 1)
        note = (
            "Apple Silicon: no separate GPU VRAM; value is "
            f"{APPLE_UNIFIED_GPU_BUDGET_FRAC:.0%} × unified RAM as a scheduling hint for gpu_vram_mb."
        )
    else:
        eff = 0.0
        if sysname == "Darwin" and ram is not None:
            note = "macOS: no NVIDIA GPU detected; install PyTorch + run again for MPS line; gpu_vram_mb stays 0 without Apple/MPS path."
        else:
            note = "No NVIDIA GPU from GPUtil; not applying Apple unified heuristic (non-Darwin or unknown)."

    return HardwareReport(
        cpu_count=cpus,
        platform_system=sysname,
        machine=machine,
        nvidia_vram_mb=nvidia,
        apple_unified_ram_mb=ram,
        effective_register_gpu_vram_mb=eff,
        apple_gpu_lines=apple_lines,
        cpu_brand=brand,
        mps_available=mps,
        cuda_available=cuda,
        torch_version=tv,
        register_note=note,
    )


def sniff_register_tuple() -> tuple[float, int]:
    """(gpu_vram_mb, cpu_count) for POST /register — matches build_hardware_report semantics."""
    r = build_hardware_report()
    return float(r.effective_register_gpu_vram_mb), int(r.cpu_count)


def hardware_report_for_register() -> dict[str, Any]:
    """JSON-serializable dict for POST /register ``hardware_report`` (any OS)."""
    return asdict(build_hardware_report())
