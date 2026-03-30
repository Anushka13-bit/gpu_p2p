#!/usr/bin/env python3
"""Print hardware (macOS Apple Silicon + NVIDIA) and /register-style fields.

Run from repo root:
  python3 scripts/print_hardware.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.hardware_sniff import build_hardware_report

try:
    import GPUtil
except ImportError:
    GPUtil = None  # type: ignore


def main() -> int:
    r = build_hardware_report()

    print("=== Machine ===")
    print(f"  platform: {r.platform_system}  machine: {r.machine}")
    if r.cpu_brand:
        print(f"  CPU:      {r.cpu_brand}")
    print(f"  cpus (logical): {r.cpu_count}")

    print("\n=== Apple GPU (displays) ===")
    if r.apple_gpu_lines:
        for ln in r.apple_gpu_lines:
            print(f"  {ln}")
    elif r.platform_system == "Darwin":
        print("  (run `system_profiler SPDisplaysDataType` if empty / slow subprocess)")

    print("\n=== Memory (macOS unified) ===")
    if r.apple_unified_ram_mb is not None:
        print(f"  system RAM (hw.memsize): {r.apple_unified_ram_mb:.0f} MiB total")
    else:
        print("  system RAM: (not macOS or sysctl unavailable)")

    print("\n=== PyTorch backends ===")
    if r.torch_version:
        print(f"  torch: {r.torch_version}")
    else:
        print("  torch: not installed")
    if r.mps_available is True:
        print("  MPS (Metal): available — training can use device `mps`")
    elif r.mps_available is False:
        print("  MPS (Metal): not available")
    else:
        print("  MPS (Metal): unknown (install torch to detect)")
    print(f"  CUDA: {r.cuda_available if r.cuda_available is not None else 'n/a'}")

    print("\n=== NVIDIA (GPUtil) ===")
    if GPUtil is None:
        print("  GPUtil not installed (`pip install gputil`)")
    else:
        try:
            gpus = GPUtil.getGPUs()
            print(f"  devices: {len(gpus)}")
            for i, g in enumerate(gpus):
                print(f"    [{i}] {g.name} — {g.memoryTotal} MiB")
        except Exception as e:
            print(f"  error: {e!r}")

    print("\n=== POST /register JSON-style (what the worker sends) ===")
    print(f"  gpu_vram_mb: {r.effective_register_gpu_vram_mb}")
    print(f"  cpu_count:   {r.cpu_count}")
    print(f"  — {r.register_note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
