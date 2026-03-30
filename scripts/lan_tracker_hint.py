#!/usr/bin/env python3
"""
Print a tracker URL your teammates can use, and a realistic note about "finding laptops".

There is no magic way to list every laptop "near you" from the terminal: Wi‑Fi / routers
do not expose a browseable list of all peers. This project also has no peer-discovery
service yet — workers must know your tracker base URL (e.g. http://192.168.1.42:8000).

Run:  python3 scripts/lan_tracker_hint.py
       python3 scripts/lan_tracker_hint.py --arp
"""

from __future__ import annotations

import argparse
import platform
import re
import subprocess
import sys


def _darwin_wifi_ip() -> str | None:
    try:
        out = subprocess.check_output(
            ["ipconfig", "getifaddr", "en0"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return None


def _darwin_all_ipv4() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    try:
        out = subprocess.check_output(["ifconfig"], text=True, timeout=10)
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return rows
    cur = ""
    for line in out.splitlines():
        if not line.startswith("\t") and not line.startswith(" "):
            cur = line.split(":")[0] if ":" in line else line.split()[0] if line.split() else ""
        m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", line)
        if m and not m.group(1).startswith("127."):
            rows.append((cur, m.group(1)))
    return rows


def _arp_table() -> str:
    if platform.system() != "Darwin":
        return "(arp -a only implemented hint for macOS in this script; use `arp -a` on Linux.)"
    try:
        return subprocess.check_output(["arp", "-a"], text=True, timeout=10)
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as e:
        return f"(arp -a failed: {e})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arp", action="store_true", help="Print ARP cache (recent LAN peers, incomplete)")
    ap.add_argument("--port", type=int, default=8000, help="Tracker port")
    args = ap.parse_args()

    print("Tracker host (you run uvicorn here); teammates point the worker at this URL.\n")

    if platform.system() == "Darwin":
        ip = _darwin_wifi_ip()
        if ip:
            print(f"  Likely Wi‑Fi IP (en0):  {ip}")
            print(f"  Example tracker URL:   http://{ip}:{args.port}")
        print("\n  Other IPv4 interfaces:")
        for iface, addr in _darwin_all_ipv4():
            print(f"    {iface or '?'}: {addr}")
    else:
        print("  On this OS, find your LAN IP with `ip a` or `hostname -I` and use http://THAT_IP:PORT")

    print(
        "\nTeammate needs: same repo, Python venv, `pip install -r requirements.txt`, "
        "then e.g.\n"
        f'  PYTHONPATH=. python mock_worker.py --tracker http://YOUR_IP:{args.port}\n'
        "For GPU Docker: Docker + NVIDIA Container Toolkit on Linux/WSL with an NVIDIA GPU "
        "(Mac teammates use CPU mock or Metal locally without this stack)."
    )

    print(
        "\n--- About 'all laptops in my vicinity' ---\n"
        "You cannot reliably enumerate every laptop on the network from a terminal without "
        "router admin tools, enterprise MDM, or building a custom discovery/broadcast service. "
        "The ARP table only shows IPs your Mac has recently talked to on the LAN — not a full roster."
    )

    if args.arp:
        print("\n--- arp -a (sample of recent Layer-2 neighbors) ---\n")
        print(_arp_table())

    return 0


if __name__ == "__main__":
    sys.exit(main())
