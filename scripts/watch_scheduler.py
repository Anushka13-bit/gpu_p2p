#!/usr/bin/env python3
"""
Watch tracker node registry + shard progress from a terminal.

Run (on the tracker host, or anywhere that can reach it):
  python3 scripts/watch_scheduler.py --tracker http://127.0.0.1:8000
  python3 scripts/watch_scheduler.py --tracker http://192.168.1.42:8000 --every 30
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

import requests


def _pct(last_index: int, start: int, end: int) -> float:
    denom = max(1, end - start)
    if last_index < start:
        return 0.0
    return 100.0 * (min(last_index, end - 1) - start + 1) / denom


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", default="http://127.0.0.1:8000")
    ap.add_argument("--every", type=float, default=30.0, help="Seconds between prints.")
    args = ap.parse_args()

    base = args.tracker.rstrip("/")
    s = requests.Session()
    key = os.environ.get("GPU_P2P_AUTH_KEY")
    if key:
        s.headers.update({"X-Auth-Key": key})

    while True:
        try:
            r = s.get(f"{base}/health", timeout=15)
            r.raise_for_status()
            payload: dict[str, Any] = r.json()
        except Exception as e:
            print(f"[watch] error fetching {base}/health: {e!r}", flush=True)
            time.sleep(args.every)
            continue

        t = payload.get("tasks", {})
        reg = t.get("node_registry", {})
        roster = t.get("worker_roster", [])
        task_table = t.get("task_table", {})

        print("")
        print(f"——— watch @ {time.strftime('%H:%M:%S')} ———")
        print(
            f"  round={t.get('round_no')}  version={t.get('version_label')}  "
            f"nodes: total={reg.get('total_nodes')} active={reg.get('active_nodes')} "
            f"(timeout={reg.get('heartbeat_timeout_sec')}s)"
        )

        if roster:
            print("  nodes:")
            for w in roster:
                wid = str(w.get("worker_id", ""))[:8]
                host = w.get("host_label") or "—"
                age = w.get("last_seen_age_sec")
                age_s = f"{age:.1f}s" if isinstance(age, (int, float)) else "—"
                print(f"    {wid}…  host={host}  last_seen={age_s}")
        else:
            print("  nodes: (none registered)")

        if task_table:
            print("  shards:")
            for tid, info in sorted(task_table.items()):
                st = info.get("status")
                worker = info.get("worker") or "—"
                rnge = info.get("range") or [0, 0]
                start, end = int(rnge[0]), int(rnge[1])
                last = int(info.get("last_index", -1))
                p = _pct(last, start, end)
                print(f"    {tid}: {st:<10} worker={str(worker)[:8] + '…' if worker!='—' else '—':<11}  {p:6.2f}%  last_index={last}")
        else:
            print("  shards: (none)")

        print("———" * 10)
        time.sleep(args.every)


if __name__ == "__main__":
    raise SystemExit(main())

