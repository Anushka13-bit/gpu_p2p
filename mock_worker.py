#!/usr/bin/env python3
"""
Mock worker for integration tests (no Docker / GPU required).

Run tracker first:
  cd "/path/to/gpu p2p" && PYTHONPATH=. uvicorn tracker.app:app --host 0.0.0.0 --port 8000

Then:
  PYTHONPATH=. python mock_worker.py --tracker http://127.0.0.1:8000

By default the mock keeps running across rounds (`--max-fed-rounds 0`).
Use `--max-fed-rounds N` to stop after N FedAvg aggregations *this worker observes*.

Simulate death mid-shard (orphan → reassignment):
  PYTHONPATH=. python mock_worker.py --tracker http://127.0.0.1:8000 --die-after-first-round

Round 2: start another worker (or the same binary) to pick up ORPHANED shards if a shard assignee stops heartbeating (see scheduler.HEARTBEAT_TIMEOUT_SEC).
"""

from __future__ import annotations

import argparse
import base64
import threading
import time
import uuid
from typing import Optional

import torch

from shared.hardware_sniff import hardware_report_for_register
from shared.models import SmallCNN, apply_state_dict
from worker.client import TrackerClient, sniff_hardware_defaults
from worker.train_utils import build_dataset_base, train_shard_batch_loop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--dataset",
        default="fashion_mnist_csv",
        help="Dataset to train on (fashion_mnist_csv, fashion_mnist, mnist).",
    )
    # Ignored as a ceiling: train_shard_batch_loop always completes --local-epochs full passes
    # over the current slice (see worker.train_utils.train_shard_batch_loop).
    parser.add_argument(
        "--steps",
        type=int,
        default=600,
        help="Legacy floor passed to trainer; full local epochs always run in one submit window.",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=15,
        help="Local epochs over the assigned shard per submit window (CNN will train multiple passes).",
    )
    parser.add_argument(
        "--heartbeat-sec",
        type=float,
        default=3.0,
        help="POST /heartbeat interval while running (tracker timeout must be larger).",
    )
    parser.add_argument(
        "--host-label",
        default="mock-worker",
        help="Shown in tracker registry / register payload.",
    )
    parser.add_argument("--quiet-training", action="store_true", help="Less worker-side training log.")
    parser.add_argument("--log-steps", action="store_true", help="Print per-mini-batch step logs (very verbose).")
    parser.add_argument("--die-after-first-round", action="store_true")
    parser.add_argument(
        "--max-fed-rounds",
        type=int,
        default=0,
        help="Stop after this many FedAvg aggregations (0 = unlimited).",
    )
    args = parser.parse_args()

    client = TrackerClient(args.tracker)
    vram, cpus = sniff_hardware_defaults()
    worker_id = f"node-{uuid.uuid4().hex[:12]}"
    reg = client.register(
        worker_id=worker_id,
        gpu_vram_mb=vram,
        cpu_count=cpus,
        host_label=args.host_label,
        hardware_report=hardware_report_for_register(),
    )
    worker_id = reg.worker_id
    print(f"registered worker_id={worker_id}", flush=True)
    client.log_event(worker_id, f"registered gpu_vram_mb={vram} cpu_count={cpus}", host_label=args.host_label)

    base = build_dataset_base(dataset=args.dataset, root="./data")
    # Prefer GPU when available (CPU-only torch builds will still fall back to CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_holder: dict[str, Optional[str]] = {"id": None}
    stop = threading.Event()

    def hb_loop() -> None:
        while not stop.is_set():
            tid = task_holder["id"]
            try:
                client.heartbeat(worker_id, tid)
            except Exception as e:
                print(f"heartbeat error: {e}", flush=True)
            stop.wait(args.heartbeat_sec)

    hb_thread = threading.Thread(target=hb_loop, daemon=True)
    hb_thread.start()

    try:
        idle_spins = 0
        # Number of aggregation events this worker *observed* (it only sees "aggregation"
        # when it submits the final shard that completes the round).
        fed_rounds_completed = 0
        while idle_spins < 120:
            tr = client.request_task(worker_id)
            if not tr.has_task or tr.task is None:
                idle_spins += 1
                # Do not exit just because we're idle; only exit when the tracker confirms
                # training has stopped (i.e., it received all required shards and ran FedAvg+eval).
                try:
                    hs = client.health().get("tasks", {})
                    if hs.get("training_stopped"):
                        print(
                            f"tracker stopped scheduling ({hs.get('stop_reason')}); exiting cleanly.",
                            flush=True,
                        )
                        stop.set()
                        return
                except Exception:
                    # If /health fails temporarily, just keep polling.
                    pass
                time.sleep(2)
                continue
            idle_spins = 0

            assign = tr.task
            task_holder["id"] = assign.task_id
            model = SmallCNN(in_channels=1, num_classes=10).to(device)
            if tr.global_model_bytes_b64:
                apply_state_dict(model, base64.b64decode(tr.global_model_bytes_b64), map_location="cpu")

            resume_next = assign.resume_next_index
            submitted_once = False
            log_prefix = (
                f"[{args.host_label}] fed_round={assign.round_no} task={assign.task_id} "
                f"rows[{assign.image_start},{assign.image_end})"
            )
            client.log_event(
                worker_id,
                f"start_task round={assign.round_no} {assign.task_id} rows[{assign.image_start},{assign.image_end}) resume_next={resume_next}",
                task_id=assign.task_id,
                host_label=args.host_label,
            )

            while True:
                if args.die_after_first_round and submitted_once:
                    print("exiting early to simulate worker failure (no more heartbeats)", flush=True)
                    stop.set()
                    return

                (
                    weights_bytes,
                    last_idx,
                    batches,
                    done,
                    last_loss,
                    run_acc,
                    eval_acc,
                    ep_planned,
                    ep_done,
                ) = train_shard_batch_loop(
                    model,
                    base,
                    assign.image_start,
                    assign.image_end,
                    resume_next,
                    device,
                    max_steps=args.steps,
                    local_epochs=args.local_epochs,
                    verbose=not args.quiet_training,
                    log_steps=bool(args.log_steps),
                    log_prefix=log_prefix,
                    on_epoch_end=lambda ep_done, ep_total, loss_last, acc_run, last_idx: client.progress_event(
                        worker_id=worker_id,
                        task_id=assign.task_id,
                        local_epoch=ep_done,
                        local_epochs_total=ep_total,
                        host_label=args.host_label,
                        shard_progress_pct=100.0
                        * (max(0, min(last_idx, assign.image_end - 1) - assign.image_start + 1))
                        / max(1, (assign.image_end - assign.image_start)),
                        train_acc_running=acc_run,
                        train_loss_last=loss_last,
                    ),
                )

                resp = client.submit_weights(
                    worker_id=worker_id,
                    task_id=assign.task_id,
                    weights_bytes=weights_bytes,
                    last_index=last_idx,
                    steps_completed=batches,
                    shard_complete=done,
                    train_loss_last=last_loss,
                    train_acc_running=run_acc,
                    shard_eval_acc=eval_acc,
                    local_epochs_planned=ep_planned,
                    local_epochs_completed=ep_done,
                )
                print(f"submit: {resp}", flush=True)
                client.log_event(
                    worker_id,
                    f"submit task={assign.task_id} last_index={last_idx} steps={batches} done={done} "
                    "checkpoint uploaded",
                    task_id=assign.task_id,
                    host_label=args.host_label,
                )
                submitted_once = True
                if "aggregation" in resp:
                    print(f"*** {resp['aggregation']}", flush=True)
                    fed_rounds_completed += 1

                # Clean exit when tracker indicates training is finished.
                if resp.get("training_stopped"):
                    print(f"tracker stopped scheduling ({resp.get('stop_reason')}); exiting cleanly.", flush=True)
                    stop.set()
                    return

                if done:
                    if args.max_fed_rounds > 0 and fed_rounds_completed >= args.max_fed_rounds:
                        print("max federation rounds reached; exiting cleanly.", flush=True)
                        return
                    break
                resume_next = last_idx + 1

            task_holder["id"] = None
    finally:
        stop.set()


if __name__ == "__main__":
    main()
