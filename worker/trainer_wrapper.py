"""
Entrypoint intended to run **inside** the training container.

Environment:
  TRACKER_URL    — e.g. http://host.docker.internal:8000
  WORKER_ID      — UUID from /register
  WORKER_TICKET  — HMAC ticket from /register
  TASK_JSON      — JSON {\"assignment\": {...}, \"weights_b64\": \"...\"}
  STEPS_PER_ROUND — defaults to 50
  MNIST_ROOT     — defaults to /app/data
"""

from __future__ import annotations

import base64
import json
import os
import sys

import torch
from pydantic import BaseModel

if "/app" not in sys.path:
    sys.path.insert(0, "/app")

from shared.models import SmallCNN, apply_state_dict
from shared.protocol import TaskAssignment

from worker.client import TrackerClient
from worker.train_utils import build_dataset_base, train_shard_batch_loop


class ContainerTaskPayload(BaseModel):
    assignment: TaskAssignment
    weights_b64: str | None = None


def main() -> None:
    tracker_url = os.environ["TRACKER_URL"]
    worker_id = os.environ["WORKER_ID"]
    worker_ticket = os.environ.get("WORKER_TICKET")
    task_raw = os.environ["TASK_JSON"]
    steps = int(os.environ.get("STEPS_PER_ROUND", "50"))
    mnist_root = os.environ.get("MNIST_ROOT", "/app/data")

    payload = ContainerTaskPayload.model_validate_json(task_raw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(in_channels=1, num_classes=10).to(device)
    if payload.weights_b64:
        apply_state_dict(model, base64.b64decode(payload.weights_b64), map_location=str(device))

    assign = payload.assignment
    dataset = os.environ.get("DATASET", "fashion_mnist")
    base = build_dataset_base(dataset=dataset, root=mnist_root)
    client = TrackerClient(tracker_url)
    if worker_ticket:
        client.ticket = worker_ticket
    resume_next = assign.resume_next_index

    while True:
        out = train_shard_batch_loop(
            model,
            base,
            assign.image_start,
            assign.image_end,
            resume_next,
            device,
            max_steps=steps,
        )
        weights_bytes, last_idx, batches, done = out[0], out[1], out[2], out[3]
        resp = client.submit_weights(
            worker_id=worker_id,
            task_id=assign.task_id,
            weights_bytes=weights_bytes,
            last_index=last_idx,
            steps_completed=batches,
            shard_complete=done,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "task_id": assign.task_id,
                    "last_index": last_idx,
                    "batches": batches,
                    "shard_complete": done,
                    "tracker": resp,
                }
            ),
            flush=True,
        )
        if done:
            break
        resume_next = last_idx + 1


if __name__ == "__main__":
    main()
