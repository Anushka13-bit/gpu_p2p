# gpu-p2p

**Turn a room full of strangers' laptops into one federated training cluster.**

`gpu-p2p` is a peer-to-peer distributed training system: a central *tracker* shards a dataset,
hands slices out to heterogeneous *worker* nodes (NVIDIA boxes, Apple Silicon laptops, plain
CPUs), collects their locally-trained weights, merges them with Federated Averaging, evaluates
the merged model, and repeats — while a live dashboard shows the whole swarm training in real
time and a Proof-of-Learning credit ledger scores every contributor.

No node ever needs another node's data. Only gradients-worth-of-weights cross the wire.

```
 laptop A (RTX 4090) ──┐                                    ┌── laptop D (M2 Air, CPU/MPS)
 laptop B (M3 Max)   ──┼──► tracker (FastAPI control plane) ◄┼── laptop E (headless CPU)
 laptop C (RTX 3060) ──┘        + React dashboard            └── mock_worker.py (CI/tests)
```

---

## Table of contents

- [Why this exists](#why-this-exists)
- [Architecture at a glance](#architecture-at-a-glance)
- [End-to-end pipeline](#end-to-end-pipeline)
- [Component tour](#component-tour)
- [Data sharding model](#data-sharding-model)
- [Federated Averaging (FedAvg)](#federated-averaging-fedavg)
- [Fault tolerance](#fault-tolerance)
- [Security model](#security-model)
- [Proof-of-Learning credits](#proof-of-learning-credits)
- [Live dashboard](#live-dashboard)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Running it](#running-it)
- [Repository layout](#repository-layout)

---

## Why this exists

Training a model usually assumes one machine (or one tightly-coupled cluster) owns all the
compute. `gpu-p2p` explores the opposite: **opportunistic, heterogeneous, trust-minimized
compute** — any laptop on the LAN (or over a tunnel like ngrok) can join, train on a slice of
data it never keeps, and leave at any point without breaking the run. The tracker treats every
worker as unreliable by default: it times out silent nodes, reassigns orphaned work, and only
rewards contributions that provably improved the shared model.

## Architecture at a glance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRACKER (control plane)                        │
│                         FastAPI · single process · in-RAM state             │
│                                                                               │
│   app.py            scheduler.py         state_manager.py                   │
│   ── REST API   ──►  ── task table   ──►  ── global model bytes             │
│   register/task/     ── watchdog          ── per-shard checkpoints          │
│   submit/health       ── round control    ── disk mirror (checkpoints/)     │
│         │                    │                                              │
│         │                    ▼                                              │
│         │             aggregator.py  ──  FedAvg (element-wise weight mean)  │
│         │                    │                                              │
│         │                    ▼                                              │
│         │             eval_utils.py  ──  val/test accuracy on held-out set  │
│         │                    │                                              │
│         │                    ▼                                              │
│         │             learning_credits.py ── Proof-of-Learning scoring      │
│         │                                                                    │
│         ▼                                                                    │
│   security.py  ── HMAC join-password + per-worker ticket auth               │
└──────────────────────────────┬──────────────────────────────────────────────┘
                                │  HTTP (register / heartbeat / task / submit_weights)
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
     ┌────────────────┐┌────────────────┐┌────────────────┐
     │   WORKER node   ││   WORKER node  ││ mock_worker.py │
     │  docker_manager  │  trainer_wrapper│  (no Docker,   │
     │  ── spins GPU    │  ── container    │   no GPU —     │
     │     container    │     entrypoint   │   CI / demo)   │
     │  client.py       │  train_utils.py  │                │
     │  ── HTTP client  │  ── local epochs │                │
     │     + retries    │     over shard   │                │
     └────────────────┘└────────────────┘└────────────────┘
              │                 │                 │
              └────────── shared/ (protocol.py, models.py, hardware_sniff.py) ──┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  dashboard/  — React + Vite + Recharts, polls tracker REST API              │
│  scripts/watch_scheduler.py — terminal equivalent (no browser needed)       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## End-to-end pipeline

1. **Register.** A worker boots, sniffs its own hardware (`shared/hardware_sniff.py` — NVIDIA
   VRAM via GPUtil, Apple unified-memory heuristic, or Windows WMIC), and `POST /register`s with
   a shared join password. The tracker issues a stateless HMAC-SHA256 **ticket** the worker
   attaches to every subsequent call.
2. **Heartbeat.** A background thread pings `POST /heartbeat` every few seconds. If a worker
   goes silent past `HEARTBEAT_TIMEOUT_SEC`, the scheduler's watchdog reclaims its shard.
3. **Get a task.** `GET /task/{worker_id}` hands out one *pending or orphaned* dataset shard —
   an index range like `[7000, 14000)` — plus the current global model weights (base64 `torch.save`
   blob), biased toward higher-reputation workers claiming higher-value shards.
4. **Train locally.** The worker (inside an isolated Docker container with `--gpus all`, or
   directly for the mock worker) runs `local_epochs` full passes over its shard slice
   (`worker/train_utils.py`), reporting per-epoch progress (`POST /progress`) so the dashboard
   can animate a live progress bar.
5. **Submit weights.** `POST /submit_weights` (multipart: JSON metadata + a raw `.pt` weights
   file) uploads the updated model, the last dataset index consumed (so a resumed/reassigned
   worker knows where to continue), and self-reported metrics (loss, running train accuracy,
   shard eval accuracy). The tracker scores this submission for **interim Proof-of-Learning
   credit** immediately.
6. **Aggregate.** Once every shard for the round is `COMPLETED`, the scheduler calls
   `aggregator.fedavg_state_dicts()` — an element-wise mean over every shard's float tensors —
   producing the next global model version.
7. **Evaluate.** The merged model is scored on held-out validation and test splits
   (`tracker/eval_utils.py`) with **zero worker involvement**, so accuracy numbers can't be gamed
   client-side.
8. **Score the round.** `learning_credits.round_pool_distribution()` compares old vs. new global
   validation accuracy and pays out (or penalizes) every contributing worker proportional to
   their shard's estimated contribution to that delta.
9. **Decide: continue or stop.** If `GPU_P2P_MAX_FED_ROUNDS` is reached, or validation accuracy
   plateaus for `GPU_P2P_EARLYSTOP_PATIENCE` rounds, the tracker sets `training_stopped=true` and
   every worker exits cleanly on its next poll. Otherwise all shards reset to `PENDING` and round
   `N+1` begins from the new global model.

## Component tour

### Tracker (`tracker/`)

| File | Responsibility |
|---|---|
| `app.py` | FastAPI app + every REST endpoint. Thin — delegates all logic to `Scheduler`/`StateManager`. Runs a background `_registry_supervisor` task that prints the live node registry to the terminal and sweeps for timed-out shards. |
| `scheduler.py` | The brain. Owns the **task table** (dataset shard → status/assignee), the **worker roster**, the heartbeat **watchdog**, shard-picking policy (reputation-weighted rarity), round/early-stop policy, and orchestrates FedAvg + eval + credit distribution when a round completes. |
| `aggregator.py` | Pure function: N `torch.save` byte blobs in → one element-wise-averaged blob out. No disk I/O, no side effects. |
| `state_manager.py` | In-RAM source of truth for the current global model and per-shard checkpoints, with a best-effort disk mirror to `checkpoints/` for crash inspection. Decides what weights a worker should start from (global vs. resumed checkpoint vs. orphan recovery). |
| `eval_utils.py` | Tracker-side-only accuracy evaluation on a held-out Fashion-MNIST slice — the ground truth that credits and early-stopping are computed from. |
| `learning_credits.py` | The incentive engine — see [Proof-of-Learning credits](#proof-of-learning-credits). |
| `security.py` | Join-password gate + deterministic, stateless `HMAC-SHA256(worker_id)` ticket issuance/validation. |

### Shared protocol (`shared/`)

| File | Responsibility |
|---|---|
| `protocol.py` | Every Pydantic request/response schema exchanged over the wire — the single source of truth for the tracker↔worker contract (`RegisterRequest`, `TaskAssignment`, `SubmitWeightsMetadata`, `ProgressEvent`, …). |
| `models.py` | The `SmallCNN` (LeNet-style Conv→Conv→FC) trained by every worker, plus `state_dict ↔ bytes` (de)serialization helpers shared by tracker and worker alike. |
| `hardware_sniff.py` | Cross-platform hardware detection: NVIDIA VRAM (GPUtil), Apple Silicon unified-memory heuristic (55% of RAM as an effective GPU budget), Windows WMIC fallback — normalized into the `gpu_vram_mb` field used for compute-tiering and resource gating. |

### Worker (`worker/`)

| File | Responsibility |
|---|---|
| `client.py` | `TrackerClient` — a thin `requests`-based HTTP client for every tracker endpoint, with automatic retry on flaky/chunked connections (e.g. ngrok tunnels). |
| `docker_manager.py` | Launches the training container with `--gpus all` (NVIDIA Container Toolkit) when a GPU is present, falling back to `FALLBACK_MODE=1` otherwise. |
| `trainer_wrapper.py` | The container's actual entrypoint: reads its assignment + starting weights from env vars, runs the train→submit loop until its shard is exhausted. |
| `train_utils.py` | Dataset loading (Fashion-MNIST CSV/torchvision, MNIST) and the local training loop — always completes the full planned local-epoch count over its assigned slice before returning, computing shard-level eval accuracy for the credit engine. |

### Everything else

- **`mock_worker.py`** — a full worker implementation with *no Docker and no GPU dependency*,
  used for local integration testing, CI, and demos (`--die-after-first-round` simulates a
  mid-training crash to exercise the orphan-reassignment path).
- **`dashboard/`** — React 19 + Vite + Recharts SPA that polls `/registry`, `/health`, and
  `/credits` to render live node status, shard progress bars, accuracy charts, and the credit
  leaderboard.
- **`scripts/`** — operational tooling: `watch_scheduler.py` (terminal dashboard),
  `print_hardware.py` (debug what a node will report), `lan_tracker_hint.py` (find your LAN IP
  to share with teammates), `dynamic_hardware_validator.py` (pre-flight VRAM/CPU manifest
  validation + live VRAM safe-kill watchdog for a launcher process).
- **`Dockerfile`** — one image, two roles: run the tracker via `uvicorn tracker.app:app`, or run
  a worker via the default `CMD` (`python -m worker.trainer_wrapper`).

## Data sharding model

The dataset (Fashion-MNIST, 70,000 images by default) is sharded into `GPU_P2P_NUM_SHARDS`
**contiguous, non-overlapping index ranges** — not a random split. Each shard is a first-class
task: `PENDING → ASSIGNED → IN_PROGRESS → COMPLETED`, or `ORPHANED` if its worker disappears.
Workers claim shards, never rows within someone else's shard, which keeps the task table O(shards)
instead of O(rows) and makes progress trivially resumable via `resume_next_index`.

```
[0 ──────────── 7000)[7000 ──────── 14000)[14000 ──── 21000) ... [63000 ──────── 70000)
      shard-0              shard-1              shard-2                shard-9
```

Shard assignment prefers workers whose reported hardware clears configurable gates
(`GPU_P2P_GPU_ONLY`, `GPU_P2P_MIN_VRAM_MB`, `GPU_P2P_REQUIRE_CUDA_TORCH`), and among eligible
shards, higher-reputation workers are steered toward higher-rarity shards
(`learning_credits.shard_rarity_multiplier`) — more responsibility to the nodes that have proven
reliable.

## Federated Averaging (FedAvg)

`tracker/aggregator.py` implements the classic FedAvg merge: load every completed shard's
`state_dict`, average each floating-point tensor **element-wise** across all N shards (integer
buffers like `num_batches_tracked` are taken as-is from the first), and re-serialize. This is
McMahan et al.'s uniform-weight FedAvg — every shard counts equally regardless of size, since
shards are equal-sized by construction. The result becomes `Global Model v{N+1}` and is what
every worker starts its next round from (`state_manager.ensure_initial_global()` guarantees
round 1 starts from one shared random init, not N independent ones — critical, since averaging
unrelated random initializations collapses to near-random accuracy).

## Fault tolerance

- **Heartbeat watchdog** (`scheduler.check_timeouts`, polled every `GPU_P2P_WATCHDOG_INTERVAL_SEC`):
  any `ASSIGNED`/`IN_PROGRESS` shard whose worker hasn't heartbeated in
  `GPU_P2P_HEARTBEAT_TIMEOUT_SEC` flips to `ORPHANED` and becomes claimable by anyone.
- **Checkpoint resume**: every `submit_weights` call persists that shard's weights + last
  consumed index. A worker that resumes an orphaned shard (or reconnects mid-shard) picks up
  exactly where the last checkpoint left off, not from scratch.
- **`mock_worker.py --die-after-first-round`** exists specifically to exercise this path in
  tests — kill a worker mid-shard and watch a second worker inherit it after the timeout.

## Security model

Deliberately minimal, LAN/trusted-group threat model:

1. A single shared `JOIN_PASSWORD` gates `POST /register` and `POST /admin/reset_session`.
2. On successful registration the tracker issues a **ticket**:
   `HMAC-SHA256(MASTER_KEY, worker_id)`, hex-encoded. This is stateless — the tracker never
   stores issued tickets — it just recomputes the HMAC and compares with `hmac.compare_digest`
   on every authenticated call (`heartbeat`, `task`, `submit_weights`, `log`, `progress`).
3. Every non-public endpoint requires `(worker_id, ticket)`, so a leaked `worker_id` alone is
   useless without the corresponding ticket, and tickets can't be forged without `MASTER_KEY`.

This is **not** protection against a malicious/Byzantine participant submitting poisoned
weights — see the credits engine below for the (partial, calibration-based) mitigation.

## Proof-of-Learning credits

`tracker/learning_credits.py` is an incentive layer that pays workers for **measured
contribution to the global model**, not just for showing up — explicitly designed around one
principle: *client-reported metrics are untrusted; only tracker-side evaluation is truth.*

**Two payout phases:**

| Phase | When | Signal |
|---|---|---|
| **Interim** (`interim_submit_credit`) | Every `submit_weights` call | `shard_eval_acc` (worker-reported) vs. the *last known global validation accuracy* — a cheap, per-checkpoint proxy for "is this worker moving in the right direction," scaled by steps-per-credit efficiency. |
| **Round** (`round_pool_distribution`) | After every FedAvg + tracker-side eval | The **actual** validation accuracy delta of the merged model, split across that round's contributing workers weighted by each shard's local gain and rarity. This is the number that can't be spoofed — it's computed entirely off tracker-controlled held-out data. |

**Anti-gaming mechanisms layered on top of both phases:**

- `anti_spam_scale` — discounts submissions with too few training steps, or where
  self-reported train accuracy and shard eval accuracy diverge suspiciously (either signals
  overfitting-to-report or fabricated metrics).
- `reputation_multiplier` — a worker's running reputation (0–100, EMA-updated via
  `update_reputation` from signed credit history) scales future payouts `0.55×–1.65×`, so bad
  actors' influence decays and reliable nodes earn more per contribution.
- `apply_streak_bonus` — a small multiplier for sustained positive contribution streaks.
- Global degradation is **penalized**, split across that round's contributors, so submitting
  weights that measurably hurt the merged model has a cost, not just zero reward.

Exposed via `GET /credits` (full leaderboard + recent event log) and
`GET /credits/leaderboard` (leaderboard only) for the dashboard's credit panel.

## Live dashboard

`dashboard/` (React 19 + Vite + Recharts) polls the tracker's read-only endpoints and renders:

- **Worker roster** — every registered node, compute tier (VRAM-based), live/stale status,
  current shard, live epoch/accuracy.
- **Shard table** — per-shard status, assigned worker, progress %, eval accuracy.
- **Charts** — global val/test accuracy across rounds.
- **Learning credits panel** — the Proof-of-Learning leaderboard.

No dashboard? `scripts/watch_scheduler.py` renders the same node registry + shard table as a
polling terminal UI — useful over SSH or when you just want `htop`-style visibility.

## API reference

All endpoints are served by `tracker/app.py`. Endpoints marked 🔒 require a valid
`(worker_id, ticket)` pair (see [Security model](#security-model)).

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/register` | Join the swarm with the shared password; returns `worker_id` + ticket. |
| `POST` | `/heartbeat` 🔒 | Liveness ping; keeps an assigned shard from timing out. |
| `GET` | `/task/{worker_id}` 🔒 | Request/re-fetch the worker's current shard assignment + starting weights. |
| `POST` | `/submit_weights` 🔒 | Upload trained weights + progress metadata for a shard. |
| `POST` | `/log` 🔒 | Structured worker log line, echoed on the tracker terminal. |
| `POST` | `/progress` 🔒 | Fine-grained per-epoch progress for live dashboard rendering. |
| `GET` | `/health` | Full tracker/task/roster snapshot (used by workers to detect `training_stopped`). |
| `GET` | `/registry` | Node registry + shard table, shaped for dashboards. |
| `GET` | `/credits` | Proof-of-Learning leaderboard + recent credit events. |
| `GET` | `/credits/leaderboard` | Leaderboard only. |
| `GET` | `/global_model` | Current global weights, base64-encoded, plus latest val/test accuracy. |
| `GET` | `/checkpoint/{task_id}` | Raw bytes of a shard's latest checkpoint. |
| `POST` | `/admin/reset_session` | Wipe roster + shards + global model; start a fresh run without restarting the process. |

## Configuration

All tuning is environment-variable driven (see `tracker/scheduler.py` for defaults):

| Variable | Default | Effect |
|---|---|---|
| `JOIN_PASSWORD` | `Antigravity-2026` | Shared secret required to register. |
| `MASTER_KEY` | `default-secret-key` | HMAC key for worker tickets. **Change both in any non-toy deployment.** |
| `GPU_P2P_TOTAL_IMAGES` | `70000` | Total dataset rows to shard over. |
| `GPU_P2P_NUM_SHARDS` | `10` | Number of contiguous shards. |
| `GPU_P2P_HEARTBEAT_TIMEOUT_SEC` | `20.0` | Seconds of silence before a shard is orphaned. |
| `GPU_P2P_WATCHDOG_INTERVAL_SEC` | `1.0` | How often the watchdog sweeps for timeouts. |
| `GPU_P2P_MAX_FED_ROUNDS` | `1` | FedAvg rounds before stopping (`0` = unlimited). |
| `GPU_P2P_EARLYSTOP_PATIENCE` | `3` | Rounds without meaningful val-acc improvement before stopping. |
| `GPU_P2P_EARLYSTOP_MIN_DELTA` | `0.1` | Minimum val-acc gain (%) to count as improvement. |
| `GPU_P2P_GPU_ONLY` | `0` | If `1`, only assign shards to GPU-capable workers. |
| `GPU_P2P_MIN_VRAM_MB` / `GPU_P2P_MIN_THREADS` | `1` / `1` | Minimum reported specs to be eligible for shards. |
| `GPU_P2P_REQUIRE_CUDA_TORCH` | `0` | If `1`, require `torch.cuda.is_available()` at registration. |
| `GPU_P2P_CHECKPOINT_DIR` | `./checkpoints` | Disk mirror location for global + per-shard weights. |
| `FASHION_MNIST_CSV_DIR` | `archive 2/` | Location of Kaggle Fashion-MNIST CSVs for CSV-mode datasets and tracker-side eval. |

## Running it

**1. Start the tracker:**

```bash
PYTHONPATH=. uvicorn tracker.app:app --host 0.0.0.0 --port 8000
```

**2. Join workers** — either a real GPU container:

```bash
docker build -t gpu_p2p .
docker run --rm --gpus all \
  -e TRACKER_URL=http://host.docker.internal:8000 \
  -e WORKER_ID=... -e WORKER_TICKET=... -e TASK_JSON='...' \
  gpu_p2p
```

or a zero-dependency mock worker for local testing:

```bash
PYTHONPATH=. python mock_worker.py --tracker http://127.0.0.1:8000
```

Run several `mock_worker.py` instances in separate terminals to simulate a multi-node swarm.

**3. Watch it train:**

```bash
# Terminal dashboard
python3 scripts/watch_scheduler.py --tracker http://127.0.0.1:8000

# Or the full React dashboard
cd dashboard && npm install && npm run dev
```

## Repository layout

```
gpu-p2p/
├── tracker/            # FastAPI control plane: scheduling, FedAvg, eval, credits, auth
├── worker/             # Container entrypoint, training loop, Docker launcher, HTTP client
├── shared/              # Wire protocol schemas, model definition, hardware detection
├── dashboard/           # React + Vite live monitoring UI
├── scripts/              # Operational CLIs (terminal dashboard, hardware probe, LAN hint)
├── mock_worker.py       # Dependency-free worker for tests/demos
├── checkpoints/         # Disk mirror of global + per-shard weights (gitignored in practice)
└── Dockerfile           # Single image serving both tracker and worker roles
```
